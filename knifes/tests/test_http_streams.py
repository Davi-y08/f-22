from __future__ import annotations

import threading
import time
import unittest
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import cv2
import numpy as np

from display.renderer import DisplayRenderer, _prepare_display_frame
from streams.rtsp_client import RTSPClient, _http_source_key, av, test_stream_source


class HTTPCamera:
    def __init__(self, disconnect_first: bool = False) -> None:
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.requests: Counter[str] = Counter()
        self.active = 0
        self.peak = 0
        self.disconnect_first = disconnect_first
        self.stall_first = False
        _, encoded = cv2.imencode(".jpg", np.full((120, 160, 3), 140, np.uint8))
        jpeg = encoded.tobytes()
        chunk = (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                 + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
        camera = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                with camera.lock:
                    camera.requests[self.path] += 1
                    request_number = camera.requests[self.path]
                    camera.active += 1
                    camera.peak = max(camera.peak, camera.active)
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    sent = 0
                    while not camera.stop.is_set():
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        sent += 1
                        if camera.stall_first and request_number == 1 and sent >= 8:
                            camera.stop.wait(8)
                            break
                        if camera.disconnect_first and request_number == 1 and sent >= 8:
                            break
                        camera.stop.wait(.03)
                except OSError:
                    pass
                finally:
                    self.close_connection = True
                    with camera.lock:
                        camera.active -= 1

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/video"

    def close(self) -> None:
        self.stop.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


@unittest.skipIf(av is None, "Install requirements-lite.txt to exercise the real HTTP decoder.")
class HTTPStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = HTTPCamera()
        self.clients: list[RTSPClient] = []

    def tearDown(self) -> None:
        for client in self.clients:
            client.stop()
        self.camera.close()

    def client(self, url: str | None = None, backend: str = "auto") -> RTSPClient:
        client = RTSPClient(url or self.camera.url, backend_preference=backend,
                            reconnect_initial_delay=.1, reconnect_max_delay=.2)
        self.clients.append(client)
        client.start()
        return client

    def assert_live(self, client: RTSPClient, timeout: float = 4.0) -> int:
        packet = client.read_latest(timeout)
        self.assertIsNotNone(packet, client.status_snapshot())
        self.assertEqual(packet.frame.shape, (120, 160, 3))
        return packet.frame_id

    def test_four_consumers_of_same_url_use_one_http_connection(self) -> None:
        clients = [self.client() for _ in range(4)]
        sequences = [self.assert_live(client) for client in clients]
        self.assertEqual(self.camera.requests["/video"], 1)
        self.assertEqual(self.camera.peak, 1)
        for client, previous in zip(clients, sequences):
            self.assertGreater(self.assert_live(client), previous)
            self.assertEqual(client.status_snapshot()["shared_consumers"], 4)

    def test_validation_reuses_live_connection(self) -> None:
        client = self.client()
        self.assert_live(client)
        self.assertTrue(test_stream_source(self.camera.url))
        self.assertEqual(self.camera.requests["/video"], 1)
        self.assert_live(client)

    def test_slow_consumer_does_not_block_fast_consumer(self) -> None:
        fast, slow = self.client(), self.client()
        previous = self.assert_live(fast)
        for _ in range(10):
            current = self.assert_live(fast)
            self.assertGreater(current, previous)
            previous = current
        self.assertLessEqual(slow.status_snapshot()["queue_depth"], 2)
        latest = self.assert_live(slow)
        self.assertGreaterEqual(latest, previous - 1)

    def test_stopping_first_consumer_keeps_others_online(self) -> None:
        first, second = self.client(), self.client()
        self.assert_live(first)
        previous = self.assert_live(second)
        first.stop()
        self.assertGreater(self.assert_live(second), previous)
        self.assertEqual(self.camera.requests["/video"], 1)
        second.stop()
        self.assertFalse(second.status_snapshot()["online"])
        replacement = self.client()
        self.assert_live(replacement)
        self.assertEqual(self.camera.requests["/video"], 2)

    def test_distinct_paths_on_same_ip_are_not_combined(self) -> None:
        clients = [self.client(self.camera.url + f"?channel={channel}") for channel in range(3)]
        for client in clients:
            self.assert_live(client)
        self.assertEqual(sum(self.camera.requests.values()), 3)
        self.assertEqual(self.camera.peak, 3)

    def test_reconnect_is_shared_and_keeps_successful_decoder(self) -> None:
        self.camera.disconnect_first = True
        first, second = self.client(), self.client()
        self.assert_live(first)
        self.assert_live(second)
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            self.assert_live(first, timeout=3)
            if first.status_snapshot()["reconnect_attempts"] and self.camera.requests["/video"] > 1:
                break
        self.assertEqual(self.camera.requests["/video"], 2)
        self.assert_live(second)
        self.assertEqual(first.status_snapshot()["backend"], "pyav")
        self.assertEqual(self.camera.peak, 1)

    def test_opencv_fallback_reads_http_with_bounded_connection(self) -> None:
        client = self.client(backend="opencv")
        self.assert_live(client, timeout=8)
        self.assertEqual(client.status_snapshot()["backend"], "opencv")

    def test_stalled_http_connection_times_out_and_both_consumers_recover(self) -> None:
        self.camera.stall_first = True
        first, second = self.client(), self.client()
        self.assert_live(first)
        self.assert_live(second)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            first.read_latest(.2)
            if self.camera.requests["/video"] >= 2 and first.status_snapshot()["online"]:
                break
        self.assertEqual(self.camera.requests["/video"], 2)
        self.assert_live(first)
        self.assert_live(second)
        self.assertGreaterEqual(first.status_snapshot()["reconnect_attempts"], 1)
        self.assertEqual(first.status_snapshot()["backend"], "pyav")


class HTTPIdentityTests(unittest.TestCase):
    def test_default_port_and_host_casing_identify_same_source(self) -> None:
        self.assertEqual(_http_source_key("http://CAMERA/video"), _http_source_key("http://camera:80/video"))

    def test_channel_credentials_and_path_are_not_discarded(self) -> None:
        base = "http://user:password@camera/video?channel=1"
        for different in [base.replace("channel=1", "channel=2"), base.replace("password", "other"),
                          base.replace("/video", "/Video"), base.replace("camera/", "camera:8080/")]:
            self.assertNotEqual(_http_source_key(base), _http_source_key(different))


class DisplayIsolationTests(unittest.TestCase):
    def test_one_bad_camera_does_not_stop_another_window(self) -> None:
        renderer = DisplayRenderer()
        frame = np.zeros((40, 40, 3), np.uint8)
        renderer.submit("broken", "Broken", frame)
        renderer.submit("working", "Working", frame)
        visited = []

        def present(item):
            visited.append(item.camera_id)
            if item.camera_id == "broken":
                raise cv2.error("invalid frame")
            renderer._stop_event.set()

        with patch.object(renderer, "_present_frame", side_effect=present), patch("display.renderer.cv2.waitKey", return_value=-1), patch("display.renderer.logging.getLogger"):
            renderer._loop()
        self.assertEqual(visited, ["broken", "working"])

    def test_preview_enhancement_operates_after_resizing(self) -> None:
        original = np.zeros((1080, 1920, 3), np.uint8)
        with patch("display.renderer._enhance_frame", side_effect=lambda frame: frame) as enhance:
            result = _prepare_display_frame(original, 640, False, "contain", "auto", True, (1920, 1080))
        self.assertEqual(result.shape, (360, 640, 3))
        self.assertEqual(enhance.call_args.args[0].shape, result.shape)
        self.assertEqual(original.shape, (1080, 1920, 3))


if __name__ == "__main__":
    unittest.main()
