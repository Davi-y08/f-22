from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from discovery.security import validate_stream_url
from discovery.service import (
    DiscoveredCamera,
    _SOURCE_RESOLUTION_CACHE,
    build_candidate_http_urls,
    build_candidate_rtsp_urls,
    _collapse_local_rtsp_aliases,
    _expand_manual_stream_candidates,
    _filter_safe_stream_candidates,
    _merge_discovered_camera,
    _parse_onvif_response,
    _probe_http_camera_endpoint,
    resolve_camera_source,
)
from streams.rtsp_client import test_stream_source


class _FakeHttpSocket:
    def __init__(self, responses: dict[str, bytes], calls: list[str]) -> None:
        self._responses = responses
        self._calls = calls
        self._path = "/"
        self._sent_response = False

    def __enter__(self) -> "_FakeHttpSocket":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def settimeout(self, *_: object) -> None:
        return None

    def sendall(self, payload: bytes) -> None:
        request_line = payload.decode("utf-8", errors="ignore").splitlines()[0]
        self._path = request_line.split()[1]
        self._calls.append(self._path)

    def recv(self, *_: object) -> bytes:
        if self._sent_response:
            return b""
        self._sent_response = True
        return self._responses.get(
            self._path,
            b"HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\n\r\n",
        )


class StreamUrlSecurityTests(unittest.TestCase):
    def test_rejects_dangerous_manual_urls(self) -> None:
        self.assertFalse(validate_stream_url("file:///etc/passwd").valid)
        self.assertFalse(validate_stream_url("http://127.0.0.1:8080/video").valid)
        self.assertFalse(validate_stream_url("http://169.254.169.254/latest/meta-data").valid)

    def test_allows_private_camera_urls(self) -> None:
        result = validate_stream_url("rtsp://user:pass@192.168.1.10:554/live")

        self.assertTrue(result.valid)
        self.assertEqual(result.host, "192.168.1.10")

    def test_loopback_can_be_allowed_for_local_stress_tests(self) -> None:
        result = validate_stream_url("rtsp://127.0.0.1:8554/cam1", allow_loopback=True)

        self.assertTrue(result.valid)


class DiscoveryParsingTests(unittest.TestCase):
    def test_expands_ip_webcam_base_url(self) -> None:
        candidates = _expand_manual_stream_candidates("http://192.168.1.244:8080/")

        self.assertIn("http://192.168.1.244:8080/video", candidates)

    def test_filters_unsafe_stream_candidates(self) -> None:
        candidates = _filter_safe_stream_candidates(
            [
                "http://127.0.0.1:8080/video",
                "rtsp://192.168.1.20:554/live",
            ]
        )

        self.assertEqual(candidates, ["rtsp://192.168.1.20:554/live"])

    def test_rtsp_path_candidates_are_prioritized(self) -> None:
        camera = DiscoveredCamera(
            key="network:192.168.1.55:rtsp:8554:cam2",
            kind="network",
            name="RTSP Stream cam2",
            host="192.168.1.55",
            rtsp_ports=[8554],
            rtsp_stream_paths=["cam2"],
        )

        candidates = build_candidate_rtsp_urls(camera)

        self.assertEqual(candidates[0], "rtsp://192.168.1.55:8554/cam2")

    def test_default_http_candidates_include_standard_camera_web_ports(self) -> None:
        camera = DiscoveredCamera(
            key="network:192.168.1.44",
            kind="network",
            name="HTTP Camera",
            host="192.168.1.44",
        )

        candidates = build_candidate_http_urls(camera)

        self.assertIn("http://192.168.1.44:80/video", candidates)
        self.assertIn("http://192.168.1.44:8080/video", candidates)

    def test_http_probe_checks_stream_paths_without_root_hint(self) -> None:
        responses = {
            "/": b"HTTP/1.1 404 Not Found\r\nServer: nginx\r\nContent-Type: text/html\r\n\r\nnot here",
            "/video": b"HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\n\r\n",
            "/videofeed": (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n"
                b"--frame"
            ),
        }
        calls: list[str] = []

        def fake_connect(*_: object, **__: object) -> _FakeHttpSocket:
            return _FakeHttpSocket(responses, calls)

        with patch("discovery.service.socket.create_connection", side_effect=fake_connect):
            host, port, is_camera, _server, stream_paths, _name, status, confidence = _probe_http_camera_endpoint(
                "192.168.1.44",
                8080,
                0.01,
            )

        self.assertEqual(host, "192.168.1.44")
        self.assertEqual(port, 8080)
        self.assertTrue(is_camera)
        self.assertIn("/videofeed", calls)
        self.assertEqual(stream_paths, ["/videofeed"])
        self.assertEqual(status, "http_stream_detected")
        self.assertGreaterEqual(confidence, 0.78)

    def test_parses_onvif_uuid_and_scopes(self) -> None:
        payload = b"""
        <e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
          xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
          xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery">
          <e:Body>
            <d:ProbeMatches>
              <d:ProbeMatch>
                <w:EndpointReference>
                  <w:Address>urn:uuid:12345678-90ab-cdef-1234-567890abcdef</w:Address>
                </w:EndpointReference>
                <d:XAddrs>http://192.168.1.50/onvif/device_service</d:XAddrs>
                <d:Scopes>
                  onvif://www.onvif.org/name/Entrada
                  onvif://www.onvif.org/hardware/IPC-HFW
                  onvif://www.onvif.org/manufacturer/Dahua
                </d:Scopes>
              </d:ProbeMatch>
            </d:ProbeMatches>
          </e:Body>
        </e:Envelope>
        """

        camera = _parse_onvif_response(payload, "192.168.1.99")

        self.assertIsNotNone(camera)
        assert camera is not None
        self.assertEqual(camera.host, "192.168.1.50")
        self.assertEqual(camera.device_uuid, "12345678-90ab-cdef-1234-567890abcdef")
        self.assertEqual(camera.name, "Entrada")
        self.assertEqual(camera.model, "IPC-HFW")
        self.assertEqual(camera.manufacturer, "Dahua")
        self.assertEqual(camera.status, "onvif_detected")

    def test_merges_duplicates_by_host(self) -> None:
        discovered: dict[str, DiscoveredCamera] = {}
        _merge_discovered_camera(
            discovered,
            DiscoveredCamera(
                key="network:192.168.1.10",
                kind="network",
                name="RTSP Camera 192.168.1.10",
                host="192.168.1.10",
                rtsp_ports=[554],
                confidence=0.7,
                status="rtsp_detected",
            ),
        )
        _merge_discovered_camera(
            discovered,
            DiscoveredCamera(
                key="network:192.168.1.10",
                kind="network",
                name="IP Webcam 192.168.1.10",
                host="192.168.1.10",
                http_ports=[8080],
                http_stream_paths=["/video"],
                confidence=0.86,
                status="http_stream_detected",
            ),
        )

        camera = next(iter(discovered.values()))
        self.assertEqual(len(discovered), 1)
        self.assertEqual(camera.rtsp_ports, [554])
        self.assertEqual(camera.http_ports, [8080])
        self.assertEqual(camera.status, "http_stream_detected")
        self.assertEqual(camera.confidence, 0.86)

    def test_collapses_local_rtsp_aliases_by_path(self) -> None:
        cameras = [
            DiscoveredCamera(
                key="network:127.0.0.1:rtsp:8554:cam1",
                kind="network",
                name="RTSP Stream cam1 (127.0.0.1)",
                host="127.0.0.1",
                rtsp_ports=[8554],
                rtsp_stream_paths=["cam1"],
                status="rtsp_stream_detected",
            ),
            DiscoveredCamera(
                key="network:192.168.1.155:rtsp:8554:cam1",
                kind="network",
                name="RTSP Stream cam1 (192.168.1.155)",
                host="192.168.1.155",
                rtsp_ports=[8554],
                rtsp_stream_paths=["cam1"],
                status="rtsp_stream_detected",
            ),
            DiscoveredCamera(
                key="network:127.0.0.1:rtsp:8554:cam2",
                kind="network",
                name="RTSP Stream cam2 (127.0.0.1)",
                host="127.0.0.1",
                rtsp_ports=[8554],
                rtsp_stream_paths=["cam2"],
                status="rtsp_stream_detected",
            ),
        ]

        with patch(
            "discovery.service._local_interface_hosts",
            return_value={"127.0.0.1", "192.168.1.155"},
        ):
            collapsed = _collapse_local_rtsp_aliases(cameras)

        self.assertEqual(len(collapsed), 2)
        self.assertEqual({camera.rtsp_stream_paths[0] for camera in collapsed}, {"cam1", "cam2"})


class SourceResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        _SOURCE_RESOLUTION_CACHE.clear()

    def test_resolution_redacts_tested_urls(self) -> None:
        camera = DiscoveredCamera(
            key="network:192.168.1.30",
            kind="network",
            name="Entrada",
            host="192.168.1.30",
            rtsp_ports=[554],
            discovery_sources={"rtsp-scan"},
        )

        with patch("discovery.service.test_stream_source", return_value=True):
            result = resolve_camera_source(
                camera,
                username="admin",
                password="senha",
                validate_stream=True,
            )

        self.assertTrue(result.validated)
        self.assertIn("admin:senha", str(result.source))
        self.assertNotIn("admin", " ".join(result.tested_urls))
        self.assertNotIn("senha", " ".join(result.tested_urls))
        self.assertIn("***:***", result.tested_urls[0])

    def test_manual_url_is_tested_before_generic_candidates(self) -> None:
        camera = DiscoveredCamera(
            key="network:192.168.1.32",
            kind="network",
            name="RTSP Server",
            host="192.168.1.32",
            rtsp_ports=[8554],
            discovery_sources={"rtsp-scan"},
        )
        calls: list[str] = []

        def fake_probe(url: str, *_: object, **__: object) -> bool:
            calls.append(url)
            return True

        with patch("discovery.service.test_stream_source", side_effect=fake_probe):
            result = resolve_camera_source(
                camera,
                manual_rtsp_url="rtsp://192.168.1.32:8554/cam4",
                validate_stream=True,
            )

        self.assertTrue(result.validated)
        self.assertEqual(calls[0], "rtsp://192.168.1.32:8554/cam4")

    def test_resolution_uses_short_lived_success_cache(self) -> None:
        camera = DiscoveredCamera(
            key="network:192.168.1.31",
            kind="network",
            name="Entrada",
            host="192.168.1.31",
            rtsp_ports=[554],
            discovery_sources={"rtsp-scan"},
        )

        with patch("discovery.service.test_stream_source", return_value=True) as probe:
            first = resolve_camera_source(camera, username="admin", password="senha")
            second = resolve_camera_source(camera, username="admin", password="senha")

        self.assertTrue(first.validated)
        self.assertTrue(second.validated)
        self.assertEqual(probe.call_count, 1)

    def test_resolution_cache_keeps_rtsp_paths_separate(self) -> None:
        cam1 = DiscoveredCamera(
            key="network:127.0.0.1:rtsp:8554:cam1",
            kind="network",
            name="Cam 1",
            host="127.0.0.1",
            rtsp_ports=[8554],
            rtsp_stream_paths=["cam1"],
        )
        cam2 = DiscoveredCamera(
            key="network:127.0.0.1:rtsp:8554:cam2",
            kind="network",
            name="Cam 2",
            host="127.0.0.1",
            rtsp_ports=[8554],
            rtsp_stream_paths=["cam2"],
        )

        with patch("discovery.service.test_stream_source", return_value=True):
            first = resolve_camera_source(cam1)
            second = resolve_camera_source(cam2)

        self.assertEqual(first.source, "rtsp://127.0.0.1:8554/cam1")
        self.assertEqual(second.source, "rtsp://127.0.0.1:8554/cam2")


class StreamSourceProbeTests(unittest.TestCase):
    def test_rtsp_validation_uses_fast_describe_before_decoder(self) -> None:
        class FakeSocket:
            def __enter__(self) -> "FakeSocket":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def settimeout(self, *_: object) -> None:
                return None

            def sendall(self, *_: object) -> None:
                return None

            def recv(self, *_: object) -> bytes:
                return b"RTSP/1.0 200 OK\r\nCSeq: 1\r\n\r\n"

        with (
            patch("streams.rtsp_client.socket.create_connection", return_value=FakeSocket()) as connect,
            patch("streams.rtsp_client.cv2.VideoCapture") as video_capture,
        ):
            self.assertTrue(test_stream_source("rtsp://127.0.0.1:8554/cam1", timeout_seconds=0.2))

        connect.assert_called_once()
        video_capture.assert_not_called()

    def test_authenticated_rtsp_401_falls_back_to_decoder(self) -> None:
        class FakeSocket:
            def __enter__(self) -> "FakeSocket":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def settimeout(self, *_: object) -> None:
                return None

            def sendall(self, *_: object) -> None:
                return None

            def recv(self, *_: object) -> bytes:
                return b"RTSP/1.0 401 Unauthorized\r\nCSeq: 1\r\n\r\n"

        fake_av = Mock()
        fake_container = Mock()
        fake_container.streams = [Mock(type="video")]
        fake_container.decode.return_value = iter([object()])
        fake_av.open.return_value = fake_container

        with (
            patch("streams.rtsp_client.socket.create_connection", return_value=FakeSocket()),
            patch("streams.rtsp_client.av", fake_av),
        ):
            self.assertTrue(test_stream_source("rtsp://user:pass@192.168.1.10:554/live", timeout_seconds=0.2))

        fake_av.open.assert_called_once()


if __name__ == "__main__":
    unittest.main()
