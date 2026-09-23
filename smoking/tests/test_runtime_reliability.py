from __future__ import annotations

import json
import tempfile
import threading
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from agent.worker import CameraWorker
from cloud.client import CloudClient, CloudClientError
from events.emitter import DetectionEvent, FileEventEmitter
from models.loader import Detection, OnnxModelSession, _SimpleTracker, _class_aware_nms, _decode_predictions, _reshape_predictions
from streams.rtsp_client import FramePacket, _StreamCapture, _OpenCVReader
from utils.config import CameraConfig, CloudConfig, DisplayConfig, save_raw_config


def detection() -> Detection:
    return Detection("knife", .9, (10, 10, 30, 30), "model", "knife_detected", False)


def event(camera_id: str = "entrada") -> DetectionEvent:
    return DetectionEvent.create("agent", camera_id, "Entrada", "alert", .9, "model", "knife", (1, 2, 3, 4))


class CloudReliabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "outbox.sqlite3"
        self.config = CloudConfig(enabled=True, api_base_url="https://example.test", agent_access_key="key")
        self.clients: list[CloudClient] = []

    def tearDown(self) -> None:
        for client in self.clients:
            client.close()
        self.directory.cleanup()

    def client(self, config: CloudConfig | None = None) -> CloudClient:
        with patch("cloud.client.threading.Thread.start"):
            client = CloudClient(config or self.config, "agent", outbox_path=self.path)
        self.clients.append(client)
        return client

    def test_retry_survives_restart_and_keeps_event_identity(self) -> None:
        first = self.client()
        alert = event()
        with patch("cloud.outbox.time.time", return_value=100):
            first.emit_event_async(alert)
            with patch.object(first, "_post_json", side_effect=CloudClientError("offline")):
                self.assertTrue(first._sync_next_event())
            self.assertFalse(first._sync_next_event())
        self.assertEqual(first.status_snapshot()["pending_events"], 1)
        first.close()
        second = self.client()
        with patch("cloud.outbox.time.time", return_value=200), patch.object(second, "_post_json", return_value={}) as send:
            self.assertTrue(second._sync_next_event())
        self.assertEqual(send.call_args.args[1]["event_id"], alert.event_id)
        self.assertEqual(second.status_snapshot()["pending_events"], 0)

    def test_rejected_event_is_retained_without_blocking_other_events(self) -> None:
        client = self.client()
        client.emit_event_async(event())
        with patch.object(client, "_post_json", side_effect=CloudClientError("invalid", retryable=False)):
            client._sync_next_event()
        client.emit_event_async(event())
        with patch.object(client, "_post_json", return_value={}):
            client._sync_next_event()
        self.assertEqual(client.status_snapshot()["failed_events"], 1)
        self.assertEqual(client.status_snapshot()["synced_events"], 1)
        self.assertFalse(client._sync_next_event())

    def test_outbox_never_sends_another_account_or_server_events(self) -> None:
        original = self.client()
        original.emit_event_async(event())
        for config in [replace(self.config, agent_access_key="other"), replace(self.config, api_base_url="https://other.test")]:
            client = self.client(config)
            self.assertEqual(client.status_snapshot()["pending_events"], 0)
            self.assertFalse(client._sync_next_event())

    def test_snapshot_is_loaded_only_by_uploader(self) -> None:
        client = self.client()
        image_path = Path(self.directory.name) / "frame.jpg"
        image_path.write_bytes(b"jpeg-bytes")
        alert = event()
        alert.snapshot_path = str(image_path)
        client.emit_event_async(alert)
        payload, _ = client._outbox.next_due()
        self.assertNotIn("snapshot_base64", payload)
        with patch.object(client, "_post_json", return_value={}) as send:
            client._sync_next_event()
        self.assertIn("snapshot_base64", send.call_args.args[1])

    def test_missing_photo_is_reported_instead_of_silently_uploaded(self) -> None:
        client = self.client()
        alert = event()
        alert.snapshot_path = str(Path(self.directory.name) / "missing.jpg")
        client.emit_event_async(alert)
        with patch.object(client, "_post_json") as send:
            client._sync_next_event()
        send.assert_not_called()
        self.assertEqual(client.status_snapshot()["failed_events"], 1)

    def test_duplicate_pending_event_is_queued_once(self) -> None:
        client = self.client()
        alert = event()
        client.emit_event_async(alert)
        client.emit_event_async(alert)
        self.assertEqual(client.status_snapshot()["pending_events"], 1)


class StorageReliabilityTests(unittest.TestCase):
    def test_snapshot_supports_unicode_and_stays_inside_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "c\u00e2meras"
            emitter = FileEventEmitter("agent", root / "events", root / "snapshots")
            alert = emitter.emit(event("../../escape"), np.zeros((32, 48, 3), np.uint8), True)
            image = Path(alert.snapshot_path)
            self.assertTrue(image.is_relative_to((root / "snapshots").resolve()))
            decoded = cv2.imdecode(np.frombuffer(image.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape, (32, 48, 3))

    def test_snapshot_encoding_failure_still_records_alert_and_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            emitter = FileEventEmitter("agent", root / "events", root / "snapshots")
            with patch("events.emitter.cv2.imencode", return_value=(False, None)):
                alert = emitter.emit(event(), np.zeros((10, 10, 3), np.uint8), True)
            record = json.loads(next((root / "events").glob("*.jsonl")).read_text())
            self.assertIsNone(alert.snapshot_path)
            self.assertIn("snapshot_error", record["metadata"])

    def test_interrupted_config_save_preserves_original(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            save_raw_config(path, {"agent_id": "original"})
            with patch("utils.storage.os.fsync", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    save_raw_config(path, {"agent_id": "changed"})
            self.assertEqual(json.loads(path.read_text())["agent_id"], "original")
            self.assertEqual(list(path.parent.glob(".pending-*")), [])


class DetectionReliabilityTests(unittest.TestCase):
    def test_overlapping_different_classes_are_kept_and_duplicates_removed(self) -> None:
        indices = _class_aware_nms([(0, 0, 50, 50)] * 3, [.95, .9, .8], [0, 1, 0], .25, .45)
        self.assertEqual(indices, [0, 1])

    def test_single_class_models_and_both_tensor_layouts(self) -> None:
        data = np.array([[20, 30, 10, 12, .9], [5, 5, 2, 2, .1]], dtype=np.float32)
        for output in [data[None], data.T[None]]:
            reshaped = _reshape_predictions(output, class_count=1)
            ids, scores, boxes = _decode_predictions(reshaped, ("knife",), .5)
            self.assertEqual(ids, [0])
            self.assertEqual(boxes, [(15, 24, 25, 36)])

    def test_invalid_model_values_do_not_generate_events(self) -> None:
        values = np.array([[10, 10, -2, 4, .9], [10, 10, 2, 4, np.nan], [np.inf, 10, 2, 4, .9]])
        self.assertEqual(_decode_predictions(values, ("knife",), .5), ([], [], []))

    def test_empty_inferences_expire_old_tracking_identity(self) -> None:
        session = object.__new__(OnnxModelSession)
        session._tracker = _SimpleTracker(max_stale_frames=2)
        original = detection()
        session._track_detections([original], 1)
        for _ in range(3):
            session._track_detections([], 1)
        returning = detection()
        session._track_detections([returning], 1)
        self.assertNotEqual(original.track_id, returning.track_id)


class CameraReliabilityTests(unittest.TestCase):
    def test_event_uses_capture_time_even_when_inference_finishes_later(self) -> None:
        emitter = Mock()
        worker = CameraWorker(SimpleNamespace(agent_id="agent"), CameraConfig("cam", "Cam", 0, ()), emitter, Mock())
        worker._model_sessions = [Mock(infer=Mock(return_value=([detection()], 500)))]
        captured_at = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        worker._analyze_frame(np.zeros((40, 40, 3), np.uint8), captured_at=captured_at)
        self.assertEqual(emitter.emit.call_args.kwargs["event"].timestamp, captured_at.isoformat())

    def test_latest_frame_skips_backlog_and_counts_discarded_frames(self) -> None:
        client = _StreamCapture(0, queue_maxsize=3)
        for index in range(3):
            client._push_frame(FramePacket(index, index, datetime.now(timezone.utc)))
        self.assertEqual(client.read_latest(timeout=0).frame_id, 2)
        self.assertEqual(client.status_snapshot()["dropped_frames"], 2)

    def test_failed_reader_is_closed_before_reconnect_wait(self) -> None:
        client = _StreamCapture("http://example.test/video", backend_preference="opencv")
        reader = Mock(backend_name="opencv")
        reader.read.side_effect = RuntimeError("disconnected")
        def wait_and_stop(seconds: float) -> None:
            reader.close.assert_called_once()
            client._stop_event.set()
        with patch.object(client, "_build_reader", return_value=reader), patch.object(client, "_sleep_with_stop", side_effect=wait_and_stop):
            client._reader_loop()
        self.assertFalse(client.status_snapshot()["online"])

    def test_failed_opencv_capture_is_released_before_fallback(self) -> None:
        failed = Mock()
        failed.isOpened.return_value = False
        working = Mock()
        working.isOpened.return_value = True
        reader = _OpenCVReader("http://example.test/video")
        with patch("streams.rtsp_client._create_network_capture", side_effect=[failed, working]):
            reader.open()
        failed.release.assert_called_once()
        reader.close()
        working.release.assert_called_once()

    def test_cooldown_starts_only_after_event_is_saved(self) -> None:
        emitter = Mock()
        emitter.emit.side_effect = [OSError("disk failure"), None]
        worker = CameraWorker(SimpleNamespace(agent_id="agent"), CameraConfig("cam", "Cam", 0, ()), emitter, Mock())
        worker._model_sessions = [Mock(infer=Mock(return_value=([detection()], 1)))]
        frame = np.zeros((40, 40, 3), np.uint8)
        with self.assertRaises(OSError):
            worker._analyze_frame(frame)
        worker._analyze_frame(frame)
        worker._analyze_frame(frame)
        self.assertEqual(emitter.emit.call_count, 2)

    def test_preview_continues_while_inference_is_blocked(self) -> None:
        started = threading.Event()
        release = threading.Event()
        preview = threading.Event()
        frame = np.zeros((40, 40, 3), np.uint8)
        def infer(image):
            started.set()
            release.wait(3)
            return [], 500
        worker = CameraWorker(
            SimpleNamespace(agent_id="agent"), CameraConfig("cam", "Cam", 0, (), display=DisplayConfig(enabled=True)),
            Mock(), Mock(),
        )
        worker.stream = Mock()
        worker.stream.status_snapshot.return_value = {
            "queue_depth": 0, "dropped_frames": 0, "reconnect_attempts": 0, "online": True,
            "backend": "test", "last_frame_at": None, "last_error": None,
        }
        worker.stream.read_latest.return_value = SimpleNamespace(frame=frame)
        def render(*args, **kwargs):
            if started.is_set() and not release.is_set():
                preview.set()
                worker.stop()
        with patch("agent.worker.build_model_sessions", return_value=[Mock(infer=infer)]), patch.object(worker, "_safe_maybe_render_frame", side_effect=render):
            worker.start()
            try:
                self.assertTrue(preview.wait(2), "preview blocked by inference")
            finally:
                release.set()
                worker.stop()
                worker.join(timeout=3)
        self.assertFalse(worker.is_alive())


if __name__ == "__main__":
    unittest.main()
