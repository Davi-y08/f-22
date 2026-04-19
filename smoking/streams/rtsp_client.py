from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import cv2

try:
    import av
except ImportError:  # pragma: no cover - optional dependency
    av = None


@dataclass(slots=True)
class FramePacket:
    frame_id: int
    frame: Any
    captured_at: datetime


class StreamReadError(RuntimeError):
    """Raised when a backend cannot provide a frame."""


class RTSPClient:
    def __init__(
        self,
        source: str | int,
        backend_preference: str = "auto",
        queue_maxsize: int = 4,
        reconnect_initial_delay: float = 2.0,
        reconnect_max_delay: float = 30.0,
        logger: Any | None = None,
    ) -> None:
        self.source = source
        self.backend_preference = backend_preference
        self.queue_maxsize = queue_maxsize
        self.reconnect_initial_delay = reconnect_initial_delay
        self.reconnect_max_delay = reconnect_max_delay
        self.logger = logger

        self._frames: queue.Queue[FramePacket] = queue.Queue(maxsize=self.queue_maxsize)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._status_lock = threading.Lock()
        self._frame_counter = 0
        self._status: dict[str, Any] = {
            "online": False,
            "backend": None,
            "last_frame_at": None,
            "last_error": None,
            "reconnect_attempts": 0,
            "dropped_frames": 0,
        }

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._reader_loop, name="rtsp-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def read_latest(self, timeout: float = 1.0) -> FramePacket | None:
        try:
            packet = self._frames.get(timeout=timeout)
        except queue.Empty:
            return None

        while True:
            try:
                packet = self._frames.get_nowait()
            except queue.Empty:
                return packet

    def status_snapshot(self) -> dict[str, Any]:
        with self._status_lock:
            snapshot = dict(self._status)
        snapshot["queue_depth"] = self._frames.qsize()
        return snapshot

    def _reader_loop(self) -> None:
        delay = self.reconnect_initial_delay
        current_preference = self.backend_preference

        while not self._stop_event.is_set():
            reader = None
            try:
                reader = self._build_reader(current_preference)
                reader.open()
                delay = self.reconnect_initial_delay
                current_preference = self.backend_preference
                self._set_status(
                    online=True,
                    backend=reader.backend_name,
                    last_error=None,
                )

                while not self._stop_event.is_set():
                    frame = reader.read()
                    packet = FramePacket(
                        frame_id=self._next_frame_id(),
                        frame=frame,
                        captured_at=datetime.now(timezone.utc),
                    )
                    self._push_frame(packet)
                    self._set_status(
                        online=True,
                        backend=reader.backend_name,
                        last_frame_at=packet.captured_at.isoformat(),
                        last_error=None,
                    )
            except Exception as exc:
                status = self.status_snapshot()
                if (
                    reader is not None
                    and reader.backend_name == "pyav"
                    and self.backend_preference.lower() == "auto"
                ):
                    current_preference = "opencv"
                self._set_status(
                    online=False,
                    last_error=str(exc),
                    reconnect_attempts=int(status["reconnect_attempts"]) + 1,
                )
                if self.logger:
                    self.logger.warning(
                        "stream_connection_error",
                        extra={
                            "source": str(self.source),
                            "error": str(exc),
                            "retry_in_seconds": delay,
                        },
                    )
                self._sleep_with_stop(delay)
                delay = min(delay * 2, self.reconnect_max_delay)
            finally:
                if reader is not None:
                    reader.close()

        self._set_status(online=False)

    def _build_reader(self, preference_override: str | None = None) -> "_BaseReader":
        if isinstance(self.source, int):
            return _OpenCVReader(self.source)

        preference = (preference_override or self.backend_preference).lower()
        if preference == "pyav" and av is None:
            raise StreamReadError("PyAV foi configurado explicitamente, mas não está instalado.")

        if preference in {"auto", "pyav"} and av is not None:
            try:
                return _PyAVReader(self.source)
            except Exception as exc:
                if preference == "pyav":
                    raise
                if self.logger:
                    self.logger.warning(
                        "pyav_unavailable_for_stream",
                        extra={"source": self.source, "error": str(exc)},
                    )

        return _OpenCVReader(self.source)

    def _push_frame(self, packet: FramePacket) -> None:
        if self._frames.full():
            try:
                self._frames.get_nowait()
                status = self.status_snapshot()
                self._set_status(dropped_frames=int(status["dropped_frames"]) + 1)
            except queue.Empty:
                pass
        self._frames.put_nowait(packet)

    def _next_frame_id(self) -> int:
        self._frame_counter += 1
        return self._frame_counter

    def _set_status(self, **updates: Any) -> None:
        with self._status_lock:
            self._status.update(updates)

    def _sleep_with_stop(self, seconds: float) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline and not self._stop_event.is_set():
            time.sleep(0.2)


class _BaseReader:
    backend_name = "base"

    def open(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def read(self) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class _OpenCVReader(_BaseReader):
    backend_name = "opencv"

    def __init__(self, source: str | int) -> None:
        self.source = source
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        if isinstance(self.source, str):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            self.capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            self.capture = cv2.VideoCapture(self.source)

        if not self.capture or not self.capture.isOpened():
            raise StreamReadError(f"Não foi possível abrir o stream '{self.source}'.")

        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    def read(self) -> Any:
        if not self.capture:
            raise StreamReadError("Stream OpenCV não inicializado.")

        success, frame = self.capture.read()
        if not success or frame is None:
            raise StreamReadError(f"Falha ao ler frame de '{self.source}'.")

        return frame

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()


class _PyAVReader(_BaseReader):
    backend_name = "pyav"

    def __init__(self, source: str) -> None:
        if av is None:
            raise StreamReadError("PyAV não está instalado.")

        self.source = source
        self.container: Any | None = None
        self.video_stream: Any | None = None
        self.frames: Any | None = None

    def open(self) -> None:
        self.container = av.open(
            self.source,
            options={
                "rtsp_transport": "tcp",
                "fflags": "nobuffer",
                "flags": "low_delay",
                "stimeout": "5000000",
            },
        )
        self.video_stream = next(stream for stream in self.container.streams if stream.type == "video")
        self.frames = self.container.decode(self.video_stream)

    def read(self) -> Any:
        if self.frames is None:
            raise StreamReadError("Stream PyAV não inicializado.")

        try:
            frame = next(self.frames)
        except StopIteration as exc:
            raise StreamReadError("Stream encerrado pelo servidor.") from exc

        return frame.to_ndarray(format="bgr24")

    def close(self) -> None:
        if self.container is not None:
            self.container.close()


def test_stream_source(
    source: str | int,
    backend_preference: str = "auto",
    timeout_seconds: float = 5.0,
) -> bool:
    if isinstance(source, int):
        capture = cv2.VideoCapture(source)
        try:
            if not capture or not capture.isOpened():
                return False
            success, _ = capture.read()
            return bool(success)
        finally:
            if capture is not None:
                capture.release()

    preference = backend_preference.lower()
    if preference in {"auto", "pyav"} and av is not None:
        try:
            container = av.open(
                source,
                options={
                    "rtsp_transport": "tcp",
                    "fflags": "nobuffer",
                    "flags": "low_delay",
                    "stimeout": str(int(timeout_seconds * 1_000_000)),
                },
            )
            video_stream = next(stream for stream in container.streams if stream.type == "video")
            frame = next(container.decode(video_stream))
            container.close()
            return frame is not None
        except Exception:
            if preference == "pyav":
                return False

    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    try:
        if not capture or not capture.isOpened():
            return False
        started = time.time()
        while (time.time() - started) < timeout_seconds:
            success, frame = capture.read()
            if success and frame is not None:
                return True
        return False
    finally:
        if capture is not None:
            capture.release()
