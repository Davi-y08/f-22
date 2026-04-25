from __future__ import annotations

import os
import queue
import random
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import cv2

from utils.redaction import redact_url_credentials

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
                    retry_in_seconds = _jittered_delay(delay)
                    self.logger.warning(
                        "stream_connection_error",
                        extra={
                            "source": redact_url_credentials(self.source),
                            "error": str(exc),
                            "retry_in_seconds": round(retry_in_seconds, 2),
                        },
                    )
                else:
                    retry_in_seconds = _jittered_delay(delay)
                self._sleep_with_stop(retry_in_seconds)
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
                        extra={"source": redact_url_credentials(self.source), "error": str(exc)},
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
            self.capture = _create_network_capture(self.source, cv2.CAP_FFMPEG)
            if not self.capture or not self.capture.isOpened():
                self.capture = _create_network_capture(self.source, None)
        else:
            self.capture, self.backend_name = _open_local_capture(int(self.source))

        if not self.capture or not self.capture.isOpened():
            raise StreamReadError(
                f"Não foi possível abrir o stream '{redact_url_credentials(self.source)}'."
            )

        try:
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass

    def read(self) -> Any:
        if not self.capture:
            raise StreamReadError("Stream OpenCV não inicializado.")

        last_error: Exception | None = None
        for _ in range(2):
            try:
                success, frame = self.capture.read()
            except Exception as exc:
                last_error = exc
                time.sleep(0.02)
                continue
            if success and frame is not None:
                return frame
            time.sleep(0.02)

        if last_error is not None:
            raise StreamReadError(
                f"Falha ao ler frame de '{redact_url_credentials(self.source)}': {last_error}"
            ) from last_error
        raise StreamReadError(f"Falha ao ler frame de '{redact_url_credentials(self.source)}'.")

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
        try:
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
        except Exception as exc:
            raise StreamReadError(
                f"Não foi possível abrir o stream '{redact_url_credentials(self.source)}' com PyAV."
            ) from exc

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
        capture = None
        try:
            capture, _ = _open_local_capture(int(source))
            return bool(capture and capture.isOpened())
        finally:
            if capture is not None:
                capture.release()

    if _is_rtsp_source(source):
        rtsp_status = _quick_rtsp_describe(source, timeout_seconds=min(timeout_seconds, 1.5))
        if rtsp_status is not None:
            return rtsp_status

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
            capture = cv2.VideoCapture(source)
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


def _is_rtsp_source(source: Any) -> bool:
    if not isinstance(source, str):
        return False
    return urlsplit(source).scheme.lower() in {"rtsp", "rtsps"}


def _quick_rtsp_describe(source: str, timeout_seconds: float) -> bool | None:
    """Validate an RTSP URL with a protocol-level DESCRIBE before opening a decoder.

    OpenCV/PyAV can take many seconds to give up on some RTSP URLs. A short
    DESCRIBE is enough for source selection because the monitoring worker will
    still verify actual frame delivery when it starts.
    """

    parsed = urlsplit(source)
    host = parsed.hostname
    if not host:
        return False

    port = parsed.port or (322 if parsed.scheme.lower() == "rtsps" else 554)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    authority = host
    if parsed.port:
        authority = f"{authority}:{parsed.port}"
    request_url = urlunsplit((parsed.scheme, authority, path, "", ""))
    request = (
        f"DESCRIBE {request_url} RTSP/1.0\r\n"
        "CSeq: 1\r\n"
        "Accept: application/sdp\r\n"
        "User-Agent: StealthLens/1.0\r\n\r\n"
    ).encode("utf-8", errors="ignore")

    try:
        with socket.create_connection((host, port), timeout=timeout_seconds) as connection:
            connection.settimeout(timeout_seconds)
            connection.sendall(request)
            response = connection.recv(2048).decode("utf-8", errors="ignore")
    except OSError:
        return None

    first_line = response.splitlines()[0] if response.splitlines() else ""
    upper_line = first_line.upper()
    if " 200 " in f" {upper_line} " or upper_line.endswith(" 200 OK"):
        return True
    if " 401 " in f" {upper_line} " or "UNAUTHORIZED" in upper_line:
        return None if parsed.username else False
    if " 404 " in f" {upper_line} " or "NOT FOUND" in upper_line:
        return False
    return None


def _open_local_capture(source_index: int) -> tuple[cv2.VideoCapture, str]:
    attempts: list[str] = []
    for backend_name, backend_id in _iter_local_backends():
        capture = _create_local_capture(source_index, backend_id)
        if capture is None or not capture.isOpened():
            attempts.append(f"{backend_name}:open_failed")
            continue

        profile_attempts = (
            ("1080p-mjpg", "MJPG", 1920, 1080, 30),
            ("720p-mjpg", "MJPG", 1280, 720, 30),
            ("native", None, None, None, None),
        )

        for profile_name, fourcc, width, height, fps in profile_attempts:
            _apply_capture_profile(capture, fourcc=fourcc, width=width, height=height, fps=fps)
            ok, reason = _warmup_capture(capture)
            if ok:
                return capture, f"opencv-{backend_name}:{profile_name}"
            attempts.append(f"{backend_name}:{profile_name}:{reason}")

        capture.release()

    details = "; ".join(attempts[-6:]) if attempts else "sem detalhes"
    raise StreamReadError(
        f"Não foi possível abrir webcam local {source_index}. Tentativas: {details}"
    )


def _iter_local_backends() -> list[tuple[str, int | None]]:
    candidates: list[tuple[str, int | None]] = []
    dshow = getattr(cv2, "CAP_DSHOW", None)
    msmf = getattr(cv2, "CAP_MSMF", None)

    if isinstance(dshow, int) and dshow > 0:
        candidates.append(("dshow", dshow))
    if isinstance(msmf, int) and msmf > 0:
        candidates.append(("msmf", msmf))
    candidates.append(("default", None))

    unique: list[tuple[str, int | None]] = []
    seen_ids: set[int | None] = set()
    for name, backend_id in candidates:
        if backend_id in seen_ids:
            continue
        seen_ids.add(backend_id)
        unique.append((name, backend_id))
    return unique


def _create_local_capture(source_index: int, backend_id: int | None) -> cv2.VideoCapture | None:
    try:
        if backend_id is None:
            return cv2.VideoCapture(source_index)
        return cv2.VideoCapture(source_index, backend_id)
    except Exception:
        return None


def _create_network_capture(source: str, backend_id: int | None) -> cv2.VideoCapture | None:
    params: list[int] = []
    open_timeout = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
    read_timeout = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
    if isinstance(open_timeout, int):
        params.extend([open_timeout, 5000])
    if isinstance(read_timeout, int):
        params.extend([read_timeout, 5000])

    try:
        if params and backend_id is not None:
            return cv2.VideoCapture(source, backend_id, params)
        if backend_id is not None:
            return cv2.VideoCapture(source, backend_id)
        return cv2.VideoCapture(source)
    except Exception:
        try:
            if backend_id is not None:
                return cv2.VideoCapture(source, backend_id)
            return cv2.VideoCapture(source)
        except Exception:
            return None


def _jittered_delay(base_delay: float) -> float:
    jitter = random.uniform(0.85, 1.25)
    return max(0.2, base_delay * jitter)


def _apply_capture_profile(
    capture: cv2.VideoCapture,
    fourcc: str | None,
    width: int | None,
    height: int | None,
    fps: int | None,
) -> None:
    if fourcc:
        try:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        except Exception:
            pass
    if width:
        try:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        except Exception:
            pass
    if height:
        try:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        except Exception:
            pass
    if fps:
        try:
            capture.set(cv2.CAP_PROP_FPS, fps)
        except Exception:
            pass

    try:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    except Exception:
        pass


def _warmup_capture(capture: cv2.VideoCapture, attempts: int = 10) -> tuple[bool, str]:
    consecutive_success = 0
    required_success = 3
    for _ in range(attempts):
        try:
            success, frame = capture.read()
        except Exception as exc:
            consecutive_success = 0
            return False, str(exc)
        if success and frame is not None and getattr(frame, "size", 0) > 0:
            consecutive_success += 1
            if consecutive_success >= required_success:
                return True, "ok"
        else:
            consecutive_success = 0
        time.sleep(0.03)
    return False, "no_frames"
