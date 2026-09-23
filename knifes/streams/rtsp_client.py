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


class _StreamCapture:
    def __init__(
        self,
        source: str | int,
        backend_preference: str = "auto",
        queue_maxsize: int = 2,
        reconnect_initial_delay: float = 2.0,
        reconnect_max_delay: float = 30.0,
        logger: Any | None = None,
    ) -> None:
        self.source = source
        self.backend_preference = backend_preference
        self.queue_maxsize = max(1, int(queue_maxsize))
        self.reconnect_initial_delay = reconnect_initial_delay
        self.reconnect_max_delay = reconnect_max_delay
        self.logger = logger

        self._frames: queue.Queue[FramePacket] = queue.Queue(maxsize=self.queue_maxsize)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed_event = threading.Event()
        self._closed_event.set()
        self._sinks: set[queue.Queue[FramePacket]] = set()
        self._sink_lock = threading.Lock()
        self._last_frame_monotonic: float | None = None
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
        self._closed_event.clear()
        self._clear_frames()
        self._thread = threading.Thread(target=self._run_capture, name="stream-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=6)
        self._clear_frames()
        self._set_status(online=False)

    def read_latest(self, timeout: float = 1.0) -> FramePacket | None:
        try:
            packet = self._frames.get(timeout=timeout)
        except queue.Empty:
            return None

        while True:
            try:
                packet = self._frames.get_nowait()
                self._increment_status_counter("dropped_frames")
            except queue.Empty:
                return packet

    def status_snapshot(self) -> dict[str, Any]:
        with self._status_lock:
            snapshot = dict(self._status)
        snapshot["queue_depth"] = self._frames.qsize()
        age = time.monotonic() - self._last_frame_monotonic if self._last_frame_monotonic is not None else None
        snapshot["frame_age_ms"] = round(age * 1000) if age is not None else None
        if age is not None and age > 5.0:
            snapshot["online"] = False
        return snapshot

    def _run_capture(self) -> None:
        try:
            self._reader_loop()
        finally:
            self._set_status(online=False)
            self._closed_event.set()

    def _reader_loop(self) -> None:
        delay = self.reconnect_initial_delay
        current_preference = self.backend_preference

        while not self._stop_event.is_set():
            reader = None
            received_frame = False
            try:
                reader = self._build_reader(current_preference)
                reader.open()
                self._set_status(
                    online=False,
                    backend=reader.backend_name,
                    last_error=None,
                )

                while not self._stop_event.is_set():
                    frame = reader.read()
                    if self._stop_event.is_set():
                        break
                    received_frame = True
                    if self.backend_preference.lower() == "auto":
                        current_preference = reader.backend_name
                    delay = self.reconnect_initial_delay
                    packet = FramePacket(
                        frame_id=self._next_frame_id(),
                        frame=frame,
                        captured_at=datetime.now(timezone.utc),
                    )
                    self._push_frame(packet)
                    self._last_frame_monotonic = time.monotonic()
                    self._set_status(
                        online=True,
                        backend=reader.backend_name,
                        last_frame_at=packet.captured_at.isoformat(),
                        last_error=None,
                    )
            except Exception as exc:
                self._clear_frames()
                status = self.status_snapshot()
                if (
                    reader is not None
                    and reader.backend_name == "pyav"
                    and self.backend_preference.lower() == "auto"
                    and not received_frame
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
            finally:
                if reader is not None:
                    try:
                        reader.close()
                    except Exception as exc:
                        if self.logger:
                            self.logger.warning("stream_close_failed", extra={"error": str(exc)})
            if not self._stop_event.is_set():
                self._sleep_with_stop(retry_in_seconds)
                delay = min(delay * 2, self.reconnect_max_delay)

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
        with self._sink_lock:
            sinks = list(self._sinks) or [self._frames]
        for sink in sinks:
            while sink.full():
                try:
                    sink.get_nowait()
                    self._increment_status_counter("dropped_frames")
                except queue.Empty:
                    break
            try:
                sink.put_nowait(packet)
            except queue.Full:
                self._increment_status_counter("dropped_frames")

    def _clear_frames(self) -> None:
        with self._sink_lock:
            sinks = [self._frames, *self._sinks]
        for sink in sinks:
            while True:
                try:
                    sink.get_nowait()
                    self._increment_status_counter("dropped_frames")
                except queue.Empty:
                    break

    def _next_frame_id(self) -> int:
        self._frame_counter += 1
        return self._frame_counter

    def _set_status(self, **updates: Any) -> None:
        with self._status_lock:
            self._status.update(updates)

    def _increment_status_counter(self, key: str) -> None:
        with self._status_lock:
            self._status[key] = int(self._status.get(key, 0)) + 1

    def _sleep_with_stop(self, seconds: float) -> None:
        self._stop_event.wait(max(0.0, seconds))


_CAPTURES_LOCK = threading.Lock()
_HTTP_CAPTURES: dict[tuple[str, str], _StreamCapture] = {}


class RTSPClient:
    """Each consumer gets current frames; identical HTTP URLs share one connection."""

    def __init__(
        self,
        source: str | int,
        backend_preference: str = "auto",
        queue_maxsize: int = 2,
        reconnect_initial_delay: float = 2.0,
        reconnect_max_delay: float = 30.0,
        logger: Any | None = None,
    ) -> None:
        self.source = source
        self.backend_preference = backend_preference
        self._frames: queue.Queue[FramePacket] = queue.Queue(maxsize=max(1, queue_maxsize))
        self._settings = {
            "source": source, "backend_preference": backend_preference,
            "queue_maxsize": queue_maxsize, "reconnect_initial_delay": reconnect_initial_delay,
            "reconnect_max_delay": reconnect_max_delay, "logger": logger,
        }
        self._key = (_http_source_key(source), backend_preference.lower()) if _is_http_source(source) else None
        self._capture: _StreamCapture | None = None
        self._last_status: dict[str, Any] = {
            "online": False, "backend": None, "last_frame_at": None, "last_error": None,
            "reconnect_attempts": 0, "dropped_frames": 0, "queue_depth": 0, "frame_age_ms": None,
        }

    def start(self) -> None:
        while True:
            with _CAPTURES_LOCK:
                if self._capture is not None:
                    return
                capture = _HTTP_CAPTURES.get(self._key) if self._key else None
                if capture is None or capture._closed_event.is_set():
                    capture = _StreamCapture(**self._settings)
                    if self._key:
                        _HTTP_CAPTURES[self._key] = capture
                if not capture._stop_event.is_set():
                    with capture._sink_lock:
                        capture._sinks.add(self._frames)
                    self._capture = capture
                    capture.start()
                    return
            # Never open a replacement while the previous connection is still closing.
            if not capture._closed_event.wait(6.0):
                raise StreamReadError("A conexao anterior da camera ainda esta encerrando.")

    def stop(self) -> None:
        with _CAPTURES_LOCK:
            capture = self._capture
            if capture is None:
                return
            self._last_status = self.status_snapshot()
            self._capture = None
            with capture._sink_lock:
                capture._sinks.discard(self._frames)
                last_consumer = not capture._sinks
            if last_consumer:
                capture._stop_event.set()
        if last_consumer:
            capture.stop()
            with _CAPTURES_LOCK:
                if capture._closed_event.is_set() and self._key and _HTTP_CAPTURES.get(self._key) is capture:
                    del _HTTP_CAPTURES[self._key]
        while not self._frames.empty():
            try:
                self._frames.get_nowait()
            except queue.Empty:
                break
        self._last_status.update(online=False, queue_depth=0, shared_consumers=0)

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
        capture = self._capture
        if capture is None:
            return dict(self._last_status)
        status = capture.status_snapshot()
        status["queue_depth"] = self._frames.qsize()
        with capture._sink_lock:
            status["shared_consumers"] = len(capture._sinks)
        return status


def _is_http_source(source: Any) -> bool:
    return isinstance(source, str) and urlsplit(source).scheme.lower() in {"http", "https"}


def _http_source_key(source: str | int) -> str:
    parsed = urlsplit(str(source).strip())
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    # Keep credentials, path and query: they may identify different streams.
    return repr((parsed.scheme.lower(), parsed.hostname, port, parsed.username, parsed.password,
                 parsed.path or "/", parsed.query))


def _active_http_stream(source: str | int, timeout: float) -> bool | None:
    key = _http_source_key(source)
    with _CAPTURES_LOCK:
        captures = [capture for (url, _), capture in _HTTP_CAPTURES.items() if url == key]
    if not captures:
        return None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(capture.status_snapshot()["online"] for capture in captures):
            return True
        time.sleep(0.05)
    return False


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
            _configure_network_capture_options()
            self.capture = _create_network_capture(self.source, cv2.CAP_FFMPEG)
            if not self.capture or not self.capture.isOpened():
                self.close()
                self.capture = _create_network_capture(self.source, None)
        else:
            self.capture, self.backend_name = _open_local_capture(int(self.source))

        if not self.capture or not self.capture.isOpened():
            raise StreamReadError(
                f"Não foi possível abrir o stream '{redact_url_credentials(self.source)}'."
            )

        try:
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def read(self) -> Any:
        if not self.capture:
            raise StreamReadError("Stream OpenCV não inicializado.")

        last_error: Exception | None = None
        attempts = 5 if isinstance(self.source, int) else 1
        for _ in range(attempts):
            try:
                success, frame = self.capture.read()
            except Exception as exc:
                last_error = exc
                time.sleep(0.015)
                continue
            if success and frame is not None:
                return frame
            time.sleep(0.015)

        if last_error is not None:
            raise StreamReadError(
                f"Falha ao ler frame de '{redact_url_credentials(self.source)}': {last_error}"
            ) from last_error
        raise StreamReadError(f"Falha ao ler frame de '{redact_url_credentials(self.source)}'.")

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


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
                timeout=(5.0, 5.0),
                options=_network_stream_options(self.source, 5.0),
            )
            self.video_stream = next(stream for stream in self.container.streams if stream.type == "video")
            if _is_http_source(self.source):
                self.video_stream.codec_context.thread_count = 1
                self.video_stream.codec_context.thread_type = "SLICE"
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
            self.container = None
            self.frames = None


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

    if _is_http_source(source):
        active = _active_http_stream(source, timeout_seconds)
        if active is not None:
            return active

    preference = backend_preference.lower()
    if preference in {"auto", "pyav"} and av is not None:
        container = None
        try:
            container = av.open(
                source,
                timeout=(timeout_seconds, timeout_seconds),
                options=_network_stream_options(source, timeout_seconds),
            )
            video_stream = next(stream for stream in container.streams if stream.type == "video")
            frame = next(container.decode(video_stream))
            return frame is not None
        except Exception:
            if preference == "pyav":
                return False
        finally:
            if container is not None:
                container.close()

    _configure_network_capture_options()
    capture = _create_network_capture(source, cv2.CAP_FFMPEG)
    try:
        if not capture or not capture.isOpened():
            if capture is not None:
                capture.release()
            capture = _create_network_capture(source, None)
            if not capture or not capture.isOpened():
                return False
        started = time.monotonic()
        while (time.monotonic() - started) < timeout_seconds:
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


def _network_stream_options(source: str, timeout: float) -> dict[str, str]:
    options = {"rw_timeout": str(int(timeout * 1_000_000))}
    if _is_http_source(source):
        options.update({
            "fflags": "discardcorrupt", "probesize": "32768",
            "analyzeduration": "500000", "fpsprobesize": "2",
        })
    else:
        options.update({
            "rtsp_transport": "tcp", "fflags": "nobuffer", "flags": "low_delay",
            "max_delay": "500000", "stimeout": str(int(timeout * 1_000_000)),
        })
    return options


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
            if capture is not None:
                capture.release()
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
    decoder_threads = getattr(cv2, "CAP_PROP_N_THREADS", None)
    if _is_http_source(source) and isinstance(decoder_threads, int):
        params.extend([decoder_threads, 1])

    try:
        if params:
            return cv2.VideoCapture(source, backend_id if backend_id is not None else cv2.CAP_ANY, params)
        return None
    except Exception:
        return None


def _configure_network_capture_options() -> None:
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|flags;low_delay|max_delay;500000|rw_timeout;5000000",
    )


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
