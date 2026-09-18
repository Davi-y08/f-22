from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import threading
import time
from typing import Any

from behaviors.smoking import SmokingBehaviorAnalyzer
from display.overlay import render_monitor_frame
from display.renderer import DisplayRenderer
from events.emitter import DetectionEvent, FileEventEmitter
from models.loader import Detection, build_model_sessions
from streams.rtsp_client import RTSPClient
from utils.config import AgentConfig, CameraConfig, ZoneConfig


class CameraWorker(threading.Thread):
    def __init__(
        self,
        agent_config: AgentConfig,
        camera_config: CameraConfig,
        emitter: FileEventEmitter,
        logger: Any,
        display_renderer: DisplayRenderer | None = None,
        display_camera_count: int = 1,
    ) -> None:
        super().__init__(name=f"camera-worker-{camera_config.id}", daemon=True)
        self.agent_config = agent_config
        self.camera_config = camera_config
        self.emitter = emitter
        self.logger = logger
        self.display_renderer = display_renderer
        self.display_camera_count = max(1, int(display_camera_count))

        self._stop_event = threading.Event()
        self._status_lock = threading.Lock()
        self._result_lock = threading.Lock()
        self._analysis_error: str | None = None
        self._last_analysis_at = 0.0
        self._last_event_at: dict[tuple[Any, ...], float] = {}
        self._last_cooldown_cleanup = 0.0
        self._model_sessions = []
        self._last_detections: list[Detection] = []
        self._last_inference_ms = 0.0
        self._inference_ema_ms = 0.0
        self._last_detection_at = 0.0
        self._recent_events: deque[str] = deque(maxlen=5)
        self._behavior_status_lines: list[str] = []
        self._started_at = time.monotonic()
        self._last_render_at = 0.0
        self._last_status_sync_at = 0.0
        self._analysis_interval = 1.0 / max(self.camera_config.fps_analysis, 0.1)
        self._consecutive_analysis_failures = 0
        self._consecutive_render_failures = 0
        self._render_interval = _resolve_render_interval(
            configured_target_fps=self.camera_config.display.target_fps,
            display_camera_count=self.display_camera_count,
        )
        self._last_stream_status_at = 0.0
        self._cached_stream_status: dict[str, Any] = {
            "online": False,
            "backend": None,
            "last_frame_at": None,
            "last_error": None,
            "reconnect_attempts": 0,
            "dropped_frames": 0,
            "queue_depth": 0,
        }
        self._zones_cache: dict[tuple[int, int], list[tuple[str, list[tuple[int, int]]]]] = {}
        self._stats = {
            "analyzed_frames": 0,
            "emitted_events": 0,
            "average_inference_ms": 0.0,
        }
        self._status = {
            "camera_id": self.camera_config.id,
            "name": self.camera_config.name,
            "state": "created",
            "online": False,
            "backend": None,
            "last_frame_at": None,
            "last_error": None,
        }

        self.stream = RTSPClient(
            source=self.camera_config.source,
            backend_preference=self.camera_config.backend_preference,
            queue_maxsize=self.camera_config.queue_maxsize,
            reconnect_initial_delay=self.camera_config.reconnect_initial_delay,
            reconnect_max_delay=self.camera_config.reconnect_max_delay,
            logger=self.logger,
        )
        self.smoking_behavior = (
            SmokingBehaviorAnalyzer(self.camera_config.smoking_behavior, self.logger)
            if self.camera_config.smoking_behavior.enabled
            else None
        )

    def stop(self) -> None:
        self._stop_event.set()

    def status_snapshot(self) -> dict[str, Any]:
        with self._status_lock:
            snapshot = dict(self._status)

        stream_status = self.stream.status_snapshot()
        uptime = max(0.0, time.monotonic() - self._started_at)
        analyzed_frames = int(self._stats["analyzed_frames"])
        snapshot.update(
            {
                "queue_depth": stream_status["queue_depth"],
                "dropped_frames": stream_status["dropped_frames"],
                "reconnect_attempts": stream_status["reconnect_attempts"],
                "analyzed_frames": analyzed_frames,
                "emitted_events": int(self._stats["emitted_events"]),
                "average_inference_ms": round(float(self._stats["average_inference_ms"]), 2),
                "analysis_fps": round(analyzed_frames / uptime, 2) if uptime > 0 else 0.0,
                "effective_analysis_fps": round(
                    1.0 / self._analysis_interval,
                    2,
                ) if self._analysis_interval > 0 else 0.0,
            }
        )
        return snapshot

    def run(self) -> None:
        executor = None
        try:
            self._set_status(state="loading_models")
            self._model_sessions = build_model_sessions(self.camera_config, self.agent_config, self.logger)

            self._set_status(state="connecting")
            self.stream.start()

            analysis_interval = 1.0 / max(self.camera_config.fps_analysis, 0.1)
            self._analysis_interval = analysis_interval
            self._set_status(state="running")
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"inference-{self.camera_config.id}")
            analysis_future = None

            while not self._stop_event.is_set():
                packet = self.stream.read_latest(timeout=1.0)
                now = time.perf_counter()
                self._sync_stream_status(now=now)

                if packet is None:
                    continue

                if analysis_future is None or analysis_future.done():
                    if analysis_future is not None:
                        analysis_future.result()
                        self._analysis_interval = self._resolve_analysis_interval(analysis_interval)
                    if now - self._last_analysis_at >= self._analysis_interval:
                        self._last_analysis_at = now
                        analysis_future = executor.submit(
                            self._safe_analyze_frame, packet.frame, getattr(packet, "captured_at", None),
                        )
                self._safe_maybe_render_frame(packet.frame, now=now, force=False)
        except Exception as exc:
            self.logger.exception(
                "camera_worker_failed",
                extra={"camera_id": self.camera_config.id, "error": str(exc)},
            )
            self._set_status(state="error", last_error=str(exc), online=False)
        finally:
            self._stop_event.set()
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            self.stream.stop()
            if self.status_snapshot()["state"] != "error":
                self._set_status(state="stopped", online=False)

    def _safe_analyze_frame(self, frame: Any, captured_at: datetime | None = None) -> None:
        try:
            self._analyze_frame(frame, captured_at=captured_at)
            self._consecutive_analysis_failures = 0
            self._analysis_error = None
        except Exception as exc:
            self._consecutive_analysis_failures += 1
            self.logger.warning(
                "frame_analysis_failed",
                extra={
                    "camera_id": self.camera_config.id,
                    "error": str(exc),
                    "consecutive_failures": self._consecutive_analysis_failures,
                },
                exc_info=True,
            )
            self._analysis_error = f"Falha na análise do frame: {exc}"
            with self._result_lock:
                self._last_detections = []
            self._set_status(state="running", last_error=self._analysis_error)
            self._stop_event.wait(min(0.25, 0.04 * self._consecutive_analysis_failures))

    def _analyze_frame(self, frame: Any, captured_at: datetime | None = None) -> None:
        total_inference_ms = 0.0
        emitted = 0
        detections: list[Detection] = []

        for session in self._model_sessions:
            session_detections, inference_ms = session.infer(frame)
            total_inference_ms += inference_ms
            detections.extend(session_detections)

        derived_event_detections: list[Detection] = []
        behavior_status_lines: list[str] = []
        if self.smoking_behavior is not None:
            behavior_result = self.smoking_behavior.process(detections)
            detections = behavior_result.detections
            derived_event_detections = behavior_result.derived_events
            behavior_status_lines = behavior_result.status_lines

        event_candidates = detections + derived_event_detections

        for detection in event_candidates:
            if not detection.emit_event:
                continue

            zone_name = self._resolve_zone_name(detection, frame)
            if detection.trigger_in_zones_only and self.camera_config.zones and zone_name is None:
                continue

            cooldown = detection.cooldown_seconds
            if cooldown is None:
                cooldown = self.camera_config.cooldown_seconds

            if not self._can_emit_event(detection, zone_name):
                continue

            height, width = frame.shape[:2]
            event = DetectionEvent.create(
                agent_id=self.agent_config.agent_id,
                camera_id=self.camera_config.id,
                camera_name=self.camera_config.name,
                event_type=detection.event_type,
                confidence=detection.confidence,
                model_alias=detection.model_alias,
                label=detection.label,
                bbox=detection.bbox,
                zone=zone_name,
                frame_size=(width, height),
                metadata=_build_event_metadata(detection, self.camera_config.fps_analysis, total_inference_ms),
            )
            if captured_at is not None:
                event.timestamp = captured_at.isoformat()
            self.emitter.emit(
                event=event,
                frame=frame,
                save_snapshot=self.camera_config.snapshot_on_event,
            )
            self._last_event_at[self._event_key(detection, zone_name)] = time.monotonic() + cooldown
            with self._result_lock:
                self._recent_events.append(_format_recent_event(event))
            emitted += 1

        with self._result_lock:
            self._last_detections = [detection for detection in detections if detection.display]
            self._behavior_status_lines = behavior_status_lines
            self._last_detection_at = time.perf_counter()
        self._last_inference_ms = total_inference_ms
        self._inference_ema_ms = (
            total_inference_ms if self._stats["analyzed_frames"] == 0
            else 0.2 * total_inference_ms + 0.8 * self._inference_ema_ms
        )
        self._stats["analyzed_frames"] += 1
        self._stats["emitted_events"] += emitted
        analyzed_frames = max(1, int(self._stats["analyzed_frames"]))
        current_average = float(self._stats["average_inference_ms"])
        self._stats["average_inference_ms"] = current_average + (
            total_inference_ms - current_average
        ) / analyzed_frames

    def _resolve_analysis_interval(self, base_interval: float) -> float:
        observed_inference_seconds = max(
            float(self._last_inference_ms) / 1000.0,
            self._inference_ema_ms / 1000.0,
        )
        if observed_inference_seconds <= 0:
            return base_interval

        max_interval = max(base_interval, min(3.0, max(1.0, base_interval * 6.0)))
        adaptive_interval = observed_inference_seconds / 0.72
        return min(max_interval, max(base_interval, adaptive_interval))

    def _safe_maybe_render_frame(self, frame: Any, now: float, force: bool) -> None:
        try:
            self._maybe_render_frame(frame, now=now, force=force)
            self._consecutive_render_failures = 0
        except Exception as exc:
            self._consecutive_render_failures += 1
            self.logger.warning(
                "frame_render_failed",
                extra={
                    "camera_id": self.camera_config.id,
                    "error": str(exc),
                    "consecutive_failures": self._consecutive_render_failures,
                },
                exc_info=True,
            )
            self._set_status(state="running", last_error=f"Falha ao renderizar frame: {exc}")

    def _maybe_render_frame(self, frame: Any, now: float, force: bool) -> None:
        if not self.display_renderer or not self.camera_config.display.enabled:
            return
        if not force and self._render_interval > 0.0:
            if (now - self._last_render_at) < self._render_interval:
                return
        self._last_render_at = now
        self._render_frame(frame, now=now)

    def _render_frame(self, frame: Any, now: float | None = None) -> None:
        if not self.display_renderer or not self.camera_config.display.enabled:
            return

        snapshot_at = now if now is not None else time.perf_counter()
        with self._result_lock:
            detections = list(self._last_detections) if snapshot_at - self._last_detection_at <= 2.0 else []
            behavior_lines = list(self._behavior_status_lines)
            recent_events = list(self._recent_events)
        if (snapshot_at - self._last_stream_status_at) >= 0.25:
            self._cached_stream_status = self.stream.status_snapshot()
            self._last_stream_status_at = snapshot_at
        stream_status = self._cached_stream_status
        effective_analysis_fps = 1.0 / self._analysis_interval if self._analysis_interval > 0 else 0.0
        status_lines = [
            f"online={stream_status['online']} backend={stream_status['backend'] or 'n/a'}",
            f"analysis_fps={effective_analysis_fps:.1f}/{self.camera_config.fps_analysis:.1f} detections={len(detections)}",
        ]

        if self.camera_config.display.show_metrics:
            status_lines.append(
                f"avg_infer={self._stats['average_inference_ms']:.1f}ms last_infer={self._last_inference_ms:.1f}ms"
            )
            status_lines.extend(behavior_lines[:1])

        zones: list[tuple[str, list[tuple[int, int]]]] = []
        if self.camera_config.display.draw_zones and self.camera_config.zones:
            height, width = frame.shape[:2]
            cache_key = (width, height)
            cached = self._zones_cache.get(cache_key)
            if cached is None:
                cached = []
                for zone in self.camera_config.zones:
                    points = [(int(x), int(y)) for x, y in _scale_polygon(zone, width, height)]
                    cached.append((zone.name, points))
                self._zones_cache[cache_key] = cached
            zones = cached

        annotated = render_monitor_frame(
            frame=frame,
            camera_name=self.camera_config.name,
            detections=detections,
            zones=zones,
            status_lines=status_lines,
            recent_events=recent_events,
        )
        self.display_renderer.submit(
            camera_id=self.camera_config.id,
            window_name=self.camera_config.display.window_name or self.camera_config.name,
            frame=annotated,
            max_width=self.camera_config.display.max_width,
            fullscreen=self.camera_config.display.fullscreen,
            fit_mode=self.camera_config.display.fit_mode,
            interpolation=self.camera_config.display.interpolation,
            enhance=self.camera_config.display.enhance,
        )

    def _can_emit_event(self, detection: Detection, zone_name: str | None) -> bool:
        now = time.monotonic()
        if now - self._last_cooldown_cleanup >= 30.0:
            self._last_event_at = {key: expiry for key, expiry in self._last_event_at.items() if expiry > now}
            self._last_cooldown_cleanup = now
        return now >= self._last_event_at.get(self._event_key(detection, zone_name), 0.0)

    @staticmethod
    def _event_key(detection: Detection, zone_name: str | None) -> tuple[Any, ...]:
        return (
            detection.event_type,
            detection.model_alias,
            detection.label,
            zone_name or "global",
            detection.track_id if detection.track_id is not None else "no-track",
        )

    def _resolve_zone_name(self, detection: Detection, frame: Any) -> str | None:
        if not self.camera_config.zones:
            return None

        height, width = frame.shape[:2]
        center_x = (detection.bbox[0] + detection.bbox[2]) / 2.0
        center_y = (detection.bbox[1] + detection.bbox[3]) / 2.0

        for zone in self.camera_config.zones:
            polygon = _scale_polygon(zone, width, height)
            if _point_in_polygon(center_x, center_y, polygon):
                return zone.name

        return None

    def _sync_stream_status(self, now: float) -> None:
        if (now - self._last_status_sync_at) < 0.25:
            return

        self._last_status_sync_at = now
        stream_status = self.stream.status_snapshot()
        self._set_status(
            online=bool(stream_status["online"]),
            backend=stream_status["backend"],
            last_frame_at=stream_status["last_frame_at"],
            last_error=stream_status["last_error"] or self._analysis_error,
        )

    def _set_status(self, **updates: Any) -> None:
        with self._status_lock:
            self._status.update(updates)


def _resolve_render_interval(configured_target_fps: float | None, display_camera_count: int) -> float:
    if configured_target_fps is not None:
        target_fps = max(5.0, min(60.0, float(configured_target_fps)))
        return 1.0 / target_fps

    # Fallback adaptativo para manter fluidez com muitas câmeras sem reduzir qualidade.
    if display_camera_count <= 1:
        return 1.0 / 30.0
    if display_camera_count == 2:
        return 1.0 / 24.0
    return 1.0 / 20.0


def _scale_polygon(zone: ZoneConfig, width: int, height: int) -> list[tuple[float, float]]:
    normalized = all(0.0 <= point[0] <= 1.0 and 0.0 <= point[1] <= 1.0 for point in zone.polygon)
    if normalized:
        return [(point[0] * width, point[1] * height) for point in zone.polygon]
    return [(point[0], point[1]) for point in zone.polygon]


def _point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    previous_index = len(polygon) - 1

    for current_index, (current_x, current_y) in enumerate(polygon):
        previous_x, previous_y = polygon[previous_index]
        intersects = ((current_y > y) != (previous_y > y)) and (
            x < (previous_x - current_x) * (y - current_y) / ((previous_y - current_y) or 1e-9) + current_x
        )
        if intersects:
            inside = not inside
        previous_index = current_index

    return inside


def _build_event_metadata(
    detection: Detection,
    analysis_target_fps: float,
    total_inference_ms: float,
) -> dict[str, Any]:
    metadata = {
        "analysis_target_fps": analysis_target_fps,
        "inference_ms": round(total_inference_ms, 2),
    }
    metadata.update(detection.metadata)
    if detection.track_id is not None:
        metadata.setdefault("track_id", detection.track_id)
    return metadata


def _format_recent_event(event: DetectionEvent) -> str:
    track_id = event.metadata.get("track_id")
    if track_id is not None:
        return f"{event.event_type.upper()} #{track_id} {event.confidence:.2f} @ {event.timestamp[11:19]}"
    return f"{event.event_type.upper()} {event.confidence:.2f} @ {event.timestamp[11:19]}"
