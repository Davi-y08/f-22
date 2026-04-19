from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

from agent.worker import CameraWorker
from display.renderer import DisplayRenderer
from events.emitter import FileEventEmitter
from utils.config import AgentConfig
from utils.logger import get_logger


class AgentManager:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.logger = get_logger("stealth_lens.manager")
        self.emitter = FileEventEmitter(
            agent_id=self.config.agent_id,
            events_dir=self.config.storage.events_dir,
            snapshots_dir=self.config.storage.snapshots_dir,
            logger=self.logger,
        )
        self.display_renderer = (
            DisplayRenderer()
            if any(camera.enabled and camera.display.enabled for camera in self.config.cameras)
            else None
        )
        self._stop_event = threading.Event()
        self._status_thread = threading.Thread(target=self._status_loop, name="status-writer", daemon=True)
        self._workers: list[CameraWorker] = []
        seen_sources: dict[str, str] = {}
        for camera_config in self.config.cameras:
            if not camera_config.enabled:
                continue

            source_key = _camera_source_key(camera_config.source)
            if source_key and source_key in seen_sources:
                self.logger.warning(
                    "duplicate_camera_source_skipped",
                    extra={
                        "camera_id": camera_config.id,
                        "camera_name": camera_config.name,
                        "source": str(camera_config.source),
                        "existing_camera_id": seen_sources[source_key],
                    },
                )
                continue

            seen_sources[source_key] = camera_config.id
            self._workers.append(
                CameraWorker(
                    agent_config=self.config,
                    camera_config=camera_config,
                    emitter=self.emitter,
                    logger=get_logger(f"stealth_lens.camera.{camera_config.id}"),
                    display_renderer=self.display_renderer,
                )
            )

    @property
    def should_stop(self) -> bool:
        return self._stop_event.is_set() or bool(self.display_renderer and self.display_renderer.stop_requested)

    def start(self) -> None:
        self.logger.info(
            "agent_starting",
            extra={
                "agent_id": self.config.agent_id,
                "enabled_cameras": len(self._workers),
                "configured_cameras": len(self.config.cameras),
            },
        )

        if self.display_renderer is not None:
            self.display_renderer.start()

        for worker in self._workers:
            worker.start()

        self._write_status_snapshot()
        self._status_thread.start()

        if not self._workers:
            self.logger.warning("agent_started_without_enabled_cameras", extra={"agent_id": self.config.agent_id})

    def stop(self) -> None:
        if self._stop_event.is_set():
            return

        self._stop_event.set()
        if self._status_thread.is_alive():
            self._status_thread.join(timeout=5)

        for worker in self._workers:
            worker.stop()

        for worker in self._workers:
            worker.join(timeout=10)

        if self.display_renderer is not None:
            self.display_renderer.stop()

        self._write_status_snapshot()
        self.emitter.close()
        self.logger.info("agent_stopped", extra={"agent_id": self.config.agent_id})

    def status_snapshot(self) -> dict[str, Any]:
        worker_statuses = [worker.status_snapshot() for worker in self._workers]
        disabled_statuses = [
            {
                "camera_id": camera.id,
                "name": camera.name,
                "state": "disabled",
                "online": False,
                "backend": None,
                "last_frame_at": None,
                "last_error": None,
                "queue_depth": 0,
                "dropped_frames": 0,
                "reconnect_attempts": 0,
                "analyzed_frames": 0,
                "emitted_events": 0,
                "average_inference_ms": 0.0,
                "analysis_fps": 0.0,
            }
            for camera in self.config.cameras
            if not camera.enabled
        ]

        cameras = worker_statuses + disabled_statuses
        online_cameras = sum(1 for camera in worker_statuses if camera["online"])

        return {
            "agent_id": self.config.agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "configured_cameras": len(self.config.cameras),
                "enabled_cameras": len(self._workers),
                "online_cameras": online_cameras,
            },
            "cameras": cameras,
        }

    def _status_loop(self) -> None:
        while not self._stop_event.wait(self.config.status_interval_seconds):
            self._write_status_snapshot()

    def _write_status_snapshot(self) -> None:
        status_payload = self.status_snapshot()
        status_path = self.config.storage.status_path
        status_path.parent.mkdir(parents=True, exist_ok=True)

        with status_path.open("w", encoding="utf-8") as handle:
            json.dump(status_payload, handle, ensure_ascii=False, indent=2)


def _camera_source_key(source: Any) -> str:
    if isinstance(source, int):
        return f"local:{source}"

    raw = str(source or "").strip()
    if not raw:
        return ""
    if raw.isdigit():
        return f"local:{int(raw)}"

    if "://" not in raw:
        return raw.lower()

    parsed = urlsplit(raw)
    host = (parsed.hostname or "").lower()
    port = parsed.port or ""
    path = parsed.path or "/"
    query = parsed.query or ""
    if query:
        return f"{parsed.scheme.lower()}://{host}:{port}{path}?{query}"
    return f"{parsed.scheme.lower()}://{host}:{port}{path}"
