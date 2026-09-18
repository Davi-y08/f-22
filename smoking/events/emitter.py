from __future__ import annotations

import json
import hashlib
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from utils.storage import atomic_write


@dataclass(slots=True)
class DetectionEvent:
    event_id: str
    agent_id: str
    timestamp: str
    camera_id: str
    camera_name: str
    event_type: str
    confidence: float
    model_alias: str
    label: str
    bbox: tuple[int, int, int, int]
    zone: str | None = None
    snapshot_path: str | None = None
    frame_size: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        agent_id: str,
        camera_id: str,
        camera_name: str,
        event_type: str,
        confidence: float,
        model_alias: str,
        label: str,
        bbox: tuple[int, int, int, int],
        zone: str | None = None,
        frame_size: tuple[int, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "DetectionEvent":
        return cls(
            event_id=uuid.uuid4().hex,
            agent_id=agent_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            camera_id=camera_id,
            camera_name=camera_name,
            event_type=event_type,
            confidence=confidence,
            model_alias=model_alias,
            label=label,
            bbox=bbox,
            zone=zone,
            frame_size=frame_size,
            metadata=metadata or {},
        )


class FileEventEmitter:
    def __init__(
        self,
        agent_id: str,
        events_dir: Path,
        snapshots_dir: Path,
        logger: Any | None = None,
        cloud_client: Any | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.events_dir = events_dir
        self.snapshots_dir = snapshots_dir
        self.logger = logger
        self.cloud_client = cloud_client
        self._lock = threading.Lock()

        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        event: DetectionEvent,
        frame: Any | None = None,
        save_snapshot: bool = False,
    ) -> DetectionEvent:
        if save_snapshot and frame is not None:
            try:
                event.snapshot_path = self._save_snapshot(event, frame)
            except (OSError, ValueError, cv2.error) as exc:
                event.snapshot_path = None
                event.metadata["snapshot_error"] = str(exc)
                if self.logger:
                    self.logger.warning("snapshot_save_failed", extra={"event_id": event.event_id, "error": str(exc)})

        payload = asdict(event)
        payload["bbox"] = list(event.bbox)
        if event.frame_size is not None:
            payload["frame_size"] = list(event.frame_size)

        target_file = self.events_dir / f"events-{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
        with self._lock:
            with target_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        if self.logger:
            self.logger.info("event_emitted", extra={"event_payload": payload})

        if self.cloud_client is not None:
            try:
                self.cloud_client.emit_event_async(event)
            except Exception as exc:
                if self.logger:
                    self.logger.error(
                        "cloud_event_enqueue_failed",
                        extra={"event_id": event.event_id, "error": str(exc), "local_file": str(target_file)},
                    )

        return event

    def close(self) -> None:
        if self.cloud_client is not None:
            self.cloud_client.close()
        return

    def _save_snapshot(self, event: DetectionEvent, frame: Any) -> str:
        camera_dir = self.snapshots_dir / _safe_component(event.camera_id)
        camera_dir.mkdir(parents=True, exist_ok=True)

        file_name = f"{_safe_component(event.event_type)}-{_safe_component(event.event_id)}.jpg"
        snapshot_path = camera_dir / file_name
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not success or encoded is None or encoded.size == 0:
            raise ValueError("Nao foi possivel codificar a foto do alerta.")
        atomic_write(snapshot_path, encoded.tobytes())
        return str(snapshot_path.resolve())


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", value)[:80]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned or 'event'}-{digest}"
