from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from utils.config import AgentConfig, CameraConfig, ModelConfig


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    model_alias: str
    event_type: str
    trigger_in_zones_only: bool
    track_id: int | None = None
    cooldown_seconds: float | None = None
    emit_event: bool = True
    display: bool = True
    overlay_label: str | None = None
    overlay_color: tuple[int, int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelSession:
    def __init__(self, model_config: ModelConfig, device: str, logger: Any) -> None:
        self.config = model_config
        self.device = device
        self.logger = logger

        if not self.config.path.exists():
            raise FileNotFoundError(
                f"Modelo '{self.config.alias}' não encontrado em '{self.config.path}'."
            )

        self._model = YOLO(str(self.config.path))
        self._names = getattr(self._model, "names", {})

        self.logger.info(
            "model_loaded",
            extra={
                "model_alias": self.config.alias,
                "model_path": str(self.config.path),
                "device": self.device,
            },
        )

    def infer(self, frame: Any) -> tuple[list[Detection], float]:
        started = time.perf_counter()
        if self.config.use_tracking:
            track_kwargs = {
                "source": frame,
                "conf": self.config.confidence,
                "iou": self.config.iou,
                "verbose": False,
                "device": self.device,
                "persist": True,
            }
            if self.config.tracker:
                track_kwargs["tracker"] = self.config.tracker
            results = self._model.track(**track_kwargs)
        else:
            results = self._model.predict(
                source=frame,
                conf=self.config.confidence,
                iou=self.config.iou,
                verbose=False,
                device=self.device,
            )
        inference_ms = (time.perf_counter() - started) * 1000.0

        if not results:
            return [], inference_ms

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return [], inference_ms

        names = getattr(result, "names", self._names)
        filters = {name.lower() for name in self.config.class_filters}
        detections: list[Detection] = []

        boxes_xyxy = boxes.xyxy.cpu().tolist() if getattr(boxes, "xyxy", None) is not None else []
        classes = boxes.cls.cpu().tolist() if getattr(boxes, "cls", None) is not None else []
        confidences = boxes.conf.cpu().tolist() if getattr(boxes, "conf", None) is not None else []
        if getattr(boxes, "id", None) is not None:
            track_ids = boxes.id.int().cpu().tolist()
        else:
            track_ids = [None] * len(boxes_xyxy)

        for box, class_id, confidence, track_id in zip(boxes_xyxy, classes, confidences, track_ids):
            label = _resolve_label(names, int(class_id))
            if filters and label.lower() not in filters:
                continue

            coordinates = tuple(int(value) for value in box)
            detections.append(
                Detection(
                    label=label,
                    confidence=float(confidence),
                    bbox=coordinates,
                    model_alias=self.config.alias,
                    event_type=self.config.event_type or label,
                    trigger_in_zones_only=self.config.trigger_in_zones_only,
                    track_id=int(track_id) if track_id is not None else None,
                    cooldown_seconds=self.config.cooldown_seconds,
                    emit_event=self.config.emit_events,
                )
            )

        return detections, inference_ms

    def clone_detection(self, detection: Detection) -> Detection:
        return copy.deepcopy(detection)


def build_model_sessions(
    camera_config: CameraConfig,
    agent_config: AgentConfig,
    logger: Any,
) -> list[ModelSession]:
    device = detect_runtime_device(agent_config.device)
    sessions: list[ModelSession] = []

    for model_reference in camera_config.models:
        model_config = agent_config.model_catalog.get(model_reference)
        if model_config is None:
            path = _resolve_runtime_path(agent_config.base_dir, model_reference)
            model_config = ModelConfig(alias=path.stem or model_reference, path=path)

        if not model_config.enabled:
            continue

        sessions.append(ModelSession(model_config=model_config, device=device, logger=logger))

    if not sessions:
        raise ValueError(f"Câmera '{camera_config.name}' não possui modelos ativos.")

    return sessions


def detect_runtime_device(preference: str = "auto") -> str:
    normalized = preference.strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized.startswith("cuda"):
        return normalized

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        return "cpu"

    return "cpu"


def _resolve_runtime_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _resolve_label(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)
