from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:  # pragma: no cover - optional dependency on lite profile
    import onnxruntime as _onnxruntime_module
except Exception:  # pragma: no cover - optional dependency
    _onnxruntime_module = None

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
    def infer(self, frame: Any) -> tuple[list[Detection], float]:  # pragma: no cover - interface
        raise NotImplementedError

    def clone_detection(self, detection: Detection) -> Detection:
        return copy.deepcopy(detection)


class UltralyticsModelSession(ModelSession):
    def __init__(self, model_config: ModelConfig, device: str, logger: Any) -> None:
        self.config = model_config
        self.device = device
        self.logger = logger

        if not self.config.path.exists():
            raise FileNotFoundError(
                f"Modelo '{self.config.alias}' não encontrado em '{self.config.path}'."
            )

        try:
            ultralytics_module = importlib.import_module("ultralytics")
            YOLO = getattr(ultralytics_module, "YOLO")
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise ImportError(
                "Ultralytics não está instalado. Use 'requirements.txt' para perfil full."
            ) from exc

        self._model = YOLO(str(self.config.path))
        self._names = getattr(self._model, "names", {})

        self.logger.info(
            "model_loaded",
            extra={
                "model_alias": self.config.alias,
                "model_path": str(self.config.path),
                "device": self.device,
                "backend": "ultralytics",
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


class OnnxModelSession(ModelSession):
    def __init__(self, model_config: ModelConfig, logger: Any) -> None:
        self.config = model_config
        self.logger = logger

        if not self.config.path.exists():
            raise FileNotFoundError(
                f"Modelo ONNX '{self.config.alias}' não encontrado em '{self.config.path}'."
            )

        self._session: Any | None = None
        self._net: Any | None = None
        self._input_name: str | None = None
        self._runtime_backend = "onnxruntime"

        ort = _onnxruntime_module
        if ort is not None:
            providers = ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(str(self.config.path), providers=providers)
            input_tensor = self._session.get_inputs()[0]
            self._input_name = input_tensor.name
            input_shape = input_tensor.shape
            metadata_source = self._session
        else:
            self._runtime_backend = "opencv-dnn"
            self._net = _load_opencv_dnn_onnx(self.config.path)
            input_shape = None
            metadata_source = None
            self.logger.warning(
                "onnxruntime_unavailable_using_opencv_dnn_fallback",
                extra={"model_alias": self.config.alias, "model_path": str(self.config.path)},
            )

        self._input_h, self._input_w = _resolve_input_size(input_shape, self.config.input_size)
        self._class_names = _resolve_onnx_class_names(metadata_source, self.config, self.logger)
        self._filters = {name.lower() for name in self.config.class_filters}
        self._tracker = _SimpleTracker(iou_threshold=0.35, max_stale_frames=45) if self.config.use_tracking else None

        self.logger.info(
            "model_loaded",
            extra={
                "model_alias": self.config.alias,
                "model_path": str(self.config.path),
                "backend": self._runtime_backend,
                "input_size": f"{self._input_w}x{self._input_h}",
                "class_names": list(self._class_names),
            },
        )

    def infer(self, frame: Any) -> tuple[list[Detection], float]:
        original_h, original_w = frame.shape[:2]
        input_tensor, ratio, pad = _preprocess_for_onnx(frame, (self._input_h, self._input_w))

        started = time.perf_counter()
        if self._session is not None and self._input_name is not None:
            outputs = self._session.run(None, {self._input_name: input_tensor})
        elif self._net is not None:
            self._net.setInput(input_tensor)
            outputs = [self._net.forward()]
        else:
            raise RuntimeError(f"Modelo ONNX '{self.config.alias}' sem backend de inferência disponível.")
        inference_ms = (time.perf_counter() - started) * 1000.0
        if not outputs:
            return [], inference_ms

        predictions = _reshape_predictions(outputs[0])
        if predictions.size == 0:
            return [], inference_ms

        class_ids, confidences, boxes = _decode_predictions(
            predictions=predictions,
            class_names=self._class_names,
            confidence_threshold=self.config.confidence,
        )
        if not boxes:
            return [], inference_ms

        indices = cv2.dnn.NMSBoxes(
            bboxes=[[x1, y1, max(1, x2 - x1), max(1, y2 - y1)] for x1, y1, x2, y2 in boxes],
            scores=confidences,
            score_threshold=self.config.confidence,
            nms_threshold=self.config.iou,
        )
        if len(indices) == 0:
            return [], inference_ms

        flattened_indices = [int(i[0]) if isinstance(i, (list, tuple, np.ndarray)) else int(i) for i in indices]
        detections: list[Detection] = []
        for prediction_index in flattened_indices:
            class_id = class_ids[prediction_index]
            confidence = float(confidences[prediction_index])
            x1, y1, x2, y2 = _scale_box_to_original(
                boxes[prediction_index],
                ratio=ratio,
                pad=pad,
                original_size=(original_w, original_h),
            )

            label = _resolve_label(self._class_names, class_id)
            if self._filters and label.lower() not in self._filters:
                continue

            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    model_alias=self.config.alias,
                    event_type=self.config.event_type or label,
                    trigger_in_zones_only=self.config.trigger_in_zones_only,
                    cooldown_seconds=self.config.cooldown_seconds,
                    emit_event=self.config.emit_events,
                )
            )

        if self._tracker is not None and detections:
            self._tracker.assign(detections)

        return detections, inference_ms


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

        resolved_model = _ensure_model_available(model_config, logger)
        backend = _resolve_backend(resolved_model)
        if backend == "onnx":
            sessions.append(OnnxModelSession(model_config=resolved_model, logger=logger))
        else:
            sessions.append(UltralyticsModelSession(model_config=resolved_model, device=device, logger=logger))

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
        torch = importlib.import_module("torch")

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        return "cpu"

    return "cpu"


def _resolve_backend(model_config: ModelConfig) -> str:
    if model_config.backend in {"onnx", "ultralytics"}:
        return model_config.backend
    if model_config.path.suffix.lower() == ".onnx":
        return "onnx"
    return "ultralytics"


def _load_opencv_dnn_onnx(model_path: Path) -> Any:
    try:
        net = cv2.dnn.readNetFromONNX(str(model_path))
    except Exception as exc:
        raise ImportError(
            "onnxruntime não está instalado e o fallback OpenCV DNN para ONNX falhou. "
            "Instale onnxruntime (requirements-lite.txt) para máxima compatibilidade."
        ) from exc

    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    except Exception:
        pass
    return net


def _ensure_model_available(model_config: ModelConfig, logger: Any) -> ModelConfig:
    if model_config.path.exists():
        if model_config.sha256:
            _validate_sha256(model_config.path, model_config.sha256)
        return model_config

    if not model_config.download_url:
        raise FileNotFoundError(
            f"Modelo '{model_config.alias}' não encontrado em '{model_config.path}' e sem download_url configurado."
        )

    _download_file(
        url=model_config.download_url,
        target=model_config.path,
        expected_sha256=model_config.sha256,
        logger=logger,
        model_alias=model_config.alias,
    )
    return model_config


def _download_file(
    url: str,
    target: Path,
    expected_sha256: str | None,
    logger: Any,
    model_alias: str,
) -> None:
    try:
        requests = importlib.import_module("requests")
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "requests não está instalado para download automático de modelos."
        ) from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(target.suffix + ".download")
    if temp_path.exists():
        temp_path.unlink()

    logger.info(
        "model_download_started",
        extra={"model_alias": model_alias, "download_url": url, "target_path": str(target)},
    )

    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                handle.write(chunk)

    if expected_sha256:
        _validate_sha256(temp_path, expected_sha256)
    temp_path.replace(target)

    logger.info(
        "model_download_completed",
        extra={"model_alias": model_alias, "target_path": str(target)},
    )


def _validate_sha256(file_path: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    actual = digest.hexdigest().lower()
    expected = expected_sha256.strip().lower()
    if actual != expected:
        raise ValueError(
            f"SHA256 inválido para '{file_path}'. Esperado={expected}, atual={actual}"
        )


def _resolve_runtime_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _resolve_label(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _resolve_input_size(shape: Any, fallback_size: int) -> tuple[int, int]:
    if isinstance(shape, (list, tuple)) and len(shape) >= 4:
        raw_h = shape[2]
        raw_w = shape[3]
        height = int(raw_h) if isinstance(raw_h, int) else fallback_size
        width = int(raw_w) if isinstance(raw_w, int) else fallback_size
        return max(160, height), max(160, width)
    return fallback_size, fallback_size


def _preprocess_for_onnx(frame: Any, target_size: tuple[int, int]) -> tuple[np.ndarray, float, tuple[float, float]]:
    resized, ratio, pad = _letterbox(frame, new_shape=target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    return tensor, ratio, pad


def _letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    shape = image.shape[:2]  # (h, w)
    if shape[0] == 0 or shape[1] == 0:
        return image, 1.0, (0.0, 0.0)

    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    bordered = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return bordered, ratio, (dw, dh)


def _reshape_predictions(output: Any) -> np.ndarray:
    data = np.asarray(output)
    if data.ndim == 3:
        data = data[0]
    if data.ndim != 2:
        return np.empty((0, 0), dtype=np.float32)

    # ONNX export can be [84, N] or [N, 84].
    if data.shape[0] < data.shape[1]:
        data = data.T

    return data.astype(np.float32, copy=False)


def _decode_predictions(
    predictions: np.ndarray,
    class_names: tuple[str, ...],
    confidence_threshold: float,
) -> tuple[list[int], list[float], list[tuple[int, int, int, int]]]:
    if predictions.shape[1] < 6:
        return [], [], []

    boxes = predictions[:, :4]
    raw_scores = predictions[:, 4:]
    if raw_scores.size == 0:
        return [], [], []

    class_count = len(class_names)
    has_objectness = class_count > 0 and raw_scores.shape[1] == class_count + 1
    if has_objectness:
        objectness = raw_scores[:, 0]
        class_scores = raw_scores[:, 1:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = objectness * class_scores[np.arange(class_scores.shape[0]), class_ids]
    else:
        class_scores = raw_scores
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]

    valid_mask = confidences >= confidence_threshold
    if not np.any(valid_mask):
        return [], [], []

    filtered_boxes = boxes[valid_mask]
    filtered_class_ids = class_ids[valid_mask]
    filtered_confidences = confidences[valid_mask]

    converted_boxes: list[tuple[int, int, int, int]] = []
    for cx, cy, width, height in filtered_boxes:
        x1 = int(cx - width / 2.0)
        y1 = int(cy - height / 2.0)
        x2 = int(cx + width / 2.0)
        y2 = int(cy + height / 2.0)
        converted_boxes.append((x1, y1, x2, y2))

    return (
        [int(value) for value in filtered_class_ids.tolist()],
        [float(value) for value in filtered_confidences.tolist()],
        converted_boxes,
    )


def _scale_box_to_original(
    box: tuple[int, int, int, int],
    ratio: float,
    pad: tuple[float, float],
    original_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    pad_w, pad_h = pad
    original_w, original_h = original_size
    gain = max(ratio, 1e-6)

    scaled_x1 = int((x1 - pad_w) / gain)
    scaled_y1 = int((y1 - pad_h) / gain)
    scaled_x2 = int((x2 - pad_w) / gain)
    scaled_y2 = int((y2 - pad_h) / gain)

    clipped_x1 = int(np.clip(scaled_x1, 0, max(0, original_w - 1)))
    clipped_y1 = int(np.clip(scaled_y1, 0, max(0, original_h - 1)))
    clipped_x2 = int(np.clip(scaled_x2, 0, max(0, original_w - 1)))
    clipped_y2 = int(np.clip(scaled_y2, 0, max(0, original_h - 1)))
    return clipped_x1, clipped_y1, clipped_x2, clipped_y2


@dataclass(slots=True)
class _Track:
    track_id: int
    label: str
    bbox: tuple[int, int, int, int]
    last_seen_frame: int


class _SimpleTracker:
    def __init__(self, iou_threshold: float = 0.35, max_stale_frames: int = 45) -> None:
        self.iou_threshold = iou_threshold
        self.max_stale_frames = max_stale_frames
        self._next_id = 1
        self._frame_index = 0
        self._tracks: dict[int, _Track] = {}

    def assign(self, detections: list[Detection]) -> None:
        self._frame_index += 1
        used_tracks: set[int] = set()

        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            best_id = self._best_track_for_detection(detection, used_tracks)
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1

            used_tracks.add(best_id)
            detection.track_id = best_id
            self._tracks[best_id] = _Track(
                track_id=best_id,
                label=detection.label.lower(),
                bbox=detection.bbox,
                last_seen_frame=self._frame_index,
            )

        stale = [
            track_id
            for track_id, track in self._tracks.items()
            if (self._frame_index - track.last_seen_frame) > self.max_stale_frames
        ]
        for track_id in stale:
            self._tracks.pop(track_id, None)

    def _best_track_for_detection(self, detection: Detection, used_tracks: set[int]) -> int | None:
        detection_label = detection.label.lower()
        best_track_id: int | None = None
        best_iou = 0.0

        for track_id, track in self._tracks.items():
            if track_id in used_tracks:
                continue
            if track.label != detection_label:
                continue

            iou = _bbox_iou(track.bbox, detection.bbox)
            if iou < self.iou_threshold:
                continue
            if iou > best_iou:
                best_iou = iou
                best_track_id = track_id

        return best_track_id


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _resolve_onnx_class_names(session: Any, model_config: ModelConfig, logger: Any) -> tuple[str, ...]:
    configured_names = tuple(name.lower() for name in model_config.class_names if str(name).strip())
    metadata_names = _extract_onnx_metadata_names(session)

    if metadata_names:
        if configured_names and configured_names != metadata_names:
            logger.warning(
                "onnx_class_names_mismatch_using_model_metadata",
                extra={
                    "model_alias": model_config.alias,
                    "config_class_names": list(configured_names),
                    "metadata_class_names": list(metadata_names),
                },
            )
        return metadata_names

    if configured_names:
        return configured_names

    raise ValueError(
        f"Modelo ONNX '{model_config.alias}' sem class_names no config e sem metadata de classes."
    )


def _extract_onnx_metadata_names(session: Any) -> tuple[str, ...]:
    if session is None:
        return ()
    try:
        modelmeta = session.get_modelmeta()
        metadata = getattr(modelmeta, "custom_metadata_map", {}) or {}
        raw_names = metadata.get("names")
    except Exception:
        return ()

    if raw_names is None:
        return ()

    parsed = _parse_names_metadata(raw_names)
    if isinstance(parsed, dict):
        ordered: list[tuple[int, str]] = []
        for key, value in parsed.items():
            try:
                index = int(key)
            except Exception:
                continue
            ordered.append((index, str(value).strip().lower()))
        ordered.sort(key=lambda item: item[0])
        return tuple(name for _, name in ordered if name)

    if isinstance(parsed, (list, tuple)):
        return tuple(str(value).strip().lower() for value in parsed if str(value).strip())

    return ()


def _parse_names_metadata(raw_names: Any) -> Any:
    if isinstance(raw_names, (dict, list, tuple)):
        return raw_names

    text = str(raw_names).strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except Exception:
        return None
