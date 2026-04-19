from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ZoneConfig:
    name: str
    polygon: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class ModelConfig:
    alias: str
    path: Path
    class_filters: tuple[str, ...] = ()
    confidence: float = 0.25
    iou: float = 0.45
    event_type: str | None = None
    cooldown_seconds: float | None = None
    trigger_in_zones_only: bool = False
    emit_events: bool = True
    use_tracking: bool = False
    tracker: str | None = None
    enabled: bool = True


@dataclass(frozen=True)
class DisplayConfig:
    enabled: bool = False
    window_name: str | None = None
    show_metrics: bool = True
    draw_zones: bool = True
    max_width: int | None = 1280


@dataclass(frozen=True)
class SmokingBehaviorConfig:
    enabled: bool = False
    model: str = "smoking_monitor"
    person_label: str = "person"
    cigarette_label: str = "cigarette"
    smoke_label: str | None = "smoke"
    event_type: str = "smoking"
    max_distance_px: int = 80
    smoke_distance_multiplier: float = 1.35
    min_frames: int = 12
    decay_frames: int = 1
    smoke_boost_frames: int = 2
    stale_track_seconds: float = 5.0
    event_cooldown_seconds: float = 20.0
    require_person_track: bool = True


@dataclass(frozen=True)
class CameraConfig:
    id: str
    name: str
    source: str | int
    models: tuple[str, ...]
    fps_analysis: float = 3.0
    enabled: bool = True
    snapshot_on_event: bool = True
    queue_maxsize: int = 4
    reconnect_initial_delay: float = 2.0
    reconnect_max_delay: float = 30.0
    cooldown_seconds: float = 10.0
    backend_preference: str = "auto"
    zones: tuple[ZoneConfig, ...] = ()
    display: DisplayConfig = field(default_factory=DisplayConfig)
    smoking_behavior: SmokingBehaviorConfig = field(default_factory=SmokingBehaviorConfig)


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    json_output: bool = True


@dataclass(frozen=True)
class StorageConfig:
    events_dir: Path
    snapshots_dir: Path
    status_path: Path


@dataclass(frozen=True)
class AgentConfig:
    agent_id: str
    base_dir: Path
    device: str
    logging: LoggingConfig
    storage: StorageConfig
    model_catalog: dict[str, ModelConfig] = field(default_factory=dict)
    cameras: tuple[CameraConfig, ...] = ()
    status_interval_seconds: float = 15.0


def load_config(config_path: str | Path) -> AgentConfig:
    path = Path(config_path).expanduser().resolve()
    raw = load_raw_config(path)

    base_dir = path.parent
    logging_cfg = _load_logging_config(raw.get("logging", {}))
    storage_cfg = _load_storage_config(base_dir, raw.get("storage", {}))
    model_catalog = _load_model_catalog(base_dir, raw.get("model_catalog", {}))

    cameras_raw = raw.get("cameras", [])
    if not isinstance(cameras_raw, list):
        raise ValueError("'cameras' deve ser uma lista.")

    cameras = tuple(
        _load_camera_config(camera_raw, index)
        for index, camera_raw in enumerate(cameras_raw)
    )

    agent_id = str(raw.get("agent_id", "stealth-lens-agent")).strip() or "stealth-lens-agent"
    device = str(raw.get("device", "auto")).strip() or "auto"
    status_interval_seconds = max(5.0, float(raw.get("status_interval_seconds", 15.0)))

    return AgentConfig(
        agent_id=agent_id,
        base_dir=base_dir,
        device=device,
        logging=logging_cfg,
        storage=storage_cfg,
        model_catalog=model_catalog,
        cameras=cameras,
        status_interval_seconds=status_interval_seconds,
    )


def load_raw_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        return build_default_raw_config()

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict):
        raise ValueError("O arquivo de configuração precisa ser um objeto JSON.")

    return _apply_raw_defaults(raw)


def save_raw_config(config_path: str | Path, raw_config: dict[str, Any]) -> Path:
    path = Path(config_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(raw_config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def upsert_camera_in_config(
    config_path: str | Path,
    camera_entry: dict[str, Any],
) -> Path:
    raw_config = load_raw_config(config_path)
    cameras = raw_config.setdefault("cameras", [])
    camera_id = str(camera_entry.get("id", "")).strip().lower()
    source_key = _source_identity(camera_entry.get("source"))

    replaced_index: int | None = None
    for index, existing in enumerate(cameras):
        existing_id = str(existing.get("id", "")).strip().lower()
        existing_source_key = _source_identity(existing.get("source"))
        if camera_id and existing_id == camera_id:
            replaced_index = index
            break
        if source_key and existing_source_key == source_key:
            replaced_index = index
            break

    if replaced_index is None:
        cameras.append(camera_entry)
    else:
        cameras[replaced_index] = camera_entry

    raw_config["cameras"] = _deduplicate_camera_entries(cameras)

    return save_raw_config(config_path, raw_config)


def build_camera_entry_from_template(
    raw_config: dict[str, Any],
    name: str,
    source: str | int,
    camera_id: str | None = None,
) -> dict[str, Any]:
    template = _select_camera_template(raw_config)
    entry = copy.deepcopy(template)

    entry["id"] = _sanitize_camera_id(camera_id) or _build_camera_id(name=name, source=source)
    entry["name"] = name
    entry["source"] = source
    entry["enabled"] = True

    models = entry.get("models")
    if not isinstance(models, list) or not models:
        model_catalog = raw_config.get("model_catalog", {})
        default_model = next(iter(model_catalog), "smoking_monitor")
        entry["models"] = [default_model]

    display = entry.setdefault("display", {})
    display.setdefault("enabled", True)
    display.setdefault("show_metrics", True)
    display.setdefault("draw_zones", True)
    display.setdefault("max_width", 1280)
    display["window_name"] = f"Stealth Lens Agent - {name}"

    if "smoking_monitor" in [str(model) for model in entry.get("models", [])]:
        smoking_behavior = entry.setdefault("smoking_behavior", {})
        smoking_behavior["enabled"] = True
        smoking_behavior.setdefault("model", "smoking_monitor")
        smoking_behavior.setdefault("person_label", "person")
        smoking_behavior.setdefault("cigarette_label", "cigarette")
        smoking_behavior.setdefault("smoke_label", "smoke")
        smoking_behavior.setdefault("event_type", "smoking")
        smoking_behavior.setdefault("max_distance_px", 80)
        smoking_behavior.setdefault("smoke_distance_multiplier", 1.35)
        smoking_behavior.setdefault("min_frames", 12)
        smoking_behavior.setdefault("decay_frames", 1)
        smoking_behavior.setdefault("smoke_boost_frames", 2)
        smoking_behavior.setdefault("stale_track_seconds", 5)
        smoking_behavior.setdefault("event_cooldown_seconds", 20)
        smoking_behavior.setdefault("require_person_track", True)

    return entry


def build_default_raw_config() -> dict[str, Any]:
    return {
        "agent_id": "stealth-lens-local",
        "device": "auto",
        "status_interval_seconds": 15,
        "logging": {
            "level": "INFO",
            "json": True,
        },
        "storage": {
            "events_dir": "artifacts/events",
            "snapshots_dir": "artifacts/snapshots",
            "status_path": "artifacts/status/agent-status.json",
        },
        "model_catalog": _default_model_catalog(),
        "cameras": [],
    }


def _load_logging_config(raw: dict[str, Any]) -> LoggingConfig:
    return LoggingConfig(
        level=str(raw.get("level", "INFO")).upper(),
        json_output=bool(raw.get("json", True)),
    )


def _load_storage_config(base_dir: Path, raw: dict[str, Any]) -> StorageConfig:
    return StorageConfig(
        events_dir=_resolve_path(base_dir, raw.get("events_dir", "artifacts/events")),
        snapshots_dir=_resolve_path(base_dir, raw.get("snapshots_dir", "artifacts/snapshots")),
        status_path=_resolve_path(base_dir, raw.get("status_path", "artifacts/status/agent-status.json")),
    )


def _load_model_catalog(base_dir: Path, raw: dict[str, Any]) -> dict[str, ModelConfig]:
    if not isinstance(raw, dict):
        raise ValueError("'model_catalog' deve ser um objeto JSON.")

    catalog: dict[str, ModelConfig] = {}
    for alias, item in raw.items():
        if not isinstance(item, dict):
            raise ValueError(f"Modelo '{alias}' inválido em 'model_catalog'.")

        path_value = item.get("path", alias)
        class_filters = tuple(str(value).lower() for value in item.get("class_filters", []))

        catalog[str(alias)] = ModelConfig(
            alias=str(alias),
            path=_resolve_path(base_dir, path_value),
            class_filters=class_filters,
            confidence=float(item.get("confidence", 0.25)),
            iou=float(item.get("iou", 0.45)),
            event_type=_optional_str(item.get("event_type")),
            cooldown_seconds=_optional_float(item.get("cooldown_seconds")),
            trigger_in_zones_only=bool(item.get("trigger_in_zones_only", False)),
            emit_events=bool(item.get("emit_events", True)),
            use_tracking=bool(item.get("use_tracking", False)),
            tracker=_optional_str(item.get("tracker")),
            enabled=bool(item.get("enabled", True)),
        )

    return catalog


def _load_camera_config(raw: dict[str, Any], index: int) -> CameraConfig:
    if not isinstance(raw, dict):
        raise ValueError("Cada item de 'cameras' deve ser um objeto JSON.")

    name = str(raw.get("name", f"Camera {index + 1}")).strip() or f"Camera {index + 1}"
    camera_id = str(raw.get("id", _slugify(name))).strip() or _slugify(name)

    models_raw = raw.get("models", [])
    if not isinstance(models_raw, list):
        raise ValueError(f"'models' da câmera '{name}' deve ser uma lista.")

    return CameraConfig(
        id=camera_id,
        name=name,
        source=_parse_source(raw.get("source", "")),
        models=tuple(str(model) for model in models_raw),
        fps_analysis=max(0.5, float(raw.get("fps_analysis", 3.0))),
        enabled=bool(raw.get("enabled", True)),
        snapshot_on_event=bool(raw.get("snapshot_on_event", True)),
        queue_maxsize=max(1, int(raw.get("queue_maxsize", 4))),
        reconnect_initial_delay=max(1.0, float(raw.get("reconnect_initial_delay", 2.0))),
        reconnect_max_delay=max(2.0, float(raw.get("reconnect_max_delay", 30.0))),
        cooldown_seconds=max(0.0, float(raw.get("cooldown_seconds", 10.0))),
        backend_preference=str(raw.get("backend_preference", "auto")).lower(),
        zones=_load_zones(raw.get("zones", [])),
        display=_load_display_config(raw.get("display", {}), name),
        smoking_behavior=_load_smoking_behavior_config(raw.get("smoking_behavior", {})),
    )


def _load_zones(raw: Any) -> tuple[ZoneConfig, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("'zones' deve ser uma lista.")

    zones: list[ZoneConfig] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError("Cada zona deve ser um objeto JSON.")

        name = str(item.get("name", f"zone-{index + 1}")).strip() or f"zone-{index + 1}"
        polygon_raw = item.get("polygon", [])
        if not isinstance(polygon_raw, list) or len(polygon_raw) < 3:
            raise ValueError(f"Zona '{name}' precisa de pelo menos 3 pontos.")

        polygon: list[tuple[float, float]] = []
        for point in polygon_raw:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError(f"Ponto inválido na zona '{name}'.")
            polygon.append((float(point[0]), float(point[1])))

        zones.append(ZoneConfig(name=name, polygon=tuple(polygon)))

    return tuple(zones)


def _load_display_config(raw: Any, camera_name: str) -> DisplayConfig:
    if raw is None:
        return DisplayConfig(window_name=f"Stealth Lens - {camera_name}")
    if not isinstance(raw, dict):
        raise ValueError("'display' deve ser um objeto JSON.")

    max_width = raw.get("max_width", 1280)
    return DisplayConfig(
        enabled=bool(raw.get("enabled", False)),
        window_name=_optional_str(raw.get("window_name")) or f"Stealth Lens - {camera_name}",
        show_metrics=bool(raw.get("show_metrics", True)),
        draw_zones=bool(raw.get("draw_zones", True)),
        max_width=int(max_width) if max_width not in (None, "") else None,
    )


def _load_smoking_behavior_config(raw: Any) -> SmokingBehaviorConfig:
    if raw is None:
        return SmokingBehaviorConfig()
    if not isinstance(raw, dict):
        raise ValueError("'smoking_behavior' deve ser um objeto JSON.")

    smoke_label = raw.get("smoke_label", "smoke")
    return SmokingBehaviorConfig(
        enabled=bool(raw.get("enabled", False)),
        model=str(raw.get("model", "smoking_monitor")),
        person_label=str(raw.get("person_label", "person")).lower(),
        cigarette_label=str(raw.get("cigarette_label", "cigarette")).lower(),
        smoke_label=str(smoke_label).lower() if smoke_label not in (None, "") else None,
        event_type=str(raw.get("event_type", "smoking")).lower(),
        max_distance_px=max(1, int(raw.get("max_distance_px", 80))),
        smoke_distance_multiplier=max(1.0, float(raw.get("smoke_distance_multiplier", 1.35))),
        min_frames=max(1, int(raw.get("min_frames", 12))),
        decay_frames=max(1, int(raw.get("decay_frames", 1))),
        smoke_boost_frames=max(0, int(raw.get("smoke_boost_frames", 2))),
        stale_track_seconds=max(1.0, float(raw.get("stale_track_seconds", 5.0))),
        event_cooldown_seconds=max(0.0, float(raw.get("event_cooldown_seconds", 20.0))),
        require_person_track=bool(raw.get("require_person_track", True)),
    )


def _resolve_path(base_dir: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _parse_source(value: Any) -> str | int:
    if isinstance(value, int):
        return value

    source = str(value).strip()
    if source.isdigit():
        return int(source)
    return source


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _slugify(value: str) -> str:
    sanitized = "".join(char.lower() if char.isalnum() else "-" for char in value)
    while "--" in sanitized:
        sanitized = sanitized.replace("--", "-")
    return sanitized.strip("-") or "camera"


def _sanitize_camera_id(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = _slugify(str(value))
    return cleaned or None


def _build_camera_id(name: str, source: str | int) -> str:
    source_key = _source_identity(source)
    if source_key.startswith("local:"):
        return f"webcam-{source_key}"
    if source_key.startswith("rtsp://"):
        digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:10]
        return f"rtsp-{digest}"
    return _slugify(name)


def _source_identity(source: Any) -> str:
    if isinstance(source, int):
        return f"local:{source}"

    raw = str(source or "").strip()
    if not raw:
        return ""
    if raw.isdigit():
        return f"local:{int(raw)}"

    lowered = raw.lower()
    if "://" not in lowered:
        return lowered

    parsed = urlsplit(raw)
    host = (parsed.hostname or "").lower()
    port = parsed.port or ""
    path = parsed.path or "/"
    query = parsed.query or ""
    if query:
        return f"{parsed.scheme.lower()}://{host}:{port}{path}?{query}"
    return f"{parsed.scheme.lower()}://{host}:{port}{path}"


def _deduplicate_camera_entries(cameras: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: list[dict[str, Any]] = []
    id_to_index: dict[str, int] = {}
    source_to_index: dict[str, int] = {}

    for camera in cameras:
        if not isinstance(camera, dict):
            continue

        camera_id = str(camera.get("id", "")).strip().lower()
        source_key = _source_identity(camera.get("source"))

        existing_index: int | None = None
        if camera_id and camera_id in id_to_index:
            existing_index = id_to_index[camera_id]
        elif source_key and source_key in source_to_index:
            existing_index = source_to_index[source_key]

        if existing_index is None:
            deduplicated.append(camera)
            index = len(deduplicated) - 1
        else:
            deduplicated[existing_index] = camera
            index = existing_index

        if camera_id:
            id_to_index[camera_id] = index
        if source_key:
            source_to_index[source_key] = index

    return deduplicated


def _select_camera_template(raw_config: dict[str, Any]) -> dict[str, Any]:
    cameras = raw_config.get("cameras", [])
    if cameras:
        return cameras[0]

    return {
        "id": "camera",
        "name": "Camera",
        "source": 0,
        "models": [next(iter(raw_config.get("model_catalog", {}) or ["smoking_monitor"]))],
        "fps_analysis": 3,
        "enabled": True,
        "snapshot_on_event": True,
        "queue_maxsize": 4,
        "cooldown_seconds": 10,
        "backend_preference": "auto",
        "zones": [],
        "display": {
            "enabled": True,
            "window_name": "Stealth Lens Agent - Camera",
            "show_metrics": True,
            "draw_zones": True,
            "max_width": 1280,
        },
        "smoking_behavior": {
            "enabled": True,
            "model": "smoking_monitor",
            "person_label": "person",
            "cigarette_label": "cigarette",
            "smoke_label": "smoke",
            "event_type": "smoking",
            "max_distance_px": 80,
            "smoke_distance_multiplier": 1.35,
            "min_frames": 12,
            "decay_frames": 1,
            "smoke_boost_frames": 2,
            "stale_track_seconds": 5,
            "event_cooldown_seconds": 20,
            "require_person_track": True,
        },
    }


def _default_model_catalog() -> dict[str, Any]:
    return {
        "smoking_monitor": {
            "path": "runs/detect/train/weights/best.pt",
            "class_filters": ["person", "cigarette", "smoke"],
            "confidence": 0.25,
            "cooldown_seconds": 8,
            "emit_events": False,
            "use_tracking": True,
            "tracker": "bytetrack.yaml",
        },
        "intrusion": {
            "path": "yolov8m.pt",
            "class_filters": ["person"],
            "event_type": "intrusion",
            "confidence": 0.35,
            "cooldown_seconds": 12,
            "trigger_in_zones_only": True,
        },
    }


def _apply_raw_defaults(raw_config: dict[str, Any]) -> dict[str, Any]:
    raw = copy.deepcopy(raw_config)
    defaults = build_default_raw_config()

    if not isinstance(raw.get("model_catalog"), dict):
        raw["model_catalog"] = {}
    if not isinstance(raw.get("cameras"), list):
        raw["cameras"] = []

    for key in ("agent_id", "device", "status_interval_seconds", "logging", "storage"):
        raw.setdefault(key, copy.deepcopy(defaults[key]))

    raw_model_catalog: dict[str, Any] = raw["model_catalog"]
    for alias, model_entry in defaults["model_catalog"].items():
        raw_model_catalog.setdefault(alias, copy.deepcopy(model_entry))

    cameras_raw = raw.get("cameras", [])
    if isinstance(cameras_raw, list):
        raw["cameras"] = _deduplicate_camera_entries(cameras_raw)

    return raw
