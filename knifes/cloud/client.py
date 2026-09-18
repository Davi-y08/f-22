from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import threading
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from utils.config import CloudConfig
from utils.redaction import redact_url_credentials
from cloud.outbox import EventOutbox

if TYPE_CHECKING:
    from discovery.service import DiscoveredCamera
    from events.emitter import DetectionEvent


MAX_SNAPSHOT_UPLOAD_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CloudSyncResult:
    success: bool
    message: str
    created: int = 0
    updated: int = 0


class CloudClientError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class CloudClient:
    def __init__(
        self,
        config: CloudConfig,
        agent_id: str,
        logger: Any | None = None,
        outbox_path: Path | None = None,
    ) -> None:
        self.config = config
        self.agent_id = agent_id
        self.logger = logger
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        scope = json.dumps([config.api_base_url.rstrip("/"), agent_id, config.agent_access_key])
        self._outbox = EventOutbox(outbox_path, hashlib.sha256(scope.encode()).hexdigest())
        self._last_sync_error: str | None = None
        self._synced_events = 0
        self._closed = False
        self._final_status: dict[str, Any] | None = None
        self._worker: threading.Thread | None = None

        if self.enabled and self.config.sync_events:
            self._worker = threading.Thread(target=self._event_worker, name="cloud-event-sync", daemon=True)
            self._worker.start()

    @property
    def enabled(self) -> bool:
        return bool(
            self.config.enabled
            and self.config.api_base_url.strip()
            and self.config.agent_access_key.strip()
        )

    def sync_discovered_cameras(self, cameras: list["DiscoveredCamera"]) -> CloudSyncResult:
        if not self.enabled or not self.config.sync_discovered_cameras:
            return CloudSyncResult(False, "Sincronização com o site desativada.")

        payload = {
            "agent_id": self.agent_id,
            "cameras": [
                camera_payload
                for camera in cameras
                if (camera_payload := _camera_payload_from_discovered(camera)) is not None
            ],
        }

        if not payload["cameras"]:
            return CloudSyncResult(False, "Nenhuma câmera válida para sincronizar.")

        return _camera_sync_result(
            self._post_json("/agent/cameras/sync", payload),
            "Site sincronizado",
        )

    def sync_configured_camera(
        self,
        *,
        external_id: str,
        name: str,
        source: str | int,
        status: str,
    ) -> CloudSyncResult:
        if not self.enabled or not self.config.sync_discovered_cameras:
            return CloudSyncResult(False, "Sincronização com o site desativada.")

        source_url = _source_to_url(source)
        camera_payload = {
            "external_id": str(external_id or source).strip(),
            "name": str(name or "Câmera local").strip(),
            "location": "Agente local",
            "url": source_url,
            "source": source_url,
            "status": _normalize_cloud_status(status),
        }

        return _camera_sync_result(
            self._post_json(
                "/agent/cameras/sync",
                {"agent_id": self.agent_id, "cameras": [camera_payload]},
            ),
            "Câmera sincronizada com o site",
        )

    def emit_event_async(self, event: "DetectionEvent") -> None:
        if not self.enabled or not self.config.sync_events:
            return

        payload = asdict(event) if is_dataclass(event) else dict(event)
        payload["agent_id"] = self.agent_id
        if "bbox" in payload:
            payload["bbox"] = list(payload["bbox"])
        if payload.get("frame_size") is not None:
            payload["frame_size"] = list(payload["frame_size"])
        if self._closed:
            raise RuntimeError("Cliente de sincronizacao encerrado.")
        if payload.get("snapshot_path"):
            payload["snapshot_path"] = str(Path(payload["snapshot_path"]).resolve())
        try:
            self._outbox.put(payload)
        except Exception as exc:
            self._last_sync_error = f"Falha ao guardar alerta para envio: {exc}"
            raise
        self._wake_event.set()

    def status_snapshot(self) -> dict[str, Any]:
        if self._final_status is not None:
            return dict(self._final_status)
        return {
            "enabled": self.enabled,
            **self._outbox.counts(),
            "synced_events": self._synced_events,
            "last_error": self._last_sync_error,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        self._wake_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=self.config.timeout_seconds + 1)
        if not self._worker or not self._worker.is_alive():
            self._final_status = self.status_snapshot()
            self._outbox.close()

    def _event_worker(self) -> None:
        while not self._stop_event.is_set():
            self._wake_event.clear()
            try:
                if self._sync_next_event():
                    continue
            except Exception as exc:
                self._last_sync_error = str(exc)
                if self.logger:
                    self.logger.warning(
                        "cloud_outbox_failed", extra={"error": str(exc)},
                    )
            self._wake_event.wait(0.5)

    def _sync_next_event(self) -> bool:
        queued = self._outbox.next_due()
        if queued is None:
            return False
        payload, attempts = queued
        event_id = payload["event_id"]
        try:
            _attach_snapshot_payload(payload)
            self._post_json("/agent/events", payload)
        except Exception as exc:
            retryable = not isinstance(exc, CloudClientError) or exc.retryable
            self._last_sync_error = str(exc)
            self._outbox.reject(event_id, attempts, str(exc), retryable)
            if self.logger:
                self.logger.warning(
                    "cloud_event_retry_pending" if retryable else "cloud_event_rejected",
                    extra={"event_id": event_id, "error": str(exc), "attempts": attempts + 1},
                )
        else:
            self._outbox.acknowledge(event_id)
            self._synced_events += 1
            self._last_sync_error = None
            if self.logger:
                self.logger.info("cloud_event_synced", extra={"event_id": event_id})
        return True

    def _post_json(self, path: str, payload: dict[str, Any]) -> Any:
        url = f"{self.config.api_base_url.rstrip('/')}/{path.lstrip('/')}"
        request = Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "StealthLensAgent/1.0",
                "X-Agent-Key": self.config.agent_access_key,
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                body = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise CloudClientError(
                f"API retornou HTTP {exc.code}: {_short_error_body(body)}",
                retryable=exc.code not in {400, 413, 415, 422},
            ) from exc
        except URLError as exc:
            raise CloudClientError(f"Não foi possível conectar em {redact_url_credentials(url)}: {exc.reason}") from exc

        if not body.strip():
            return {}

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise CloudClientError("API retornou uma resposta inválida.") from exc


def _camera_payload_from_discovered(camera: "DiscoveredCamera") -> dict[str, Any] | None:
    source = _source_from_discovered(camera)
    if source is None:
        return None

    external_id = camera.device_uuid or camera.key or source
    return {
        "external_id": str(external_id),
        "name": camera.name or "Câmera descoberta",
        "location": "Agente local",
        "url": source,
        "source": source,
        "status": _normalize_cloud_status(camera.status),
    }


def _source_from_discovered(camera: "DiscoveredCamera") -> str | None:
    if camera.kind == "local" and camera.local_index is not None:
        return f"local://{camera.local_index}"

    if not camera.host:
        return None

    if camera.http_ports:
        port = camera.http_ports[0]
        path = camera.http_stream_paths[0] if camera.http_stream_paths else "/video"
        return f"http://{camera.host}:{port}/{path.lstrip('/')}"

    if camera.rtsp_ports:
        port = camera.rtsp_ports[0]
        path = camera.rtsp_stream_paths[0] if camera.rtsp_stream_paths else ""
        suffix = f"/{path.lstrip('/')}" if path else "/"
        return f"rtsp://{camera.host}:{port}{suffix}"

    if camera.onvif_xaddrs:
        parsed = urlparse(camera.onvif_xaddrs[0])
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return camera.onvif_xaddrs[0]

    return None


def _source_to_url(source: str | int) -> str:
    if isinstance(source, int):
        return f"local://{source}"

    raw = str(source).strip()
    if raw.isdigit():
        return f"local://{raw}"
    if raw.startswith("local://"):
        return raw
    return raw


def _attach_snapshot_payload(payload: dict[str, Any]) -> None:
    snapshot_path = payload.get("snapshot_path")
    if not snapshot_path:
        return

    path = Path(str(snapshot_path)).expanduser()
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_SNAPSHOT_UPLOAD_BYTES + 1)
    except OSError as exc:
        raise CloudClientError("Nao foi possivel ler a foto do alerta.", retryable=False) from exc

    if not data or len(data) > MAX_SNAPSHOT_UPLOAD_BYTES:
        raise CloudClientError("Foto vazia ou maior que 5 MB.", retryable=False)

    mime_type, _ = mimetypes.guess_type(str(path))
    payload["snapshot_base64"] = base64.b64encode(data).decode("ascii")
    payload["snapshot_mime_type"] = mime_type or "image/jpeg"
    payload["snapshot_filename"] = path.name


def _normalize_cloud_status(status: str) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in {"online", "http_stream_detected", "rtsp_stream_detected"}:
        return "online"
    if normalized == "offline":
        return "offline"
    return "unknown"


def _camera_sync_result(response: Any, prefix: str) -> CloudSyncResult:
    created = int(response.get("created", 0)) if isinstance(response, dict) else 0
    updated = int(response.get("updated", 0)) if isinstance(response, dict) else 0
    return CloudSyncResult(
        True,
        f"{prefix}: {created} criada(s), {updated} atualizada(s).",
        created=created,
        updated=updated,
    )


def _short_error_body(body: str) -> str:
    compact = " ".join(str(body or "").split())
    return compact[:220] if compact else "sem corpo de resposta"
