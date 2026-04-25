from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import dataclass, field
import getpass
import hashlib
import ipaddress
import re
import socket
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse, urlunparse
import xml.etree.ElementTree as ET

import cv2

from discovery.cache import DiscoveryCache
from discovery.security import redacted_urls, validate_stream_url
from streams.rtsp_client import test_stream_source
from utils.config import (
    build_camera_entry_from_template,
    load_raw_config,
    upsert_camera_in_config,
)

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


COMMON_RTSP_PORTS = (554, 8554, 10554)
COMMON_HTTP_CAMERA_PORTS = (8080, 8081)
LOCAL_TEST_RTSP_PATHS = tuple(f"cam{index}" for index in range(1, 9))
GENERIC_RTSP_PATHS = (
    "",
    *LOCAL_TEST_RTSP_PATHS,
    "live",
    "live/main",
    "stream",
    "stream1",
    "Streaming/Channels/101",
    "Streaming/Channels/102",
    "cam/realmonitor?channel=1&subtype=0",
    "cam/realmonitor?channel=1&subtype=1",
    "h264Preview_01_main",
    "h264Preview_01_sub",
    "axis-media/media.amp",
)
GENERIC_HTTP_STREAM_PATHS = (
    "/video",
    "/videofeed",
    "/?action=stream",
    "/mjpegfeed",
    "/mjpeg",
)


@dataclass(slots=True)
class DiscoveredCamera:
    key: str
    kind: str
    name: str
    host: str | None = None
    local_index: int | None = None
    manufacturer: str | None = None
    model: str | None = None
    rtsp_ports: list[int] = field(default_factory=list)
    rtsp_stream_paths: list[str] = field(default_factory=list)
    rtsp_server: str | None = None
    http_ports: list[int] = field(default_factory=list)
    http_stream_paths: list[str] = field(default_factory=list)
    http_server: str | None = None
    onvif_xaddrs: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)
    discovery_sources: set[str] = field(default_factory=set)
    device_uuid: str | None = None
    confidence: float = 0.0
    status: str = "discovered"
    status_reason: str | None = None


@dataclass(slots=True)
class DiscoverySetupResult:
    configured: bool
    run_after_setup: bool = False
    selected_camera_name: str | None = None
    source: str | int | None = None
    saved_config_path: Path | None = None


@dataclass(slots=True)
class SourceResolutionResult:
    source: str | int | None
    validated: bool
    error: str | None = None
    tested_urls: list[str] = field(default_factory=list)


_DISCOVERY_CACHE: DiscoveryCache[list[DiscoveredCamera]] = DiscoveryCache()
_SOURCE_RESOLUTION_CACHE: DiscoveryCache[SourceResolutionResult] = DiscoveryCache()


def discover_cameras(
    logger: Any,
    include_local_webcams: bool = True,
    rtsp_scan_timeout: float = 0.35,
    http_scan_timeout: float = 0.35,
    onvif_timeout: float = 2.5,
    use_cache: bool = True,
    cache_ttl_seconds: float = 20.0,
    max_workers: int = 64,
    rtsp_scan_budget_seconds: float = 8.0,
    http_scan_budget_seconds: float = 6.0,
) -> list[DiscoveredCamera]:
    cache_key = (
        include_local_webcams,
        round(rtsp_scan_timeout, 2),
        round(http_scan_timeout, 2),
        round(onvif_timeout, 2),
    )
    if use_cache:
        cached = _DISCOVERY_CACHE.get(cache_key, ttl_seconds=cache_ttl_seconds)
        if cached is not None:
            logger.info("camera_discovery_cache_hit", extra={"devices_found": len(cached)})
            return cached

    discovery_started_at = time.monotonic()
    discovered: dict[str, DiscoveredCamera] = {}
    hosts_to_probe = _enumerate_hosts_for_network_scan(logger)
    network_workers = max(16, max_workers // 2)

    discovery_jobs = {
        "onvif": lambda: _discover_onvif_devices(timeout=onvif_timeout, logger=logger),
        "rtsp": lambda: _discover_rtsp_devices(
            hosts_to_probe=hosts_to_probe,
            timeout=rtsp_scan_timeout,
            logger=logger,
            max_workers=network_workers,
            scan_budget_seconds=rtsp_scan_budget_seconds,
        ),
        "http": lambda: _discover_http_devices(
            hosts_to_probe=hosts_to_probe,
            timeout=http_scan_timeout,
            logger=logger,
            max_workers=network_workers,
            scan_budget_seconds=http_scan_budget_seconds,
        ),
    }

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(job): name for name, job in discovery_jobs.items()}
        for future in as_completed(futures):
            job_name = futures[future]
            try:
                cameras = future.result()
            except Exception as exc:
                logger.warning("camera_discovery_job_failed", extra={"job": job_name, "error": str(exc)})
                continue
            for camera in cameras:
                _merge_discovered_camera(discovered, camera)

    results = list(discovered.values())

    if include_local_webcams:
        for camera in _discover_local_webcams(logger):
            _merge_discovered_camera(discovered, camera)
        results = list(discovered.values())

    results = _collapse_local_rtsp_aliases(results)
    results.sort(key=lambda item: (-item.confidence, item.kind != "network", item.host or "", item.local_index or -1))

    elapsed_ms = int((time.monotonic() - discovery_started_at) * 1000)
    logger.info(
        "camera_discovery_completed",
        extra={
            "devices_found": len(results),
            "hosts_scanned": len(hosts_to_probe),
            "elapsed_ms": elapsed_ms,
            "network_workers_per_probe": network_workers,
        },
    )

    if use_cache:
        _DISCOVERY_CACHE.set(cache_key, results)

    return results


def run_interactive_camera_setup(
    config_path: str | Path,
    logger: Any,
) -> DiscoverySetupResult:
    cameras = discover_cameras(logger=logger)
    if not cameras:
        print("Nenhuma câmera foi encontrada automaticamente na rede ou localmente.")
        return DiscoverySetupResult(configured=False)

    _print_discovered_cameras(cameras)
    selected = _prompt_camera_selection(cameras)
    if selected is None:
        print("Seleção cancelada.")
        return DiscoverySetupResult(configured=False)

    source = _resolve_camera_source(selected)
    if source is None:
        return DiscoverySetupResult(configured=False)

    default_name = selected.name
    chosen_name = input(f"Nome da câmera [{default_name}]: ").strip() or default_name

    raw_config = load_raw_config(config_path)
    camera_entry = build_camera_entry_from_template(
        raw_config,
        chosen_name,
        source,
        camera_id=selected.key,
    )
    saved_path = upsert_camera_in_config(config_path, camera_entry)

    print(f"Câmera salva em: {saved_path}")
    run_after_setup = _prompt_yes_no("Iniciar monitoramento agora?", default=True)
    return DiscoverySetupResult(
        configured=True,
        run_after_setup=run_after_setup,
        selected_camera_name=chosen_name,
        source=source,
        saved_config_path=saved_path,
    )


def resolve_camera_source(
    camera: DiscoveredCamera,
    username: str = "",
    password: str = "",
    manual_rtsp_url: str | None = None,
    validate_stream: bool = True,
    timeout_seconds: float = 4.0,
) -> SourceResolutionResult:
    if camera.kind == "local" and camera.local_index is not None:
        return SourceResolutionResult(source=camera.local_index, validated=True)

    host = camera.host
    if not host:
        return SourceResolutionResult(
            source=None,
            validated=False,
            error="A câmera escolhida não possui host utilizável.",
        )

    resolution_cache_key = _source_resolution_cache_key(
        camera=camera,
        username=username,
        password=password,
        manual_rtsp_url=manual_rtsp_url,
        validate_stream=validate_stream,
    )
    if validate_stream:
        cached_resolution = _SOURCE_RESOLUTION_CACHE.get(resolution_cache_key, ttl_seconds=30.0)
        if cached_resolution is not None:
            return cached_resolution

    tested_urls: list[str] = []
    manual_candidates = (
        _filter_safe_manual_stream_candidates(_expand_manual_stream_candidates(manual_rtsp_url))
        if manual_rtsp_url
        else []
    )
    if manual_rtsp_url and not manual_candidates:
        return SourceResolutionResult(
            source=None,
            validated=False,
            error="URL manual inválida ou bloqueada por segurança. Use RTSP/HTTP(S) para IP de câmera válido.",
            tested_urls=[],
        )

    if manual_candidates:
        for candidate in manual_candidates:
            tested_urls.append(candidate)

        if not validate_stream:
            return SourceResolutionResult(
                source=manual_candidates[0],
                validated=False,
                tested_urls=redacted_urls(tested_urls),
            )

        for manual_candidate in manual_candidates:
            if test_stream_source(manual_candidate, backend_preference="auto", timeout_seconds=timeout_seconds):
                result = SourceResolutionResult(
                    source=manual_candidate,
                    validated=True,
                    tested_urls=redacted_urls(tested_urls),
                )
                _SOURCE_RESOLUTION_CACHE.set(resolution_cache_key, result)
                return result

    rtsp_candidates = build_candidate_rtsp_urls(camera, username=username, password=password)
    http_candidates = build_candidate_http_urls(camera, username=username, password=password)
    candidates = _filter_safe_stream_candidates(
        _prioritize_stream_candidates(camera, rtsp_candidates, http_candidates),
        allow_loopback=_is_loopback_host(host),
    )

    if validate_stream:
        for url in candidates:
            tested_urls.append(url)
            if test_stream_source(url, backend_preference="auto", timeout_seconds=timeout_seconds):
                result = SourceResolutionResult(
                    source=url,
                    validated=True,
                    tested_urls=redacted_urls(tested_urls),
                )
                _SOURCE_RESOLUTION_CACHE.set(resolution_cache_key, result)
                return result
    elif candidates:
        return SourceResolutionResult(
            source=candidates[0],
            validated=False,
            tested_urls=redacted_urls(candidates[:1]),
        )

    error = "Nenhuma URL de stream (RTSP/HTTP) foi validada para esta câmera."
    if camera.status == "credentials_required" and not username:
        error = "Câmera detectada, mas o stream provavelmente exige usuário e senha."
    elif camera.status == "onvif_detected":
        error = "ONVIF detectado, mas nenhum stream RTSP/HTTP comum respondeu."

    return SourceResolutionResult(
        source=None,
        validated=False,
        error=error,
        tested_urls=redacted_urls(tested_urls),
    )


def save_camera_selection(
    config_path: str | Path,
    camera_name: str,
    source: str | int,
    camera_key: str | None = None,
) -> Path:
    raw_config = load_raw_config(config_path)
    camera_entry = build_camera_entry_from_template(
        raw_config,
        camera_name,
        source,
        camera_id=camera_key,
    )
    return upsert_camera_in_config(config_path, camera_entry)


def build_candidate_rtsp_urls(
    camera: DiscoveredCamera,
    username: str = "",
    password: str = "",
) -> list[str]:
    return _build_candidate_rtsp_urls(camera=camera, username=username, password=password)


def build_candidate_http_urls(
    camera: DiscoveredCamera,
    username: str = "",
    password: str = "",
) -> list[str]:
    return _build_candidate_http_urls(camera=camera, username=username, password=password)


def describe_camera(camera: DiscoveredCamera) -> str:
    if camera.kind == "local":
        return f"{camera.name} | local | index={camera.local_index} | status={camera.status}"

    sources = ",".join(sorted(camera.discovery_sources)) or "network"
    details: list[str] = [
        f"host={camera.host}",
        f"src={sources}",
        f"status={camera.status}",
        f"conf={camera.confidence:.2f}",
    ]
    if camera.rtsp_ports:
        rtsp_ports = ",".join(str(port) for port in sorted(camera.rtsp_ports))
        rtsp_paths = ",".join(camera.rtsp_stream_paths[:4])
        details.append(f"rtsp={rtsp_ports}" + (f" paths={rtsp_paths}" if rtsp_paths else ""))
    if camera.http_ports:
        http_ports = ",".join(str(port) for port in sorted(camera.http_ports))
        details.append(f"http={http_ports}")
    return f"{camera.name} | " + " | ".join(details)


def _discover_onvif_devices(timeout: float, logger: Any) -> list[DiscoveredCamera]:
    message_id = uuid.uuid4()
    payload = f"""<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
 xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
 xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
 xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
  <e:Header>
    <w:MessageID>uuid:{message_id}</w:MessageID>
    <w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
    <w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
  </e:Header>
  <e:Body>
    <d:Probe>
      <d:Types>dn:NetworkVideoTransmitter</d:Types>
    </d:Probe>
  </e:Body>
</e:Envelope>""".encode("utf-8")

    cameras: list[DiscoveredCamera] = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        sock.settimeout(timeout)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        sock.sendto(payload, ("239.255.255.250", 3702))

        while True:
            try:
                data, address = sock.recvfrom(65535)
            except socket.timeout:
                break

            camera = _parse_onvif_response(data, address[0])
            if camera is not None:
                cameras.append(camera)
    except OSError as exc:
        logger.warning("onvif_discovery_failed", extra={"error": str(exc)})
    finally:
        sock.close()

    return cameras


def _discover_rtsp_devices(
    hosts_to_probe: set[str],
    timeout: float,
    logger: Any,
    max_workers: int,
    scan_budget_seconds: float,
) -> list[DiscoveredCamera]:
    if not hosts_to_probe:
        return []

    cameras: dict[str, DiscoveredCamera] = {}
    futures = []
    worker_count = _scan_worker_count(hosts_to_probe, max_workers=max_workers)
    executor = ThreadPoolExecutor(max_workers=worker_count)
    timed_out = 0
    try:
        for host in hosts_to_probe:
            for port in COMMON_RTSP_PORTS:
                futures.append(executor.submit(_probe_rtsp_endpoint, host, port, timeout))

        try:
            completed = as_completed(futures, timeout=max(1.0, scan_budget_seconds))
            for future in completed:
                try:
                    host, port, is_rtsp, server, status = future.result()
                except Exception:
                    continue
                if not is_rtsp:
                    continue

                stream_paths = _probe_common_rtsp_stream_paths(host, port, server, timeout)
                if stream_paths:
                    for stream_path in stream_paths:
                        key = f"network:{host}:rtsp:{port}:{stream_path}"
                        camera = cameras.setdefault(
                            key,
                            DiscoveredCamera(
                                key=key,
                                kind="network",
                                name=f"RTSP Stream {stream_path} ({host})",
                                host=host,
                                rtsp_ports=[port],
                                rtsp_stream_paths=[stream_path],
                                rtsp_server=server,
                                confidence=0.86,
                                status="rtsp_stream_detected",
                                discovery_sources={"rtsp-scan", "rtsp-path"},
                            ),
                        )
                        camera.confidence = max(camera.confidence, 0.86)
                        camera.status = _best_camera_status(camera.status, "rtsp_stream_detected")
                    continue

                key = f"network:{host}"
                camera = cameras.setdefault(
                    key,
                    DiscoveredCamera(
                        key=key,
                        kind="network",
                        name=f"RTSP Camera {host}",
                        host=host,
                        confidence=0.65,
                        status=status,
                    ),
                )
                if port not in camera.rtsp_ports:
                    camera.rtsp_ports.append(port)
                camera.rtsp_server = camera.rtsp_server or server
                camera.confidence = max(
                    camera.confidence,
                    0.7 if status == "rtsp_detected" else 0.62,
                )
                camera.status = _best_camera_status(camera.status, status)
                camera.discovery_sources.add("rtsp-scan")
        except FuturesTimeoutError:
            timed_out = sum(1 for future in futures if not future.done())
            for future in futures:
                future.cancel()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    if cameras:
        logger.info(
            "rtsp_discovery_completed",
            extra={
                "devices_found": len(cameras),
                "hosts_scanned": len(hosts_to_probe),
                "workers": worker_count,
                "timed_out_probes": timed_out,
            },
        )

    return list(cameras.values())


def _discover_http_devices(
    hosts_to_probe: set[str],
    timeout: float,
    logger: Any,
    max_workers: int,
    scan_budget_seconds: float,
) -> list[DiscoveredCamera]:
    if not hosts_to_probe:
        return []

    cameras: dict[str, DiscoveredCamera] = {}
    futures = []
    worker_count = _scan_worker_count(hosts_to_probe, max_workers=max_workers)
    executor = ThreadPoolExecutor(max_workers=worker_count)
    timed_out = 0
    try:
        for host in hosts_to_probe:
            for port in COMMON_HTTP_CAMERA_PORTS:
                futures.append(executor.submit(_probe_http_camera_endpoint, host, port, timeout))

        try:
            completed = as_completed(futures, timeout=max(1.0, scan_budget_seconds))
            for future in completed:
                try:
                    host, port, is_camera, server, stream_paths, name, status, confidence = future.result()
                except Exception:
                    continue
                if not is_camera:
                    continue

                key = f"network:{host}"
                camera = cameras.setdefault(
                    key,
                    DiscoveredCamera(
                        key=key,
                        kind="network",
                        name=name or f"IP Webcam {host}",
                        host=host,
                        confidence=confidence,
                        status=status,
                    ),
                )
                if name:
                    camera.name = name
                if port not in camera.http_ports:
                    camera.http_ports.append(port)
                for stream_path in stream_paths:
                    if stream_path not in camera.http_stream_paths:
                        camera.http_stream_paths.append(stream_path)
                camera.http_server = camera.http_server or server
                camera.confidence = max(camera.confidence, confidence)
                camera.status = _best_camera_status(camera.status, status)
                camera.discovery_sources.add("http-scan")
        except FuturesTimeoutError:
            timed_out = sum(1 for future in futures if not future.done())
            for future in futures:
                future.cancel()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    if cameras:
        logger.info(
            "http_camera_discovery_completed",
            extra={
                "devices_found": len(cameras),
                "hosts_scanned": len(hosts_to_probe),
                "workers": worker_count,
                "timed_out_probes": timed_out,
            },
        )

    return list(cameras.values())


def _discover_local_webcams(logger: Any, max_indices: int = 5) -> list[DiscoveredCamera]:
    cameras: list[DiscoveredCamera] = []
    for index in range(max_indices):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(index)
        try:
            if not capture or not capture.isOpened():
                continue
            success, _ = capture.read()
            if not success:
                continue
        finally:
            if capture is not None:
                capture.release()

        cameras.append(
            DiscoveredCamera(
                key=f"local:{index}",
                kind="local",
                name=f"Webcam Local {index}",
                local_index=index,
                discovery_sources={"local-device"},
                confidence=0.78,
                status="online",
            )
        )

    if cameras:
        logger.info("local_webcams_discovered", extra={"count": len(cameras)})

    return cameras


def _resolve_camera_source(camera: DiscoveredCamera) -> str | int | None:
    if camera.kind == "local" and camera.local_index is not None:
        return camera.local_index

    host = camera.host
    if not host:
        print("A câmera escolhida não possui host utilizável.")
        return None

    username = input("Usuário (Enter se não houver): ").strip()
    password = getpass.getpass("Senha (Enter se não houver): ")
    print("Testando URLs de stream candidatas (RTSP/HTTP). Isso pode levar alguns segundos...")

    rtsp_candidates = _build_candidate_rtsp_urls(camera, username, password)
    http_candidates = _build_candidate_http_urls(camera, username, password)
    candidates = _filter_safe_stream_candidates(_prioritize_stream_candidates(camera, rtsp_candidates, http_candidates))

    for url in candidates:
        if test_stream_source(url, backend_preference="auto", timeout_seconds=4.0):
            print(f"Stream validado com sucesso: {redacted_urls([url])[0]}")
            return url

    print("Nenhuma URL de stream comum respondeu com sucesso.")
    manual = input("Cole a URL manualmente (RTSP/HTTP) ou pressione Enter para cancelar: ").strip()
    if not manual:
        print("Configuração cancelada.")
        return None

    manual_candidates = _filter_safe_manual_stream_candidates(_expand_manual_stream_candidates(manual))
    if not manual_candidates:
        print("URL manual inválida ou bloqueada por segurança.")
        return None

    for manual_candidate in manual_candidates:
        if test_stream_source(manual_candidate, backend_preference="auto", timeout_seconds=4.0):
            print(f"URL manual validada com sucesso: {redacted_urls([manual_candidate])[0]}")
            return manual_candidate

    if _prompt_yes_no("Não foi possível validar a URL manual. Deseja salvar mesmo assim?", default=False):
        return manual_candidates[0] if manual_candidates else manual

    print("Configuração cancelada.")
    return None


def _build_candidate_rtsp_urls(
    camera: DiscoveredCamera,
    username: str,
    password: str,
) -> list[str]:
    host = camera.host or ""
    ports = camera.rtsp_ports or list(COMMON_RTSP_PORTS)
    auth = ""
    if username:
        auth = quote(username, safe="") + ":" + quote(password, safe="") + "@"

    vendor_paths = _vendor_specific_paths(camera)
    paths = list(dict.fromkeys([*camera.rtsp_stream_paths, *vendor_paths, *GENERIC_RTSP_PATHS]))
    urls: list[str] = []
    for port in ports:
        for path in paths:
            suffix = f"/{path.lstrip('/')}" if path else "/"
            urls.append(f"rtsp://{auth}{host}:{port}{suffix}")

    return _dedupe_preserve_order(urls)


def _build_candidate_http_urls(
    camera: DiscoveredCamera,
    username: str,
    password: str,
) -> list[str]:
    host = camera.host or ""
    ports = camera.http_ports or list(COMMON_HTTP_CAMERA_PORTS)
    paths = camera.http_stream_paths or list(GENERIC_HTTP_STREAM_PATHS)
    auth = ""
    if username:
        auth = quote(username, safe="") + ":" + quote(password, safe="") + "@"

    urls: list[str] = []
    for port in ports:
        for path in paths:
            normalized_path = "/" + str(path or "").lstrip("/")
            urls.append(f"http://{auth}{host}:{port}{normalized_path}")

    deduplicated: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduplicated.append(url)
    return deduplicated


def _expand_manual_stream_candidates(manual_url: str) -> list[str]:
    value = (manual_url or "").strip()
    if not value:
        return []

    candidates: list[str] = [value]
    parsed = urlparse(value)
    if parsed.scheme.lower() not in {"http", "https"}:
        return candidates

    has_specific_path = bool(parsed.path and parsed.path.strip("/") and parsed.path not in {"/"})
    has_query = bool(parsed.query)
    if has_specific_path or has_query:
        return candidates

    for stream_path in GENERIC_HTTP_STREAM_PATHS:
        normalized_path = "/" + stream_path.lstrip("/")
        path_part, _, query_part = normalized_path.partition("?")
        expanded = urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                path_part,
                "",
                query_part,
                "",
            )
        )
        candidates.append(expanded)

    deduplicated: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduplicated.append(candidate)
    return deduplicated


def _prioritize_stream_candidates(
    camera: DiscoveredCamera,
    rtsp_candidates: list[str],
    http_candidates: list[str],
) -> list[str]:
    if not rtsp_candidates:
        return http_candidates
    if not http_candidates:
        return rtsp_candidates

    sources = {value.lower() for value in camera.discovery_sources}
    if "http-scan" in sources and "rtsp-scan" not in sources:
        return [*http_candidates, *rtsp_candidates]

    return [*rtsp_candidates, *http_candidates]


def _filter_safe_stream_candidates(candidates: list[str], allow_loopback: bool = False) -> list[str]:
    safe_candidates: list[str] = []
    for candidate in candidates:
        validation = validate_stream_url(candidate, allow_loopback=allow_loopback)
        if validation.valid:
            safe_candidates.append(candidate)
    return _dedupe_preserve_order(safe_candidates)


def _filter_safe_manual_stream_candidates(candidates: list[str]) -> list[str]:
    safe_candidates: list[str] = []
    for candidate in candidates:
        validation = validate_stream_url(candidate, allow_loopback=True)
        if validation.valid:
            safe_candidates.append(candidate)
    return _dedupe_preserve_order(safe_candidates)


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    normalized = str(host).strip().lower()
    if normalized in {"localhost", "localhost.localdomain"}:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _source_resolution_cache_key(
    camera: DiscoveredCamera,
    username: str,
    password: str,
    manual_rtsp_url: str | None,
    validate_stream: bool,
) -> tuple[str, str, str, bool]:
    credential_fingerprint = hashlib.sha256(
        f"{username}\0{password}".encode("utf-8", errors="ignore")
    ).hexdigest()[:16]
    path_identity = ",".join(camera.rtsp_stream_paths + camera.http_stream_paths)
    identity = camera.device_uuid or f"{camera.key}|{camera.host}|{path_identity}"
    manual_fingerprint = hashlib.sha256(
        str(manual_rtsp_url or "").encode("utf-8", errors="ignore")
    ).hexdigest()[:16]
    return str(identity), credential_fingerprint, manual_fingerprint, validate_stream


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    deduplicated: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduplicated.append(value)
    return deduplicated


def _vendor_specific_paths(camera: DiscoveredCamera) -> list[str]:
    blob = " ".join(
        value.lower()
        for value in (
            camera.manufacturer or "",
            camera.model or "",
            camera.rtsp_server or "",
            camera.name or "",
            *camera.scopes,
        )
    )

    if "hik" in blob:
        return ["Streaming/Channels/101", "Streaming/Channels/102"]
    if "dahua" in blob:
        return ["cam/realmonitor?channel=1&subtype=0", "cam/realmonitor?channel=1&subtype=1"]
    if "reolink" in blob:
        return ["h264Preview_01_main", "h264Preview_01_sub"]
    if "axis" in blob:
        return ["axis-media/media.amp"]
    return []


def _enumerate_local_networks(logger: Any) -> list[tuple[str, ipaddress.IPv4Network]]:
    networks = _enumerate_with_psutil()
    if not networks:
        networks = _enumerate_with_ipconfig()

    if networks:
        logger.info("local_networks_detected", extra={"count": len(networks)})
    return networks


def _enumerate_hosts_for_network_scan(logger: Any) -> set[str]:
    hosts_to_probe: set[str] = set()
    hosts_to_probe.add("127.0.0.1")
    for host, network in _enumerate_local_networks(logger):
        hosts_to_probe.add(host)
        for candidate in network.hosts():
            hosts_to_probe.add(str(candidate))
    return hosts_to_probe


def _scan_worker_count(hosts_to_probe: set[str], max_workers: int) -> int:
    endpoint_count = max(1, len(hosts_to_probe))
    requested = max(8, min(int(max_workers), endpoint_count * 2))
    return min(128, requested)


def _enumerate_with_psutil() -> list[tuple[str, ipaddress.IPv4Network]]:
    if psutil is None:
        return []

    results: list[tuple[str, ipaddress.IPv4Network]] = []
    stats = psutil.net_if_stats()

    for interface_name, addresses in psutil.net_if_addrs().items():
        interface_stats = stats.get(interface_name)
        if interface_stats is not None and not interface_stats.isup:
            continue

        for address in addresses:
            if address.family != socket.AF_INET or not address.address or not address.netmask:
                continue

            ip_addr = ipaddress.ip_address(address.address)
            if ip_addr.is_loopback or not (ip_addr.is_private or ip_addr.is_link_local):
                continue

            network = ipaddress.ip_network(f"{address.address}/{address.netmask}", strict=False)
            results.append((str(ip_addr), _limit_network_size(ip_addr, network)))

    return _unique_networks(results)


def _enumerate_with_ipconfig() -> list[tuple[str, ipaddress.IPv4Network]]:
    try:
        output = subprocess.run(
            ["ipconfig"],
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="ignore",
        ).stdout
    except OSError:
        return []

    ipv4_matches = re.findall(r"IPv4[^\:]*:\s*([0-9\.]+)", output, flags=re.IGNORECASE)
    mask_matches = re.findall(r"(?:Subnet Mask|Máscara[^\:]*):\s*([0-9\.]+)", output, flags=re.IGNORECASE)

    results: list[tuple[str, ipaddress.IPv4Network]] = []
    for ip_raw, mask_raw in zip(ipv4_matches, mask_matches):
        ip_addr = ipaddress.ip_address(ip_raw)
        if ip_addr.is_loopback or not (ip_addr.is_private or ip_addr.is_link_local):
            continue
        network = ipaddress.ip_network(f"{ip_raw}/{mask_raw}", strict=False)
        results.append((str(ip_addr), _limit_network_size(ip_addr, network)))

    return _unique_networks(results)


def _unique_networks(items: list[tuple[str, ipaddress.IPv4Network]]) -> list[tuple[str, ipaddress.IPv4Network]]:
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, ipaddress.IPv4Network]] = []
    for host, network in items:
        key = (host, str(network))
        if key in seen:
            continue
        seen.add(key)
        unique.append((host, network))
    return unique


def _limit_network_size(ip_addr: ipaddress.IPv4Address, network: ipaddress.IPv4Network) -> ipaddress.IPv4Network:
    if network.num_addresses <= 256:
        return network
    return ipaddress.ip_network(f"{ip_addr}/24", strict=False)


def _probe_rtsp_endpoint(host: str, port: int, timeout: float) -> tuple[str, int, bool, str | None, str]:
    request = (
        f"OPTIONS rtsp://{host}:{port}/ RTSP/1.0\r\n"
        "CSeq: 1\r\n"
        "User-Agent: StealthLens/1.0\r\n\r\n"
    ).encode("utf-8")

    try:
        with socket.create_connection((host, port), timeout=timeout) as connection:
            connection.settimeout(timeout)
            connection.sendall(request)
            response = connection.recv(1024).decode("utf-8", errors="ignore")
    except OSError:
        return host, port, False, None, "offline"

    if "RTSP/" not in response.upper():
        return host, port, False, None, "not_rtsp"

    server = None
    status = "rtsp_detected"
    first_line = response.splitlines()[0] if response.splitlines() else ""
    if " 401 " in f" {first_line} " or "UNAUTHORIZED" in first_line.upper():
        status = "credentials_required"
    for line in response.splitlines():
        if line.lower().startswith("server:"):
            server = line.split(":", 1)[1].strip()
            break

    return host, port, True, server, status


def _probe_common_rtsp_stream_paths(
    host: str,
    port: int,
    server: str | None,
    timeout: float,
) -> list[str]:
    server_blob = str(server or "").lower()
    should_probe_test_paths = port == 8554 or "mediamtx" in server_blob or "gortsplib" in server_blob
    if not should_probe_test_paths:
        return []

    discovered_paths: list[str] = []
    for path in LOCAL_TEST_RTSP_PATHS:
        status = _probe_rtsp_stream_path(host, port, path, timeout)
        if status in {"rtsp_stream_detected", "credentials_required"}:
            discovered_paths.append(path)
    return discovered_paths


def _probe_rtsp_stream_path(host: str, port: int, path: str, timeout: float) -> str:
    normalized_path = str(path or "").strip().lstrip("/")
    if not normalized_path:
        return "not_found"

    request = (
        f"DESCRIBE rtsp://{host}:{port}/{normalized_path} RTSP/1.0\r\n"
        "CSeq: 2\r\n"
        "Accept: application/sdp\r\n"
        "User-Agent: StealthLens/1.0\r\n\r\n"
    ).encode("utf-8")

    try:
        with socket.create_connection((host, port), timeout=timeout) as connection:
            connection.settimeout(timeout)
            connection.sendall(request)
            response = connection.recv(2048).decode("utf-8", errors="ignore")
    except OSError:
        return "offline"

    first_line = response.splitlines()[0] if response.splitlines() else ""
    upper_line = first_line.upper()
    if " 200 " in f" {upper_line} " or upper_line.endswith(" 200 OK"):
        return "rtsp_stream_detected"
    if " 401 " in f" {upper_line} " or "UNAUTHORIZED" in upper_line:
        return "credentials_required"
    return "not_found"


def _probe_http_camera_endpoint(
    host: str,
    port: int,
    timeout: float,
) -> tuple[str, int, bool, str | None, list[str], str | None, str, float]:
    stream_paths: list[str] = []
    server: str | None = None
    name: str | None = None
    status = "offline"
    confidence = 0.0

    root_status, root_headers, root_body = _probe_http_endpoint(host, port, "/", timeout)
    server = root_headers.get("server")
    root_has_camera_hint = _looks_like_camera_http_response(root_status, root_headers, root_body)
    if _looks_like_ip_webcam_body(root_body):
        name = f"IP Webcam {host}"
        confidence = max(confidence, 0.8)
    elif root_has_camera_hint:
        name = f"HTTP Camera {host}"
        confidence = max(confidence, 0.55)

    paths_to_probe = ["/video"]
    if root_has_camera_hint:
        paths_to_probe.extend(path for path in GENERIC_HTTP_STREAM_PATHS if path != "/video")

    for path in _dedupe_preserve_order(paths_to_probe):
        status_code, headers, body = _probe_http_endpoint(host, port, path, timeout)
        if server is None:
            server = headers.get("server")

        content_type = headers.get("content-type", "").lower()
        is_stream = ("multipart/x-mixed-replace" in content_type) or ("image/jpeg" in content_type)
        requires_auth = status_code == 401 and _looks_like_camera_headers(headers)
        if status_code in {200, 401} and (is_stream or requires_auth):
            normalized_path = "/" + path.lstrip("/")
            if normalized_path not in stream_paths:
                stream_paths.append(normalized_path)
            confidence = max(confidence, 0.78 if status_code == 200 else 0.62)
            status = "http_stream_detected" if status_code == 200 else "credentials_required"

        if _looks_like_ip_webcam_body(body):
            name = f"IP Webcam {host}"
            confidence = max(confidence, 0.86)

    is_camera = bool(stream_paths) or bool(name)
    if is_camera and not stream_paths:
        stream_paths.append("/video")
        status = "http_camera_detected"
        confidence = max(confidence, 0.58)

    return host, port, is_camera, server, stream_paths, name, status, confidence


def _probe_http_endpoint(
    host: str,
    port: int,
    path: str,
    timeout: float,
) -> tuple[int | None, dict[str, str], str]:
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "User-Agent: StealthLens/1.0\r\n"
        "Connection: close\r\n\r\n"
    ).encode("utf-8")

    raw_response = b""
    try:
        with socket.create_connection((host, port), timeout=timeout) as connection:
            connection.settimeout(timeout)
            connection.sendall(request)
            while True:
                chunk = connection.recv(2048)
                if not chunk:
                    break
                raw_response += chunk
                if len(raw_response) >= 8192:
                    break
    except OSError:
        return None, {}, ""

    decoded = raw_response.decode("utf-8", errors="ignore")
    if not decoded:
        return None, {}, ""

    head, _, body = decoded.partition("\r\n\r\n")
    lines = [line for line in head.splitlines() if line.strip()]
    if not lines:
        return None, {}, body

    status_code: int | None = None
    try:
        parts = lines[0].split()
        if len(parts) >= 2:
            status_code = int(parts[1])
    except Exception:
        status_code = None

    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        headers[key.strip().lower()] = value.strip()

    return status_code, headers, body


def _looks_like_ip_webcam_body(body: str) -> bool:
    lowered = body.lower()
    return ("ip webcam" in lowered) or ("pavel khlebovich" in lowered)


def _looks_like_camera_http_response(
    status_code: int | None,
    headers: dict[str, str],
    body: str,
) -> bool:
    if _looks_like_ip_webcam_body(body):
        return True
    if _looks_like_camera_headers(headers):
        return True
    if status_code == 401 and "www-authenticate" in headers:
        return _looks_like_camera_headers(headers)
    return False


def _looks_like_camera_headers(headers: dict[str, str]) -> bool:
    blob = " ".join(
        str(value).lower()
        for key, value in headers.items()
        if key in {"server", "www-authenticate", "x-powered-by"}
    )
    camera_markers = (
        "ip webcam",
        "mjpg-streamer",
        "motion",
        "webcam",
        "goahead",
        "hikvision",
        "dahua",
        "axis",
        "rtsp",
        "onvif",
    )
    return any(marker in blob for marker in camera_markers)


def _parse_onvif_response(data: bytes, fallback_host: str) -> DiscoveredCamera | None:
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return None

    namespaces = {
        "soap": "http://www.w3.org/2003/05/soap-envelope",
        "d2005": "http://schemas.xmlsoap.org/ws/2005/04/discovery",
        "d2009": "http://docs.oasis-open.org/ws-dd/ns/discovery/2009/01",
    }

    xaddrs_text = (
        root.findtext(".//d2005:XAddrs", default="", namespaces=namespaces)
        or root.findtext(".//d2009:XAddrs", default="", namespaces=namespaces)
    )
    scopes_text = (
        root.findtext(".//d2005:Scopes", default="", namespaces=namespaces)
        or root.findtext(".//d2009:Scopes", default="", namespaces=namespaces)
    )

    xaddrs = [item.strip() for item in xaddrs_text.split() if item.strip()]
    scopes = [item.strip() for item in scopes_text.split() if item.strip()]
    endpoint_address = _find_xml_text_by_local_name(root, "Address")
    device_uuid = _extract_uuid(endpoint_address)

    host = fallback_host
    if xaddrs:
        parsed_host = _extract_host_from_url(xaddrs[0])
        host = parsed_host or fallback_host

    if not host:
        return None

    name = _extract_scope_value(scopes, "name") or f"ONVIF Camera {host}"
    model = _extract_scope_value(scopes, "hardware")
    manufacturer = _extract_scope_value(scopes, "manufacturer") or _extract_scope_value(scopes, "vendor")

    camera = DiscoveredCamera(
        key=f"network:{host}",
        kind="network",
        name=name,
        host=host,
        model=model,
        manufacturer=manufacturer,
        onvif_xaddrs=xaddrs,
        scopes=scopes,
        discovery_sources={"onvif"},
        device_uuid=device_uuid,
        confidence=0.88,
        status="onvif_detected",
    )
    return camera


def _extract_scope_value(scopes: list[str], key: str) -> str | None:
    prefix = f"onvif://www.onvif.org/{key}/"
    for scope in scopes:
        if scope.startswith(prefix):
            return scope[len(prefix) :].replace("%20", " ")
    return None


def _find_xml_text_by_local_name(root: ET.Element, local_name: str) -> str | None:
    for element in root.iter():
        tag = str(element.tag)
        if tag.rsplit("}", 1)[-1] == local_name and element.text:
            return element.text.strip()
    return None


def _extract_uuid(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(
        r"(?:urn:)?uuid:([0-9a-fA-F\-]{8,})",
        value,
        flags=re.IGNORECASE,
    )
    return match.group(1).lower() if match else None


def _extract_host_from_url(url: str) -> str | None:
    match = re.search(r"^[a-z]+://\[?([^\]/:]+)\]?(?::\d+)?", url, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _merge_discovered_camera(
    discovered: dict[str, DiscoveredCamera],
    camera: DiscoveredCamera,
) -> None:
    existing_key = _find_existing_camera_key(discovered, camera)
    existing = discovered.get(existing_key or camera.key)
    if existing is None:
        discovered[camera.key] = camera
        return

    existing.name = existing.name if existing.name and not existing.name.startswith("RTSP Camera") else camera.name
    existing.manufacturer = existing.manufacturer or camera.manufacturer
    existing.model = existing.model or camera.model
    existing.rtsp_server = existing.rtsp_server or camera.rtsp_server
    existing.http_server = existing.http_server or camera.http_server
    existing.device_uuid = existing.device_uuid or camera.device_uuid
    existing.confidence = max(existing.confidence, camera.confidence)
    existing.status = _best_camera_status(existing.status, camera.status)
    existing.status_reason = existing.status_reason or camera.status_reason

    for port in camera.rtsp_ports:
        if port not in existing.rtsp_ports:
            existing.rtsp_ports.append(port)

    for stream_path in camera.rtsp_stream_paths:
        if stream_path not in existing.rtsp_stream_paths:
            existing.rtsp_stream_paths.append(stream_path)

    for xaddr in camera.onvif_xaddrs:
        if xaddr not in existing.onvif_xaddrs:
            existing.onvif_xaddrs.append(xaddr)

    for port in camera.http_ports:
        if port not in existing.http_ports:
            existing.http_ports.append(port)

    for path in camera.http_stream_paths:
        if path not in existing.http_stream_paths:
            existing.http_stream_paths.append(path)

    for scope in camera.scopes:
        if scope not in existing.scopes:
            existing.scopes.append(scope)

    existing.discovery_sources.update(camera.discovery_sources)


def _collapse_local_rtsp_aliases(cameras: list[DiscoveredCamera]) -> list[DiscoveredCamera]:
    local_hosts = _local_interface_hosts()
    chosen_by_path: dict[tuple[int, str], DiscoveredCamera] = {}
    collapsed: list[DiscoveredCamera] = []

    for camera in cameras:
        host = str(camera.host or "").lower()
        is_local_rtsp_path = (
            camera.kind == "network"
            and host in local_hosts
            and bool(camera.rtsp_ports)
            and bool(camera.rtsp_stream_paths)
        )
        if not is_local_rtsp_path:
            collapsed.append(camera)
            continue

        port = int(camera.rtsp_ports[0])
        path = camera.rtsp_stream_paths[0]
        key = (port, path)
        existing = chosen_by_path.get(key)
        if existing is None or _local_host_preference(host) < _local_host_preference(str(existing.host or "")):
            chosen_by_path[key] = camera

    collapsed.extend(chosen_by_path.values())
    return collapsed


def _local_interface_hosts() -> set[str]:
    hosts = {"127.0.0.1", "::1", "localhost"}
    for host, _network in _enumerate_with_psutil() or _enumerate_with_ipconfig():
        hosts.add(str(host).lower())
    return hosts


def _local_host_preference(host: str) -> int:
    normalized = str(host or "").lower()
    if normalized in {"127.0.0.1", "localhost", "::1"}:
        return 0
    if normalized.startswith("192.168.") or normalized.startswith("10."):
        return 1
    return 2


def _find_existing_camera_key(
    discovered: dict[str, DiscoveredCamera],
    camera: DiscoveredCamera,
) -> str | None:
    if camera.key in discovered:
        return camera.key

    for existing_key, existing in discovered.items():
        if camera.device_uuid and existing.device_uuid and camera.device_uuid == existing.device_uuid:
            return existing_key
        if camera.host and existing.host and camera.host.lower() == existing.host.lower():
            camera_paths = set(camera.rtsp_stream_paths)
            existing_paths = set(existing.rtsp_stream_paths)
            if camera_paths or existing_paths:
                if camera_paths and existing_paths and camera_paths.intersection(existing_paths):
                    return existing_key
                continue
            return existing_key
        if camera.kind == "local" and existing.kind == "local" and camera.local_index == existing.local_index:
            return existing_key

    return None


def _best_camera_status(current: str, incoming: str) -> str:
    priority = {
        "online": 90,
        "http_stream_detected": 80,
        "rtsp_stream_detected": 78,
        "rtsp_detected": 76,
        "credentials_required": 70,
        "onvif_detected": 60,
        "http_camera_detected": 55,
        "discovered": 10,
        "offline": 0,
    }
    return incoming if priority.get(incoming, 0) > priority.get(current, 0) else current


def _print_discovered_cameras(cameras: list[DiscoveredCamera]) -> None:
    print("\nCâmeras encontradas:\n")
    for index, camera in enumerate(cameras, start=1):
        if camera.kind == "local":
            print(f"[{index}] {camera.name} | origem=local | índice={camera.local_index} | status={camera.status}")
            continue

        sources = ",".join(sorted(camera.discovery_sources)) or "network"
        ports = ",".join(str(port) for port in sorted(camera.rtsp_ports)) or "desconhecido"
        http_ports = ",".join(str(port) for port in sorted(camera.http_ports)) if camera.http_ports else ""
        identity = " | ".join(
            item
            for item in [
                camera.manufacturer,
                camera.model,
                camera.rtsp_server,
                camera.http_server,
            ]
            if item
        )
        line = (
            f"[{index}] {camera.name} | host={camera.host} | fontes={sources} "
            f"| status={camera.status} | confiança={camera.confidence:.2f} | rtsp={ports}"
        )
        if http_ports:
            line += f" | http={http_ports}"
        if camera.rtsp_stream_paths:
            line += f" | paths={','.join(camera.rtsp_stream_paths[:6])}"
        print(line)
        if identity:
            print(f"    {identity}")
        if camera.onvif_xaddrs:
            print(f"    ONVIF: {camera.onvif_xaddrs[0]}")

    print("")


def _prompt_camera_selection(cameras: list[DiscoveredCamera]) -> DiscoveredCamera | None:
    while True:
        choice = input("Escolha a câmera pelo número (Enter cancela): ").strip()
        if not choice:
            return None
        if not choice.isdigit():
            print("Digite apenas o número da câmera.")
            continue

        index = int(choice)
        if 1 <= index <= len(cameras):
            return cameras[index - 1]

        print("Número inválido.")


def _prompt_yes_no(prompt: str, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{prompt} {suffix}: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes", "s", "sim"}:
            return True
        if answer in {"n", "no", "nao", "não"}:
            return False
        print("Responda com y/s ou n.")
