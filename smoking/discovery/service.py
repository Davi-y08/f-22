from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import getpass
import ipaddress
import re
import socket
import subprocess
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse, urlunparse
import xml.etree.ElementTree as ET

import cv2

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
GENERIC_RTSP_PATHS = (
    "",
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
    rtsp_server: str | None = None
    http_ports: list[int] = field(default_factory=list)
    http_stream_paths: list[str] = field(default_factory=list)
    http_server: str | None = None
    onvif_xaddrs: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)
    discovery_sources: set[str] = field(default_factory=set)


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


def discover_cameras(
    logger: Any,
    include_local_webcams: bool = True,
    rtsp_scan_timeout: float = 0.35,
    http_scan_timeout: float = 0.35,
    onvif_timeout: float = 2.5,
) -> list[DiscoveredCamera]:
    discovered: dict[str, DiscoveredCamera] = {}

    for camera in _discover_onvif_devices(timeout=onvif_timeout, logger=logger):
        _merge_discovered_camera(discovered, camera)

    for camera in _discover_rtsp_devices(timeout=rtsp_scan_timeout, logger=logger):
        _merge_discovered_camera(discovered, camera)

    for camera in _discover_http_devices(timeout=http_scan_timeout, logger=logger):
        _merge_discovered_camera(discovered, camera)

    results = list(discovered.values())
    results.sort(key=lambda item: (item.kind != "network", item.host or "", item.local_index or -1))

    if include_local_webcams:
        results.extend(_discover_local_webcams(logger))

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

    tested_urls: list[str] = []
    rtsp_candidates = build_candidate_rtsp_urls(camera, username=username, password=password)
    http_candidates = build_candidate_http_urls(camera, username=username, password=password)
    candidates = _prioritize_stream_candidates(camera, rtsp_candidates, http_candidates)

    if validate_stream:
        for url in candidates:
            tested_urls.append(url)
            if test_stream_source(url, backend_preference="auto", timeout_seconds=timeout_seconds):
                return SourceResolutionResult(
                    source=url,
                    validated=True,
                    tested_urls=tested_urls,
                )
    elif candidates:
        return SourceResolutionResult(
            source=candidates[0],
            validated=False,
            tested_urls=candidates[:1],
        )

    manual_candidates = _expand_manual_stream_candidates(manual_rtsp_url) if manual_rtsp_url else []
    if manual_candidates:
        for candidate in manual_candidates:
            if candidate not in tested_urls:
                tested_urls.append(candidate)

        if not validate_stream:
            return SourceResolutionResult(
                source=manual_candidates[0],
                validated=False,
                tested_urls=tested_urls,
            )

        for manual_candidate in manual_candidates:
            if test_stream_source(manual_candidate, backend_preference="auto", timeout_seconds=timeout_seconds):
                return SourceResolutionResult(
                    source=manual_candidate,
                    validated=True,
                    tested_urls=tested_urls,
                )

    return SourceResolutionResult(
        source=None,
        validated=False,
        error="Nenhuma URL de stream (RTSP/HTTP) foi validada para esta câmera.",
        tested_urls=tested_urls,
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
        return f"{camera.name} | local | index={camera.local_index}"

    sources = ",".join(sorted(camera.discovery_sources)) or "network"
    details: list[str] = [f"host={camera.host}", f"src={sources}"]
    if camera.rtsp_ports:
        rtsp_ports = ",".join(str(port) for port in sorted(camera.rtsp_ports))
        details.append(f"rtsp={rtsp_ports}")
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


def _discover_rtsp_devices(timeout: float, logger: Any) -> list[DiscoveredCamera]:
    hosts_to_probe = _enumerate_hosts_for_network_scan(logger)

    if not hosts_to_probe:
        return []

    cameras: dict[str, DiscoveredCamera] = {}
    futures = []
    with ThreadPoolExecutor(max_workers=64) as executor:
        for host in hosts_to_probe:
            for port in COMMON_RTSP_PORTS:
                futures.append(executor.submit(_probe_rtsp_endpoint, host, port, timeout))

        for future in as_completed(futures):
            host, port, is_rtsp, server = future.result()
            if not is_rtsp:
                continue

            key = f"network:{host}"
            camera = cameras.setdefault(
                key,
                DiscoveredCamera(
                    key=key,
                    kind="network",
                    name=f"RTSP Camera {host}",
                    host=host,
                ),
            )
            if port not in camera.rtsp_ports:
                camera.rtsp_ports.append(port)
            camera.rtsp_server = camera.rtsp_server or server
            camera.discovery_sources.add("rtsp-scan")

    if cameras:
        logger.info(
            "rtsp_discovery_completed",
            extra={"devices_found": len(cameras), "hosts_scanned": len(hosts_to_probe)},
        )

    return list(cameras.values())


def _discover_http_devices(timeout: float, logger: Any) -> list[DiscoveredCamera]:
    hosts_to_probe = _enumerate_hosts_for_network_scan(logger)
    if not hosts_to_probe:
        return []

    cameras: dict[str, DiscoveredCamera] = {}
    futures = []
    with ThreadPoolExecutor(max_workers=64) as executor:
        for host in hosts_to_probe:
            for port in COMMON_HTTP_CAMERA_PORTS:
                futures.append(executor.submit(_probe_http_camera_endpoint, host, port, timeout))

        for future in as_completed(futures):
            host, port, is_camera, server, stream_paths, name = future.result()
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
            camera.discovery_sources.add("http-scan")

    if cameras:
        logger.info(
            "http_camera_discovery_completed",
            extra={"devices_found": len(cameras), "hosts_scanned": len(hosts_to_probe)},
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
    candidates = _prioritize_stream_candidates(camera, rtsp_candidates, http_candidates)

    for url in candidates:
        if test_stream_source(url, backend_preference="auto", timeout_seconds=4.0):
            print(f"Stream validado com sucesso: {url}")
            return url

    print("Nenhuma URL de stream comum respondeu com sucesso.")
    manual = input("Cole a URL manualmente (RTSP/HTTP) ou pressione Enter para cancelar: ").strip()
    if not manual:
        print("Configuração cancelada.")
        return None

    manual_candidates = _expand_manual_stream_candidates(manual)
    for manual_candidate in manual_candidates:
        if test_stream_source(manual_candidate, backend_preference="auto", timeout_seconds=4.0):
            print(f"URL manual validada com sucesso: {manual_candidate}")
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
    paths = list(dict.fromkeys([*vendor_paths, *GENERIC_RTSP_PATHS]))
    urls: list[str] = []
    for port in ports:
        for path in paths:
            suffix = f"/{path.lstrip('/')}" if path else "/"
            urls.append(f"rtsp://{auth}{host}:{port}{suffix}")

    return urls


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
    for host, network in _enumerate_local_networks(logger):
        hosts_to_probe.add(host)
        for candidate in network.hosts():
            hosts_to_probe.add(str(candidate))
    return hosts_to_probe


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


def _probe_rtsp_endpoint(host: str, port: int, timeout: float) -> tuple[str, int, bool, str | None]:
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
        return host, port, False, None

    if "RTSP/" not in response.upper():
        return host, port, False, None

    server = None
    for line in response.splitlines():
        if line.lower().startswith("server:"):
            server = line.split(":", 1)[1].strip()
            break

    return host, port, True, server


def _probe_http_camera_endpoint(
    host: str,
    port: int,
    timeout: float,
) -> tuple[str, int, bool, str | None, list[str], str | None]:
    stream_paths: list[str] = []
    server: str | None = None
    name: str | None = None

    for path in GENERIC_HTTP_STREAM_PATHS:
        status_code, headers, body = _probe_http_endpoint(host, port, path, timeout)
        if server is None:
            server = headers.get("server")

        content_type = headers.get("content-type", "").lower()
        is_stream = ("multipart/x-mixed-replace" in content_type) or ("image/jpeg" in content_type)
        if status_code in {200, 401} and is_stream:
            normalized_path = "/" + path.lstrip("/")
            if normalized_path not in stream_paths:
                stream_paths.append(normalized_path)

        if _looks_like_ip_webcam_body(body):
            name = f"IP Webcam {host}"

    status_code, headers, body = _probe_http_endpoint(host, port, "/", timeout)
    if server is None:
        server = headers.get("server")
    if _looks_like_ip_webcam_body(body):
        name = f"IP Webcam {host}"

    is_camera = bool(stream_paths) or bool(name)
    if is_camera and not stream_paths:
        stream_paths.append("/video")

    return host, port, is_camera, server, stream_paths, name


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

    host = fallback_host
    if xaddrs:
        parsed_host = _extract_host_from_url(xaddrs[0])
        host = parsed_host or fallback_host

    if not host:
        return None

    name = _extract_scope_value(scopes, "name") or f"ONVIF Camera {host}"
    model = _extract_scope_value(scopes, "hardware")
    location = _extract_scope_value(scopes, "location")

    camera = DiscoveredCamera(
        key=f"network:{host}",
        kind="network",
        name=name,
        host=host,
        model=model,
        manufacturer=location,
        onvif_xaddrs=xaddrs,
        scopes=scopes,
        discovery_sources={"onvif"},
    )
    return camera


def _extract_scope_value(scopes: list[str], key: str) -> str | None:
    prefix = f"onvif://www.onvif.org/{key}/"
    for scope in scopes:
        if scope.startswith(prefix):
            return scope[len(prefix) :].replace("%20", " ")
    return None


def _extract_host_from_url(url: str) -> str | None:
    match = re.search(r"^[a-z]+://\[?([^\]/:]+)\]?(?::\d+)?", url, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _merge_discovered_camera(
    discovered: dict[str, DiscoveredCamera],
    camera: DiscoveredCamera,
) -> None:
    existing = discovered.get(camera.key)
    if existing is None:
        discovered[camera.key] = camera
        return

    existing.name = existing.name if existing.name and not existing.name.startswith("RTSP Camera") else camera.name
    existing.manufacturer = existing.manufacturer or camera.manufacturer
    existing.model = existing.model or camera.model
    existing.rtsp_server = existing.rtsp_server or camera.rtsp_server
    existing.http_server = existing.http_server or camera.http_server

    for port in camera.rtsp_ports:
        if port not in existing.rtsp_ports:
            existing.rtsp_ports.append(port)

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


def _print_discovered_cameras(cameras: list[DiscoveredCamera]) -> None:
    print("\nCâmeras encontradas:\n")
    for index, camera in enumerate(cameras, start=1):
        if camera.kind == "local":
            print(f"[{index}] {camera.name} | origem=local | índice={camera.local_index}")
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
        line = f"[{index}] {camera.name} | host={camera.host} | fontes={sources} | rtsp={ports}"
        if http_ports:
            line += f" | http={http_ports}"
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
