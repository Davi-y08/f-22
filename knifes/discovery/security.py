from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from urllib.parse import urlparse

from utils.redaction import redact_url_credentials


STREAM_SCHEMES = {"rtsp", "rtsps", "http", "https"}
BLOCKED_HOSTNAMES = {"localhost", "localhost.localdomain"}
BLOCKED_IPS = {ipaddress.ip_address("169.254.169.254")}


@dataclass(frozen=True, slots=True)
class StreamUrlValidation:
    valid: bool
    reason: str | None = None
    host: str | None = None


def validate_stream_url(
    value: str,
    *,
    allow_public_hosts: bool = True,
    allow_link_local: bool = True,
    allow_loopback: bool = False,
) -> StreamUrlValidation:
    raw = str(value or "").strip()
    if not raw:
        return StreamUrlValidation(False, "url_empty")
    if any(ord(char) < 32 for char in raw):
        return StreamUrlValidation(False, "url_control_characters")

    parsed = urlparse(raw)
    scheme = parsed.scheme.lower()
    if scheme not in STREAM_SCHEMES:
        return StreamUrlValidation(False, "url_scheme_not_supported")
    if not parsed.hostname:
        return StreamUrlValidation(False, "url_missing_host")

    host = parsed.hostname.strip().lower()
    if not allow_loopback and (host in BLOCKED_HOSTNAMES or host.endswith(".localhost")):
        return StreamUrlValidation(False, "url_loopback_host", host)

    try:
        ip_addr = ipaddress.ip_address(host)
    except ValueError:
        return StreamUrlValidation(True, host=host) if allow_public_hosts else StreamUrlValidation(
            False,
            "url_hostname_not_allowed",
            host,
        )

    if ip_addr.is_loopback and not allow_loopback:
        return StreamUrlValidation(False, "url_loopback_ip", host)
    if ip_addr.is_multicast:
        return StreamUrlValidation(False, "url_multicast_ip", host)
    if ip_addr.is_unspecified:
        return StreamUrlValidation(False, "url_unspecified_ip", host)
    if ip_addr.is_reserved:
        return StreamUrlValidation(False, "url_reserved_ip", host)
    if ip_addr in BLOCKED_IPS:
        return StreamUrlValidation(False, "url_blocked_metadata_ip", host)
    if ip_addr.is_link_local and not allow_link_local:
        return StreamUrlValidation(False, "url_link_local_ip", host)
    if not allow_public_hosts and not (ip_addr.is_private or ip_addr.is_link_local):
        return StreamUrlValidation(False, "url_public_ip_not_allowed", host)

    return StreamUrlValidation(True, host=host)


def redacted_urls(urls: list[str]) -> list[str]:
    return [redact_url_credentials(url) for url in urls]
