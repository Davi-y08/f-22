from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit


def redact_url_credentials(value: object) -> str:
    text = str(value)
    if "://" not in text:
        return text

    try:
        parsed = urlsplit(text)
    except Exception:
        return text

    if not parsed.username and not parsed.password:
        return text

    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        parsed_port = parsed.port
    except ValueError:
        parsed_port = None
    port = f":{parsed_port}" if parsed_port is not None else ""
    netloc = f"***:***@{host}{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
