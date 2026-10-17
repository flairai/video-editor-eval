"""Standard-library HTTP for the providers without an SDK here (fal, GPT Image), so the image models need
nothing installed beyond Pillow."""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
import uuid

from evals.core.interfaces import ProviderError

TRANSIENT = (408, 409, 425, 429, 500, 502, 503, 504)


def http(method: str, url: str, headers: dict, body: bytes | None = None, timeout: float = 600.0) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        payload = err.read().decode("utf-8", "replace")
        try:  # fal echoes the whole request (prompt and data URIs) in its validation errors: drop the echo
            value = json.loads(payload)
            for item in value.get("detail", []) if isinstance(value, dict) and isinstance(value.get("detail"), list) else []:
                if isinstance(item, dict):
                    item.pop("input", None)
            payload = json.dumps(value)
        except (json.JSONDecodeError, AttributeError):
            pass
        raise ProviderError(f"HTTP {err.code}: {payload[:1200]}", err.code in TRANSIENT, err.code) from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
        raise ProviderError(f"network: {err}", transient=True) from None


def post_json(url: str, headers: dict, payload: dict, timeout: float = 600.0) -> dict:
    _, raw = http("POST", url, {**headers, "Content-Type": "application/json"}, json.dumps(payload).encode(), timeout)
    return json.loads(raw) if raw.strip() else {}


def get_json(url: str, headers: dict, timeout: float = 120.0) -> dict:
    _, raw = http("GET", url, headers, None, timeout)
    return json.loads(raw)


def multipart(fields: list[tuple[str, str]], files: list[tuple[str, str, bytes, str]]) -> tuple[bytes, str]:
    """Encode form fields and files (field, filename, bytes, content type) as multipart/form-data."""
    boundary = uuid.uuid4().hex
    out = io.BytesIO()
    for name, value in fields:
        out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for name, filename, data, ctype in files:
        out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                  f'Content-Type: {ctype}\r\n\r\n'.encode())
        out.write(data)
        out.write(b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def fetch_bytes(url: str) -> bytes:
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    _, raw = http("GET", url, {}, None, 300.0)
    return raw


def require_key(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ProviderError(f"{name} is not set (pass --env-file or export it)")
    return value
