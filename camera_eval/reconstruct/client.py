"""This machine's side of the reconstruction server."""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request

import numpy as np


class ReconError(RuntimeError):
    """The reconstruction server failed or could not be reached; transient ones are worth retrying."""

    def __init__(self, message: str, transient: bool):
        super().__init__(message)
        self.transient = transient


class ReconClient:
    def __init__(self, url: str, token: str, timeout: float = 600.0, retries: int = 3):
        self.url, self.token, self.timeout, self.retries = url.rstrip("/"), token, timeout, retries

    def _request(self, path: str, body: bytes | None = None, timeout: float | None = None) -> bytes:
        req = urllib.request.Request(self.url + path, data=body, method="POST" if body is not None else "GET",
                                     headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return r.read()
        except urllib.error.HTTPError as err:
            text = err.read()[:500].decode(errors="replace")
            raise ReconError("HTTP %d from the reconstruction server: %s" % (err.code, text), err.code >= 500) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as err:
            raise ReconError("reconstruction server unreachable at %s: %s" % (self.url, err), True) from None

    def health(self, timeout: float = 10.0) -> dict:
        return json.loads(self._request("/health", timeout=timeout))

    def reconstruct(self, cells: list[bytes], resolution: int = 512) -> dict[str, np.ndarray]:
        body = json.dumps({"images": [base64.b64encode(c).decode() for c in cells], "resolution": resolution}).encode()
        for attempt in range(self.retries + 1):
            try:
                raw = self._request("/reconstruct", body)
                with np.load(io.BytesIO(raw)) as z:
                    return {k: z[k] for k in z.files}
            except ReconError as err:
                if not err.transient or attempt == self.retries:
                    raise
                time.sleep(3 * (attempt + 1))
        raise AssertionError("unreachable")
