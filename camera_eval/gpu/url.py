"""A reconstruction server already running somewhere this machine can reach, such as one behind your
own load balancer or a tunnel you manage: --gpu https://recon.example.com, with its token in
CAMERA_EVAL_GPU_TOKEN. Nothing is started or stopped.
"""

from __future__ import annotations

import os

from camera_eval.gpu.base import Backend, Endpoint, wait_healthy


class URLBackend(Backend):
    kind = "url"

    def __init__(self, cfg, state_dir, log, url: str):
        super().__init__(cfg, state_dir, log)
        self.url = url.rstrip("/")

    def up(self) -> Endpoint:
        token = os.environ.get("CAMERA_EVAL_GPU_TOKEN", "")
        if not token:
            raise RuntimeError("set CAMERA_EVAL_GPU_TOKEN to the reconstruction server's token")
        health = wait_healthy(self.url, token, 60, self.log, self.url)
        self.endpoint = Endpoint(self.url, token, {"kind": "url", "url": self.url, "gpu": health.get("gpu")})
        return self.endpoint

    def down(self) -> None:
        self.endpoint = None
