"""This machine's own GPU: bootstrap.sh into ~/.cache/camera-eval/local, then the server as a child
process. Without CUDA it falls back to Apple's MPS or the CPU, which is slow (minutes per grid) and
meant for checking the pipeline, not for benchmarks.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from camera_eval.gpu.base import RECON_DIR, Backend, Endpoint, free_port, new_token, wait_healthy


class LocalBackend(Backend):
    kind = "local"

    def __init__(self, cfg, state_dir, log, home: str | None = None):
        super().__init__(cfg, state_dir, log)
        self.home = Path(home or "~/.cache/camera-eval/local").expanduser()
        self.proc: subprocess.Popen | None = None

    def up(self) -> Endpoint:
        if self.endpoint and self.proc and self.proc.poll() is None:
            return self.endpoint
        g = self.gpu_cfg
        env = {**os.environ, "CE_HOME": str(self.home), "CE_VGGT_SHA": g["vggt_commit"], "CE_CKPT_REPO": g["checkpoint_repo"],
               "CE_CKPT_FILE": g["checkpoint_file"], "CE_CKPT_SHA256": g["checkpoint_sha256"], "CE_ALLOW_NO_CUDA": "1"}
        t0 = time.time()
        p = subprocess.run(["bash", str(RECON_DIR / "bootstrap.sh")], env=env, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError("bootstrap failed: %s %s" % (p.stdout[-1500:], p.stderr[-800:]))
        self.log("gpu_setup", host="localhost", seconds=round(time.time() - t0, 1), detail=p.stdout.strip().splitlines()[-1:])
        token, port = new_token(), free_port()
        token_file = self.home / "token"
        token_file.write_text(token)
        token_file.chmod(0o600)
        logf = open(self.home / "logs" / "server.log", "ab")
        self.proc = subprocess.Popen([str(self.home / "venv" / "bin" / "python"), str(RECON_DIR / "server.py"),
                                      "--ckpt", str(self.home / "ckpt" / g["checkpoint_file"]), "--port", str(port),
                                      "--token-file", str(token_file)], stdout=logf, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        url = "http://127.0.0.1:%d" % port
        health = wait_healthy(url, token, 900, self.log, "localhost")
        self.endpoint = Endpoint(url, token, {"kind": "local", "gpu": health.get("gpu"), "device": health.get("device")})
        return self.endpoint

    def down(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc, self.endpoint = None, None
        (self.home / "token").unlink(missing_ok=True)
