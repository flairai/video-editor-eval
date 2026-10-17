"""What every GPU platform provides, and the SSH plumbing the SSH-based ones share.

A backend brings up the reconstruction server (camera_eval/reconstruct/server.py) somewhere and
returns an Endpoint this machine can call: a URL and the bearer token the server checks. Remote
servers listen on their own loopback only and are reached through an SSH tunnel, so neither images
nor the token cross the network unencrypted and no port is opened on the GPU host. A backend never
sends a credential of yours to the host: the token is generated per start and the host needs no key.
"""

from __future__ import annotations

import json
import os
import secrets
import shlex
import socket
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from camera_eval.reconstruct.client import ReconClient, ReconError

HERE = Path(__file__).resolve().parent
RECON_DIR = HERE.parent / "reconstruct"


@dataclass
class Endpoint:
    url: str
    token: str
    info: dict = field(default_factory=dict)


class Backend(ABC):
    """One GPU platform. up() is safe to call again: it reuses what is already running."""

    kind = "?"

    def __init__(self, cfg: dict, state_dir: Path, log: Callable[..., None]):
        self.cfg, self.gpu_cfg, self.state_dir, self.log = cfg, cfg["gpu"], state_dir, log
        self.endpoint: Endpoint | None = None

    @abstractmethod
    def up(self) -> Endpoint:
        """Bring the server up (renting or starting what is needed) and return how to reach it."""

    @abstractmethod
    def down(self) -> None:
        """Release everything up() started or rented. Safe to call more than once."""

    def detach(self) -> None:
        """Let go without releasing: close local connections and leave the GPU for the next up() to reuse.
        Backends that cannot be reattached release instead."""
        self.down()

    def ensure(self) -> None:
        """Called by the service between requests: repair what can break while idle, such as a tunnel."""

    def describe(self) -> dict:
        return {"kind": self.kind, **(self.endpoint.info if self.endpoint else {})}

    def cost_estimate_usd(self) -> float:
        return 0.0

    # ---------------------------------------------------------------- shared helpers

    def state_path(self) -> Path:
        return self.state_dir / ("gpu-%s.json" % self.kind)

    def read_state(self) -> dict:
        p = self.state_path()
        return json.loads(p.read_text()) if p.exists() else {}

    def write_state(self, state: dict) -> None:
        p = self.state_path()
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(p)

    def clear_state(self) -> None:
        self.state_path().unlink(missing_ok=True)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def new_token() -> str:
    return secrets.token_urlsafe(32)


def wait_healthy(url: str, token: str, timeout: float, log, what: str) -> dict:
    client = ReconClient(url, token)
    t0, last = time.time(), None
    while time.time() - t0 < timeout:
        try:
            info = client.health()
            log("gpu_ready", where=what, seconds=round(time.time() - t0, 1),
                **{k: info.get(k) for k in ("model", "gpu", "torch", "device", "load_seconds")})
            return info
        except ReconError as err:
            last = str(err)
        time.sleep(3)
    raise TimeoutError("reconstruction server at %s not healthy after %ds: %s" % (what, timeout, last))


class RemoteHost:
    """A machine reached over SSH: run commands, copy files, set up and start the server, tunnel to it."""

    def __init__(self, host: str, port: int | None, user: str | None, key: Path | None, known_hosts: Path | None,
                 home: str, log, accept_new: bool = True):
        self.host, self.port, self.user, self.key, self.known_hosts = host, port, user, key, known_hosts
        self.home = home.replace("~", "$HOME", 1) if home.startswith("~") else home
        self.log = log
        self.accept_new = accept_new
        self.tunnel_proc: subprocess.Popen | None = None
        self.local_port: int | None = None

    def target(self) -> str:
        return "%s@%s" % (self.user, self.host) if self.user else self.host

    def ssh_args(self) -> list[str]:
        args = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4"]
        if self.accept_new:
            args += ["-o", "StrictHostKeyChecking=accept-new"]
        if self.known_hosts:
            args += ["-o", "UserKnownHostsFile=%s" % self.known_hosts]
        if self.key:
            args += ["-i", str(self.key), "-o", "IdentitiesOnly=yes"]
        if self.port:
            args += ["-p", str(self.port)]
        return args

    def run(self, command: str, timeout: float = 600, check: bool = True, input: bytes | str | None = None) -> subprocess.CompletedProcess:
        p = subprocess.run(["ssh", *self.ssh_args(), self.target(), command], capture_output=True, timeout=timeout,
                           input=input.encode() if isinstance(input, str) else input)
        if check and p.returncode != 0:
            raise RuntimeError("ssh %s %r failed (%d): %s %s" % (self.target(), command[:100], p.returncode,
                                                                p.stdout.decode(errors="replace")[-1200:],
                                                                p.stderr.decode(errors="replace")[-1200:]))
        return p

    def wait_ssh(self, timeout: float) -> None:
        t0, err = time.time(), ""
        while time.time() - t0 < timeout:
            p = self.run("echo ok", timeout=60, check=False)
            if p.returncode == 0 and b"ok" in p.stdout:
                return
            err = p.stderr.decode(errors="replace")[-300:]
            time.sleep(8)
        raise TimeoutError("SSH to %s not reachable after %ds: %s" % (self.target(), timeout, err))

    def put(self, data: bytes, remote_path: str, mode: str = "644") -> None:
        self.run("umask 077; mkdir -p $(dirname %s) && cat > %s && chmod %s %s" % (remote_path, remote_path, mode, remote_path),
                 input=data, timeout=300)

    def setup(self, gpu_cfg: dict, extra_env: dict | None = None) -> None:
        """Copy the server over and run bootstrap.sh (installs what is missing, checks the checkpoint)."""
        t0 = time.time()
        self.put((RECON_DIR / "server.py").read_bytes(), "%s/server.py" % self.home)
        env = {"CE_HOME": self.home, "CE_VGGT_SHA": gpu_cfg["vggt_commit"], "CE_CKPT_REPO": gpu_cfg["checkpoint_repo"],
               "CE_CKPT_FILE": gpu_cfg["checkpoint_file"], "CE_CKPT_SHA256": gpu_cfg["checkpoint_sha256"], **(extra_env or {})}
        exports = " ".join("%s=%s" % (k, v if k == "CE_HOME" else shlex.quote(v)) for k, v in env.items())
        p = self.run("%s bash -s" % exports, input=(RECON_DIR / "bootstrap.sh").read_bytes(), timeout=3600, check=False)
        out = p.stdout.decode(errors="replace").strip()
        if p.returncode != 0:
            raise RuntimeError("bootstrap on %s failed: %s %s" % (self.target(), out[-2000:], p.stderr.decode(errors="replace")[-1500:]))
        self.log("gpu_setup", host=self.host, seconds=round(time.time() - t0, 1), detail=out.splitlines()[-1:] if out else [])

    def start_server(self, gpu_cfg: dict, token: str) -> None:
        self.put(token.encode(), "%s/token" % self.home, mode="600")
        h = self.home
        self.run("cd %s && if [ -f server.pid ] && kill -0 $(cat server.pid) 2>/dev/null; then kill $(cat server.pid); sleep 2; fi; "
                 "nohup venv/bin/python server.py --ckpt ckpt/%s --port %d --token-file token >> logs/server.log 2>&1 < /dev/null & "
                 "echo $! > server.pid" % (h, gpu_cfg["checkpoint_file"], gpu_cfg["remote_port"]), timeout=60)
        self.log("gpu_server_started", host=self.host, port=gpu_cfg["remote_port"])

    def stop_server(self) -> None:
        self.run("cd %s && if [ -f server.pid ]; then kill $(cat server.pid) 2>/dev/null || true; rm -f server.pid; fi; rm -f token" % self.home,
                 timeout=60, check=False)

    def server_log_tail(self, n: int = 30) -> str:
        p = self.run("tail -n %d %s/logs/server.log" % (n, self.home), timeout=60, check=False)
        return p.stdout.decode(errors="replace")

    def open_tunnel(self, remote_port: int) -> str:
        """Forward a free local port to the server's port on the host's loopback; returns the local URL."""
        self.close_tunnel()
        self.local_port = free_port()
        self.tunnel_proc = subprocess.Popen(
            ["ssh", *self.ssh_args(), "-N", "-o", "ExitOnForwardFailure=yes",
             "-L", "127.0.0.1:%d:127.0.0.1:%d" % (self.local_port, remote_port), self.target()],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
        time.sleep(2)
        if self.tunnel_proc.poll() is not None:
            raise RuntimeError("SSH tunnel to %s exited: %s" % (self.target(), self.tunnel_proc.stderr.read().decode(errors="replace")[-500:]))
        return "http://127.0.0.1:%d" % self.local_port

    def tunnel_alive(self) -> bool:
        return self.tunnel_proc is not None and self.tunnel_proc.poll() is None

    def close_tunnel(self) -> None:
        if self.tunnel_proc and self.tunnel_proc.poll() is None:
            self.tunnel_proc.terminate()
            try:
                self.tunnel_proc.wait(10)
            except subprocess.TimeoutExpired:
                self.tunnel_proc.kill()
        self.tunnel_proc = None


def ssh_keygen(path: Path, comment: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    if not path.exists():
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", comment, "-f", str(path)], check=True)
    return path.with_suffix(".pub").read_text().strip()
