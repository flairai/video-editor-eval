"""Any GPU machine you can SSH into: a cloud VM (AWS, GCP, Azure, Lambda, CoreWeave, RunPod...), a
workstation, a cluster node. The VM is yours: it is never created or deleted here.

    camera-eval up --gpu ssh://ubuntu@203.0.113.7            # user@host, port 22, your ssh-agent or ~/.ssh/config
    camera-eval up --gpu ssh://ubuntu@203.0.113.7:2222 --ssh-key ~/.ssh/gpu_vm
    camera-eval up --gpu ssh://my-gpu-box                    # a Host alias from ~/.ssh/config

up() runs bootstrap.sh (a venv under ~/camera-eval, the pinned vggt-omega, the checkpoint checked
against its sha256; torch is installed only when no CUDA torch >= 2.6 is there), starts the server
on the VM's loopback and tunnels to it. down() stops the server and leaves the VM and its files, so
the next up() starts in seconds. Host keys go through your known_hosts as usual.
"""

from __future__ import annotations

import urllib.parse
from pathlib import Path

from camera_eval.gpu.base import Backend, Endpoint, RemoteHost, new_token, wait_healthy


def parse(spec: str) -> tuple[str, int | None, str | None]:
    u = urllib.parse.urlsplit(spec)
    if u.scheme != "ssh" or not u.hostname:
        raise ValueError("expected ssh://[user@]host[:port], got %r" % spec)
    return u.hostname, u.port, u.username


class SSHBackend(Backend):
    kind = "ssh"

    def __init__(self, cfg, state_dir, log, spec: str, key: str | None = None):
        super().__init__(cfg, state_dir, log)
        host, port, user = parse(spec)
        self.host = RemoteHost(host, port, user, Path(key).expanduser() if key else None, None, self.gpu_cfg["remote_home"], log)
        self.spec = spec

    def up(self) -> Endpoint:
        if self.endpoint and self.host.tunnel_alive():
            return self.endpoint
        self.host.wait_ssh(120)
        self.host.setup(self.gpu_cfg)
        token = new_token()
        self.host.start_server(self.gpu_cfg, token)
        url = self.host.open_tunnel(self.gpu_cfg["remote_port"])
        try:
            health = wait_healthy(url, token, 600, self.log, self.host.target())
        except TimeoutError:
            tail = self.host.server_log_tail()
            self.down()
            raise RuntimeError("server did not come up on %s: %s" % (self.host.target(), tail)) from None
        self.write_state({"spec": self.spec})
        self.endpoint = Endpoint(url, token, {"kind": "ssh", "host": self.host.target(), "gpu": health.get("gpu")})
        return self.endpoint

    def ensure(self) -> None:
        if self.endpoint and not self.host.tunnel_alive():
            self.log("gpu_tunnel_reopened", host=self.host.target())
            self.endpoint.url = self.host.open_tunnel(self.gpu_cfg["remote_port"])

    def detach(self) -> None:
        self.host.close_tunnel()
        self.endpoint = None

    def down(self) -> None:
        self.host.close_tunnel()
        try:
            self.host.stop_server()
            self.log("gpu_server_stopped", host=self.host.target())
        except Exception as err:  # noqa: BLE001 (an unreachable VM has nothing left to stop)
            self.log("gpu_stop_error", host=self.host.target(), error=str(err)[:300])
        self.clear_state()
        self.endpoint = None
