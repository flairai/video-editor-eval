"""Where VGGT-Omega runs. Pick a backend with --gpu:

    vast                          rent the cheapest fitting GPU on vast.ai, destroy it at the end (vast.py)
    ssh://[user@]host[:port]      any VM or machine you can SSH into, on any cloud (ssh.py)
    local                         this machine (local.py)
    http(s)://...                 a server already running, token in CAMERA_EVAL_GPU_TOKEN (url.py)
    cache                         no GPU: only the stored grades of the oracles and controls (cache.py)

A new platform is a Backend subclass (base.py) with up() returning an Endpoint and down() releasing
what up() took; RemoteHost does the SSH setup, server start and tunnel for anything with SSH.
"""

from __future__ import annotations

from pathlib import Path

from camera_eval.gpu.base import Backend, Endpoint


def make_backend(spec: str, cfg: dict, state_dir: Path, log, ssh_key: str | None = None, gpu_name: str | None = None,
                 max_dph: float | None = None) -> Backend:
    if spec == "vast" or spec.startswith("vast:"):
        from camera_eval.gpu.vast import VastBackend
        return VastBackend(cfg, state_dir, log, gpu_name=gpu_name or (spec.split(":", 1)[1] if ":" in spec else None), max_dph=max_dph)
    if spec.startswith("ssh://"):
        from camera_eval.gpu.ssh import SSHBackend
        return SSHBackend(cfg, state_dir, log, spec, ssh_key)
    if spec == "local":
        from camera_eval.gpu.local import LocalBackend
        return LocalBackend(cfg, state_dir, log)
    if spec.startswith(("http://", "https://")):
        from camera_eval.gpu.url import URLBackend
        return URLBackend(cfg, state_dir, log, spec)
    if spec == "cache":
        from camera_eval.gpu.cache import CacheBackend
        return CacheBackend(cfg, state_dir, log)
    raise ValueError("unknown --gpu %r: use vast, ssh://user@host[:port], local, an http(s) URL, or cache" % spec)


__all__ = ["Backend", "Endpoint", "make_backend"]
