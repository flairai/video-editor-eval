"""evals doctor: what this machine has, what each kind of run needs from it, and what is missing.

    uv run evals doctor

It names every key variable a model, subject or judge panel uses and says whether it is set and who needs
it, never its value. Nothing it reports stops the free runs except a FAIL line; a missing key only stops the
runs that need that key, and `evals run` checks again before it starts anything.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys

from evals.core import config
from evals.core.config import ROOT, ConfigError

OK, WARN, FAIL = "ok", "warn", "FAIL"

# Where each key comes from. A provider's own page, so a new user knows what to sign up for.
KEY_SOURCES = {
    "ANTHROPIC_API_KEY": "https://console.anthropic.com/settings/keys",
    "OPENAI_API_KEY": "https://platform.openai.com/api-keys",
    "GEMINI_API_KEY": "https://aistudio.google.com/apikey",
    "FAL_KEY": "https://fal.ai/dashboard/keys",
    "VAST_API_KEY": "the vast.ai console, or `vastai set api-key`",
    "HF_TOKEN": "https://huggingface.co/settings/tokens (write access), or `hf auth login`",
}
PROVIDER_KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY", "google": "GEMINI_API_KEY"}
FFMPEG_INSTALL = "macOS: brew install ffmpeg; Debian or Ubuntu: sudo apt-get install ffmpeg"


def _version(cmd: list[str]) -> str | None:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (res.stdout or res.stderr).strip().splitlines()[0] if res.returncode == 0 and (res.stdout or res.stderr).strip() else None


def tools(probe_network: bool = True) -> list[tuple[str, str]]:
    lines = [(OK, "python %s" % sys.version.split()[0])]
    uv = _version(["uv", "--version"])
    lines.append((OK, uv) if uv else (FAIL, "uv not found: https://docs.astral.sh/uv/ (the scorers and Harbor run through it)"))
    from evals.core import images
    if not images.docker_ok():
        lines.append((WARN, "Docker not answering: needed for Harbor runs and `evals controls --docker` only; the smoke run, "
                            "local grading and the direct executor work without it"))
    else:
        docker = _version(["docker", "version", "--format", "Docker {{.Server.Version}}"]) or "Docker"
        buildx = _version(["docker", "buildx", "version"])
        lines.append((OK, docker) if buildx else (WARN, "%s without buildx: Harbor needs the buildx plugin to allow an agent's hosts" % docker))
        if probe_network:
            from evals.core import harbor
            if harbor.egress_supported():
                lines.append((OK, "Docker can enforce the tasks' no-network rules"))
            else:
                lines.append((WARN, "this Docker engine cannot enforce no-network (its kernel lacks CONFIG_NFT_FIB_INET): Harbor runs "
                                    "use copies of the tasks with the network open, and say so. Image models are unaffected; "
                                    "for enforced runs use a Linux Docker host, OrbStack or Colima"))
    ffmpeg, ffprobe = _version(["ffmpeg", "-version"]), _version(["ffprobe", "-version"])
    lines.append((OK, " ".join(ffmpeg.split()[:3])) if ffmpeg and ffprobe else
                 (FAIL, "%s not found: the reports' video previews and the video judge need ffmpeg and ffprobe (%s)" % (
                     "ffprobe" if ffmpeg else "ffmpeg", FFMPEG_INSTALL)))
    free = shutil.disk_usage(ROOT).free / 1e9
    lines.append((OK, "disk: %.0f GB free" % free) if free >= 15 else
                 (WARN, "disk: %.1f GB free; the media, the images and the grading environment take about 10 GB" % free))
    return lines


def project(tasks) -> list[tuple[str, str]]:
    from evals.core import images, media
    lines = [(OK, "tasks: %d in tasks/eval/ (%s)" % (len(tasks), ", ".join(sorted({t.category for t in tasks}))))]
    if images.docker_ok():
        for name in images.needed(tasks):
            state = images.built(name)
            lines.append((OK, "image %s is current" % images.tag(name)) if state == images.content(name) else
                         (WARN, "image %s %s: a Harbor run builds it, or uv run evals setup" % (
                             images.tag(name), "is not built" if state is None else "is out of date")))
    absent = media.missing(tasks)
    total = len(media.wanted(media.lock()["files"], tasks))
    lines.append((OK, "media: all %d files here (`evals media check` hashes them)" % total) if not absent else
                 (OK, "media: %d of %d files not fetched yet; the commands that need them fetch them" % (len(absent), total)))
    lines.append(dataset())
    if any(t.grader == "camera-eval" for t in tasks):
        lines.append((OK, "camera-eval: the oracles and controls have stored grades, so checking them needs no GPU or key; "
                          "grading a model's grids needs --gpu and the judges' keys"))
    return lines


def dataset() -> tuple[str, str]:
    from evals.core import media
    lk = media.lock()
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return FAIL, "huggingface_hub is not installed: run `uv sync`"
    try:
        HfApi().dataset_info(lk["repo"], revision=lk["revision"])
    except Exception as err:  # not found, offline, rate limited: the hub's message says which
        return WARN, "dataset %s@%s not reachable: %s" % (lk["repo"], lk["revision"][:12], str(err).splitlines()[0][:160])
    return OK, "dataset huggingface.co/datasets/%s at %s: reachable, no login needed" % (lk["repo"], lk["revision"][:12])


def env_file() -> list[tuple[str, str]]:
    path = ROOT / ".env"
    if not path.exists():
        return [(OK, ".env: none (keys come from the environment; `cp .env.example .env` to keep them in a file)")]
    lines = []
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", ".env"], capture_output=True, check=False).returncode == 0
    ignored = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", ".env"], capture_output=True, check=False).returncode == 0
    if tracked or not ignored:
        lines.append((FAIL, ".env is %s: keys in it could be committed. Remove it from git and keep it in .gitignore" % (
            "tracked by git" if tracked else "not ignored by git")))
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        lines.append((WARN, ".env can be read by other users (mode %o): chmod 600 .env" % mode))
    if not lines:
        lines.append((OK, ".env: ignored by git, readable only by you"))
    return lines


def key_users() -> dict[str, dict[str, list[str]]]:
    """Every key variable the configs name, and what needs it: {key: {"subjects": [...], "judge panels": [...], ...}}."""
    from evals.core import registry
    from evals.core.config import load_toml
    users: dict[str, dict[str, list[str]]] = {}

    def need(key, kind, who):
        if key and who not in users.setdefault(key, {}).setdefault(kind, []):
            users[key][kind].append(who)

    for name in registry.subject_names():
        s = registry.subject(name)
        if s.kind == "image-model":
            need(registry.model_entry("image", s.model).get("key_env"), "subjects", name)
        elif s.kind == "container" and s.model:
            need(registry.model_entry("chat", s.model).get("key_env"), "subjects", name)
        elif s.kind == "harbor" and s.model and "/" in s.model:
            need(PROVIDER_KEYS.get(s.model.split("/")[0]), "subjects", name)
    for panel, entry in registry.judges_config().items():
        for model in entry["models"]:
            need(registry.model_entry("chat", model).get("key_env"), "judge panels", panel)
    camera = config.CONFIGS / "camera-eval.toml"
    if camera.exists():
        panel = (load_toml(camera).get("judges") or {}).get("panel")
        for model in registry.panel_entry(panel)["models"] if panel else []:
            need(registry.model_entry("chat", model).get("key_env"), "camera tasks", "camera-eval's judges")
    need("VAST_API_KEY", "camera tasks", "--gpu vast, a rented GPU")
    need("HF_TOKEN", "media", "maintainers only, to publish media with `evals media push`")
    return users


def keys() -> list[tuple[str, str, str]]:
    from pathlib import Path
    out = []
    for key, kinds in sorted(key_users().items()):
        is_set = bool(os.environ.get(key)) or (key == "VAST_API_KEY" and (Path.home() / ".config/vastai/vast_api_key").exists())
        who = "; ".join("%s %s" % (kind, ", ".join(names)) if kind in ("subjects", "judge panels") else ", ".join(names)
                        for kind, names in kinds.items())
        out.append(("set" if is_set else "missing", key, who + ("" if is_set else "  (%s)" % KEY_SOURCES.get(key, "see .env.example"))))
    return out


def doctor(env_files=None, probe_network: bool = True) -> int:
    from evals.core import tasks as T
    config.load_env_files(env_files)
    print("evals doctor: %s\n" % ROOT)
    lines = tools(probe_network)
    try:
        lines += project(T.discover("eval")) + env_file()
        key_lines = keys()
    except ConfigError as err:
        lines.append((FAIL, str(err)))
        key_lines = []
    for status, text in lines:
        print("  %-5s %s" % (status, text))
    if key_lines:
        print("\nkeys (from the environment and .env; names only, never values)")
        for status, key, who in key_lines:
            print("  %-8s %-20s %s" % (status, key, who))
    failed = any(s == FAIL for s, _ in lines)
    print("\n%s" % ("Fix the FAIL lines first." if failed else
                    "The free runs need no key: uv run evals quickstart. A run that needs a missing key stops before it starts "
                    "and names it."))
    return 1 if failed else 0
