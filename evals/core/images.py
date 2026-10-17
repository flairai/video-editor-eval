"""The images task environments build FROM: images/<name>/, tagged as configs/evals.toml [images] says.

    from evals.core import images
    images.needed(tasks)       # ["grader"]: the ones these tasks' environment/Dockerfiles are FROM
    images.build("grader")     # builds it unless the image already carries this folder's content hash

Each image is labelled with a hash of its folder (and of the image it is FROM, when that is one of these), so
a build is skipped when nothing changed and redone when anything did.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from functools import lru_cache

from evals.core.config import ROOT, ConfigError, settings, which

LABEL = "org.evals.content"


def configured() -> dict[str, str]:
    """name -> tag."""
    return dict(settings().get("images") or {})


def tag(name: str) -> str:
    tags = configured()
    if name not in tags:
        raise ConfigError("no image %r in configs/evals.toml [images]; have: %s" % (name, ", ".join(tags)))
    return tags[name]


def base(name: str) -> str | None:
    """The image of ours that images/<name>/Dockerfile is FROM, if any."""
    m = re.search(r"(?mi)^FROM\s+(\S+)", (ROOT / "images" / name / "Dockerfile").read_text())
    by_tag = {t: n for n, t in configured().items()}
    return by_tag.get(m.group(1)) if m else None


@lru_cache(maxsize=None)
def content(name: str) -> str:
    """A hash of images/<name>/ and, through its FROM line, of the image it builds on."""
    digest = hashlib.sha256()
    folder = ROOT / "images" / name
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(folder)).encode() + b"\0" + path.read_bytes() + b"\0")
    parent = base(name)
    if parent:
        digest.update(content(parent).encode())
    return digest.hexdigest()[:16]


def needed(tasks) -> list[str]:
    """The images of ours these tasks' environments are FROM, bases first."""
    by_tag = {t: n for n, t in configured().items()}
    out: list[str] = []

    def add(name):
        parent = base(name)
        if parent:
            add(parent)
        if name not in out:
            out.append(name)

    for t in tasks:
        dockerfile = t.path / "environment" / "Dockerfile"
        m = re.search(r"(?mi)^FROM\s+(\S+)", dockerfile.read_text()) if dockerfile.exists() else None
        if m and m.group(1) in by_tag:
            add(by_tag[m.group(1)])
    return out


def docker_ok() -> bool:
    return bool(which("docker")) and subprocess.run(["docker", "info"], capture_output=True, check=False).returncode == 0


def built(name: str) -> str | None:
    """The content hash the image was built from, "" for an image built without one, None when it is absent."""
    res = subprocess.run(["docker", "image", "inspect", "-f", '{{index .Config.Labels "%s"}}' % LABEL, tag(name)],
                         capture_output=True, text=True, check=False)
    if res.returncode != 0:
        return None
    value = res.stdout.strip()
    return "" if value in ("", "<no value>") else value


def current(name: str) -> bool:
    return built(name) == content(name)


def build(name: str, say=print) -> bool:
    """Build images/<name>/ (its base first) unless the image is current; returns whether it built anything."""
    did = False
    parent = base(name)
    if parent:
        did = build(parent, say)
    if current(name):
        return did
    say("building %s from images/%s/ (a few minutes the first time)" % (tag(name), name))
    res = subprocess.run(["docker", "build", "-t", tag(name), "--label", "%s=%s" % (LABEL, content(name)), str(ROOT / "images" / name)],
                         check=False)
    if res.returncode != 0:
        raise ConfigError("docker build of images/%s/ failed (exit %d); its output is above" % (name, res.returncode))
    return True
