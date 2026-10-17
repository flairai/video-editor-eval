"""The media git does not hold (videos, audio, images, arrays, and the stored grades of the camera tasks'
controls) lives in one public Hugging Face dataset, at the same paths as in the repo, pinned by
configs/media.lock.json: the dataset, the dataset commit, and every file's size and sha256.

Nothing is downloaded by hand: every command that needs a task's files (run, controls, grade, judge, setup,
quickstart) fetches those that are missing (ensure), with no login, token or prompt, and checks each against
the lock. A file that does not match is reported and never used.

    uv run evals media pull                            # every task's files, ahead of time
    uv run evals media pull --tasks 'camera-*'         # only these tasks'
    uv run evals media pull --generators               # the generators' sources and build material too
    uv run evals media pull --from ../other-checkout   # copy them from another checkout, by the same paths
    uv run evals media check                           # what is missing, differs, or is not in the lock
    uv run evals media push [--tasks ...] [--dry-run]  # maintainers: upload new or changed files, rewrite the lock

A task's files are those under tasks/<split>/<task>/ and controls/<task>/; the generators' (generators/<name>/,
about 6 GB) come only with --generators, for rebuilding tasks. A push uploads only files that are new or differ
from the lock and never deletes from the dataset, so every older commit stays complete; commit the lock it
writes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from evals.core.config import CONFIGS, ROOT, ConfigError, rel

LOCK = CONFIGS / "media.lock.json"
KINDS = ("tasks", "controls", "generators")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def lock() -> dict:
    """configs/media.lock.json: {"repo", "repo_type", "revision", "files": {path: {"bytes", "sha256"}}}."""
    try:
        return json.loads(LOCK.read_text())
    except FileNotFoundError:
        raise ConfigError("missing %s" % rel(LOCK)) from None


def revision() -> str | None:
    """The dataset commit the lock pins, as run.json records it."""
    try:
        return lock().get("revision")
    except ConfigError:
        return None


def owner(path: str) -> tuple[str, str]:
    """(kind, name) of a media path: ("tasks", task), ("controls", task) or ("generators", "generators/<name>")."""
    parts = path.split("/")
    if parts[0] == "tasks":
        return "tasks", parts[2]
    if parts[0] == "controls":
        return "controls", parts[2] if parts[1] == "train" else parts[1]
    return parts[0], "/".join(parts[:2])


def wanted(files: dict, tasks=None, generators: bool = False) -> list[str]:
    """The lock's paths for these tasks (every task when None): their inputs, oracle and controls, and with
    `generators` the sources and build material of the generators that built them."""
    names = None if tasks is None else {t.name for t in tasks}
    built_by = None if tasks is None else {t.source for t in tasks if t.source}
    out = []
    for p in sorted(files):
        kind, name = owner(p)
        if kind == "generators":
            if generators and (built_by is None or name in built_by or any(n in name for n in names or ())):
                out.append(p)
        elif kind in ("tasks", "controls") and (names is None or name in names):
            out.append(p)
    return out


def verify(paths: list[str], files: dict, full: bool = True) -> dict[str, str]:
    """path -> "missing" or "differs" for each file that is not as the lock says; `full` hashes every file,
    otherwise a file of the right size is taken as it is."""
    bad = {}
    for p in paths:
        local = ROOT / p
        if not local.is_file():
            bad[p] = "missing"
        elif local.stat().st_size != files[p]["bytes"] or (full and sha256(local) != files[p]["sha256"]):
            bad[p] = "differs"
    return bad


def missing(tasks=None) -> list[str]:
    """The lock's files for these tasks that are not on disk at their size: a quick check, no hashing."""
    files = lock()["files"]
    return sorted(verify(wanted(files, tasks), files, full=False))


def _download(lk: dict, paths: list[str], say=print) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ConfigError("fetching the media needs huggingface_hub: run `uv sync`") from None
    size = sum(lk["files"][p]["bytes"] for p in paths)
    say("fetching %d media file(s), %.2f GB, from huggingface.co/datasets/%s at %s" % (len(paths), size / 1e9, lk["repo"], lk["revision"][:12]))
    escaped = [re.sub(r"([*?\[])", r"[\1]", p) for p in paths]  # allow_patterns are globs; these are exact paths
    try:
        snapshot_download(lk["repo"], repo_type=lk.get("repo_type", "dataset"), revision=lk["revision"], local_dir=ROOT,
                          allow_patterns=escaped)
    except Exception as err:  # the hub's message says what is wrong: not found, offline, rate limited
        raise ConfigError("could not fetch the media from %s@%s: %s" % (lk["repo"], lk["revision"][:12], str(err).splitlines()[0])) from None


def _copy_from(paths: list[str], files: dict, other: Path) -> dict[str, str]:
    """Copy files from another checkout, by the same paths; returns why it could not, by path."""
    why = {}
    for p in paths:
        src = other / p
        if not src.is_file():
            why[p] = "not in %s:" % other
        elif src.stat().st_size != files[p]["bytes"] or sha256(src) != files[p]["sha256"]:
            why[p] = "differs from the lock in %s:" % other
        else:
            dest = ROOT / p
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            shutil.copyfile(src, tmp)  # a copy, never a link: nothing here may write through to the other checkout
            os.replace(tmp, dest)
    return why


def ensure(tasks=None, generators: bool = False, say=print) -> None:
    """Fetch the files these tasks need that are not here, and check them; raise ConfigError naming any file that
    still is not as the lock says. Files already here at their size are taken as they are (`media check` hashes them)."""
    lk = lock()
    paths = wanted(lk["files"], tasks, generators)
    todo = sorted(verify(paths, lk["files"], full=False))
    if not todo:
        return
    _download(lk, todo, say)
    bad = verify(todo, lk["files"])
    if bad:
        raise ConfigError("media not as %s pins it: %s" % (rel(LOCK), "; ".join("%s %s" % (s, p) for p, s in sorted(bad.items())[:5])))


def pull(tasks=None, generators: bool = False, other: Path | None = None, say=print) -> list[str]:
    """Fetch every file of these tasks that is missing or differs, from the dataset or from `other` (another
    checkout), and check them all by sha256; returns the problems (empty when every file matches)."""
    lk = lock()
    paths = wanted(lk["files"], tasks, generators)
    todo = sorted(verify(paths, lk["files"]))
    why: dict[str, str] = {}
    if todo and other:
        say("copying %d of %d media file(s) from %s" % (len(todo), len(paths), other))
        why = _copy_from(todo, lk["files"], Path(other).resolve())
    elif todo:
        _download(lk, todo, say)
    bad = verify(todo, lk["files"]) if todo else {}
    say("media: %d of %d file(s) present and checked against %s" % (len(paths) - len(bad), len(paths), rel(LOCK)))
    return ["%s %s" % (why.get(p, status), p) for p, status in sorted(bad.items())]


def local_media() -> list[str]:
    """What git ignores under tasks/, controls/ and generators/ here: the media on disk, minus caches and the
    score.json a generator writes beside each broken deliverable (expected.json records those)."""
    res = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--others", "--ignored", "--exclude-standard", "-z", "--", *KINDS],
                         capture_output=True, check=False)
    out = []
    for p in res.stdout.decode().split("\0"):
        parts = p.split("/")
        if not p or "__pycache__" in parts or p.endswith((".pyc", ".DS_Store", ".part")):
            continue
        if parts[0] == "controls" and parts[-1] == "score.json" and "grades" not in parts:
            continue
        out.append(p)
    return sorted(out)


def check(tasks=None, generators: bool = False) -> tuple[list[str], list[str]]:
    """(problems: files missing or different, by sha256; notes: media on disk the lock does not list)."""
    lk = lock()
    paths = wanted(lk["files"], tasks, generators)
    problems = ["%s %s" % (status, p) for p, status in sorted(verify(paths, lk["files"]).items())]
    notes = ["not in %s: %s" % (rel(LOCK), p) for p in local_media() if p not in lk["files"]]
    return problems, notes


def push(tasks=None, dry_run: bool = False, say=print) -> None:
    """Upload the media on disk that is new or differs from the lock, then rewrite the lock. With `tasks`, only
    their files, and the other entries stay as they are; without, the lock lists exactly what is on disk."""
    lk = lock()
    paths = local_media()
    if tasks is not None:
        names = {t.name for t in tasks} | {t.source for t in tasks if t.source}
        paths = [p for p in paths if owner(p)[1] in names]
    special = [p for p in paths if any(c in p for c in "*?[]")]
    if special:
        raise ConfigError("rename these first; their names read as globs: %s" % ", ".join(special))
    files = {p: {"bytes": (ROOT / p).stat().st_size, "sha256": sha256(ROOT / p)} for p in paths}
    upload = [p for p in paths if lk["files"].get(p) != files[p]]
    say("%d of %d file(s) new or changed, %.2f GB, for %s" % (len(upload), len(paths), sum(files[p]["bytes"] for p in upload) / 1e9, lk["repo"]))
    if dry_run:
        for p in upload:
            say("  %8.1f MB  %s" % (files[p]["bytes"] / 1e6, p))
        return
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ConfigError("publishing the media needs huggingface_hub: run `uv sync`") from None
    api = HfApi()
    if upload:
        api.upload_large_folder(repo_id=lk["repo"], folder_path=ROOT, repo_type=lk.get("repo_type", "dataset"), allow_patterns=upload)
    new = {**lk["files"], **files} if tasks is not None else files
    lk.update(revision=api.repo_info(lk["repo"], repo_type=lk.get("repo_type", "dataset")).sha, files=dict(sorted(new.items())))
    LOCK.write_text(json.dumps(lk, indent=1) + "\n")
    say("pinned %s@%s in %s: commit it" % (lk["repo"], lk["revision"], rel(LOCK)))
