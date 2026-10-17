"""What keeps a fresh clone working and its keys private: redaction, the config key check, the env file parser,
the media lock and fetching on demand.

    uv run pytest evals/tests/test_install.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os

import pytest

from evals.core import config, media, secrets, tasks
from evals.core.config import ConfigError

KEY = "sk-test-" + "a1B2c3D4e5F6g7H8i9J0" * 2


@pytest.fixture
def key_env(monkeypatch):
    monkeypatch.setenv("EVALS_TEST_API_KEY", KEY)
    return KEY


def test_redact_masks_every_key_variable(key_env):
    assert secrets.redact("auth %s done" % KEY) == "auth [EVALS_TEST_API_KEY] done"
    assert KEY not in secrets.dumps({"headers": {"x-api-key": KEY}, "note": "fine"})
    stream = io.StringIO()
    secrets.RedactingStream(stream).write("Traceback: bad key %s\n" % KEY)
    assert KEY not in stream.getvalue() and "[EVALS_TEST_API_KEY]" in stream.getvalue()


def test_short_or_unnamed_values_are_left_alone(monkeypatch):
    monkeypatch.setenv("EVALS_TEST_TOKEN", "1")                 # too short to be a key
    monkeypatch.setenv("EVALS_TEST_NAME", "hello-world-value")  # not named like a key
    assert secrets.redact("1 hello-world-value") == "1 hello-world-value"


def test_a_config_that_holds_a_key_is_refused_by_field(tmp_path, key_env):
    path = tmp_path / "c.toml"
    for body in ('[a]\napi_key = "anything"\n', '[a]\nmodel = "%s"\n' % KEY, '[a]\nopts = { header = "x %s" }\n' % KEY):
        path.write_text(body)
        with pytest.raises(ConfigError) as err:
            config.load_toml(path)
        assert "looks like a key" in str(err.value) and KEY not in str(err.value)
    path.write_text('[a]\nkey_env = "OPENAI_API_KEY"\nsha256 = "%s"\n' % ("ab" * 32))
    assert config.load_toml(path)["a"]["key_env"] == "OPENAI_API_KEY"


def test_every_shipped_config_loads():
    for path in config.CONFIGS.rglob("*.toml"):
        config.load_toml(path)


def test_env_file_comments_are_never_values(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))  # load_env_file writes os.environ; keep it to this test
    for name in ("EVALS_T_A", "EVALS_T_B", "EVALS_T_C"):
        os.environ.pop(name, None)
    path = tmp_path / ".env"
    path.write_text('EVALS_T_A=   # a note\nEVALS_T_B=value # a note\nEVALS_T_C="quoted # kept"\n')
    assert config.load_env_file(path) == ["EVALS_T_B", "EVALS_T_C"]
    assert os.environ["EVALS_T_B"] == "value" and os.environ["EVALS_T_C"] == "quoted # kept"


# ---------------------------------------------------------------- media


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repo of its own, with a lock listing two files of task t1 and one of its generator."""
    files = {"tasks/eval/t1/environment/workspace/a.png": b"input", "controls/t1/v1/out/grid.png": b"broken",
             "generators/g1/sources/s.webm": b"source"}
    lock = {"repo": "org/data", "repo_type": "dataset", "revision": "abc123",
            "files": {p: {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for p, b in files.items()}}
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "media.lock.json").write_text(json.dumps(lock))
    task = tmp_path / "tasks" / "eval" / "t1"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('source = "generators/g1"\n')
    monkeypatch.setattr(media, "ROOT", tmp_path)
    monkeypatch.setattr(media, "LOCK", tmp_path / "configs" / "media.lock.json")
    return tmp_path, files, tasks.Task(task)


def test_the_lock_selects_a_tasks_files(repo):
    _, files, task = repo
    lock = media.lock()["files"]
    assert media.wanted(lock, [task]) == ["controls/t1/v1/out/grid.png", "tasks/eval/t1/environment/workspace/a.png"]
    assert "generators/g1/sources/s.webm" in media.wanted(lock, [task], generators=True)
    assert media.owner("controls/train/t9/v1/out/grid.png") == ("controls", "t9")


def test_ensure_fetches_only_what_is_missing_and_checks_it(repo, monkeypatch):
    root, files, task = repo
    (root / "tasks/eval/t1/environment/workspace").mkdir(parents=True)
    (root / "tasks/eval/t1/environment/workspace/a.png").write_bytes(b"input")
    asked = []

    def download(lk, paths, say=print):
        asked.extend(paths)
        for p in paths:
            (root / p).parent.mkdir(parents=True, exist_ok=True)
            (root / p).write_bytes(files[p])

    monkeypatch.setattr(media, "_download", download)
    media.ensure([task], say=lambda *a: None)
    assert asked == ["controls/t1/v1/out/grid.png"]
    media.ensure([task], say=lambda *a: None)  # nothing left to fetch
    assert asked == ["controls/t1/v1/out/grid.png"]
    (root / "controls/t1/v1/out/grid.png").unlink()
    monkeypatch.setattr(media, "_download", lambda lk, paths, say=print: [(root / p).write_bytes(b"tampered") for p in paths])
    with pytest.raises(ConfigError, match="not as"):
        media.ensure([task], say=lambda *a: None)


def test_copies_are_checked_and_never_links(repo, tmp_path_factory):
    root, files, task = repo
    other = tmp_path_factory.mktemp("other")
    (other / "tasks/eval/t1/environment/workspace").mkdir(parents=True)
    (other / "tasks/eval/t1/environment/workspace/a.png").write_bytes(b"input")
    (other / "controls/t1/v1/out").mkdir(parents=True)
    (other / "controls/t1/v1/out/grid.png").write_bytes(b"tampered")
    problems = media.pull([task], other=other, say=lambda *a: None)
    assert problems == ["differs from the lock in %s: controls/t1/v1/out/grid.png" % other]
    copied = root / "tasks/eval/t1/environment/workspace/a.png"
    assert copied.read_bytes() == b"input" and copied.stat().st_nlink == 1


def test_the_shipped_lock_covers_every_task():
    listed = {media.owner(p)[1] for p in media.lock()["files"]}
    for t in tasks.discover("eval"):
        assert t.name in listed, t.name
