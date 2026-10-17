"""Paths, TOML configs and KEY=value env files: the one copy that the runner, the graders, the camera-eval
service and the judge tools share.

configs/evals.toml holds the settings (the pinned Harbor, the grading environment, the images, where runs
go); configs/media.lock.json pins the media (evals.core.media); configs/models.toml, agents.toml, judges.toml
and suites/*.toml hold what can be run (evals.core.registry). No file under configs/ holds a key: an entry
names the environment variable that holds one, keys come from the shell or from a gitignored .env
(load_env_files), and load_toml refuses a config value that looks like a key.
"""

from __future__ import annotations

import os
import re
import shutil
import tomllib
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
TASKS = ROOT / "tasks"
CONTROLS = ROOT / "controls"
JOBS = ROOT / "jobs"

# Keys in the formats providers issue them (OpenAI and Anthropic, Google, Hugging Face, GitHub, Slack, AWS,
# fal, PEM), and the field names that would hold one.
KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,}|hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}"
                         r"|github_pat_\w{30,}|xox[abprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY"
                         r"|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}:[0-9a-f]{32}")
KEY_FIELD = re.compile(r"(^|_)(api_?key|token|secret|password|private_?key|credentials?)$", re.IGNORECASE)


class ConfigError(RuntimeError):
    """A config file, an entry in one, or something it needs is missing or wrong."""


def key_like(data) -> list[str]:
    """Where a parsed config holds what looks like a key: a field named for one, a value in a provider's key
    format, or the value of a key variable in this environment. Returns the fields, never the values."""
    from evals.core.secrets import secret_values
    known = [v for v, _ in secret_values() if len(v) >= 16]
    found = []

    def walk(value, path):
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, path + [str(k)])
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, path + [str(i)])
        elif isinstance(value, str) and value.strip():
            if (path and KEY_FIELD.search(path[-1])) or KEY_PATTERN.search(value) or any(s in value for s in known):
                found.append(".".join(path))

    walk(data, [])
    return found


def load_toml(path: Path) -> dict:
    """A config file. One that holds what looks like a key is refused, naming the field and not the value."""
    try:
        data = tomllib.loads(Path(path).read_text())
    except FileNotFoundError:
        raise ConfigError("missing %s" % rel(path)) from None
    except tomllib.TOMLDecodeError as err:
        raise ConfigError("%s: %s" % (rel(path), err)) from None
    bad = key_like(data)
    if bad:
        raise ConfigError("%s: %s looks like a key. Keys belong in the environment or .env (.env.example lists them), "
                          "never in a config file" % (rel(path), ", ".join(bad)))
    return data


@lru_cache(maxsize=None)
def settings() -> dict:
    """configs/evals.toml."""
    return load_toml(CONFIGS / "evals.toml")


def rel(path: Path | str) -> str:
    """The path relative to the repo root when it is inside it, as the reports and logs show it."""
    p = Path(path).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(path)


def resolve(path: Path | str) -> Path:
    """A path from a config file: relative ones are from the repo root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def runs_dir() -> Path:
    return resolve(settings()["runs"]["dir"])


# ---------------------------------------------------------------- keys


def load_env_file(path: Path, override: bool = False) -> list[str]:
    """Read KEY=value lines into os.environ; returns the names set (values are never printed).

    A quoted value keeps everything inside its quotes; an unquoted one ends at a comment, so `KEY=  # note`
    is empty. An empty value never hides one set earlier, and a variable already in the environment wins
    unless `override`.
    """
    names = []
    for line in Path(path).expanduser().read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.removeprefix("export ").strip()
        value = value.strip()
        if value[:1] in ('"', "'") and value.count(value[0]) >= 2:
            value = value[1:value.index(value[0], 1)]
        elif value.startswith("#"):
            value = ""
        else:
            value = value.split(" #", 1)[0].split("\t#", 1)[0].strip()
        if value and (override or name not in os.environ):
            os.environ[name] = value
            names.append(name)
    return names


def load_env_files(paths: list[Path] | None) -> None:
    """--env-file arguments, or .env at the repo root when there is one."""
    for p in paths or ([ROOT / ".env"] if (ROOT / ".env").exists() else []):
        if not Path(p).expanduser().is_file():
            raise ConfigError("no env file at %s" % p)
        load_env_file(p)


def missing_env(names) -> list[str]:
    return sorted({n for n in names if n and not os.environ.get(n)})


def which(program: str) -> str | None:
    return shutil.which(program)
