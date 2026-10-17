"""Names in configs/*.toml, resolved to the classes and entries they stand for.

    configs/models.toml   [chat.<name>] and [image.<name>]: impl, key_env, prices, options
    configs/agents.toml   [<name>]: a subject, an agent and the model it drives
    configs/judges.toml   [<name>]: a judge panel, its models, rule and effort
    configs/suites/*.toml one suite each: which tasks, which subjects, attempts, judging

An `impl` or `agent` is a short name registered below or any "module:Class" path, the convention
Harbor's --agent-import-path uses, so a model or agent can live in any importable module and be swapped
by editing one line of config.
"""

from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass, field
from functools import lru_cache

from evals.core.config import CONFIGS, ConfigError, load_toml

BUILTIN = {
    "chat": {
        "anthropic": "evals.models.anthropic:AnthropicChat",
        "openai": "evals.models.openai:OpenAIChat",
        "google": "evals.models.google:GeminiChat",
        "openai_compat": "evals.models.openai_compat:OpenAICompatChat",
    },
    "image": {
        "gpt-image": "evals.models.openai:GPTImage",
        "seedream": "evals.models.fal:Seedream",
        "uni-1": "evals.models.fal:Uni1",
        "nano-banana": "evals.models.google:NanoBanana",
        "oracle": "evals.models.local:OracleImage",
    },
}


def load_object(spec: str, builtins: dict[str, str] | None = None):
    """A registered short name, or "module:attr"."""
    target = (builtins or {}).get(spec, spec)
    if ":" not in target:
        raise ConfigError("%r is neither a registered name (%s) nor a module:Class path" % (spec, ", ".join(builtins or {}) or "none"))
    module, attr = target.split(":", 1)
    try:
        obj = importlib.import_module(module)
    except ImportError as err:
        raise ConfigError("cannot import %s for %r: %s" % (module, spec, err)) from None
    for part in attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            raise ConfigError("%s has no %s (for %r)" % (module, attr, spec)) from None
    return obj


# ---------------------------------------------------------------- models


@lru_cache(maxsize=None)
def models_config() -> dict:
    return load_toml(CONFIGS / "models.toml")


def model_entry(kind: str, name: str) -> dict:
    """configs/models.toml [<kind>.<name>]; a "provider/" prefix, as Harbor's -m gives, is ignored."""
    entries = models_config().get(kind, {})
    key = name if name in entries else name.split("/")[-1]
    if key not in entries:
        raise ConfigError("no %s model %r in configs/models.toml; have: %s" % (kind, name, ", ".join(entries)))
    return entries[key]


def model_names(kind: str) -> list[str]:
    return list(models_config().get(kind, {}))


_instances: dict[tuple[str, str], object] = {}
_lock = threading.Lock()


def _model(kind: str, name: str):
    key = name.split("/")[-1] if name not in models_config().get(kind, {}) else name
    with _lock:  # one instance (and one SDK client) per model, shared by the worker threads
        if (kind, key) not in _instances:
            entry = model_entry(kind, key)
            cls = load_object(entry.get("impl", ""), BUILTIN[kind])
            _instances[kind, key] = cls(key, entry)
        return _instances[kind, key]


def chat_model(name: str):
    """The ChatModel configs/models.toml [chat.<name>] describes."""
    return _model("chat", name)


def image_model(name: str):
    """The ImageModel configs/models.toml [image.<name>] describes."""
    return _model("image", name)


# ---------------------------------------------------------------- subjects (agents and the models they drive)


@dataclass(frozen=True)
class Subject:
    """What a suite runs on its tasks: configs/agents.toml [<name>].

    agent = "harbor:<name>"            one of Harbor's agents (oracle, nop, codex, claude-code, terminus-2...)
            "harbor:<module:Class>"    any Harbor BaseAgent, by import path
            "image-model"              an image model from configs/models.toml [image.*], as a Harbor agent
            "<module:Class>"           our own agent: a subclass of evals.agents.ContainerAgent
    """
    name: str
    agent: str
    model: str | None = None
    options: dict = field(default_factory=dict)       # the agent's kwargs (Harbor's --ak)
    allow_hosts: tuple[str, ...] = ()                 # hosts the container may reach while the agent runs
    host_env: dict = field(default_factory=dict)      # variables for Harbor's own process
    agent_env: dict = field(default_factory=dict)     # variables for the agent phase; never keys
    max_concurrent: int | None = None
    label: str = ""

    @property
    def kind(self) -> str:
        """"harbor", "image-model" or "container"."""
        if self.agent.startswith("harbor:"):
            return "harbor"
        return "image-model" if self.agent == "image-model" else "container"

    @property
    def reference(self) -> bool:
        """Harbor's oracle and nop: references for a task, not subjects under test."""
        return self.agent in ("harbor:oracle", "harbor:nop")

    @property
    def display(self) -> str:
        return self.label or self.name


@lru_cache(maxsize=None)
def agents_config() -> dict:
    return load_toml(CONFIGS / "agents.toml")


def subject(name: str) -> Subject:
    entries = agents_config()
    if name not in entries:
        raise ConfigError("no subject %r in configs/agents.toml; have: %s" % (name, ", ".join(entries)))
    e = dict(entries[name])
    secretish = [k for k in e.get("agent_env", {}) if any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET"))]
    if secretish:  # Harbor saves each agent's env in the job folder
        raise ConfigError("%s: agent_env must not carry keys (%s); keys reach Harbor through its environment" % (
            name, ", ".join(secretish)))
    unknown = set(e) - {"agent", "model", "options", "allow_hosts", "host_env", "agent_env", "max_concurrent", "label"}
    if unknown:
        raise ConfigError("%s: unknown field(s) in configs/agents.toml: %s" % (name, ", ".join(sorted(unknown))))
    if "agent" not in e:
        raise ConfigError("%s: configs/agents.toml entry needs an agent" % name)
    return Subject(name=name, agent=e["agent"], model=e.get("model"), options=dict(e.get("options", {})),
                   allow_hosts=tuple(e.get("allow_hosts", ())), host_env=dict(e.get("host_env", {})),
                   agent_env=dict(e.get("agent_env", {})), max_concurrent=e.get("max_concurrent"), label=e.get("label", ""))


def subject_names() -> list[str]:
    return list(agents_config())


# ---------------------------------------------------------------- judges and suites


@lru_cache(maxsize=None)
def judges_config() -> dict:
    return load_toml(CONFIGS / "judges.toml")


def panel_entry(name: str) -> dict:
    entries = judges_config()
    if name not in entries:
        raise ConfigError("no judge panel %r in configs/judges.toml; have: %s" % (name, ", ".join(entries)))
    return entries[name]


def suite(name: str) -> dict:
    path = CONFIGS / "suites" / ("%s.toml" % name)
    if not path.exists():
        raise ConfigError("no suite %r: configs/suites/ has %s" % (name, ", ".join(suite_names()) or "none"))
    return {"name": name, **load_toml(path)}


def suite_names() -> list[str]:
    return sorted(p.stem for p in (CONFIGS / "suites").glob("*.toml"))
