"""Every interface in the pipeline, in one place.

    ImageModel   prompt + reference images + the output's shape -> one image          evals/models
    ChatModel    messages of text, image, audio and video parts -> text, JSON or tool calls
    Grader       a deliverables folder -> score.json and the reward, as tests/test.sh   evals/core/grading.py
    Resource     a service graders need: up() returns the environment that reaches it  evals/resources
    Executor     subjects x tasks x attempts -> trial folders                           evals/executors
    View         one task's trials -> the data its report panel draws                   evals/report/views

Agents are Harbor's own BaseAgent (evals/agents/harbor.py adapts ours), and judges are a Panel of
ChatModels over a question source (evals/judging). An implementation is named in configs/*.toml by a
registered short name or any "module:Class" path (evals/core/registry.py), so swapping one is a config
edit. Standard library only: implementations import their SDKs themselves.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ProviderError(Exception):
    """A model call that failed. Transient ones (rate limits, 5xx, the network) are worth retrying."""

    def __init__(self, message: str, transient: bool = False, status: int | None = None):
        super().__init__(message)
        self.transient = transient
        self.status = status


class Model(ABC):
    """What chat and image models share: a name in configs/models.toml and the entry under it."""

    requires: tuple[str, ...] = ()  # the packages it imports, for the environments it runs in (Harbor's)

    def __init__(self, name: str, entry: dict):
        self.name, self.entry = name, entry
        self.model = entry.get("model", name)  # the provider's id, when it differs from the name
        self.key_env = entry.get("key_env") or None
        self.label = entry.get("label", name)
        self.price_source = entry.get("price_source", "")

    def __repr__(self) -> str:
        return "%s(%s)" % (type(self).__name__, self.name)


# ---------------------------------------------------------------- image models


@dataclass(frozen=True)
class ImageRequest:
    task: str
    prompt: str                          # instruction.md, exactly as written
    references: tuple[Path, ...] = ()    # the task's [metadata.image] inputs
    aspect: str = "1:1"                  # "16:9", "2:1" or "1:1"
    min_width: int = 0
    output: str = "out/grid.png"         # where the image goes, relative to /workspace
    task_dir: Path | None = None         # only the oracle reads it, for the task's own solution

    @property
    def reference(self) -> Path | None:
        return self.references[0] if self.references else None

    def meta(self, **fields) -> dict:
        """What was sent, for the records: no key and no image bytes."""
        return {"prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest(),
                "reference": self.reference.name if self.reference else None, **fields}


@dataclass(frozen=True)
class ImageResult:
    image: bytes                         # as the provider returned it; the caller re-encodes to PNG
    meta: dict = field(default_factory=dict)   # what was sent and returned, without keys or image bytes
    usd: float | None = None             # the provider's own count, when it gives one


class ImageModel(Model):
    @abstractmethod
    def generate(self, req: ImageRequest) -> ImageResult:
        """One image, or ProviderError."""

    def list_price(self, req: ImageRequest) -> float:
        """The list price of one attempt, reference input included."""
        return float(self.entry.get("usd_per_image", 0.0))

    def release(self, req: ImageRequest) -> None:
        """Release what this one request holds (its uploaded references), for a caller whose files go away after it."""

    def close(self) -> None:
        """Release what is held across requests (every uploaded reference)."""


# ---------------------------------------------------------------- chat models


@dataclass(frozen=True)
class Part:
    kind: str                            # "text", "image", "audio" or "video"
    data: str | bytes
    mime: str = ""                       # for media: "image/png", "audio/wav", "video/mp4"

    @staticmethod
    def text(s: str) -> "Part":
        return Part("text", s)

    @staticmethod
    def image(data: bytes, mime: str = "image/png") -> "Part":
        return Part("image", data, mime)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    role: str                            # "user", "assistant" or "tool"
    parts: list[Part] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)   # an assistant's
    tool_call_id: str | None = None      # a tool result's
    native: Any = None                   # the provider's own form of an assistant turn, sent back as it was


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict                     # JSON schema of the arguments


@dataclass
class ChatRequest:
    messages: list[Message]
    system: str = ""
    schema: dict | None = None           # the JSON schema the answer must follow
    schema_name: str = "answer"
    tools: list[Tool] = field(default_factory=list)
    effort: str | None = None            # "low", "medium" or "high", in each provider's own terms
    max_tokens: int = 16000


@dataclass(frozen=True)
class Usage:
    input: int = 0                       # includes cached input
    output: int = 0                      # includes reasoning
    cached: int = 0


@dataclass
class ChatResponse:
    text: str
    message: Message                     # the assistant turn, to append to the conversation
    usage: Usage
    usd: float
    stop: str = ""

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self.message.tool_calls

    def json(self) -> dict:
        return json.loads(self.text)


class ChatModel(Model):
    def __init__(self, name: str, entry: dict):
        super().__init__(name, entry)
        self.usd_in, self.usd_out = (entry.get("usd_per_mtok") or [0.0, 0.0])[:2]

    @abstractmethod
    def complete(self, req: ChatRequest) -> ChatResponse:
        """One reply, or an exception (ProviderError when the provider refused or failed)."""

    def cost(self, usage: Usage) -> float:
        return round((usage.input * self.usd_in + usage.output * self.usd_out) / 1e6, 5)


# ---------------------------------------------------------------- grading


@dataclass
class Grade:
    """What a verifier writes, read back: score.json, reward.json and reward-details.json in `out`."""
    out: Path
    reward: float | None
    details: dict
    score: dict | None
    seconds: float = 0.0
    error: str | None = None             # an infrastructure failure: no reward was written, which is not a 0

    @property
    def checks(self) -> dict[str, str]:
        return {c["id"]: c["status"] for c in (self.score or {}).get("checks", []) if "id" in c}


class Grader(ABC):
    name = "?"

    @abstractmethod
    def grade(self, task, deliverables: Path, out: Path) -> Grade:
        """Grade the files in `deliverables` as the verifier would see them, writing into `out`."""


# ---------------------------------------------------------------- resources


class Resource(ABC):
    """A service some tasks' graders call, named by their task.toml [metadata] grader."""

    name = "?"

    @abstractmethod
    def running(self) -> dict | None:
        """The service's environment if it is up already: {"host": {...}, "docker": {...}}."""

    @abstractmethod
    def up(self, **options) -> dict:
        """Start it (or reuse it) and return its environment, as running() does."""

    @abstractmethod
    def down(self) -> None:
        """Stop what up() started."""

    def missing(self, gpu: str | None, judges: bool = True) -> list[str]:
        """What would stop up(gpu=..., judges=...) from starting, such as a key, so a run can say so before it starts
        (gpu None: no GPU chosen yet)."""
        return []


# ---------------------------------------------------------------- executors


class Executor(ABC):
    name = "?"

    @abstractmethod
    def supports(self, subject) -> str | None:
        """None when this executor can run the subject, else why not."""

    @abstractmethod
    def run(self, plan) -> list[Path]:
        """Run the plan's trials; returns their folders."""


# ---------------------------------------------------------------- report views


class View(ABC):
    """One kind of report panel: task_data() and trial_data() return plain data, and the view's script
    (evals/report/views/<module>.js) draws it. `media` turns files into the page's media (evals/report/build.py)."""

    name = "?"

    def task_data(self, task, media, controls: dict[str, Path]) -> dict | None:
        """What every trial of the task shares (an answer key's timeline, the oracle's video). `controls` maps
        each control graded in this run to its folder (score.json, evidence/)."""
        return None

    @abstractmethod
    def trial_data(self, task, trial, media) -> dict | None:
        """What the panel draws for one trial (a TrialRecord: its verifier/ and deliverables folders);
        None when there is nothing to draw."""
