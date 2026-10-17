"""The base for our own agents: a policy over a Sandbox (the task's container) and any ChatModel.

    class MyAgent(ContainerAgent):
        defaults = {"max_steps": 100}

        async def run(self, instruction: str, box: Sandbox) -> None:
            reply = await self.ask(ChatRequest([Message("user", [Part.text(instruction)])], tools=[...]))
            result = await box.exec("ls /workspace")
            ...

and in configs/agents.toml:

    [my-agent]
    agent = "mypkg.agents:MyAgent"
    model = "claude-opus-5-5"          # any [chat.*] in configs/models.toml
    options = { max_steps = 50 }

Harbor runs it through evals.agents.harbor:HarborContainerAgent, in Harbor's own process on the host,
so the task container keeps no network: only the model calls leave, and they leave from the host.
Nothing here imports Harbor, so an agent can be tried against a LocalSandbox (a folder on this machine)
and unit-tested without Docker. ask() records every model call in an ATIF trajectory, the format
Harbor and the report read, with its tokens and cost.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from evals.core.config import ConfigError
from evals.core.interfaces import ChatModel, ChatRequest, ChatResponse, Usage
from evals.core.secrets import dumps


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ---------------------------------------------------------------- the container


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int


class Sandbox(ABC):
    """The task's container, as an agent sees it: commands run in /workspace."""

    workdir = "/workspace"

    @abstractmethod
    async def exec(self, command: str, timeout: float | None = None) -> ExecResult:
        """Run a shell command in the working directory."""

    @abstractmethod
    async def read(self, path: str) -> bytes:
        """A file's bytes; relative paths are from the working directory."""

    @abstractmethod
    async def write(self, path: str, data: bytes) -> None:
        """Write a file, making its folder; relative paths are from the working directory."""

    def absolute(self, path: str) -> str:
        return path if path.startswith("/") else str(PurePosixPath(self.workdir) / path)


class LocalSandbox(Sandbox):
    """A folder on this machine standing in for /workspace, for tests and for trying an agent without Docker.
    It isolates nothing: run only agents and commands you trust here."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.workdir = str(self.root)

    def _path(self, path: str) -> Path:
        if path.startswith("/workspace"):
            path = path[len("/workspace"):].lstrip("/")
        p = Path(path)
        return p if p.is_absolute() else self.root / p

    async def exec(self, command: str, timeout: float | None = None) -> ExecResult:
        def run():
            try:
                p = subprocess.run(["bash", "-c", command.replace("/workspace", str(self.root))], cwd=self.root,
                                   capture_output=True, text=True, timeout=timeout,
                                   env={**os.environ, "HOME": str(self.root)})
                return ExecResult(p.stdout, p.stderr, p.returncode)
            except subprocess.TimeoutExpired as err:
                return ExecResult(err.stdout or "", "timed out after %s s" % timeout, 124)
        return await asyncio.to_thread(run)

    async def read(self, path: str) -> bytes:
        return self._path(path).read_bytes()

    async def write(self, path: str, data: bytes) -> None:
        p = self._path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


# ---------------------------------------------------------------- the trajectory


class Trajectory:
    """An ATIF trajectory (Harbor's Agent Trajectory Interchange Format), the steps the report reads."""

    def __init__(self, agent: str, version: str, model: str | None, tools: list[dict] | None = None):
        self.agent = {"name": agent, "version": version, "model_name": model}
        if tools:
            self.agent["tool_definitions"] = tools
        self.steps: list[dict] = []

    def _add(self, step: dict) -> dict:
        step = {"step_id": len(self.steps) + 1, "timestamp": now(), **step}
        self.steps.append(step)
        return step

    def user(self, text: str) -> None:
        self._add({"source": "user", "message": text})

    def reply(self, model: str | None, r: ChatResponse) -> dict:
        step = {"source": "agent", "model_name": model, "message": r.text,
                "metrics": {"prompt_tokens": r.usage.input, "completion_tokens": r.usage.output,
                            "cached_tokens": r.usage.cached, "cost_usd": r.usd}}
        if r.tool_calls:
            step["tool_calls"] = [{"tool_call_id": c.id, "function_name": c.name, "arguments": c.arguments} for c in r.tool_calls]
        return self._add(step)

    def observe(self, results: list[tuple[str, str]]) -> None:
        """What the tools returned for the last reply's calls: (call id, text)."""
        for step in reversed(self.steps):
            if step["source"] == "agent":
                step["observation"] = {"results": [{"source_call_id": cid, "content": text} for cid, text in results]}
                return

    def final(self, text: str) -> None:
        self._add({"source": "agent", "message": text})

    def to_dict(self) -> dict:
        m = [s.get("metrics") or {} for s in self.steps]
        return {"schema_version": "ATIF-v1.8", "agent": self.agent, "steps": self.steps,
                "final_metrics": {"total_prompt_tokens": sum(x.get("prompt_tokens", 0) for x in m),
                                  "total_completion_tokens": sum(x.get("completion_tokens", 0) for x in m),
                                  "total_cached_tokens": sum(x.get("cached_tokens", 0) for x in m),
                                  "total_cost_usd": round(sum(x.get("cost_usd", 0) or 0 for x in m), 6),
                                  "total_steps": len(self.steps)}}

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps(self.to_dict(), indent=1))


# ---------------------------------------------------------------- the agent


class ContainerAgent(ABC):
    """Subclass this, implement run(), and name the class in configs/agents.toml (see the module docstring)."""

    defaults: dict = {}      # every option the agent takes, with its default; an unknown option is an error
    version = "0.1.0"

    def __init__(self, llm: ChatModel | None, options: dict | None = None, logs_dir: Path | None = None):
        unknown = set(options or {}) - set(self.defaults)
        if unknown:
            raise ConfigError("%s takes no option(s) %s; it takes: %s" % (type(self).__name__, ", ".join(sorted(unknown)),
                                                                          ", ".join(self.defaults) or "none"))
        self.llm, self.options, self.logs_dir = llm, {**self.defaults, **(options or {})}, logs_dir
        self.trajectory = Trajectory(type(self).__name__, self.version, llm.name if llm else None)
        self.usage = Usage()
        self.spent_usd = 0.0
        self.final_message = ""

    @abstractmethod
    async def run(self, instruction: str, box: Sandbox) -> None:
        """Do the task in `box`. Return when done; what is in /workspace then is what gets graded."""

    async def ask(self, req: ChatRequest) -> ChatResponse:
        """One model call, from a worker thread, recorded in the trajectory with its tokens and cost."""
        if self.llm is None:
            raise ConfigError("%s needs a model: set model in its configs/agents.toml entry" % type(self).__name__)
        r = await asyncio.to_thread(self.llm.complete, req)
        self.usage = Usage(self.usage.input + r.usage.input, self.usage.output + r.usage.output, self.usage.cached + r.usage.cached)
        self.spent_usd = round(self.spent_usd + r.usd, 6)
        self.trajectory.reply(self.llm.name, r)
        return r

    def save_image(self, name: str, data: bytes) -> None:
        """Keep an image the agent looked at beside its trajectory (agent/images/), for the report."""
        if self.logs_dir:
            p = self.logs_dir / "images" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)

    def finish(self) -> None:
        if self.logs_dir:
            self.trajectory.write(self.logs_dir / "trajectory.json")
