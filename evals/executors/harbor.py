"""Run a plan's trials with Harbor: one job config, written to the run folder and handed to `harbor run -c`.

Harbor schedules; this only configures. Each subject becomes an agents[] entry: Harbor's own agents by
name, an image model as evals.agents.harbor:ImageModelAgent, our own agents through
evals.agents.harbor:HarborContainerAgent. Keys never enter the config: Harbor reads them from its
environment, and a subject's host_env (no keys) and the grading services' addresses are added to it.

On a Docker engine that cannot enforce the tasks' network rules (evals/core/harbor.py), Harbor refuses
the tasks as written; the run then uses copies with the network open, and run.json and the report say so.
An image model is called from Harbor's process on this machine and only drops a PNG into the container,
so it is unaffected; a harness inside the container is not.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from evals.core import harbor, registry
from evals.core.config import ROOT, ConfigError, rel
from evals.core.interfaces import Executor
from evals.core.registry import Subject
from evals.core.secrets import dumps


def _absolute(options: dict) -> dict:
    """Option values that name a file in the repo, made absolute: Harbor reads them from its own cwd."""
    out = {}
    for k, v in options.items():
        out[k] = str(ROOT / v) if isinstance(v, str) and not v.startswith("/") and "/" in v and (ROOT / v).exists() else v
    return out


def agent_config(s: Subject, concurrency: int) -> dict:
    """The subject as one of Harbor's agents[] entries."""
    if s.kind == "harbor":
        target = s.agent.split(":", 1)[1]
        cfg = {"import_path": target} if ":" in target else {"name": target}
        if s.model:
            cfg["model_name"] = s.model
        if s.options:
            cfg["kwargs"] = _absolute(s.options)
    elif s.kind == "image-model":
        cfg = {"import_path": "evals.agents.harbor:ImageModelAgent", "model_name": s.model}
    else:
        cfg = {"import_path": "evals.agents.harbor:HarborContainerAgent", "kwargs": {"agent": s.agent, "options": dict(s.options)}}
        if s.model:
            cfg["model_name"] = s.model
    if s.allow_hosts:
        cfg["extra_allowed_hosts"] = list(s.allow_hosts)
    if s.agent_env:
        cfg["env"] = dict(s.agent_env)
    cfg["n_concurrent"] = min(s.max_concurrent or concurrency, concurrency)
    return cfg


def packages(subjects: list[Subject]) -> list[str]:
    """What Harbor's environment must hold for our agents' models."""
    out = set()
    for s in subjects:
        if s.kind == "image-model":
            out.update(registry.image_model(s.model).requires)
        elif s.kind == "container" and s.model:
            out.update(registry.chat_model(s.model).requires)
    return sorted(out)


def auto_concurrency(subjects: list[Subject]) -> int:
    """Docker's memory over 10 GB a trial (the README's rule), capped by what the subjects allow at once."""
    try:
        mem = int(subprocess.run(["docker", "info", "--format", "{{.MemTotal}}"], capture_output=True, text=True).stdout.strip())
    except ValueError:
        mem = 0
    by_memory = max(1, mem // (10 * 1024 ** 3)) if mem else 4
    return max(1, min(by_memory, sum(s.max_concurrent or 4 for s in subjects)))


class HarborExecutor(Executor):
    name = "harbor"

    def supports(self, subject: Subject) -> str | None:
        return None

    def run(self, plan) -> list[Path]:
        from evals.run import say
        concurrency = plan.concurrency if isinstance(plan.concurrency, int) else auto_concurrency(plan.subjects)
        groups: dict[tuple, list[Subject]] = {}
        for s in plan.subjects:  # an image model runs only the tasks that say what to send it, so it gets its own job
            key = tuple(t.name for t in plan.tasks if s.kind != "image-model" or t.image is not None)
            groups.setdefault(key, []).append(s)
        task_paths = {t.name: t.path for t in plan.tasks}
        if not plan.dry_run and not harbor.egress_supported():
            plan.network = harbor.OPEN_NETWORK
            say("WARNING: this Docker engine cannot enforce the tasks' network rules (its kernel lacks CONFIG_NFT_FIB_INET), "
                "so Harbor runs copies of the tasks with the network open. For enforced runs use a Linux Docker host, OrbStack "
                "or Colima.")
            task_paths = {p.name: p for p in harbor.relaxed_copies([t.path for t in plan.tasks], plan.run_dir / "tasks")}
            info = json.loads((plan.run_dir / "run.json").read_text())
            (plan.run_dir / "run.json").write_text(dumps({**info, "network": plan.network}, indent=1))
        env = {**plan.service_env("docker")}
        for s in plan.subjects:
            env.update(s.host_env)
        trials = []
        for k, (names, subjects) in enumerate(sorted(groups.items(), key=lambda g: g[1][0].name)):
            job = "harbor" if len(groups) == 1 else "harbor-%d" % (k + 1)
            config = {"job_name": job, "jobs_dir": str(plan.run_dir), "n_attempts": plan.attempts,
                      "n_concurrent_trials": concurrency, "tasks": [harbor.task_entry(task_paths[n]) for n in names],
                      "agents": [agent_config(s, concurrency) for s in subjects], "environment": {"type": "docker"}}
            path = plan.run_dir / ("%s-config.json" % job)
            path.write_text(dumps(config, indent=1))
            say("Harbor job %s: %d tasks x %s, %d at once; trials go to %s" % (
                job, len(names), ", ".join(s.name for s in subjects), concurrency, rel(plan.run_dir / job)))
            args = ["run", "-c", str(path), "-y"] + (["--dry-run"] if plan.dry_run else [])
            code = harbor.run(args, packages(subjects), env)
            if code != 0:
                if plan.dry_run:
                    raise ConfigError("harbor rejected the job config (exit %d)" % code)
                say("harbor exited %d; the trials it finished are kept" % code)
            trials += sorted(p.parent for p in (plan.run_dir / job).glob("*/result.json"))
        return trials
