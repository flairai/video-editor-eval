"""Build a run's interactive page.

    uv run evals report jobs/runs/<run>                   # <run>/report/index.html
    uv run evals report jobs/<harbor-job> --name codex    # any Harbor job or trial folders, the image runner's
                                                          # folders: jobs/reports/<name>/index.html

The page is a pure function of its inputs, in three stages:

  collect(paths)   reads the trial records (evals/core/records.py), their tasks and their files, and returns
                   the page data, a media plan and the sha256 of every file read. It writes nothing.
  render(data)     fills shell.html with the data and the views' scripts. No I/O.
  materialize(...) the only stage that writes. Each build is a new folder, <out>/<name>-<digest>/, where the
                   digest covers every input file, this code, the shell, the views and the encoders' versions.
                   It is built in a temporary folder and renamed into place, so a crash leaves nothing
                   half-written; <out>/<name> links to the newest. Rebuilding unchanged inputs does nothing.

Browser previews (H.264/AAC video and audio, JPEG images) are cached by content in jobs/.report-cache/<key>,
where the key hashes the source's sha256, the encode settings and the encoder's version: a changed input
never reuses a stale preview and an unchanged one is never encoded twice. Reports hard-link from the cache.
Serve the folder over HTTP to watch the videos (the `reports` launch config serves jobs/).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from evals.core import records, registry
from evals.core.config import JOBS, ROOT, load_env_files, rel, resolve, settings
from evals.core.records import TrialRecord
from evals.core.secrets import dumps
from evals.core.tasks import Task, discover
from evals.report import detail
from evals.report.views import view as view_for

HERE = Path(__file__).resolve().parent
SHELL = HERE / "shell.html"
VIDEO_ARGS = ("-vf", "scale='min(1280,iw)':-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
              "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart")
AUDIO_ARGS = ("-vn", "-c:a", "aac", "-b:a", "192k")
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm")
AUDIO_SUFFIXES = (".wav", ".flac", ".m4a", ".mp3", ".aac")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
TEXT_SUFFIXES = (".md", ".json", ".txt", ".csv")
REFERENCES = ("oracle", "image-oracle", "nop")  # shown after the subjects under test
CATEGORIES = {"video-editing": "Video editing", "image-generation": "Image generation"}  # in page order
HARNESSES = {"codex": "Codex", "claude-code": "Claude Code", "evals.agents.tool_loop:ToolLoopAgent": "Tool loop"}
BOOTSTRAP = 4000


# ---------------------------------------------------------------------------------------- reading


class Reader:
    """Reads input files and remembers the sha256 of each, so the report can name its inputs."""

    def __init__(self, root: Path):
        self.root = root
        self.hashes: dict[str, str] = {}

    def _key(self, path) -> str:
        path = Path(path).resolve()
        return str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path)

    def sha256(self, path) -> str:
        key = self._key(path)
        if key not in self.hashes:
            digest = hashlib.sha256()
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    digest.update(block)
            self.hashes[key] = digest.hexdigest()
        return self.hashes[key]

    def bytes(self, path) -> bytes:
        self.sha256(path)
        return Path(path).read_bytes()

    def text(self, path) -> str:
        self.sha256(path)
        return Path(path).read_text(errors="replace")

    def json(self, path):
        try:
            return json.loads(self.text(path)) if Path(path).is_file() else None
        except json.JSONDecodeError:
            return None


class Media:
    """Turns files into the page's media: each call plans one output and returns its path in the page."""

    def __init__(self, reader: Reader, versions: dict):
        self.reader, self.versions = reader, versions
        self.plan: dict[str, dict] = {}
        self.blobs: dict[str, bytes] = {}

    def _key(self, sha: str, args) -> str:
        return hashlib.sha256(json.dumps([sha, list(args), self.versions]).encode()).hexdigest()[:20]

    def video(self, src: Path) -> str:
        dst = "media/%s.mp4" % self._key(self.reader.sha256(src), VIDEO_ARGS)
        self.plan[dst] = {"kind": "video", "src": str(src), "args": list(VIDEO_ARGS)}
        return dst

    def audio(self, src: Path) -> str:
        dst = "media/%s.m4a" % self._key(self.reader.sha256(src), AUDIO_ARGS)
        self.plan[dst] = {"kind": "audio", "src": str(src), "args": list(AUDIO_ARGS)}
        return dst

    def image(self, src: Path, width: int = 1600) -> str:
        """A JPEG preview at most `width` pixels wide (a PNG grid can be several megabytes)."""
        dst = "media/%s.jpg" % self._key(self.reader.sha256(src), ["image", width])
        self.plan[dst] = {"kind": "image", "src": str(src), "width": width}
        return dst

    def image_data(self, content: bytes, width: int = 960) -> str:
        """An image made of bytes (one an agent looked at, inline in its trajectory) as a JPEG preview."""
        sha = hashlib.sha256(content).hexdigest()
        dst = "media/%s.jpg" % self._key(sha, ["image", width])
        self.plan[dst] = {"kind": "image", "src": None, "sha256": sha, "width": width}
        self.blobs[dst] = content
        return dst

    def poster(self, src: Path, width: int = 480) -> str:
        """A JPEG still from a third of the way into a video, for its thumbnail."""
        dst = "media/%s.jpg" % self._key(self.reader.sha256(src), ["poster", width])
        self.plan[dst] = {"kind": "poster", "src": str(src), "width": width}
        return dst

    def copy(self, src: Path) -> str:
        dst = "frames/%s%s" % (self.reader.sha256(src)[:20], Path(src).suffix.lower())
        self.plan[dst] = {"kind": "copy", "src": str(src)}
        return dst

    def data(self, content: bytes, suffix: str) -> str:
        """Bytes made here (a point cloud), as a file the page fetches when it needs it."""
        dst = "media/%s%s" % (hashlib.sha256(content).hexdigest()[:20], suffix)
        self.plan[dst] = {"kind": "bytes", "sha256": hashlib.sha256(content).hexdigest()}
        self.blobs[dst] = content
        return dst

    def json(self, path: Path):
        return self.reader.json(path)

    def bytes(self, path: Path) -> bytes:
        return self.reader.bytes(path)


# ------------------------------------------------------------------------------------- pure helpers


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def rubric_question(rubric: str, cid: str) -> str:
    """The question for check `cid`, from a numbered list item, a bullet, or a table row (with <play>-style templates)."""
    m = re.search(r"\*\*" + re.escape(cid) + r"\*\*[^:\n]*: (.+?)(?=\n\s*(?:\d+\.|-) \*\*|\n\n|\n##)", rubric, re.S)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    for line in rubric.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or not cells[0].startswith("`"):
            continue
        pattern = re.sub(r"<[^>]+>", "[^-]+", re.escape(cells[0].split("`")[1]).replace(r"\<", "<").replace(r"\>", ">"))
        if re.fullmatch(pattern, cid):
            return cells[1] + (f" Passes when: {cells[2]}" if len(cells) > 2 and cells[2] else "")
    return ""


def summarize(trials: list[dict]) -> dict:
    valid = [t["reward"] for t in trials if t["valid"]]
    return {"n": len(trials), "n_valid": len(valid), "mean": round(sum(valid) / len(valid), 4) if valid else None,
            "min": min(valid) if valid else None, "max": max(valid) if valid else None, "solved": sum(1 for v in valid if v >= 1.0)}


# ------------------------------------------------------------------------------------ collect


def collect_trial(rec: TrialRecord, task: Task, view, media: Media, rd: Reader, spec: dict, rubric: str) -> dict:
    score = rd.json(rec.verifier / "score.json") or {}
    judged = {}
    j = rd.json(rec.path / "judge" / "judge.json") if not rec.flat else None
    for q in (j or {}).get("questions", []):
        judged[q["id"]] = {"verdict": q["verdict"], "votes": q["votes"],
                           "answers": [{"model": a["model"], "answer": a["answer"],
                                        "reason": a.get("reason") or a.get("observation") or a.get("error", ""),
                                        **({"item": a["item"]} if a.get("item") else {})}  # a clips check's item
                                       for a in q["answers"]]}
    checks = []
    for c in score.get("checks", []):
        cid = c["id"]
        kind = ("gate" if cid in spec.get("gates", []) else "scored" if cid in spec.get("scored", [])
                else "process" if cid.startswith("process:") else "judge" if c.get("status") == "needs_judge" or cid.startswith(("judge:", "evidence:"))
                else "unscored")
        checks.append({"id": cid, "status": c["status"], "kind": kind, "question": rubric_question(rubric, cid),
                       "evidence": c.get("evidence"), "judged": judged.get(cid)})

    video, audio, texts, images, thumb, grid_png = None, [], [], [], None, None
    for decl in task.artifacts:
        local = rec.deliverables / decl
        files = sorted(local.rglob("*")) if local.is_dir() else [local]
        for f in (f for f in files if f.is_file()):
            name = str(f.relative_to(rec.deliverables))
            suffix = f.suffix.lower()
            if suffix in VIDEO_SUFFIXES:
                if video is None:
                    video, thumb = media.video(f), media.poster(f)
            elif suffix in AUDIO_SUFFIXES:
                audio.append({"name": name, "src": media.audio(f)})
            elif suffix in IMAGE_SUFFIXES:
                from PIL import Image
                with Image.open(f) as im:
                    size = list(im.size)
                images.append({"name": name, "src": media.image(f), "size": size})
                if thumb is None:
                    thumb, grid_png = media.image(f, 480), f
            elif suffix in TEXT_SUFFIXES and f.stat().st_size < 2_000_000:
                texts.append({"name": name, "text": rd.text(f)})

    process = {"kind": "agent", "commands": [], "final_msg": "", "images": []}
    traj = rd.json(rec.agent_dir / "trajectory.json")
    meta = rd.json(rec.agent_dir / "image-model.json") if not (rec.agent_dir / "trajectory.json").exists() else None
    if rec.flat and rec.executor == "image-runner":
        meta = rd.json(rec.path / "meta.json")
    if traj:
        t0 = None
        for s in traj.get("steps", []):
            t0 = t0 or ts(s.get("timestamp"))
            for tc in s.get("tool_calls") or []:
                arg = tc.get("arguments") or {}
                text = arg.get("cmd") or arg.get("command") or arg.get("input") or json.dumps(arg, sort_keys=True)
                text = " ".join(map(str, text)) if isinstance(text, list) else str(text)
                at = ts(s.get("timestamp"))
                process["commands"].append({"min": round((at - t0).total_seconds() / 60, 1) if at and t0 else None,
                                            "tool": tc.get("function_name", ""), "text": re.sub(r"^/bin/bash -lc ", "", text)[:600],
                                            "flag": any(p.search(text) for _, p in records.CONTAMINATION)})
        process["final_msg"] = next((s.get("message") for s in reversed(traj.get("steps", []))
                                     if s.get("source") == "agent" and isinstance(s.get("message"), str) and s.get("message")), "")
        process["trajectory"] = detail.trajectory(rec.agent_dir / "trajectory.json", rec.agent_dir, media)
    elif meta:
        process = {"kind": "image-model", "status": meta.get("status"), "request": meta.get("request"), "image": meta.get("image"),
                   "tries": meta.get("tries", []), "seconds": meta.get("seconds"), "usd": meta.get("usd"),
                   "usd_list_price": meta.get("usd_list_price"), "price_source": meta.get("price_source"),
                   "returned": meta.get("returned"), "usage": meta.get("usage")}
    elif rec.executor == "control":
        process = {"kind": "control", "what": ((task.calibration().get("runs") or {}).get(rec.model) or {}).get("what", "")}

    label = ((((task.calibration().get("runs") or {}).get(rec.model) or {}).get("label") or rec.model)
             if rec.executor == "control" else "%s %d" % (rec.subject, rec.attempt))
    judges = detail.stage_judges(rec.path, media)
    if judges is None and grid_png is not None and task.image:
        ev = rec.verifier / "evidence"
        evidence = {p.name: media.image(p, 1200) for p in sorted(ev.glob("*.png"))} if ev.is_dir() else {}
        ref = task.workspace / task.image.inputs[0] if task.image.inputs else None
        judges = detail.camera_judges(task, rec.verifier, grid_png, media, evidence,
                                      media.image(ref, 1200) if ref and ref.is_file() else None)
    return {"id": rec.trial, "run": rec.run, "subject": rec.subject, "attempt": rec.attempt, "label": label,
            "control": rec.executor == "control", "executor": rec.executor, "status": rec.status, "reward": rec.reward,
            "valid": rec.valid and rec.executor != "control", "details": rec.details, "checks": checks, "error": rec.error,
            "flags": rec.flags, "seconds": rec.seconds, "cost": rec.cost_usd, "tokens": rec.tokens, "started": rec.started,
            "metrics": {k: v for k, v in (rec.details or {}).items() if k not in {"reward", "deliverable_found", "inputs_unaltered",
                        "gates_passed", "checks_passed", "checks_scored", "checks_needing_judge"}},
            "judged": bool(judged), "video": video, "audio": audio, "texts": texts, "images": images, "process": process,
            "viz": view.trial_data(task, rec, media) if view else None, "thumb": thumb, "judges": judges}


SUMMARY = ("id", "run", "subject", "attempt", "label", "control", "executor", "status", "reward", "valid", "seconds", "cost",
           "started", "thumb", "flags")


def split(trial: dict, media: Media) -> dict:
    """What the overview needs inline; the rest (checks, media, trajectory, judges) in a file fetched when it opens."""
    out = {k: trial[k] for k in SUMMARY}
    out["error"] = (trial["error"] or {}).get("type") if trial["error"] else None
    out["failed"] = sorted(c["id"] for c in trial["checks"] if c["status"] == "fail")
    rest = {k: v for k, v in trial.items() if k not in SUMMARY}
    out["detail"] = media.data(dumps(rest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(), ".json")  # keys masked
    return out


def collect_task(task: Task, recs: list[TrialRecord], controls: list[TrialRecord], media: Media, rd: Reader) -> dict:
    view = view_for(task.view)
    spec = task.reward_spec()
    rubric = rd.text(task.tests / "rubric.md") if (task.tests / "rubric.md").exists() else ""
    rd.sha256(task.path / "task.toml")
    rd.sha256(task.tests / "answer_key.json")
    rank = lambda r: (r.subject in REFERENCES, r.subject == "nop", r.subject, r.attempt, r.trial)  # noqa: E731  (references last)
    trials = [split(collect_trial(r, task, view, media, rd, spec, rubric), media) for r in sorted(recs, key=rank)]
    graded = [split(collect_trial(r, task, view, media, rd, spec, rubric), media) for r in controls]
    calib = task.calibration().get("runs") or {}
    expected = task.expected_controls()
    in_run = {t["id"].split("__control-", 1)[1]: t for t in graded if "__control-" in t["id"]}
    ctl = []
    for case in sorted(set(expected) | set(in_run), key=lambda c: (c != "oracle", c == "no-deliverable", c)):
        rec = in_run.get(case)
        info = calib.get(case) or {}
        statuses = (expected.get(case) or {}).get("checks") or {}
        failed = rec["failed"] if rec else sorted(k for k, v in statuses.items() if v == "fail")
        ctl.append({"name": case, "label": info.get("label") or {"oracle": "Oracle (the reference answer)", "no-deliverable": "No deliverable"}.get(case, case),
                    "what": info.get("what", ""), "breaks": info.get("built_to_break", []),
                    "reward": rec["reward"] if rec else (expected.get(case) or {}).get("reward"),
                    "failed": failed, "graded_here": bool(rec),
                    "trial": rec["id"] if rec else None})
    summary = {}
    for t in trials:
        summary.setdefault(t["subject"], []).append(t)
    inputs = []
    for f in sorted(p for p in task.workspace.rglob("*") if p.is_file()) if task.workspace.is_dir() else []:
        item, suffix = {"name": str(f.relative_to(task.workspace)), "bytes": f.stat().st_size}, f.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            item["image"] = media.image(f, 1200)
        elif suffix in VIDEO_SUFFIXES:
            item["poster"] = media.poster(f, 320)
        elif suffix in AUDIO_SUFFIXES:
            item["audio"] = media.audio(f)
        elif suffix in TEXT_SUFFIXES and item["bytes"] < 200_000:
            item["text"] = rd.text(f)
        inputs.append(item)
    rep = task.report
    return {"id": task.name, "title": task.title, "short": rep.get("short") or task.title, "summary": rep.get("summary") or task.description,
            "tests": rep.get("tests", ""), "category": task.category, "family": task.family, "view": task.view, "split": task.split,
            "instruction": task.instruction, "inputs": inputs, "artifacts": task.artifacts, "gates": spec.get("gates", []), "scored": spec.get("scored", []),
            "viz": view.task_data(task, media, {c.model: c.path for c in controls}) if view else None,
            "trials": trials + graded, "controls": ctl, "subjects": {s: summarize(ts_) for s, ts_ in summary.items()}}


def model_label(model: str) -> str:
    """configs/models.toml's label for a model, else one read off its id: claude-sonnet-5-5 is Claude Sonnet 5.5,
    gpt-6-luna is GPT-6 Luna."""
    e = (registry.models_config().get("chat") or {}).get(model) or {}
    if e.get("label"):
        return e["label"]
    m = re.fullmatch(r"claude-([a-z]+)-(\d+)-(\d+)", model)
    if m:
        return "Claude %s %s.%s" % (m.group(1).capitalize(), m.group(2), m.group(3))
    m = re.fullmatch(r"gpt-([\d.]+)-([a-z]+)", model)
    if m:
        return "GPT-%s %s" % (m.group(1), m.group(2).capitalize())
    return model


def subject_names(subject: str, agent: str | None, model: str | None) -> dict:
    """A subject's name on the page (its model's label) and its harness (Codex, Claude Code; none for an image model)."""
    cfg = registry.models_config()
    prof = registry.agents_config().get(subject) or {}
    agent = prof.get("agent") or agent or ""
    model = (prof.get("model") or model or "").split("/")[-1]
    if subject in REFERENCES or agent in ("harbor:oracle", "harbor:nop", "oracle", "nop") or model == "oracle":
        return {"name": prof.get("label") or subject.capitalize(), "harness": None, "reference": True}
    if agent == "image-model":
        e = (cfg.get("image") or {}).get(model) or {}
        return {"name": e.get("label") or model, "harness": None, "reference": False}
    harness = HARNESSES.get(agent.removeprefix("harbor:"), HARNESSES.get(agent, agent.removeprefix("harbor:")))
    return {"name": model_label(model) if model else subject, "harness": harness, "reference": False}


def interval(per_task: list[list[float]], rng) -> tuple[float, float]:
    """95% bootstrap interval of the mean over tasks, resampling attempts within each task."""
    means = sorted(sum(sum(rng.choices(xs, k=len(xs))) / len(xs) for xs in per_task) / len(per_task) for _ in range(BOOTSTRAP))
    return means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP) - 1]


def benchmark(tasks: list[dict], subjects: dict) -> list[dict]:
    """Per category: each subject's mean reward (every task weighted equally, with a bootstrap interval over
    attempts), cost and time per attempt. An attempt with an infrastructure error or reaching outside its
    container is left out, not counted as 0."""
    import random
    scopes = []
    for cat in sorted({t["category"] for t in tasks}, key=lambda c: list(CATEGORIES).index(c) if c in CATEGORIES else 9):
        members = [t for t in tasks if t["category"] == cat]
        points = []
        for s in sorted({x["subject"] for t in members for x in t["trials"] if not x["control"]}):
            if subjects.get(s, {}).get("reference"):
                continue
            per_task, usd, secs, n = [], [], [], 0
            for t in members:
                mine = [x for x in t["trials"] if x["subject"] == s and not x["control"] and x["status"] != "error" and not x["flags"]]
                if mine:
                    per_task.append([x["reward"] or 0.0 for x in mine])
                    usd += [x["cost"] for x in mine if x["cost"] is not None]
                    secs += [x["seconds"].get("agent") for x in mine if (x["seconds"] or {}).get("agent") is not None]
                    n += len(mine)
            if not per_task:
                continue
            lo, hi = interval(per_task, random.Random(hash((cat, s)) & 0xFFFF))
            points.append({"subject": s, "mean": sum(sum(r) / len(r) for r in per_task) / len(per_task), "lo": lo, "hi": hi,
                           "usd": sum(usd) / len(usd) if usd else None, "seconds": sum(secs) / len(secs) if secs else None,
                           "attempts": n, "tasks": len(per_task)})
        if points:
            scopes.append({"id": cat, "label": CATEGORIES.get(cat, cat), "tasks": [t["id"] for t in members],
                           "points": sorted(points, key=lambda p: -p["mean"])})
    return scopes


def run_info(path: Path, info: dict) -> dict:
    """What the page shows of a run.json, filled in for the runs camera-eval bench wrote before evals. Not its
    command line, git state or notes: those name this machine's paths, and the page is meant to be published."""
    started = info.get("started")
    if isinstance(started, (int, float)):
        started = datetime.fromtimestamp(started).astimezone().isoformat(timespec="seconds")
    return {"name": info.get("name") or path.name, "title": info.get("title") or info.get("suite") or path.name,
            "executor": info.get("executor") or ("harbor" if (path / "harbor").is_dir() else "?"), "attempts": info.get("attempts") or 1,
            "tasks": info.get("tasks") or [], "network": info.get("network", "enforced"), "subjects": info.get("subjects") or [],
            "started": started}


def collect(paths: list[Path], versions: dict, title: str | None = None) -> tuple[dict, dict, dict, dict]:
    """Read every trial and its task. Returns (data, media plan, input hashes, blobs); writes nothing."""
    rd = Reader(ROOT)
    media = Media(rd, versions)
    recs = records.find(paths)
    runs = [run_info(Path(p), info) for p in paths if (info := rd.json(Path(p) / "run.json"))]
    by_task: dict[str, list[TrialRecord]] = {}
    for r in recs:
        by_task.setdefault(r.task, []).append(r)
    order = [t.name for t in discover("eval")] + [t.name for t in discover("train")]
    tasks, skipped = [], []
    for name in sorted(by_task, key=lambda n: (order.index(n) if n in order else len(order), n)):
        path = records.task_dir(name)
        if path is None:
            skipped.append(name)
            continue
        task = Task(path)
        controls = [c for p in paths for c in records.controls_of(Path(p), task)]
        tasks.append(collect_task(task, by_task[name], controls, media, rd))
    subjects = {}
    for info in runs:
        for s in info.get("subjects", []):
            subjects[s["name"]] = {"label": s.get("label") or s["name"], "agent": s.get("agent"), "model": s.get("model")}
    for r in recs:
        if r.subject not in subjects:
            label = registry.models_config().get("image", {}).get(r.subject, {}).get("label") if r.agent == "image-model" else None
            subjects[r.subject] = {"label": label or r.subject, "agent": r.agent, "model": r.model}
    for s, info in subjects.items():
        info.update(subject_names(s, info.get("agent"), info.get("model")))
    caveats = []
    for info in runs:
        if str(info.get("network", "enforced")).startswith("open"):
            caveats.append("Run %s used copies of the tasks with the network open: %s. An image model only drops a PNG into the "
                           "container, so it is unaffected; a harness inside the container could reach the network." % (
                               info["name"], info["network"].split(":", 1)[-1].strip()))
    errors = [r for r in recs if r.status == "error"]
    if errors:
        caveats.append("%d trial(s) ended in an infrastructure error and have no reward: %s." % (
            len(errors), "; ".join("%s %s: %s" % (r.task, r.subject, (r.error or {}).get("type")) for r in errors[:8])))
    flagged = [r for r in recs if r.flags]
    if flagged:
        caveats.append("%d run(s) reached outside their container (%s) and are left out of the means." % (
            len(flagged), "; ".join(sorted({f for r in flagged for f in r.flags}))))
    if skipped:
        caveats.append("Trials of tasks not on this branch are left out: %s." % ", ".join(sorted(skipped)))
    order = sorted({r.subject for r in recs}, key=lambda s: (s in REFERENCES, s == "nop", s))  # the page's sorted keys lose it
    names = [info.get("title") or info.get("name") for info in runs]
    data = {"title": title or (names[0] if len(names) == 1 else "Benchmark"), "runs": runs, "subjects": subjects, "order": order,
            "tasks": tasks, "caveats": caveats, "views": sorted({t["view"] for t in tasks if t["view"]}),
            "paths": [rel(p) for p in paths], "benchmark": benchmark(tasks, subjects)}
    return data, dict(sorted(media.plan.items())), dict(sorted(rd.hashes.items())), media.blobs


# ------------------------------------------------------------------------------------- render


def view_scripts(names: list[str]) -> str:
    """Each view module's script, once, in a stable order."""
    from evals.report.views import VIEWS
    modules = sorted({VIEWS[n].split(":")[0] for n in names if n in VIEWS})
    return "\n".join((HERE / "views" / ("%s.js" % m.rsplit(".", 1)[1])).read_text() for m in modules)


def render(data: dict, shell: str, scripts: str) -> str:
    """The page: the shell with the data and the views inlined. `</` is escaped so no string can close a script."""
    payload = dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    return shell.replace("__VIEWS__", scripts.replace("</script", "<\\/script")).replace("__DATA__", payload)


def generator_hashes(scripts: str) -> dict:
    files = [Path(__file__), SHELL, *sorted((HERE / "views").glob("*.py")), HERE.parent / "core" / "records.py"]
    out = {rel(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    out["views.js"] = hashlib.sha256(scripts.encode()).hexdigest()
    return out


# -------------------------------------------------------------------------------- materialize


def versions() -> dict:
    out = {}
    try:
        out["ffmpeg"] = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        out["ffmpeg"] = None
    try:
        import PIL
        out["pillow"] = PIL.__version__
    except ImportError:
        out["pillow"] = None
    return out


def ensure_cached(dst: str, item: dict, cache: Path, blobs: dict) -> Path:
    """The cached file for one plan item, made first if missing (atomically)."""
    target = cache / Path(dst).name
    if target.exists():
        return target
    cache.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=cache, suffix=target.suffix)
    os.close(fd)
    try:
        if item["kind"] == "copy":
            shutil.copyfile(item["src"], tmp)
        elif item["kind"] == "bytes":
            Path(tmp).write_bytes(blobs[dst])
        elif item["kind"] == "image":
            from PIL import Image
            with Image.open(item["src"] if item.get("src") else io.BytesIO(blobs[dst])) as im:
                im = im.convert("RGB")
                if im.width > item["width"]:
                    im = im.resize((item["width"], round(im.height * item["width"] / im.width)), Image.Resampling.LANCZOS)
                im.save(tmp, "JPEG", quality=85)
        elif item["kind"] == "poster":
            probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", item["src"]],
                                   capture_output=True, text=True)
            at = float(probe.stdout.strip() or 0) / 3
            subprocess.run(["nice", "ffmpeg", "-v", "error", "-y", "-ss", "%.2f" % at, "-i", item["src"], "-frames:v", "1",
                            "-vf", "scale=%d:-2" % item["width"], "-q:v", "4", tmp], check=True)
        else:
            subprocess.run(["nice", "ffmpeg", "-v", "error", "-y", "-threads", "4", "-i", item["src"], *item["args"], tmp], check=True)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return target


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def materialize(html: str, plan: dict, blobs: dict, manifest: dict, out: Path, name: str, cache: Path) -> Path:
    """Write the report as a new folder <out>/<name>-<digest>/ and point <out>/<name> at it."""
    final = out / ("%s-%s" % (name, manifest["digest"][:12]))
    if not final.exists():
        out.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=out, prefix=".%s-" % name))
        try:
            for i, (dst, item) in enumerate(plan.items(), 1):
                print("  [%d/%d] %s <- %s" % (i, len(plan), dst, Path(item.get("src") or "data").name), file=sys.stderr)
                cached = ensure_cached(dst, item, cache, blobs)
                (tmp / dst).parent.mkdir(parents=True, exist_ok=True)
                link_or_copy(cached, tmp / dst)
            (tmp / "index.html").write_text(html)
            (tmp / "manifest.json").write_text(dumps(manifest, indent=1, sort_keys=True) + "\n")
            os.chmod(tmp, 0o755)
            os.rename(tmp, final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    link = out / name
    tmp_link = out / (".%s.link" % name)
    if tmp_link.is_symlink() or tmp_link.exists():
        tmp_link.unlink()
    tmp_link.symlink_to(final.name)
    os.replace(tmp_link, link)
    return final


def build(paths: list[Path], out: Path | None = None, name: str = "report", cache: Path | None = None,
          title: str | None = None) -> Path:
    """Build the page for these runs (or job or trial folders); returns its index.html."""
    load_env_files(None)  # .env's keys too, so the page masks every one an agent printed
    paths = [Path(p).resolve() for p in paths]
    v = versions()
    data, plan, inputs, blobs = collect(paths, v, title)
    if not data["tasks"]:
        raise SystemExit("no trials of a known task under " + " ".join(rel(p) for p in paths))
    if any(item["kind"] in ("video", "audio") for item in plan.values()) and not v["ffmpeg"]:
        raise SystemExit("the report needs ffmpeg for its video and audio previews")
    scripts = view_scripts(data["views"])
    generator = generator_hashes(scripts)
    digest = hashlib.sha256(json.dumps([inputs, generator, v], sort_keys=True).encode()).hexdigest()
    data["build"] = {"digest": digest[:12], "trials": sum(len(t["trials"]) for t in data["tasks"])}
    manifest = {"digest": digest, "generator": generator, "versions": v, "inputs": inputs, "paths": data["paths"],
                "media": {dst: {k: (rel(x) if k == "src" and x else x) for k, x in item.items()} for dst, item in plan.items()}}
    html = render(data, SHELL.read_text(), scripts)
    final = materialize(html, plan, blobs, manifest, out or JOBS / "reports", name, cache or resolve(settings()["report"]["cache"]))
    return final / "index.html"
