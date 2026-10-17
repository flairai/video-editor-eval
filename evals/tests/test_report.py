"""The page's own reading and arithmetic, with no runs, services or keys: an agent's trajectory (images it looked
at, as files or inline), the benchmark's means and intervals, and the names the page gives each subject.

    uv run pytest evals/tests/test_report.py
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from PIL import Image

from evals.report import build, detail


def jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 40, 40)).save(buf, "JPEG")
    return buf.getvalue()


def test_a_trajectory_keeps_every_step_and_the_images_in_place(tmp_path):
    agent = tmp_path / "agent"
    (agent / "images").mkdir(parents=True)
    (agent / "images" / "seen.jpg").write_bytes(jpeg())
    inline = json.dumps({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(jpeg()).decode()}})
    steps = [
        {"timestamp": "2026-10-17T10:00:00Z", "source": "user", "message": "Cut the footage."},
        {"timestamp": "2026-10-17T10:00:05Z", "source": "agent", "message": "Looking at the clips.", "reasoning_content": "First the footage.",
         "tool_calls": [{"function_name": "Bash", "arguments": {"command": "ls footage"}}],
         "observation": {"results": [{"content": [{"type": "text", "text": "clip_1.mp4"},
                                                  {"type": "image", "source": {"media_type": "image/jpeg", "path": "images/seen.jpg"}}]}]},
         "metrics": {"prompt_tokens": 1000, "completion_tokens": 20}},
        {"timestamp": "2026-10-17T10:01:05Z", "source": "agent", "message": "",
         "tool_calls": [{"function_name": "Read", "arguments": {"file_path": "/tmp/frame.jpg"}}],
         "observation": {"results": [{"content": inline}]}},
    ]
    (agent / "trajectory.json").write_text(json.dumps({"steps": steps, "final_metrics": {"total_prompt_tokens": 1000}}))
    media = build.Media(build.Reader(tmp_path), {})
    t = detail.trajectory(agent / "trajectory.json", agent, media)
    assert [s["src"] for s in t["steps"]] == ["user", "agent", "agent"] and [s["t"] for s in t["steps"]] == [0.0, 5.0, 65.0]
    second = t["steps"][1]
    assert second["think"] == "First the footage." and second["calls"] == [{"tool": "Bash", "input": "ls footage"}]
    assert second["out"][0] == {"text": "clip_1.mp4"} and second["out"][1]["image"].startswith("media/") and second["tokens"] == [1000, 20]
    third = t["steps"][2]
    assert third["calls"][0]["input"] == "/tmp/frame.jpg" and third["out"][0]["image"].startswith("media/")  # decoded from base64
    item = media.plan[third["out"][0]["image"]]
    assert item["kind"] == "image" and item["src"] is None and t["seconds"] == 65.0  # a JPEG preview made from the bytes


def trial(subject, reward, cost=1.0, seconds=60.0, status="graded", flags=()):
    return {"subject": subject, "reward": reward, "cost": cost, "seconds": {"agent": seconds}, "status": status, "flags": list(flags),
            "control": False}


def test_the_benchmark_weighs_tasks_equally_and_leaves_errors_out():
    tasks = [{"id": "a", "category": "video-editing", "trials": [trial("s", 1.0), trial("s", 0.0), trial("s", None, status="error")]},
             {"id": "b", "category": "video-editing", "trials": [trial("s", 0.5, cost=3.0), trial("s", 0.5, flags=["curl"])]},
             {"id": "c", "category": "video-editing", "trials": [trial("oracle", 1.0)]}]
    [scope] = build.benchmark(tasks, {"s": {"reference": False}, "oracle": {"reference": True}})
    [p] = scope["points"]
    assert p["subject"] == "s" and p["mean"] == 0.5 and p["attempts"] == 3  # (0.5 + 0.5) / 2: the error and the flagged run are out
    assert p["usd"] == (1.0 + 1.0 + 3.0) / 3 and p["lo"] <= 0.5 <= p["hi"] and scope["tasks"] == ["a", "b", "c"]


def test_subjects_are_named_by_their_model_and_harness():
    assert build.subject_names("claude-code · claude-sonnet-5-5", "claude-code", "claude-sonnet-5-5") == \
        {"name": "Claude Sonnet 5.5", "harness": "Claude Code", "reference": False}
    assert build.subject_names("codex · gpt-6.1-sol", "codex", "gpt-6.1-sol")["name"] == "GPT-6.1 Sol"
    assert build.subject_names("gpt-image", "image-model", "gpt-image-2.5-sunburst") == \
        {"name": "GPT Image 2.5 Sunburst", "harness": None, "reference": False}
    assert build.subject_names("oracle", "harbor:oracle", None)["reference"] is True


def test_the_page_carries_no_glyphs():
    """The page says pass and fail in words: no check marks, crosses, bullets, middle dots, arrows or ellipses."""
    here = Path(build.__file__).parent
    for f in [here / "shell.html", *sorted((here / "views").glob("*.js"))]:
        text = f.read_text()
        assert not any(g in text for g in "✓✗✕◌⚠•·→←…›‹"), f.name


def test_an_attempts_detail_file_masks_keys(tmp_path, monkeypatch):
    """A key an agent printed in a tool's output never reaches the page's files."""
    monkeypatch.setenv("MY_SERVICE_API_KEY", "sk-test-0123456789abcdef")
    media = build.Media(build.Reader(tmp_path), {})
    trial = {k: None for k in build.SUMMARY} | {"error": None, "checks": [], "flags": [],
                                                 "process": {"trajectory": {"steps": [{"out": [{"text": "env: sk-test-0123456789abcdef"}]}]}}}
    out = build.split(trial, media)
    written = media.blobs[out["detail"]].decode()
    assert "sk-test-0123456789abcdef" not in written and "[MY_SERVICE_API_KEY]" in written
