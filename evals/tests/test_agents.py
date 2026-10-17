"""Our agents against a LocalSandbox (a folder standing in for /workspace) and a scripted chat model."""

from __future__ import annotations

import asyncio
import io
import json

from PIL import Image

from evals.agents.base import LocalSandbox
from evals.agents.tool_loop import ToolLoopAgent
from evals.core.interfaces import ChatModel, ChatResponse, Message, ToolCall, Usage


class Scripted(ChatModel):
    """Replies with the tool calls it is given, one turn each, and records what it was sent."""

    def __init__(self, turns):
        super().__init__("scripted", {"usd_per_mtok": [1.0, 2.0]})
        self.turns, self.seen = list(turns), []

    def complete(self, req):
        req.messages = list(req.messages)  # what was sent then: the agent keeps appending to its own list
        self.seen.append(req)
        calls = self.turns.pop(0)
        usage = Usage(100, 10)
        return ChatResponse("", Message("assistant", [], [ToolCall("c%d" % len(self.seen), n, a) for n, a in calls]), usage, self.cost(usage))


def test_tool_loop_runs_commands_looks_and_finishes(tmp_path):
    work, logs = tmp_path / "workspace", tmp_path / "logs"
    work.mkdir()
    buf = io.BytesIO()
    Image.new("RGB", (64, 32), (0, 200, 0)).save(buf, format="PNG")
    (work / "frame.png").write_bytes(buf.getvalue())
    llm = Scripted([[("shell", {"command": "echo hello > /workspace/out.txt && ls /workspace"})],
                    [("look", {"paths": ["frame.png", "missing.png"]}), ("write_file", {"path": "notes/a.md", "content": "# hi"})],
                    [("done", {"report": "made out.txt"})]])
    agent = ToolLoopAgent(llm, {"max_steps": 5}, logs)
    asyncio.run(agent.run("Make out.txt.", LocalSandbox(work)))
    agent.finish()
    assert (work / "out.txt").read_text() == "hello\n" and (work / "notes/a.md").read_text() == "# hi"
    assert agent.final_message == "made out.txt"
    second = llm.seen[1].messages  # the shell's output came back as a tool result
    assert second[-1].role == "tool" and "out.txt" in second[-1].parts[0].data and "exit code 0" in second[-1].parts[0].data
    third = llm.seen[2].messages
    look = [m for m in third if m.role == "tool"][-2]
    assert [p.kind for p in look.parts] == ["text", "image", "text"] and "cannot show it" in look.parts[2].data
    traj = json.loads((logs / "trajectory.json").read_text())
    assert traj["schema_version"].startswith("ATIF") and traj["final_metrics"]["total_prompt_tokens"] == 300
    assert [t["function_name"] for s in traj["steps"] for t in s.get("tool_calls", [])] == ["shell", "look", "write_file", "done"]
    assert traj["steps"][-1]["message"] == "made out.txt" and list((logs / "images").glob("*.png"))
    assert agent.spent_usd == round(3 * (100 * 1 + 10 * 2) / 1e6, 6)


def test_tool_loop_stops_at_max_steps(tmp_path):
    llm = Scripted([[("shell", {"command": "true"})]] * 3)
    agent = ToolLoopAgent(llm, {"max_steps": 2})
    asyncio.run(agent.run("x", LocalSandbox(tmp_path)))
    assert len(llm.seen) == 2 and "max_steps" in agent.final_message


def test_an_unknown_option_is_an_error():
    import pytest
    with pytest.raises(Exception, match="takes no option"):
        ToolLoopAgent(None, {"max_stepz": 3})


def test_a_failing_tool_goes_back_to_the_model(tmp_path):
    llm = Scripted([[("look", {"paths": ["/nowhere/x.png"]}), ("nonsense", {})], [("done", {"report": ""})]])
    asyncio.run(ToolLoopAgent(llm, {}).run("x", LocalSandbox(tmp_path)))
    results = [m for m in llm.seen[1].messages if m.role == "tool"]
    assert "cannot show it" in results[0].parts[0].data and "no tool named" in results[1].parts[0].data
