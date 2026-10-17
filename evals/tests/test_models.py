"""The providers' requests and replies, with fake clients: nothing is sent.

What each judge request carries is pinned to the requests the code sent before the evals package existed
(checked byte for byte when the clients moved); these tests keep the shape and the tool-calling
translation from drifting.
"""

from __future__ import annotations

import base64
import io
import json

import pytest
from PIL import Image

from evals.agents.image_model import make_image
from evals.core import registry
from evals.core.interfaces import ChatRequest, ImageModel, ImageRequest, ImageResult, Message, Part, ProviderError, Tool
from evals.judging.panel import ANSWER_SCHEMA, Panel, Question


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def model_dump(self, exclude_none=False):
        return {k: v for k, v in self.__dict__.items() if not (exclude_none and v is None)}


def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


ANSWER = json.dumps({"cells": [{"cell": 1, "seen": "x"}], "answer": "yes", "reason": "r"})


@pytest.fixture
def calls():
    return []


def fake(model_name: str, calls: list, reply):
    model = registry.chat_model(model_name)
    old = model._client

    class Messages:
        def create(self, **kw):
            calls.append(kw)
            return reply(kw)

    model._client = Obj(messages=Messages(), responses=Messages(), models=Obj(generate_content=Messages().create),
                        chat=Obj(completions=Messages()))
    return model, old


def test_judge_request_anthropic(calls):
    model, old = fake("claude-opus-5-5", calls, lambda kw: Obj(stop_reason="end_turn", content=[Obj(type="text", text=ANSWER)],
                                                                usage=Obj(input_tokens=1000, output_tokens=100)))
    try:
        q = Question("judge:x", "SYSTEM", [Part.text("Cell 1:"), Part.image(png()), Part.text("Question: q?")], "q?")
        a = Panel(["claude-opus-5-5"]).ask("claude-opus-5-5", q)
    finally:
        model._client = old
    kw = calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["system"] == "SYSTEM" and kw["max_tokens"] == 16000
    assert kw["output_config"] == {"effort": "medium", "format": {"type": "json_schema", "schema": ANSWER_SCHEMA}}
    content = kw["messages"][0]["content"]
    assert content[1]["type"] == "image" and content[1]["source"]["media_type"] == "image/png"
    assert a["answer"] == "yes" and a["usd"] == round((1000 * 4 + 100 * 20) / 1e6, 5)


def test_judge_request_openai(calls):
    model, old = fake("gpt-6.1-sol", calls, lambda kw: Obj(status="completed", output_text=ANSWER, output=[],
                                                            usage=Obj(input_tokens=10, output_tokens=5, input_tokens_details=None)))
    try:
        Panel(["gpt-6.1-sol"]).ask("gpt-6.1-sol", Question("judge:x", "S", [Part.image(png())], "q"))
    finally:
        model._client = old
    kw = calls[0]
    assert kw["instructions"] == "S" and kw["reasoning"] == {"effort": "medium"} and kw["max_output_tokens"] == 16000
    assert kw["text"]["format"] == {"type": "json_schema", "name": "judge_answer", "schema": ANSWER_SCHEMA, "strict": True}
    assert kw["input"][0]["content"][0]["detail"] == "high"


def test_a_failed_request_is_never_a_yes(calls):
    model, old = fake("claude-opus-5-5", calls, lambda kw: Obj(stop_reason="max_tokens", content=[], usage=Obj(input_tokens=1, output_tokens=1)))
    try:
        out = Panel(["claude-opus-5-5"]).judge([Question("judge:x", "S", [Part.text("q")], "q")])
    finally:
        model._client = old
    assert out[0]["answers"][0]["answer"] == "error" and out[0]["verdict"] == "no"


def test_anthropic_tool_calls_round_trip(calls):
    def reply(kw):
        return Obj(stop_reason="tool_use", content=[Obj(type="text", text="let me look"), Obj(type="tool_use", id="t1", name="shell", input={"command": "ls"})],
                   usage=Obj(input_tokens=5, output_tokens=5, cache_read_input_tokens=2))
    model, old = fake("claude-opus-5-5", calls, reply)
    try:
        tools = [Tool("shell", "run", {"type": "object", "properties": {"command": {"type": "string"}}})]
        r = model.complete(ChatRequest([Message("user", [Part.text("go")])], tools=tools))
        assert [(c.id, c.name, c.arguments) for c in r.tool_calls] == [("t1", "shell", {"command": "ls"})]
        msgs = [Message("user", [Part.text("go")]), r.message, Message("tool", [Part.text("a.txt")], tool_call_id="t1"),
                Message("tool", [Part.text("b"), Part.image(png())], tool_call_id="t2")]
        model.complete(ChatRequest(msgs, tools=tools))
    finally:
        model._client = old
    sent = calls[1]["messages"]
    assert sent[1]["role"] == "assistant" and sent[1]["content"][1]["type"] == "tool_use"
    assert len(sent) == 3 and [b["tool_use_id"] for b in sent[2]["content"]] == ["t1", "t2"]  # one user turn holds every result
    assert sent[2]["content"][1]["content"][1]["type"] == "image"


def test_openai_compatible_tool_calls(calls, monkeypatch):
    from evals.models.openai_compat import OpenAICompatChat
    m = OpenAICompatChat("mine", {"impl": "openai_compat", "base_url": "http://localhost:9/v1", "model": "my-model", "usd_per_mtok": [1, 2]})

    def create(**kw):
        calls.append(kw)
        msg = Obj(content=None, tool_calls=[Obj(id="c1", function=Obj(name="look", arguments='{"paths": ["f.png"]}'))])
        return Obj(choices=[Obj(message=msg, finish_reason="tool_calls")], usage=Obj(prompt_tokens=3, completion_tokens=4, prompt_tokens_details=None))
    m._client = Obj(chat=Obj(completions=Obj(create=create)))
    r = m.complete(ChatRequest([Message("user", [Part.text("hi")])], system="S", tools=[Tool("look", "d", {"type": "object"})]))
    assert r.tool_calls[0].arguments == {"paths": ["f.png"]} and r.usd == round((3 * 1 + 4 * 2) / 1e6, 5)
    m.complete(ChatRequest([Message("user", [Part.text("hi")]), r.message, Message("tool", [Part.text("ok"), Part.image(png())], tool_call_id="c1")]))
    sent = calls[1]["messages"]
    assert sent[1]["tool_calls"][0]["function"]["name"] == "look" and sent[2] == {"role": "tool", "tool_call_id": "c1", "content": "ok"}
    assert sent[3]["role"] == "user" and sent[3]["content"][0]["type"] == "image_url"  # a tool message is text: its image follows


def test_fal_reference_is_uploaded_once_and_deleted(monkeypatch, tmp_path):
    import evals.models.fal as F
    import evals.models.http as H
    monkeypatch.setenv("FAL_KEY", "k")
    seen = []

    def http(method, url, headers, body=None, timeout=600.0):
        seen.append(url)
        if "auth/token" in url:
            return 200, b'{"token": "t", "token_type": "Bearer"}'
        if "files/upload" in url:
            return 200, b'{"access_url": "https://cdn/ref.png"}'
        if url.endswith("/status"):
            return 200, b'{"status": "COMPLETED"}'
        if url == "https://cdn/out.png":
            return 200, png()
        if "files/delete" in url:
            return 200, b"{}"
        if url.startswith(F.FAL_QUEUE) and method == "POST":
            return 200, b'{"status_url": "https://q/s/status", "response_url": "https://q/r"}'
        return 200, b'{"images": [{"url": "https://cdn/out.png", "width": 4, "height": 4}]}'
    for mod in (H, F):
        monkeypatch.setattr(mod, "http", http)
    ref = tmp_path / "reference.png"
    ref.write_bytes(png())
    req = ImageRequest(task="t", prompt="a `b`", references=(ref,), aspect="16:9")
    for name in ("seedream-5.0-pro", "uni-1-max"):
        registry.image_model(name).generate(req)
    assert sum("files/upload" in u for u in seen) == 1  # one upload per reference, shared by the fal models
    registry.image_model("uni-1-max").release(req)
    assert any("files/delete" in u for u in seen) and not F._uploads


def test_nano_banana_request_cost_and_unsupported_aspect(monkeypatch, tmp_path):
    import evals.models.google as G
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    sent = []
    data = base64.b64encode(png()).decode()

    def post_json(url, headers, payload, timeout=600.0):
        sent.append((url, headers, payload))
        return {"id": "i1", "status": "completed",
                "usage": {"total_tokens": 4000, "total_input_tokens": 1500, "total_output_tokens": 2000, "total_thought_tokens": 500,
                          "output_tokens_by_modality": [{"modality": "image", "tokens": 1680}]},
                "steps": [{"type": "thought", "signature": "s"},
                          {"type": "model_output", "content": [{"type": "image", "mime_type": "image/jpeg", "data": "interim"},
                                                               {"type": "image", "mime_type": "image/jpeg", "data": data}]}]}
    monkeypatch.setattr(G, "post_json", post_json)
    ref = tmp_path / "reference.png"
    ref.write_bytes(png())
    model = registry.image_model("nano-banana-2.1")
    out = model.generate(ImageRequest(task="t", prompt="p", references=(ref,), aspect="16:9"))
    url, headers, body = sent[0]
    assert url.endswith("/v1beta/interactions") and headers == {"x-goog-api-key": "k"} and body["model"] == "gemini-nano-banana-2.1"
    assert body["response_format"] == {"type": "image", "aspect_ratio": "16:9", "image_size": "2K", "mime_type": "image/jpeg"}
    assert body["generation_config"] == {"thinking_level": "high"}
    assert body["input"][0] == {"type": "text", "text": "p"} and body["input"][1]["type"] == "image"
    assert out.image == png()  # the last image of the output
    assert out.usd == round(1500 * 1.5e-6 + 1680 * 30e-6 + (4000 - 1500 - 1680) * 7.5e-6, 4)
    model.generate(ImageRequest(task="t", prompt="p", aspect="1:1"))
    assert sent[1][2]["input"] == "p"  # no reference: the prompt alone
    png_bytes, meta = make_image(model, ImageRequest(task="t", prompt="p", aspect="2:1"), sleep=lambda s: None)
    assert png_bytes is None and meta["status"] == "no_image" and len(sent) == 2  # 2:1 is never sent
    assert "no 2:1 output" in meta["tries"][0]["error"] and model.list_price(ImageRequest(task="t", prompt="p", aspect="2:1")) == 0.0


class Flaky(ImageModel):
    def __init__(self, errors):
        super().__init__("flaky", {})
        self.errors = errors

    def generate(self, req):
        if self.errors:
            raise self.errors.pop(0)
        return ImageResult(png(), {"request": req.meta()}, 0.01)


def test_make_image_retries_transient_errors_then_raises():
    png_bytes, meta = make_image(Flaky([ProviderError("429", True, 429)]), ImageRequest(task="t", prompt="p"), sleep=lambda s: None)
    assert png_bytes and meta["status"] == "image" and len(meta["tries"]) == 2 and meta["usd"] == 0.01
    with pytest.raises(ProviderError) as err:
        make_image(Flaky([ProviderError("503", True, 503)] * 4), ImageRequest(task="t", prompt="p"), sleep=lambda s: None)
    assert err.value.meta["status"] == "error"
    png_bytes, meta = make_image(Flaky([ProviderError("refused", False, 400)]), ImageRequest(task="t", prompt="p"), sleep=lambda s: None)
    assert png_bytes is None and meta["status"] == "no_image"


def test_to_png_keeps_pixels(tmp_path):
    from evals.agents.image_model import to_png
    buf = io.BytesIO()
    Image.new("RGB", (8, 6), (200, 10, 10)).save(buf, format="JPEG", quality=95)
    out, info = to_png(buf.getvalue())
    assert info == {"format": "JPEG", "mode": "RGB", "size": [8, 6]} and out[:8] == b"\x89PNG\r\n\x1a\n"
    assert to_png(png())[0] == png()  # a PNG is kept byte for byte
