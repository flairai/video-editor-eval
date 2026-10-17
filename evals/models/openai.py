"""OpenAI: GPT Image (an image model, over plain HTTP) and the Responses API (a chat model, through the
openai SDK), as configs/models.toml names them with impl = "gpt-image" and impl = "openai"."""

from __future__ import annotations

import base64
import json
import os
import threading

from evals.core.interfaces import (ChatModel, ChatRequest, ChatResponse, ImageModel, ImageRequest, ImageResult,
                                   Message, Part, ToolCall, Usage)
from evals.models.http import http, multipart, post_json, require_key

OPENAI_URL = "https://api.openai.com/v1"

# GPT Image 2.5 Sunburst, checked 2026-10-16 on developers.openai.com (models/gpt-image-2.5-sunburst,
# guides/image-generation, reference/resources/images, pricing). Sizes: WxH, edges multiples of 16, the
# smallest at each aspect that meets the instructions' minimum, so 16:9 is 2048x1152. Price: $30/M image
# output tokens, $8/M image input, $5/M text input; output tokens at quality high from OpenAI's own
# calculator. A reference image costs about 2,048 input tokens.
GPT_IMAGE_SIZES = {"1:1": "1536x1536", "2:1": "2048x1024", "16:9": "2048x1152"}
GPT_IMAGE_HIGH_USD = {"1:1": 0.0753, "2:1": 0.0354, "16:9": 0.0424}
GPT_IMAGE_REFERENCE_USD = 2048 * 8e-6


def gpt_image_usd(usage: dict | None) -> float | None:
    if not usage:
        return None
    details = usage.get("input_tokens_details") or {}
    return round(usage.get("output_tokens", 0) * 30e-6 + details.get("image_tokens", 0) * 8e-6 +
                 details.get("text_tokens", 0) * 5e-6, 4)


class GPTImage(ImageModel):
    """images/generations without a reference, images/edits with one. Entry options: quality (high), and
    usd_by_aspect, the list price of one image at each aspect for a model whose output tokens per size differ
    from Sunburst's (GPT_IMAGE_HIGH_USD). Either way an attempt costs what its returned usage comes to."""

    provider = "openai"

    def list_price(self, req: ImageRequest) -> float:
        table = self.entry.get("usd_by_aspect") or GPT_IMAGE_HIGH_USD
        return table[req.aspect] + (GPT_IMAGE_REFERENCE_USD if req.reference else 0.0)

    def generate(self, req: ImageRequest) -> ImageResult:
        size, quality = GPT_IMAGE_SIZES[req.aspect], self.entry.get("quality", "high")
        headers = {"Authorization": "Bearer " + require_key(self.key_env or "OPENAI_API_KEY")}
        fields = {"model": self.model, "prompt": req.prompt, "size": size, "quality": quality, "output_format": "png", "n": 1}
        if req.reference:
            body, ctype = multipart([(k, str(v)) for k, v in fields.items()],
                                    [("image[]", req.reference.name, req.reference.read_bytes(), "image/png")])
            _, raw = http("POST", f"{OPENAI_URL}/images/edits", {**headers, "Content-Type": ctype}, body)
            value, endpoint = json.loads(raw), "images/edits"
        else:
            value, endpoint = post_json(f"{OPENAI_URL}/images/generations", headers, fields), "images/generations"
        usage = value.get("usage")
        usd = gpt_image_usd(usage)
        return ImageResult(base64.b64decode(value["data"][0]["b64_json"]),
                           {"request": req.meta(endpoint=endpoint, model=self.model, size=size, quality=quality),
                            "usage": usage, "usd": usd}, usd)


class OpenAIChat(ChatModel):
    """The Responses API. Images go at detail high; a JSON schema is enforced strictly; effort is the
    reasoning effort. Entry options: base_url, for a proxy that speaks the Responses API."""

    provider = "openai"
    requires = ("openai",)

    def __init__(self, name: str, entry: dict):
        super().__init__(name, entry)
        self._client = None
        self._lock = threading.Lock()

    def client(self):
        with self._lock:  # a client built twice is closed when the spare is collected, mid-request
            if self._client is None:
                import openai
                kw = {"max_retries": 5}
                if self.key_env and self.key_env != "OPENAI_API_KEY":
                    kw["api_key"] = os.environ.get(self.key_env)
                if self.entry.get("base_url"):
                    kw["base_url"] = self.entry["base_url"]
                self._client = openai.OpenAI(**kw)
            return self._client

    @staticmethod
    def content(parts) -> list[dict]:
        out = []
        for p in parts:
            if p.kind == "text":
                out.append({"type": "input_text", "text": p.data})
            elif p.kind == "image":
                out.append({"type": "input_image", "detail": "high",
                            "image_url": "data:%s;base64,%s" % (p.mime or "image/png", base64.standard_b64encode(p.data).decode())})
            else:
                raise ValueError("the Responses API takes no %s parts here" % p.kind)
        return out

    def items(self, messages: list[Message]) -> list:
        out = []
        for m in messages:
            if m.role == "user":
                out.append({"role": "user", "content": self.content(m.parts)})
            elif m.role == "assistant":
                out += m.native if m.native is not None else [
                    {"role": "assistant", "content": [{"type": "output_text", "text": p.data} for p in m.parts if p.kind == "text"]}]
            else:  # a tool result is text; its images follow as a user turn
                out.append({"type": "function_call_output", "call_id": m.tool_call_id,
                            "output": "\n".join(p.data for p in m.parts if p.kind == "text")})
                media = [p for p in m.parts if p.kind != "text"]
                if media:
                    out.append({"role": "user", "content": self.content(media)})
        return out

    def complete(self, req: ChatRequest) -> ChatResponse:
        kw = {"model": self.model, "input": self.items(req.messages), "max_output_tokens": req.max_tokens}
        if req.system:
            kw["instructions"] = req.system
        if req.effort:
            kw["reasoning"] = {"effort": req.effort}
        if req.schema:
            kw["text"] = {"format": {"type": "json_schema", "name": req.schema_name, "schema": req.schema, "strict": True}}
        if req.tools:
            kw["tools"] = [{"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters,
                            "strict": False} for t in req.tools]
        r = self.client().responses.create(**kw)
        if r.status != "completed":
            raise RuntimeError("status %s" % r.status)
        calls = [ToolCall(o.call_id, o.name, json.loads(o.arguments or "{}")) for o in getattr(r, "output", None) or []
                 if getattr(o, "type", "") == "function_call"]
        native = [o.model_dump(exclude_none=True) for o in r.output] if req.tools else None
        u = r.usage
        cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
        usage = Usage(u.input_tokens, u.output_tokens, cached)
        text = r.output_text or ""
        return ChatResponse(text, Message("assistant", [Part.text(text)] if text else [], calls, native=native), usage,
                            self.cost(usage), "tool_calls" if calls else "completed")
