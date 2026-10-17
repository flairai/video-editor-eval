"""Google: Gemini through google-genai (a chat model, impl = "google") and Nano Banana through the
Interactions API over plain HTTP (an image model, impl = "nano-banana"). For Gemini, a JSON schema goes in
response_json_schema and effort is the thinking level. Media parts of any kind (image, audio, video) go inline."""

from __future__ import annotations

import base64
import os
import threading

from evals.core.interfaces import (ChatModel, ChatRequest, ChatResponse, ImageModel, ImageRequest, ImageResult, Message,
                                   Part, ProviderError, ToolCall, Usage)
from evals.models.http import post_json, require_key

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiChat(ChatModel):
    provider = "google"
    requires = ("google-genai",)

    def __init__(self, name: str, entry: dict):
        super().__init__(name, entry)
        self._client = None
        self._lock = threading.Lock()

    def client(self):
        with self._lock:  # a client built twice is closed when the spare is collected, mid-request
            if self._client is None:
                from google import genai
                from google.genai import types
                kw = {"http_options": types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=6))}
                if self.key_env and self.key_env not in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                    kw["api_key"] = os.environ.get(self.key_env)
                self._client = genai.Client(**kw)
            return self._client

    @staticmethod
    def parts(parts):
        from google.genai import types
        return [types.Part.from_text(text=p.data) if p.kind == "text" else types.Part.from_bytes(data=p.data, mime_type=p.mime or "image/png")
                for p in parts]

    def contents(self, messages: list[Message]):
        from google.genai import types
        out, names = [], {}
        for m in messages:
            if m.role == "user":
                out.append(types.Content(role="user", parts=self.parts(m.parts)))
            elif m.role == "assistant":
                for c in m.tool_calls:
                    names[c.id] = c.name
                out.append(m.native if m.native is not None else types.Content(role="model", parts=self.parts(
                    [p for p in m.parts if p.kind == "text"])))
            else:
                text = "\n".join(p.data for p in m.parts if p.kind == "text")
                parts = [types.Part.from_function_response(name=names.get(m.tool_call_id, m.tool_call_id or "tool"), response={"output": text})]
                out.append(types.Content(role="user", parts=parts + self.parts([p for p in m.parts if p.kind != "text"])))
        return out

    def complete(self, req: ChatRequest) -> ChatResponse:
        from google.genai import types
        config = {}
        if req.system:
            config["system_instruction"] = req.system
        if req.schema:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = req.schema
        if req.effort:
            config["thinking_config"] = types.ThinkingConfig(thinking_level=req.effort.upper())
        if req.tools:
            config["tools"] = [types.Tool(function_declarations=[types.FunctionDeclaration(
                name=t.name, description=t.description, parameters_json_schema=t.parameters) for t in req.tools])]
        r = self.client().models.generate_content(model=self.model, contents=self.contents(req.messages),
                                                  config=types.GenerateContentConfig(**config))
        if req.tools:
            content = r.candidates[0].content if r.candidates else None
            text = "".join(p.text for p in (content.parts if content and content.parts else []) if getattr(p, "text", None) and not getattr(p, "thought", False))
            calls = [ToolCall(c.id or "%s-%d" % (c.name, i), c.name, dict(c.args or {})) for i, c in enumerate(r.function_calls or [])]
            native = content
        else:
            text, calls, native = r.text, [], None
        u = r.usage_metadata
        usage = Usage(u.prompt_token_count, (u.candidates_token_count or 0) + (u.thoughts_token_count or 0),
                      getattr(u, "cached_content_token_count", 0) or 0)
        return ChatResponse(text or "", Message("assistant", [Part.text(text)] if text else [], calls, native=native), usage,
                            self.cost(usage), "tool_calls" if calls else "stop")


class NanoBanana(ImageModel):
    """Nano Banana 2.1 (gemini-nano-banana-2.1) through the Interactions API, checked 2026-10-17 on
    ai.google.dev (models/gemini-nano-banana-2.1, image-generation, pricing): POST v1beta/interactions with
    response_format {type: image, aspect_ratio, image_size, mime_type}, references inline after the prompt.

    It renders 1K, 2K or 4K at the ratios in ASPECTS, with no 2:1, so a 2:1 task gets no request and no image
    (it scores 0, at no cost). 2K is the smallest size that meets the instructions' minimum (1K is 1024 px on a
    side). It returns JPEG only: it refuses image/png. Price: $1.50/M input tokens, $30/M image output tokens
    (1,680 at 2K, $0.0504), $7.50/M text and thinking tokens; an attempt costs what the usage it returns
    comes to. Entry options: image_size (2K), thinking_level (unset: Google's default, medium)."""

    provider = "google"
    ASPECTS = ("1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9")
    IMAGE_USD = {"1K": 0.0336, "2K": 0.0504, "4K": 0.113}
    USD_INPUT, USD_IMAGE, USD_TEXT = 1.50e-6, 30e-6, 7.50e-6           # per token
    REFERENCE_TOKENS = 1120   # input tokens per reference image, as Google bills them
    THINKING_TOKENS = 4500    # text and thinking tokens the list price allows for: high thinking returned 2,900 to 5,800 on the multi-view grids

    def list_price(self, req: ImageRequest) -> float:
        if req.aspect not in self.ASPECTS:
            return 0.0
        return self.IMAGE_USD[self.entry.get("image_size", "2K")] + self.THINKING_TOKENS * self.USD_TEXT \
            + self.REFERENCE_TOKENS * len(req.references) * self.USD_INPUT

    @classmethod
    def usd(cls, usage: dict | None) -> float | None:
        """What a returned usage costs. Every output token that is not an image token (text or thinking, however
        the response splits them) is billed at the text rate."""
        if not usage:
            return None
        image = sum(m.get("tokens", 0) for m in usage.get("output_tokens_by_modality") or [] if m.get("modality") == "image")
        given = usage.get("total_input_tokens", 0)
        return round(given * cls.USD_INPUT + image * cls.USD_IMAGE + (usage.get("total_tokens", 0) - given - image) * cls.USD_TEXT, 4)

    def generate(self, req: ImageRequest) -> ImageResult:
        if req.aspect not in self.ASPECTS:
            raise ProviderError("%s has no %s output (its ratios: %s), so nothing was sent" % (self.label, req.aspect, ", ".join(self.ASPECTS)))
        fmt = {"aspect_ratio": req.aspect, "image_size": self.entry.get("image_size", "2K"), "mime_type": "image/jpeg"}
        body = {"model": self.model, "response_format": {"type": "image", **fmt},
                "input": [{"type": "text", "text": req.prompt}] + [
                    {"type": "image", "mime_type": "image/png", "data": base64.b64encode(p.read_bytes()).decode()} for p in req.references]
                if req.references else req.prompt}
        if self.entry.get("thinking_level"):
            body["generation_config"] = {"thinking_level": self.entry["thinking_level"]}
            fmt["thinking_level"] = self.entry["thinking_level"]
        value = post_json(f"{GEMINI_URL}/interactions", {"x-goog-api-key": require_key(self.key_env or "GEMINI_API_KEY")}, body)
        outputs = [c for s in value.get("steps") or [] if s.get("type") == "model_output" for c in s.get("content") or []]
        images = [c for c in outputs if c.get("type") == "image" and c.get("data")]
        status = value.get("status")
        if status != "completed" or not images:
            # Thinking renders interim images it never returns; an interaction that ends without one either stopped
            # on Google's side (worth retrying) or answered in text instead (a refusal).
            text = " ".join(c.get("text", "") for c in outputs if c.get("type") == "text").strip()
            raise ProviderError("the interaction ended %s without an image%s" % (status, ": " + text[:600] if text else ""),
                                transient=status in ("failed", "incomplete", "cancelled"))
        usage = value.get("usage")
        usd = self.usd(usage)
        return ImageResult(base64.b64decode(images[-1]["data"]),  # the last image of the output is the final one
                           {"request": req.meta(endpoint="interactions", model=self.model, **fmt),
                            "returned": {"mime_type": images[-1].get("mime_type"), "interaction": value.get("id")},
                            "usage": usage, "usd": usd}, usd)
