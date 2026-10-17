"""Any OpenAI-compatible chat completions endpoint (impl = "openai_compat"): a model of our own behind
vLLM, SGLang or a gateway, OpenRouter, a local server, or Gemini's own compatible endpoint.

    [chat.my-llm]
    impl = "openai_compat"
    base_url = "https://llm.example.com/v1"
    model = "my-model"              # the id the endpoint knows (default: the entry's name)
    key_env = "MY_LLM_API_KEY"
    usd_per_mtok = [0.5, 1.5]
    effort_param = "reasoning_effort"   # only if the endpoint takes one; otherwise effort is not sent

Images go as data URLs; a tool result's images follow it as a user turn, since a tool message is text.
"""

from __future__ import annotations

import base64
import json
import os
import threading

from evals.core.config import ConfigError
from evals.core.interfaces import ChatModel, ChatRequest, ChatResponse, Message, Part, ToolCall, Usage


class OpenAICompatChat(ChatModel):
    provider = "openai-compatible"
    requires = ("openai",)

    def __init__(self, name: str, entry: dict):
        super().__init__(name, entry)
        if not entry.get("base_url"):
            raise ConfigError("[chat.%s] impl = \"openai_compat\" needs a base_url" % name)
        self._client = None
        self._lock = threading.Lock()

    def client(self):
        with self._lock:
            if self._client is None:
                import openai
                self._client = openai.OpenAI(base_url=self.entry["base_url"], max_retries=5,
                                             api_key=os.environ.get(self.key_env or "", "") or "none")
            return self._client

    @staticmethod
    def content(parts) -> list[dict]:
        out = []
        for p in parts:
            if p.kind == "text":
                out.append({"type": "text", "text": p.data})
            elif p.kind == "image":
                out.append({"type": "image_url", "image_url": {
                    "url": "data:%s;base64,%s" % (p.mime or "image/png", base64.standard_b64encode(p.data).decode())}})
            else:
                raise ValueError("chat completions take no %s parts here" % p.kind)
        return out

    def messages(self, req: ChatRequest) -> list[dict]:
        out = [{"role": "system", "content": req.system}] if req.system else []
        for m in req.messages:
            if m.role == "user":
                out.append({"role": "user", "content": self.content(m.parts)})
            elif m.role == "assistant":
                msg = {"role": "assistant", "content": "".join(p.data for p in m.parts if p.kind == "text") or None}
                if m.tool_calls:
                    msg["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                                         for c in m.tool_calls]
                out.append(msg)
            else:
                out.append({"role": "tool", "tool_call_id": m.tool_call_id,
                            "content": "\n".join(p.data for p in m.parts if p.kind == "text")})
                media = [p for p in m.parts if p.kind != "text"]
                if media:
                    out.append({"role": "user", "content": self.content(media)})
        return out

    def complete(self, req: ChatRequest) -> ChatResponse:
        kw = {"model": self.model, "messages": self.messages(req), self.entry.get("max_tokens_param", "max_tokens"): req.max_tokens}
        if req.effort and self.entry.get("effort_param"):
            kw[self.entry["effort_param"]] = req.effort
        if req.schema:
            kw["response_format"] = {"type": "json_schema", "json_schema": {"name": req.schema_name, "schema": req.schema, "strict": True}}
        if req.tools:
            kw["tools"] = [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                           for t in req.tools]
        r = self.client().chat.completions.create(**kw)
        choice = r.choices[0]
        if req.schema and choice.finish_reason != "stop":
            raise RuntimeError("finish_reason %s" % choice.finish_reason)
        text = choice.message.content or ""
        calls = [ToolCall(c.id, c.function.name, json.loads(c.function.arguments or "{}")) for c in choice.message.tool_calls or []]
        u = r.usage
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        usage = Usage(u.prompt_tokens if u else 0, u.completion_tokens if u else 0, cached)
        return ChatResponse(text, Message("assistant", [Part.text(text)] if text else [], calls), usage, self.cost(usage),
                            choice.finish_reason or "")
