"""Claude through the Messages API (impl = "anthropic"). A JSON schema goes in output_config.format and
effort in output_config.effort. A key that is not scoped to a workspace needs ANTHROPIC_WORKSPACE_ID."""

from __future__ import annotations

import base64
import os
import threading

from evals.core.interfaces import ChatModel, ChatRequest, ChatResponse, Message, Part, ToolCall, Usage


class AnthropicChat(ChatModel):
    provider = "anthropic"
    requires = ("anthropic",)

    def __init__(self, name: str, entry: dict):
        super().__init__(name, entry)
        self._client = None
        self._lock = threading.Lock()

    def client(self):
        with self._lock:  # a client built twice is closed when the spare is collected, mid-request
            if self._client is None:
                import anthropic
                workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
                kw = {"max_retries": 5, "default_headers": {"anthropic-workspace-id": workspace} if workspace else None}
                if self.key_env and self.key_env != "ANTHROPIC_API_KEY":
                    kw["api_key"] = os.environ.get(self.key_env)
                if self.entry.get("base_url"):
                    kw["base_url"] = self.entry["base_url"]
                self._client = anthropic.Anthropic(**kw)
            return self._client

    @staticmethod
    def content(parts) -> list[dict]:
        out = []
        for p in parts:
            if p.kind == "text":
                out.append({"type": "text", "text": p.data})
            elif p.kind == "image":
                out.append({"type": "image", "source": {"type": "base64", "media_type": p.mime or "image/png",
                                                        "data": base64.standard_b64encode(p.data).decode()}})
            else:
                raise ValueError("the Messages API takes no %s parts here" % p.kind)
        return out

    def messages(self, messages: list[Message]) -> list[dict]:
        out = []
        for m in messages:
            if m.role == "user":
                out.append({"role": "user", "content": self.content(m.parts)})
            elif m.role == "assistant":
                blocks = m.native if m.native is not None else self.content([p for p in m.parts if p.kind == "text"]) + [
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in m.tool_calls]
                out.append({"role": "assistant", "content": blocks})
            else:  # every result for one assistant turn goes in the next user turn
                block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": self.content(m.parts)}
                if out and out[-1]["role"] == "user" and out[-1]["content"] and out[-1]["content"][0].get("type") == "tool_result":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    def complete(self, req: ChatRequest) -> ChatResponse:
        kw = {"model": self.model, "max_tokens": req.max_tokens, "messages": self.messages(req.messages)}
        if req.system:
            kw["system"] = req.system
        config = {}
        if req.effort:
            config["effort"] = req.effort
        if req.schema:
            config["format"] = {"type": "json_schema", "schema": req.schema}
        if config:
            kw["output_config"] = config
        if req.tools:
            kw["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in req.tools]
        r = self.client().messages.create(**kw)
        if r.stop_reason not in ("end_turn", "tool_use", "stop_sequence") or (req.schema and r.stop_reason != "end_turn"):
            raise RuntimeError("stop_reason %s" % r.stop_reason)
        texts = [b.text for b in r.content if b.type == "text"]
        text = texts[0] if req.schema else "".join(texts)  # a structured answer is the one text block
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in r.content if b.type == "tool_use"]
        native = [b.model_dump(exclude_none=True) for b in r.content] if req.tools else None
        usage = Usage(r.usage.input_tokens, r.usage.output_tokens, getattr(r.usage, "cache_read_input_tokens", 0) or 0)
        return ChatResponse(text, Message("assistant", [Part.text(text)] if text else [], calls, native=native), usage,
                            self.cost(usage), r.stop_reason)
