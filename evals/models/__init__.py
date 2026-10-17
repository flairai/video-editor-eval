"""The model providers, one module each. Each class implements evals.core.interfaces.ImageModel or
ChatModel and is named in configs/models.toml by `impl` (the short names are in evals.core.registry).

    openai.py          GPT Image (images); the Responses API (chat: judges and our agents)
    fal.py             Seedream and Uni-1 through fal's queue
    anthropic.py       the Messages API
    google.py          Gemini through google-genai; Nano Banana through the Interactions API
    openai_compat.py   any OpenAI-compatible chat completions endpoint: a custom LLM, vLLM, OpenRouter, a local server
    local.py           the oracle: the task's own solution image, at no cost

A new provider is one class here (or in any module), with `impl = "module:Class"` in configs/models.toml.
"""
