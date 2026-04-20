"""Factory for an OpenAI client that can route via a LiteLLM proxy.

If `LITELLM_BASE_URL` is set in the environment, the client is pointed at
that URL using `LITELLM_API_KEY` (falling back to `OPENAI_API_KEY` or a
placeholder). Otherwise we construct a plain OpenAI client using
`OPENAI_API_KEY` the way the stock FDB scripts do.
"""

from __future__ import annotations

import os

from openai import OpenAI


def make_client() -> OpenAI:
    base_url = os.getenv("LITELLM_BASE_URL")
    if base_url:
        api_key = (
            os.getenv("LITELLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "sk-litellm-placeholder"
        )
        return OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
