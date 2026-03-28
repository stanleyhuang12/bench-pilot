"""
client.py — litellm-backed client factory.

All models (GPT, Claude, Gemini, etc.) are accessed through the same interface
via OpenRouter or direct provider endpoints.
"""

import time
import litellm
import asyncio 

from typing import Coroutine

import logging
from dataclasses import dataclass

litellm_logger = logging.getLogger("LiteLLM")
litellm_logger.setLevel(logging.WARNING)

# Models that require max_completion_tokens instead of max_tokens
_COMPLETION_TOKENS_MODELS = {"gpt-5.2", "o3", "o3-mini", "o4-mini", "o1", "o1-mini"}

_MAX_RETRIES = 5
_RETRY_DELAY = 10  # seconds, multiplied by attempt number


@dataclass
class LiteLLMClient:
    """
    Thin config holder returned by make_client().

    Replaces the openai.OpenAI client while keeping the public API identical.
    base_url and api_key are forwarded to every litellm.completion() call.
    """
    base_url: str
    api_key: str


def make_client(model_config: dict) -> LiteLLMClient:
    return LiteLLMClient(
        base_url=model_config["base_url"],
        api_key=model_config["api_key"],
    )


async def _create_with_retry(client: LiteLLMClient, **kwargs) -> object:
    """Call litellm.completion with retries on 5xx errors."""
    for attempt in range(_MAX_RETRIES):
        try:
            return await litellm.acompletion(
                api_base=client.base_url,
                api_key=client.api_key,
                **kwargs,
            )
        except Exception as e:
            is_server_error = "500" in str(e) or "502" in str(e) or "503" in str(e)
            if is_server_error and attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_RETRY_DELAY * (attempt + 1))
            else:
                raise


async def chat(client: LiteLLMClient, model: str, messages: list[dict], **kwargs) -> str:
    response = await _create_with_retry(client, model=model, messages=messages, **kwargs)
    return response.choices[0].message.content

async def chat_json(client: LiteLLMClient, model: str, messages: list[dict], **kwargs) -> str:
    """Send a chat request in JSON mode. Caller is responsible for json.loads()."""
    response = await _create_with_retry(
        client,
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        **kwargs,
    )
    return response.choices[0].message.content