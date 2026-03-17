"""
client.py — OpenAI-compatible client factory.

All models (GPT, Claude, Gemini, etc.) are accessed through the same interface
via OpenRouter or direct provider endpoints.
"""

import time
from openai import OpenAI

# Models that require max_completion_tokens instead of max_tokens
_COMPLETION_TOKENS_MODELS = {"gpt-5.2", "o3", "o3-mini", "o4-mini", "o1", "o1-mini"}

_MAX_RETRIES = 5
_RETRY_DELAY = 10  # seconds, multiplied by attempt number


def make_client(model_config: dict) -> OpenAI:
    return OpenAI(
        base_url=model_config["base_url"],
        api_key=model_config["api_key"],
    )


def _create_with_retry(client: OpenAI, **kwargs) -> object:
    """Call chat.completions.create with retries on 5xx errors."""
    for attempt in range(_MAX_RETRIES):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            is_server_error = "500" in str(e) or "502" in str(e) or "503" in str(e)
            if is_server_error and attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY * (attempt + 1))
            else:
                raise


def chat(client: OpenAI, model: str, messages: list[dict], **kwargs) -> str:
    response = _create_with_retry(client, model=model, messages=messages, **kwargs)
    return response.choices[0].message.content


def chat_json(client: OpenAI, model: str, messages: list[dict], **kwargs) -> str:
    """Send a chat request in JSON mode. Caller is responsible for json.loads()."""
    response = _create_with_retry(
        client,
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        **kwargs,
    )
    return response.choices[0].message.content
