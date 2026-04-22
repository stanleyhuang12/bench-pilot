"""
client.py — litellm-backed client factory.

All models (GPT, Claude, Gemini, etc.) are accessed through the same interface
via OpenRouter or direct provider endpoints.
"""
from __future__ import annotations
import litellm
import asyncio 


import json
import os
import logging
from dataclasses import dataclass
import re 


litellm_logger = logging.getLogger("LiteLLM")
litellm_logger.setLevel(logging.WARNING)

# Models that require max_completion_tokens instead of max_tokens

_MAX_RETRIES = 5
_RETRY_DELAY = 10  # seconds, multiplied by attempt number

_CUSTOM_MODEL_PRICING = {
    "deepinfra/google/gemma-4-31B-it": {
        "input_cost_per_token":  0.00000013,
        "output_cost_per_token": 0.00000038, 
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "deepinfra/Qwen/Qwen3.6-35B-A3B": {
        "input_cost_per_token":  0.00000006,   # placeholder — check deepinfra.com/pricing
        "output_cost_per_token": 0.00000006,
        "litellm_provider": "openai",
        "mode": "chat",
    },
     "deepinfra/deepseek-ai/DeepSeek-V3.2": {
        "input_cost_per_token":  0.00000020, 
        "output_cost_per_token": 0.00000077, 
    },
}

def register_custom_pricing() -> None:
    """Register pricing for models not in LiteLLM's model_prices_and_context_window.json."""
    litellm.register_model(_CUSTOM_MODEL_PRICING)
    print(f"Registered custom pricing for {len(_CUSTOM_MODEL_PRICING)} models.")

@dataclass
class LiteLLMClient:
    """
    Thin config holder returned by make_client().

    Replaces the openai.OpenAI client while keeping the public API identical.
    base_url and api_key are forwarded to every litellm.completion() call.
    """
    base_url: str
    api_key: str


@dataclass 
class LiteLLMCostTracker: 
    cost: float = 0
    input_tokens: int = 0 
    output_tokens: int = 0 
    
    def add(self, c:dict): 
        if not c: return
        self.cost += c.get("cost", 0)
        self.input_tokens += c.get("input_tokens", 0)
        self.output_tokens += c.get("output_tokens", 0)
            
    def merge(self, other: "LiteLLMCostTracker"):
        self.cost += other.cost
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
    
    def to_json(self): 
        return { 
            "cost": self.cost,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens
        }
        
    def write_out_costs(self, step_name: str, abs_path_file: str, metadata: dict | None = None, cost_pathname: str | None = None): 
        os.makedirs(abs_path_file, exist_ok=True)
        
        if cost_pathname is None: 
            cost_pathname = "cost.json" 
        
        cost_path = os.path.join(abs_path_file, cost_pathname)
        
        if os.path.exists(cost_path): 
            try:    
                with open(cost_path, "r") as f: 
                    data = json.load(f)
            except json.JSONDecodeError:
                data = {}
        else: 
            data = {}
        
        data[step_name] = self.to_json()
        data[step_name]["metadata"] = metadata or {}
        
        with open(cost_path, "w") as f: 
            json.dump(data, f, indent=4)
        
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
        except litellm.RateLimitError:
            if attempt < _MAX_RETRIES - 1:
                wait = _RETRY_DELAY * (attempt + 1)
                print(f"[RateLimit] attempt {attempt+1}/{_MAX_RETRIES}, retrying in {wait}s...")
                await asyncio.sleep(wait)
        else:
                raise

def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()

async def chat(client: LiteLLMClient, model: str, messages: list[dict], **kwargs) -> tuple[str, "LiteLLMCostTracker"]:
    response = await _create_with_retry(client, model=model, messages=messages, **kwargs)
    usage = response.usage
    try:
        cost = litellm.completion_cost(response)
    except Exception:
        cost = 0.0
    tracker = {
        "cost": cost,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
    }

    return response.choices[0].message.content, LiteLLMCostTracker(**tracker)

async def chat_json(client: LiteLLMClient, model: str, messages: list[dict], **kwargs) -> tuple[str, "LiteLLMCostTracker"]:
    """Send a chat request in JSON mode. Caller is responsible for json.loads()."""
    response = await _create_with_retry(
        client,
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        **kwargs,
    )
    
    usage = response.usage
    tracker = {
        "cost": litellm.completion_cost(response),
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
    }
    
    return response.choices[0].message.content, LiteLLMCostTracker(**tracker)