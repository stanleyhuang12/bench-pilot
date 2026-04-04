"""
config.py — Load and validate config.json.
"""

import json
import os
from typing import Optional, List

REQUIRED_MODEL_ROLES = ["generator", "user", "target", "evaluator"]
REQUIRED_MODEL_FIELDS = ["model", "base_url", "api_key"]


def load_config(path: str = "config.json") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        config = json.load(f)
    _validate(config)
    return config


def get_model_name(config: dict, role: str) -> Optional[str, list[str]]:
    if role= "target": 
        model_dicts = config["models"][role]
        return [d["model"] for d in model_dicts]
    else: 
        return config["models"][role]["model"]


def _validate(config: dict) -> None:
    for key in ["models", "generation", "paths"]:
        if key not in config:
            raise ValueError(f"config.json missing top-level key: '{key}'")

    for role in REQUIRED_MODEL_ROLES:
        if role not in config["models"]:
            raise ValueError(f"config.json missing model role: '{role}'")
        for field in REQUIRED_MODEL_FIELDS:
            if field not in config["models"][role]:
                raise ValueError(f"config.json model '{role}' missing field: '{field}'")

    for field in ["num_scenarios", "turns_per_conversation"]:
        if field not in config["generation"]:
            raise ValueError(f"config.json missing generation field: '{field}'")

    for field in ["goal_prompt", "test_file", "conversations_dir", "results_file"]:
        if field not in config["paths"]:
            raise ValueError(f"config.json missing path field: '{field}'")
