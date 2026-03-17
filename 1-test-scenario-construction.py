"""
generate.py — Phase 1: Generate test.json from goal.md and config.json.

Usage:
    python generate.py
    python generate.py --config path/to/config.json
"""

import argparse
import json
import os

from client import make_client, chat_json
from config import load_config, get_model_name


SYSTEM_PROMPT = """\
You are an expert benchmark designer for LLM evaluation.
Generate a structured benchmark test file based on the user's evaluation goal.
Output valid JSON only — no markdown, no text outside the JSON object.\
"""


def build_prompt(goal: str, num_scenarios: int) -> str:
    return f"""\
Based on the following evaluation goal, generate a benchmark test file.

EVALUATION GOAL:
{goal}

Generate exactly {num_scenarios} scenarios and an appropriate set of metrics.

Each scenario must have:
- id: "scenario_001", "scenario_002", etc. (zero-padded 3 digits)
- title: short descriptive title (5-10 words)
- description: 2-3 sentence description of the scenario
- user_persona: who the AI user simulator should roleplay as (2-4 sentences, specific and detailed)
- user_goal: what the simulated user is trying to achieve
- target_system_prompt: the system prompt for the target model being evaluated
- landmarks: list of 0-3 objects, each with:
    - turn: integer (1-indexed turn number to inject this behavior)
    - instruction: specific instruction to the user simulator for that turn
      (e.g. "Express frustration and say the issue is urgent")

Each metric must have:
- id: "metric_001", "metric_002", etc.
- name: short metric name
- description: what is being measured and how to judge pass vs fail
- type: "binary"
- applies_to: list of scenario IDs this metric applies to, or the string "all"

Return JSON with exactly this structure:
{{
  "scenarios": [ ... ],
  "metrics": [ ... ]
}}

Make scenarios diverse and realistic. Make metrics specific and unambiguous.
Metrics must be evaluatable by reading the conversation transcript alone.\
"""


def generate(config_path: str = "config.json") -> None:
    config = load_config(config_path)
    goal_path = config["paths"]["goal_prompt"]
    test_path = config["paths"]["test_file"]
    num_scenarios = config["generation"]["num_scenarios"]

    if not os.path.exists(goal_path):
        raise FileNotFoundError(f"Goal file not found: {goal_path}")

    with open(goal_path) as f:
        goal = f.read().strip()

    print(f"Goal:       {goal_path}")
    print(f"Scenarios:  {num_scenarios}")

    client = make_client(config["models"]["generator"])
    model = get_model_name(config, "generator")
    print(f"Model:      {model}\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(goal, num_scenarios)},
    ]

    raw = chat_json(client, model, messages)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Generator returned invalid JSON: {e}\n\n{raw}")

    if "scenarios" not in data or "metrics" not in data:
        raise ValueError(f"Output missing 'scenarios' or 'metrics' keys.\n\n{raw}")

    os.makedirs(os.path.dirname(test_path) or ".", exist_ok=True)
    with open(test_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Generated {len(data['scenarios'])} scenarios, {len(data['metrics'])} metrics.")
    print(f"Saved to:  {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Generate test.json")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    generate(args.config)
