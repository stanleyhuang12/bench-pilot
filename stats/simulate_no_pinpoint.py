"""
simulate_no_pinpoint.py — Phase 2 variant: free-form user simulation.

Same as simulate.py but the user simulator receives no landmark instructions
and no turn counter. It acts freely based only on persona and goal.

Usage:
    python simulate_no_pinpoint.py
    python simulate_no_pinpoint.py --config path/to/config.json
"""

import argparse
import json
import os

from client import make_client
from config import load_config, get_model_name
from simulate import run_conversation


def simulate_no_pinpoint(config_path: str = "config.json") -> None:
    config = load_config(config_path)
    test_path = config["paths"]["test_file"]
    conv_dir = config["paths"]["conversations_dir"]
    total_turns = config["generation"]["turns_per_conversation"]

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"{test_path} not found — run generate.py first.")

    with open(test_path) as f:
        scenarios = json.load(f)["scenarios"]

    os.makedirs(conv_dir, exist_ok=True)

    user_client = make_client(config["models"]["user"])
    user_model = get_model_name(config, "user")
    target_client = make_client(config["models"]["target"])
    target_model = get_model_name(config, "target")

    print(f"User model:    {user_model}")
    print(f"Target model:  {target_model}")
    print(f"Turns:         {total_turns}")
    print(f"Scenarios:     {len(scenarios)}")
    print(f"Mode:          no-pinpoint (free-form)\n")

    for i, scenario in enumerate(scenarios):
        sid = scenario["id"]
        print(f"[{i+1}/{len(scenarios)}] {sid} — {scenario['title']}")

        try:
            turns = run_conversation(
                scenario=scenario,
                user_client=user_client,
                user_model=user_model,
                target_client=target_client,
                target_model=target_model,
                total_turns=total_turns,
                pinpoint=False,
            )

            data = {"scenario_id": sid, "scenario": scenario, "turns": turns}
            out_path = os.path.join(conv_dir, f"{sid}.json")

            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)

            print(f"  Saved: {out_path} ({len(turns) // 2} turns)")

        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone. Conversations saved to: {conv_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 (no-pinpoint): Simulate free-form conversations")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    simulate_no_pinpoint(args.config)
