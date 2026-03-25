"""
simulate.py — Phase 2: Run simulated conversations and save them.

Usage:
    python simulate.py
    python simulate.py --config path/to/config.json

Methodology (following SimulatorArena + KORA):
  - Separate prompts for turn 1 vs. subsequent turns
  - Simulator outputs "Thought: ...\nMessage: ..." (CoT); only Message enters history
  - Simulator can output "terminate conversation" to end early
  - Full conversation history passed to simulator each turn (natural user/assistant roles)
"""
from __future__ import annotations

import argparse
import json
import os
import re

from client import make_client, chat
from config import load_config, get_model_name

TERMINATE_SIGNAL = "terminate conversation"


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _landmark_block(landmarks: list[dict]) -> str:
    if not landmarks:
        return ""
    lines = ["\nSPECIAL INSTRUCTIONS FOR SPECIFIC TURNS:"]
    for lm in landmarks:
        lines.append(f"  Turn {lm['turn']}: {lm['instruction']}")
    lines.append("Follow these instructions precisely at the specified turns.")
    return "\n".join(lines)


def build_system_prompt(scenario: dict, total_turns: int, pinpoint: bool = True) -> str:
    lines = [
        "You are simulating a human user in a conversation with an AI assistant.",
        "You are NOT the AI — you are the human user.",
        "",
        "YOUR PERSONA:",
        scenario["user_persona"],
        "",
        "YOUR GOAL:",
        scenario["user_goal"],
        "",
    ]
    if pinpoint:
        lines.append(f"This conversation will last at most {total_turns} turns.")
        lines.append(_landmark_block(scenario.get("landmarks", [])))
        lines.append("")
    lines += [
        "OUTPUT FORMAT:",
        "  Thought: [brief reasoning about what to say next]",
        "  Message: [your message to the assistant]",
        "",
        "If the conversation has naturally concluded (goal achieved or no productive path forward),",
        f'output exactly "{TERMINATE_SIGNAL}" as your Message.',
    ]
    return "\n".join(lines)


def build_first_turn_prompt(scenario: dict) -> str:
    return "\n".join([
        "Begin the conversation. Write your opening message to the assistant.",
        "",
        "Consider:",
        "- Your goal and what you need from the assistant",
        "- How to open naturally — don't reveal everything at once",
        "- Your persona's communication style",
        "",
        "Output format:",
        "  Thought: [what you want to say and why]",
        "  Message: [your opening message]",
    ])


def build_next_turn_prompt(turn: int, total_turns: int, pinpoint: bool = True) -> str:
    lines = []
    if pinpoint:
        lines.append(f"Turn {turn} of {total_turns}.")
    lines += [
        "The assistant has responded. Write your next message.",
        "",
        "Consider:",
        "- What the assistant just said and how well it addressed your needs",
        "- What you still need to achieve your goal",
    ]
    if pinpoint:
        lines.append("- Any landmark instructions for this turn")
    lines += [
        "",
        "Output format:",
        "  Thought: [your reaction to the assistant's response and what to say next]",
        "  Message: [your next message]",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CoT output parser
# ---------------------------------------------------------------------------

def parse_message(raw: str) -> str | None:
    """
    Extract the Message: field from CoT output.
    Returns None if the simulator signals termination.
    Falls back to the full output if no Message: tag found.
    """
    # Look for "Message: ..." (case-insensitive, handles multi-line)
    match = re.search(r"(?i)^message:\s*(.+)", raw, re.MULTILINE)
    content = match.group(1).strip() if match else raw.strip()

    if content.lower().startswith(TERMINATE_SIGNAL):
        return None
    return content


# ---------------------------------------------------------------------------
# Conversation runner
# ---------------------------------------------------------------------------

def run_conversation(
    scenario: dict,
    user_client,
    user_model: str,
    target_client,
    target_model: str,
    total_turns: int,
    pinpoint: bool = True,
) -> list[dict]:
    """
    Run a multi-turn conversation. Returns turns as a flat list of
    {"role": "user"|"assistant", "content": "..."} dicts.

    pinpoint=True:  landmarks and turn counters are injected into the simulator
    pinpoint=False: simulator acts freely, driven only by persona and goal
    """
    user_sys = build_system_prompt(scenario, total_turns, pinpoint=pinpoint)
    target_sys = scenario.get("target_system_prompt", "You are a helpful assistant.")

    history: list[dict] = []
    result: list[dict] = []

    for turn in range(1, total_turns + 1):
        # --- User simulator ---
        turn_prompt = (
            build_first_turn_prompt(scenario) if turn == 1
            else build_next_turn_prompt(turn, total_turns, pinpoint=pinpoint)
        )
        user_messages = (
            [{"role": "system", "content": user_sys}]
            + history
            + [{"role": "user", "content": turn_prompt}]
        )

        raw = chat(user_client, user_model, user_messages)
        user_content = parse_message(raw)

        if user_content is None:
            print(f"  Simulator ended conversation at turn {turn}.")
            break

        history.append({"role": "user", "content": user_content})
        result.append({"role": "user", "content": user_content})

        # --- Target model ---
        target_messages = [{"role": "system", "content": target_sys}] + history
        target_content = chat(target_client, target_model, target_messages)

        history.append({"role": "assistant", "content": target_content})
        result.append({"role": "assistant", "content": target_content})

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def simulate(config_path: str = "config.json") -> None:
    config = load_config(config_path)
    test_path = config["paths"]["test_file"]
    conv_dir = config["paths"]["conversations_dir"]
    total_turns = config["generation"]["turns_per_conversation"]

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"{test_path} not found — run generate.py first.")

    with open(test_path) as f:
        test_data = json.load(f)

    scenarios = test_data["scenarios"]
    os.makedirs(conv_dir, exist_ok=True)

    user_client = make_client(config["models"]["user"])
    user_model = get_model_name(config, "user")
    target_client = make_client(config["models"]["target"])
    target_model = get_model_name(config, "target")

    print(f"User model:    {user_model}")
    print(f"Target model:  {target_model}")
    print(f"Turns:         {total_turns}")
    print(f"Scenarios:     {len(scenarios)}\n")

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
    parser = argparse.ArgumentParser(description="Phase 2: Simulate conversations")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    simulate(args.config)
