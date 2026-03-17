# LLM Benchmark

A 3-phase pipeline for evaluating AI assistants using LLM-as-user simulation.

```
python generate.py   # goal.md + config.json → test.json
python simulate.py   # test.json → conversations/
python evaluate.py   # conversations/ → results.json
```

---

## Setup

```bash
pip install -r requirements.txt
```

Edit `config.json` with your API keys and model choices, then write your evaluation goal in `goal.md`.

---

## Files

| File | Role |
|------|------|
| `goal.md` | Describe what you want to evaluate (plain text) |
| `config.json` | Model config, paths, generation settings |
| `generate.py` | Phase 1 — generate scenarios and metrics |
| `simulate.py` | Phase 2 — run simulated conversations |
| `evaluate.py` | Phase 3 — score conversations against metrics |
| `client.py` | Shared OpenAI-compatible client |
| `config.py` | Shared config loader and validator |

---

## config.json

```json
{
  "models": {
    "generator": { "model": "gpt-4o", "base_url": "...", "api_key": "..." },
    "user":      { "model": "gpt-4o", "base_url": "...", "api_key": "..." },
    "target":    { "model": "gpt-4o-mini", "base_url": "...", "api_key": "..." },
    "evaluator": { "model": "gpt-4o", "base_url": "...", "api_key": "..." }
  },
  "generation": {
    "num_scenarios": 5,
    "turns_per_conversation": 6
  },
  "paths": {
    "goal_prompt":      "goal.md",
    "test_file":        "test.json",
    "conversations_dir": "conversations/",
    "results_file":     "results.json"
  }
}
```

All four model roles can point to any OpenAI-compatible endpoint (OpenAI, Anthropic via OpenRouter, Gemini, etc.).

---

## Phase 1 — Generate (`generate.py`)

Reads `goal.md` and calls the generator model to produce `test.json` containing:

- **Scenarios** — each with a user persona, user goal, target system prompt, and optional landmarks
- **Metrics** — binary pass/fail criteria evaluatable from the conversation transcript alone

**Landmarks** are per-turn instructions injected into the user simulator's system prompt upfront. Example:

```json
"landmarks": [
  { "turn": 4, "instruction": "Express frustration and say the issue is urgent" }
]
```

---

## Phase 2 — Simulate (`simulate.py`)

For each scenario, runs a multi-turn conversation between:

- **User simulator** — an LLM roleplaying as a human with the given persona and goal
- **Target model** — the AI assistant being evaluated

Saves each conversation to `conversations/{scenario_id}.json`:

```json
{
  "scenario_id": "scenario_001",
  "scenario": { ... },
  "turns": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

---

## Phase 3 — Evaluate (`evaluate.py`)

For each conversation × applicable metric, calls the evaluator model and asks for a binary pass/fail verdict with justification.

Saves `results.json`:

```json
{
  "summary":     { "total": 20, "passed": 15, "pass_rate": 0.75 },
  "by_metric":   { "metric_001": { "total": 10, "passed": 8, "pass_rate": 0.8 } },
  "by_scenario": { "scenario_001": { "total": 2, "passed": 1, "pass_rate": 0.5 } },
  "details": [
    { "scenario_id": "...", "metric_id": "...", "result": "pass", "justification": "..." }
  ]
}
```
