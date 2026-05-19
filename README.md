# LLM Benchmark

A numbered, file-driven pipeline for evaluating AI assistants using LLM-as-user simulation.

```
python 0-parse-through-xlsx.py          # benchmark_submission.csv → results/{bench}/goal.json
python 1-test-scenario-construction.py  # goal.json → test.json
python 2-simulation.py                  # test.json → conversations/
python 3-evaluation.py                  # conversations/ → results.json
python 4-export.py                      # results.json → results.docx
```

Each stage reads the previous stage's JSON and writes the next, so phases can be re-run independently. Everything is keyed by **benchmark slug** under a results root (default `results/`):

```
results/
  {benchmark}/
    goal.json            # phase 0 output
    test.json            # phase 1 output (scenarios + metrics)
    conversations/
      {model_name}/      # phase 2 output, one JSON per conversation
    results.json         # phase 3 aggregate
    results_details.json # phase 3 per-judgment detail
```

---

## Setup

```bash
pip install -r requirements.txt
```

Create `config.json` with your API keys and model choices (it is gitignored). Helper scripts `init.sh`, `eval.sh`, `metric_gen.sh`, and `cost.sh` wrap common runs.

All four model roles can point to any OpenAI-compatible endpoint (OpenAI, Anthropic, Gemini, etc.) — calls are routed through `litellm` in `client.py`, with per-run cost tracking via `LiteLLMCostTracker`.

---

## Files

| File | Role |
|------|------|
| `0-parse-through-xlsx.py` | Phase 0 — LLM-normalize `benchmark_submission.csv` rows into per-benchmark `goal.json` |
| `1-test-scenario-construction.py` | Phase 1 — generate `test.json` (scenarios + metrics) from `goal.json` |
| `1-test-scenario-distilled.py` | Phase 1 variant — same, with two-stage demographic expansion |
| `2-simulation.py` | Phase 2 — run simulated conversations concurrently |
| `3-evaluation.py` | Phase 3 — score conversations against metrics |
| `3-evaluation-batch.py` | Phase 3 variant — batched evaluation |
| `4-export.py` | Phase 4 — compile all benchmark data into `results.docx` |
| `aggregate-results.py` | Post — combine per-model/per-benchmark results into `results_aggregate.json` |
| `client.py` | Shared litellm client + cost tracker |
| `config.py` | Config loader and validator |
| `demographics.py` | Demographic combination expansion (used by the distilled phase 1) |

---

## config.json

`target` is a **list** of model dicts (multiple target models are evaluated in one run); the other roles are single dicts.

```json
{
  "models": {
    "generator": { "model": "gpt-4o",      "base_url": "...", "api_key": "..." },
    "user":      { "model": "gpt-4o",      "base_url": "...", "api_key": "..." },
    "target":    [
      { "model": "gpt-4o-mini",  "base_url": "...", "api_key": "..." },
      { "model": "gpt-3.5-turbo", "base_url": "...", "api_key": "..." }
    ],
    "evaluator": { "model": "gpt-4o",      "base_url": "...", "api_key": "..." }
  },
  "generation": {
    "num_scenarios": 5,
    "turns_per_conversation": 6
  },
  "paths": {
    "goal_prompt":       "goal.json",
    "test_file":         "test.json",
    "conversations_dir": "conversations/",
    "results_file":      "results.json"
  }
}
```

`config.py` validates that every model role has `model`, `base_url`, and `api_key`, and that `generation` and `paths` keys are present.

## goal.json

```json
{
  "benchmark_name": "...",
  "description": "...",
  "metadata": {
    "physical_health": [],
    "psychological_wellbeing": [],
    "self_actualization": []
  },
  "target_population": {
    "age": []
  },
  "scenario": {
    "user_context": "",
    "implicit_context": ""
  },
  "metric": [
    {
      "id": "metric_001",
      "metric_name": "Metric Name",
      "type": "binary",
      "definition": "Code 'yes' if the model exhibits the defined behavior. Clearly specify what constitutes a positive vs negative case using observable language patterns.",
      "applies_to": "all",
      "examples": []
    }
  ]
}
```

---

## Phase 0 — Parse (`0-parse-through-xlsx.py`)

Reads `benchmark_submission.csv` and uses an LLM to normalize each row into a per-benchmark `goal.json` under the results root.

```bash
python 0-parse-through-xlsx.py --csv benchmark_submission.csv --results-root results
python 0-parse-through-xlsx.py --row-entry 3 --dry-run    # preview a single row
```

Key flags: `--benchmark` (single slug), `--row-entry N`, `--dry-run`.

---

## Phase 1 — Generate (`1-test-scenario-construction.py`)

Reads `goal.json` and calls the generator model to produce `test.json` containing:

- **Scenarios** — each with a user persona, user goal, target system prompt, and optional landmarks
- **Metrics** — binary pass/fail criteria evaluatable from the conversation transcript alone

**Landmarks** are per-turn instructions injected into the user simulator's system prompt. Example:

```json
"landmarks": [
  { "turn": 4, "instruction": "Express frustration and say the issue is urgent" }
]
```

Run multiple generation batches (mind provider rate limits):

```bash
python 1-test-scenario-construction.py --b emotional-dependency --num-batch 5 --results-root results
```

Key flags: `--b/--benchmark` (omit to run all discovered benchmarks), `--num-batch`, `--overwrite` (replace instead of append), `--overwrite-model` / `--overwrite-api-key`.

### Distilled variant (`1-test-scenario-distilled.py`)

Same output, plus two-stage demographic expansion:

- `--demographic-base {gender,age,race}` — full LLM expansion, one variant per group (IDs become `_v01`, `_v02`, …)
- `--demographic-random {gender,age,race} ...` — randomly assigns one group as metadata to each already-expanded scenario (no LLM call, seeded)
- `--quality-check` — print demographic distribution without generating
- `--exclude-bench foo bar` — skip benchmark slugs

---

## Phase 2 — Simulate (`2-simulation.py`)

For each scenario, runs a multi-turn conversation between:

- **User simulator** — an LLM roleplaying as a human with the given persona and goal
- **Target model(s)** — every model in `config.models.target`, evaluated in the same run

All `(model × scenario × sample)` tasks run concurrently under one semaphore; each conversation is written to disk as it completes, with cost checkpoints flushed periodically. An optional adversarial block targets one metric per scenario.

```bash
python 2-simulation.py --b emotional-dependency --ns 3 --semaphore 8
```

Key flags: `--ns/--num-samples` (independent samples per scenario), `--semaphore` (max concurrency), `--flush-every`, `--p/--perfunctory`, `--downsample` with `--test-size N`.

Each conversation is saved under `conversations/{model_name}/`:

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

## Phase 3 — Evaluate (`3-evaluation.py`)

For each conversation × applicable metric, calls the evaluator model and asks for a binary pass/fail verdict with justification. Writes `results.json` (aggregate) and `results_details.json` (per-judgment).

```bash
python 3-evaluation.py --b emotional-dependency --concurrency 5
python 3-evaluation.py --b emotional-dependency --aggregate   # re-aggregate, no new LLM calls
```

`--aggregate` rebuilds `results.json` from the existing `results_details.json` without re-running the evaluator. Use `3-evaluation-batch.py` for batched evaluation.

`results.json`:

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

---

## Phase 4 — Export (`4-export.py`)

Compiles all benchmark data into a `results.docx` report.

```bash
python 4-export.py
python 4-export.py --results-root results --benchmark emotional-dependency --output report.docx
```

---

## Post — Aggregate (`aggregate-results.py`)

Combines every `results_details.json` (falling back to `results.json`) across models and benchmarks into a single `results_aggregate.json`.

```bash
python aggregate-results.py                                  # auto-discover config.json
python aggregate-results.py --benchmark emotional-dependency
python aggregate-results.py --output my_aggregate.json
```
