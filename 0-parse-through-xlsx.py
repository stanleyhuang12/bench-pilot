"""
0-parse-through-xlsx.py — Convert benchmark_submission.csv into per-benchmark goal.json files.

This version uses an LLM to normalize each row into the project's goal.json schema.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re

from typing import Any
import pandas as pd

from client import make_client, chat_json
from config import load_config, get_model_name
from demographics import AGE_BANK, GENDER_BANK, RACE_BANK

COLUMN_MAP = {
    "Timestamp": "timestamp", 
    "Email Address": "email", 
    'Define the construct: What are you attempting to measure? Is there a literature on this construct that you can point to?': 'construct', 
    'Relate the construct: Conceptually, how does this benchmark relate to other existing constructs, especially those that are already represented in this project?': "construct_relate", 
    'Justify the construct: Please provide a narrative description, including citations, for why this LLM behavior that is being benchmarked, relates to the dimension(s) of human flourishing you identified above.': "construct_justify",
    'User demographic(s): What demographic(s) of interest would you be interested in testing? We may simulate conversations that reflect this demographic. We may provide benchmarks where the model explicitly knows the demographics or when the model has to infer these demographics from previous conversations.': "demographics",
    'User context(s) - known: Define what the model knows about the user from previous interactions. What has the user previously expressed that may elicit the behavior of interest. ': "user_context", 
    'User context(s) - unknown: Define characteristics of the user that the model cannot directly observe. We may simulate conversations that reflect this context without providing this context to the model.': "implicit_context", 
    'User message(s): Provide one or more messages that are designed to elicit the behavior of interest.': "user_message", 
    'LLM-as judge prompt: Provide one or more LLM prompts that allows for a yes/no judgment as to whether a behavior is present in any LLM response. If you provide more than one response, we may test various versions.': "llm_as_judge_prompt", 
    'Positively Scored Examples: Provide one or more specific examples of this behavior being present in a response that we can use to validate the LLM-as-judge prompt.': "pos_examples", 
    'Negatively Scored Examples: Provide one or more specific examples of this behavior being absent in a response that we can use to validate the LLM-as-judge prompt.': "neg_examples", 
    'Name the Construct: Provide a name for your proposed benchmark idea (Name/Concept).  ': "construct_name"
}
GOAL_NORMALIZATION_PROMPT = """\
You are an expert in LLM evaluation benchmark design.

Convert a raw benchmark submission into a structured goal.json format.
- Make sure to properly decompose the description into discrete evaluatable tasks.
- If no explicit demographic group is described, then don't include keys or values.
- If there is explicit demographic groups described, then include the keys and the list. 
- If the description emphasizes a specific demographic but does not exclude others, still include all groups such as age: "{AGE_BANK}", gender: "{GENDER_BANK}", and race/ethnicity: {RACE_BANK}"
- Look at the LLM-as-a-judge information context, use these to construct well-defined, discrete, individual, and evaluatable metrics. 
- Make sure metrics align with the kind of user messages that we may see. 

STRICT REQUIREMENTS:
- Output valid JSON only
- Follow schema exactly
- Expand broad metrics into multiple granular ones
- Do NOT hallucinate

Schema:
{{
  "benchmark_name": str,
  "description": str,
  "metadata": {{
    "physical_health": list[str],
    "psychological_wellbeing": list[str],
    "self_actualization": list[str]
  }},
  "target_population": {{ 
    "age": list[str],
    "gender": list[str],
    "race": list[str]
  }},
  "scenario": {{
    "user_context": str,
    "implicit_context": str
  }},
  "metric": [
    {{
      "id": "metric_001",
      "metric_name": str,
      "type": "binary",
      "definition": str,
      "applies_to": "all",
      "examples": list[str]
    }}
  ]
}}
""".format(
    AGE_BANK=AGE_BANK,
    RACE_BANK=RACE_BANK,
    GENDER_BANK=GENDER_BANK,
)


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")

def _clean(val: Any) -> str:
    return "" if pd.isna(val) else str(val).strip()

def _split_examples(raw: str) -> list[str]:
    if not raw:
        return []
    parts = re.split(r"\n\s*\d+[.)]\s*", raw)
    if len(parts) > 1:
        return [p.strip() for p in parts if p.strip()]
    return [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]


def _row_to_llm_input(row: pd.Series) -> dict[str, Any]:
    return { 
        "benchmark_name": row.get("construct_name", ""),
        "construct": row.get("construct", ""),
        "construct_relate": row.get("construct_relate", ""),
        "construct_justify": row.get("construct_justify", ""),
        "user_context": row.get("user_context", ""),
        "implicit_context": row.get("implicit_context", ""),
        "user_messages": _split_examples(row.get("user_message", "")),
        "positive_examples": _split_examples(row.get("pos_examples", "")),
        "negative_examples": _split_examples(row.get("neg_examples", "")),
        "judge_prompt": row.get("llm_as_judge_prompt", ""),
    }


def _validate_and_fix(goal: dict[str, Any]) -> dict[str, Any]:
    required = ["benchmark_name", "description", "metadata", "target_population", "scenario", "metric"]
    for key in required:
        if key not in goal:
            raise ValueError(f"Missing '{key}' field in goal.json")

    if not isinstance(goal["metric"], list) or not goal["metric"]:
        raise ValueError("'metric' must be a non-empty list")

    for i, metric in enumerate(goal["metric"], 1):
        metric["id"] = f"metric_{i:03d}"
        metric.setdefault("type", "binary")
        metric.setdefault("applies_to", "all")
        metric.setdefault("examples", [])

    return goal


async def _build_goal_with_llm(client, model: str, row: pd.Series) -> dict[str, Any]:
    payload = _row_to_llm_input(row)
    messages = [
        {"role": "system", "content": GOAL_NORMALIZATION_PROMPT},
        {"role": "user", "content": json.dumps(payload, indent=2)},
    ]

    raw, costs = await chat_json(client, model, messages)
    try:
        goal = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from LLM:\n{raw}") from exc

    return _validate_and_fix(goal)



# ---------- main ----------

async def ingest(
    csv_path: str,
    results_root: str,
    dry_run: bool,
    benchmark: str | None,
    row_entry: int, 
) -> list[str]:

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    config = load_config("config.json")
    client = make_client(config["models"]["generator"])
    model = get_model_name(config, "generator")

    df = pd.read_csv(csv_path)
    print(df)
    
    df = df.iloc[[row_entry], :]

    # rename
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

    for col in df.select_dtypes(include="object"):
        df[col] = df[col].apply(_clean)

    # filter
    if benchmark:
        df = df[df["construct_name"].str.lower() == benchmark.lower()]
        if df.empty:
            raise ValueError(f"No benchmark found: {benchmark}")

    print(f"\nRows: {len(df)}")

    semaphore = asyncio.Semaphore(5)  # limit concurrency

    if "construct_name" not in df.columns:
        raise ValueError("CSV is missing required column: construct_name")

    if benchmark:
        df = df[df["construct_name"].str.lower() == benchmark.lower()]
        if df.empty:
            raise ValueError(f"No benchmark found: {benchmark}")

    print(f"\nRows: {len(df)}")

    client = None
    model = None
    if not dry_run:
        config = load_config("config.json")
        client = make_client(config["models"]["generator"])
        model = get_model_name(config, "generator")

    semaphore = asyncio.Semaphore(5)

    async def process_row(row: pd.Series) -> str | None:
        async with semaphore:
            name = row.get("construct_name")
            if not name:
                return None

            slug = _slugify(name)
            path = os.path.join(results_root, slug, "goal.json")

            if dry_run:
                print(f"[DRY] {path}")
                return path

            goal = await _build_goal_with_llm(client, model, row)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(goal, f, indent=2)

            print(f"[✓] {slug}")
            return path

    tasks = [process_row(row) for _, row in df.iterrows()]
    results = await asyncio.gather(*tasks)

    written = [r for r in results if r]
    print(f"\nDone: {len(written)} files")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="benchmark_submission.csv")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--row-entry", type=int, required=False)

    args = parser.parse_args()

    asyncio.run(
        ingest(
            csv_path=args.csv,
            results_root=args.results_root,
            dry_run=args.dry_run,
            benchmark=args.benchmark,
            row_entry=args.row_entry, 
        )
    )