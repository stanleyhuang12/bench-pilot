"""
0-ingest.py — Phase 0: Convert benchmark_submission.csv into per-benchmark goal.json files.
 
Each row in the CSV becomes one goal.json, written to:
    results/<benchmark-slug>/goal.json
 
The output format is exactly what generate.py (Phase 1) expects.
 
Usage:
    python 0-ingest.py
    python 0-ingest.py --csv path/to/benchmark_submission.csv
    python 0-ingest.py --csv benchmark_submission.csv --results-root results --dry-run
"""
from __future__ import annotations
 
import argparse
import json
import os
import re
 
import pandas as pd
 
df = pd.read_csv('benchmark_submission.csv')

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
    'Name the Construct: Provide a name for your proposed benchmark idea (Name/Concept).  ': "construct_name", 
}


def _slugify(name: str) -> str:
    """'Emotional Dependency' → 'emotional-dependency'"""
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
 
 
def _clean(val) -> str:
    """Return empty string for NaN/None, otherwise strip."""
    if pd.isna(val):
        return ""
    return str(val).strip()
 
 
def _split_examples(raw: str) -> list[str]:
    """
    Split a free-text examples field into a list.
    Handles newlines, numbered lists (1. / 1) ), and bullet-separated entries.
    """
    if not raw:
        return []
    # Try splitting on numbered list markers first: "1." / "1)"
    parts = re.split(r"\n\s*\d+[.)]\s*", raw)
    if len(parts) > 1:
        return [p.strip() for p in parts if p.strip()]
    # Fall back to splitting on blank lines
    parts = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
    return parts if parts else [raw.strip()]
 
 
def _build_description(row: pd.Series) -> str:
    """
    Combine construct + relate + justify into a rich description paragraph.
    Only non-empty fields are included.
    """
    parts = []
    if row.construct:
        parts.append(row.construct)
    if row.construct_relate:
        parts.append(f"Related constructs: {row.construct_relate}")
    if row.construct_justify:
        parts.append(f"Justification: {row.construct_justify}")
    return "\n\n".join(parts)
 
 
def _build_metric(row: pd.Series) -> dict:
    """
    Build a single metric dict in the format _normalise_metrics() expects:
      { metric_name, definition, type, examples }
    """
    pos = _split_examples(row.pos_examples)
    neg = _split_examples(row.neg_examples)
 
    # Tag examples so the evaluator knows polarity
    examples = (
        [f"[PASS] {e}" for e in pos]
        + [f"[FAIL] {e}" for e in neg]
    )
 
    # The judge prompt is the definition; fall back to construct if missing
    definition = row.llm_as_judge_prompt or row.construct
 
    return {
        "metric_name": row.construct_name,
        "definition":  definition,
        "type":        "binary",
        "examples":    examples,
    }
 
 
def _build_goal(row: pd.Series) -> dict:
    """
    Assemble the full goal.json structure that generate.py consumes.
 
    """
    return {
        "benchmark_name": row.construct_name,
        "description": _build_description(row),
 
        # Passed into build_scenario_prompt → scenario_cfg
        "scenario": {
            "user_context": row.user_context,
            "implicit_context": row.implicit_context,
            "example_messages": _split_examples(row.user_message),
        },
 
        # Consumed by sample_demographics_batch (demographics.py)
        "demographics": {
            "description": row.demographics,
        },
 
        # Consumed by _normalise_metrics → passed to evaluator
        "metric": [_build_metric(row)],
 
        # Metadata — not used by generate.py but useful for auditing
        "_meta": {
            "submitter_email": row.email,
            "timestamp": row.timestamp,
        },
    }
 

def ingest(
    csv_path: str = "benchmark_submission.csv",
    results_root: str = "results",
    dry_run: bool = False,
) -> list[str]:
    """
    Read the CSV, convert every row to a goal.json, and write it under
    results/<benchmark-slug>/goal.json.
 
    Returns a list of written file paths.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
 
    df = pd.read_csv(csv_path)
 
    # Only rename columns that actually exist in this CSV
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
 
    # Normalise all string columns
    str_cols = [c for c in df.columns if df[c].dtype == object]
    for col in str_cols:
        df[col] = df[col].apply(_clean)
 
    written: list[str] = []
    skipped: list[str] = []
 
    print(f"\n  CSV:          {csv_path}")
    print(f" Results root: {results_root}")
    print(f" Rows found: {len(df)}\n")
 
    for idx, row in df.iterrows():
        name = row.get("construct_name", "")
        if not name:
            print(f"  [{idx+1}] SKIP — missing construct_name")
            skipped.append(str(idx))
            continue
 
        slug      = _slugify(name)
        bench_dir = os.path.join(results_root, slug)
        goal_path = os.path.join(bench_dir, "goal.json")
 
        goal = _build_goal(row)
 
        if dry_run:
            print(f"  [{idx+1}] DRY RUN — would write: {goal_path}")
            print(f"   benchmark_name: {goal['benchmark_name']}")
            print(f"   metric:  {goal['metric'][0]['metric_name']}")
            written.append(goal_path)
            continue
 
        os.makedirs(bench_dir, exist_ok=True)
        with open(goal_path, "w") as f:
            json.dump(goal, f, indent=2)
 
        print(f"  [{idx+1}] ✓  {slug}")
        print(f"  → {goal_path}")
        written.append(goal_path)
 
    print(f"\n  Done — {len(written)} written, {len(skipped)} skipped.\n")
 
    # Print the next-step command for convenience
    if written and not dry_run:
        example_slug = _slugify(df.iloc[0].get("construct_name", "benchmark"))
        print("  Next step — run Phase 1 for a single benchmark:")
        print(f"    python generate.py --results-root {results_root} --config config.json")
        print(f"    (set goal_prompt in config.json to: {results_root}/{example_slug}/goal.json)\n")
 
    return written
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0: Ingest CSV → goal.json files")
    parser.add_argument("--csv", default="benchmark_submission.csv")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written without creating files")
    args = parser.parse_args()
 
    ingest(
        csv_path=args.csv,
        results_root=args.results_root,
        dry_run=args.dry_run,
    )


