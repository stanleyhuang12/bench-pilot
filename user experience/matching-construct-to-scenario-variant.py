"""
matching-construct-to-scenario-variant.py

Take the top-5 construct strings produced by `construct-string-matching.py`
and find the closest matching scenario variants across the bench-pilot
benchmarks, using an LLM to do the semantic match.

For each of the 5 input construct strings, the script:
  1. Loads every scenario (and its variants) from each benchmark's
     `results-p/<benchmark>/test.json`.
  2. Filters to scenarios whose benchmark slug matches the construct's
     `benchmark_slug` (since each construct is anchored to a benchmark).
  3. Asks the LLM to pick the single best-matching variant — preferring the
     variant whose demographic best aligns with the construct's
     `demographic_label` (e.g. teenagers vs adults), and whose scenario
     content most clearly exercises the construct.
  4. Returns a flat JSON of (construct_string -> best variant) pairs with
     a short rationale and the variant's full payload.

Usage:
    export OPENAI_API_KEY=sk-...
    python 'user experience/matching-construct-to-scenario-variant.py' \\
        --input 'user experience/testapr21-construct-string-matching.json' \\
        --tests-root results-p \\
        --output 'user experience/test-matching-construct-to-scenario-variant.json'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT = """\
You match a single benchmark "construct string" of the form
"<construct item> for <demographic>" to the single best scenario variant
from the supplied list.

Selection rules:
1. Demographic alignment is the first filter. If the construct says
   "for teenagers", strongly prefer variants whose age field falls in 6-17
   (or labelled "teenager", "13-17", "6-12"). If "for adults", prefer 18+.
   If "for general users", any demographic is acceptable.
2. Among demographically-aligned variants, pick the one whose scenario title,
   description, user_persona, and landmarks most directly exercise the
   construct. Look at what behavior the LLM under test would have to perform.
3. Tie-breaker: pick the variant with the richest landmarks / most explicit
   probe of the construct.

Return STRICT JSON, nothing else:

{
  "best_variant_id": "<exact scenario id, e.g. scenario_002_v03>",
  "rationale": "<1-2 sentences explaining why this variant best matches>"
}
"""


def load_top5(path: Path) -> dict:
    return json.loads(path.read_text())


def load_test_json(tests_root: Path, benchmark_slug: str) -> dict:
    return json.loads((tests_root / benchmark_slug / "test.json").read_text())


def variant_blob(scenario: dict) -> str:
    demo = scenario.get("demographic", "")
    if isinstance(demo, dict):
        demo = ", ".join(f"{k}={v}" for k, v in demo.items())
    landmarks = scenario.get("landmarks", [])
    lm_str = "; ".join(
        f"turn {lm.get('turn')}: {lm.get('instruction','')[:120]}"
        for lm in landmarks
    ) if landmarks else "(none)"
    return (
        f"id: {scenario.get('id')}\n"
        f"  title: {scenario.get('title','')}\n"
        f"  description: {scenario.get('description','')}\n"
        f"  user_persona: {scenario.get('user_persona','')[:300]}\n"
        f"  user_goal: {scenario.get('user_goal','')}\n"
        f"  demographic: {demo}\n"
        f"  landmarks: {lm_str}"
    )


def build_user_message(construct: dict, scenarios: list[dict]) -> str:
    parts = [
        f"CONSTRUCT STRING: {construct['construct_string']}",
        f"  pillar: {construct.get('pillar')}",
        f"  benchmark: {construct.get('benchmark_slug')}",
        f"  demographic_label: {construct.get('demographic_label')}",
        "",
        "CANDIDATE SCENARIO VARIANTS:",
    ]
    for sc in scenarios:
        parts.append(variant_blob(sc))
        parts.append("")
    return "\n".join(parts)


def call_openai(system: str, user: str, model: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("openai package not installed. `pip install openai`.")
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return resp.choices[0].message.content


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input", required=True,
        help="Path to construct-string-matching output JSON.",
    )
    p.add_argument(
        "--tests-root", default=str(REPO_ROOT / "results-p"),
        help="Root containing <benchmark_slug>/test.json folders.",
    )
    p.add_argument("--model", default="gpt-4o", help="LLM model id.")
    p.add_argument("--output", required=True, help="Where to write JSON.")
    args = p.parse_args()

    top5 = load_top5(Path(args.input))
    tests_root = Path(args.tests_root)

    matches = []
    for sel in top5["selections"]:
        bslug = sel["benchmark_slug"]
        try:
            test_data = load_test_json(tests_root, bslug)
        except FileNotFoundError:
            matches.append({
                "construct_string": sel["construct_string"],
                "benchmark_slug": bslug,
                "error": f"no test.json found at {tests_root}/{bslug}",
            })
            continue

        scenarios = test_data.get("scenarios", [])
        if not scenarios:
            matches.append({
                "construct_string": sel["construct_string"],
                "benchmark_slug": bslug,
                "error": "no scenarios in test.json",
            })
            continue

        user_msg = build_user_message(sel, scenarios)
        raw = call_openai(SYSTEM_PROMPT, user_msg, args.model)
        parsed = json.loads(raw)

        best_id = parsed.get("best_variant_id")
        best_variant = next((s for s in scenarios if s.get("id") == best_id), None)

        matches.append({
            "construct_string": sel["construct_string"],
            "benchmark_slug": bslug,
            "demographic_label": sel.get("demographic_label"),
            "best_variant_id": best_id,
            "rationale": parsed.get("rationale", ""),
            "variant": best_variant,
        })

    output = {
        "input_file": str(args.input),
        "model": args.model,
        "system_prompt": SYSTEM_PROMPT,
        "matches": matches,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"wrote {out_path} ({len(matches)} matches)")


if __name__ == "__main__":
    main()
