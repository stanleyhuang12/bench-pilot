"""
construct-string-matching.py

Given a natural-language user prompt and the flat list of construct strings
produced by `results/contextual-constructs/contextual-constructs.json`,
ask an LLM to return the 5 most relevant-yet-diverse construct strings.

Usage:
    export OPENAI_API_KEY=sk-...
    python user\ experience/construct-string-matching.py \
        --prompt "I am a school teacher worried about the use of AI chatbots..." \
        --output user\ experience/testapr21-construct-string-matching.json

Notes:
- "Diverse" means the returned strings should collectively cover different
  facets of the user's concern (e.g. learning, attention, autonomy, emotional
  wellbeing), not five near-duplicate items about the user's single stated
  worry. The LLM is prompted accordingly.
- Uses the OpenAI SDK by default. The core picking logic is delegated to the
  model — this script is the plumbing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONSTRUCTS = (
    REPO_ROOT / "results" / "contextual-constructs" / "contextual-constructs.json"
)

SYSTEM_PROMPT = """\
You help designers map a user's free-text concern onto a curated catalog of
"construct strings" drawn from AI safety & wellbeing benchmarks. Each construct
string has the shape "<construct item> for <demographic>" (e.g.
"Encourage Learning for teenagers", "parasocial attachment for teenagers").

Given a user prompt and the full catalog, return the 5 construct strings that
are:

1. RELEVANT — substantively connected to the user's situation, stakeholders,
   or underlying concerns (including concerns the user may not have named
   explicitly but that clearly apply to their context).
2. DIVERSE — collectively cover different facets of the situation. Do not
   return 5 near-duplicates of the most obvious match. Aim to span different
   pillars (psychological / physical / social) and different underlying
   constructs. The goal is a well-rounded picture of what the user should
   care about, not just a literal restatement of their prompt.

Return STRICT JSON with this shape and nothing else:

{
  "selections": [
    {
      "construct_string": "<exact string from the catalog>",
      "rationale": "<1-2 sentence justification>"
    },
    ... 5 total ...
  ]
}

Only emit construct strings that appear verbatim in the supplied catalog.
"""


def load_catalog(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["constructs"]


def build_user_message(user_prompt: str, catalog: list[dict]) -> str:
    lines = [f"USER PROMPT:\n{user_prompt}", "", "CATALOG:"]
    for row in catalog:
        lines.append(
            f"- {row['construct_string']} "
            f"[pillar={row['pillar_label']}; benchmark={row['benchmark_slug']}]"
        )
    return "\n".join(lines)


def call_openai(system: str, user: str, model: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit(
            "openai package not installed. `pip install openai` and set "
            "OPENAI_API_KEY."
        )
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return resp.choices[0].message.content


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True, help="Natural-language user prompt.")
    p.add_argument(
        "--catalog",
        default=str(DEFAULT_CONSTRUCTS),
        help="Path to contextual-constructs.json.",
    )
    p.add_argument("--model", default="gpt-4o", help="LLM model id.")
    p.add_argument(
        "--output", required=True, help="Where to write the JSON result."
    )
    args = p.parse_args()

    catalog = load_catalog(Path(args.catalog))
    user_msg = build_user_message(args.prompt, catalog)

    raw = call_openai(SYSTEM_PROMPT, user_msg, args.model)
    parsed = json.loads(raw)

    # Enrich each selection with the catalog row so the designer has full context.
    lookup = {row["construct_string"]: row for row in catalog}
    enriched = []
    for sel in parsed.get("selections", []):
        cs = sel.get("construct_string")
        row = lookup.get(cs, {})
        enriched.append({
            "construct_string": cs,
            "rationale": sel.get("rationale", ""),
            "pillar": row.get("pillar_label"),
            "benchmark_slug": row.get("benchmark_slug"),
            "benchmark_name": row.get("benchmark_name"),
            "demographic_label": row.get("demographic_label"),
        })

    output = {
        "prompt": args.prompt,
        "model": args.model,
        "system_prompt": SYSTEM_PROMPT,
        "selections": enriched,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"wrote {out_path} ({len(enriched)} selections)")


if __name__ == "__main__":
    main()
