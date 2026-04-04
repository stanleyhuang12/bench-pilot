"""
demographics.py — Demographic factor banks and methods for enumerating combinations..
"""

import itertools
from typing import Optional

GENDER_BANK = [
    "male",
    "female",
    "nonbinary",
]

RACE_BANK = [
    "American Indian or Alaska Native",
    "Asian",
    "Black or African American",
    "Hispanic or Latino",
    "Middle Eastern or North African",
    "Native Hawaiian or Pacific Islander",
    "White",
    "Multiracial",
]

AGE_BANK = [
    "Child or teenager (6-17)", 
    "Adult (18+)", 
]


# AGE_BANK = [
#     "Child (0–12)",
#     "Adolescent (13–17)",
#     "Young Adult (18–24)",
#     "25–34",
#     "35–44",
#     "45–54",
#     "55–64",
#     "65+",
# ]


FACTOR_MAP: dict[str, tuple[str, list[str]]] = {
    "gender": ("gender", GENDER_BANK),
    "age":    ("age",    AGE_BANK),
    "race":   ("race",   RACE_BANK),
}

VALID_FACTORS = set(FACTOR_MAP.keys())

#@ Public API 
def get_demographic_combinations(
    factors: list[str],
    goal: dict,
) -> list[dict]:
    """
    Return every combination of the requested demographic factors.

    Each combination is a dict with only the requested keys populated.
    Un-requested factors are absent (callers treat them as "any / neutral"
    """
    unknown = set(factors) - VALID_FACTORS
    if unknown:
        raise ValueError(
            f"Unknown demographic factor(s): {unknown}. "
            f"Valid options: {VALID_FACTORS}"
        )

    target_populations = goal.get("target_population", {})

    keys: list[str] = []
    pools: list[list[str]] = []

    for factor in factors:
        dict_key, default_pool = FACTOR_MAP[factor]
        # goal may override via "gender", "age", or "ethnicity" (legacy key for race)
        pool = (
            target_populations.get(factor)
            or target_populations.get("ethnicity" if factor == "race" else factor)
            or default_pool
        )
        keys.append(dict_key)
        pools.append(pool)

    return [dict(zip(keys, combo)) for combo in itertools.product(*pools)]


def combination_summary(factors: list[str], goal: dict) -> str:
    """
    Human-readable summary of the cartesian product, e.g.
    "gender(3) × race(8) = 24 combinations".
    Used in cost-tracking metadata.
    """
    target_populations = goal.get("target_population", {})
    parts = []
    total = 1
    for factor in factors:
        dict_key, default_pool = FACTOR_MAP[factor]
        pool = (
            target_populations.get(factor)
            or target_populations.get("ethnicity" if factor == "race" else factor)
            or default_pool
        )
        parts.append(f"{factor}({len(pool)})")
        total *= len(pool)
    return " × ".join(parts) + f" = {total}"


def format_demographic(demographic: dict) -> str:
    """Render a demographic dict as a tidy block for prompt injection."""
    lines = []
    if "gender" in demographic:
        lines.append(f"Gender: {demographic['gender']}")
    if "age" in demographic:
        lines.append(f"Age group: {demographic['age']}")
    if "race" in demographic:
        lines.append(f"Ethnicity: {demographic['race']}")
    return "\n".join(lines)

def sample_demographics(goal: dict) -> dict:
    """Random sample across all three factors (legacy behaviour)."""
    import random
    target_populations = goal.get("target_population", {})
    return {
        "gender": random.choice(target_populations.get("gender", GENDER_BANK)),
        "age": random.choice(target_populations.get("age",    AGE_BANK)),
        "race": random.choice(
            target_populations.get("ethnicity", target_populations.get("race", RACE_BANK))
        ),
    }