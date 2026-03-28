
"""
Helper module to create diverse scenarios prompts 
"""
import random

# Demographics dictionary 
# Gender 
GENDER_BANK = [
    "male",
    "female",
    "nonbinary",
]

# Race / Ethnicity
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

# Age Bank 
AGE_BANK = [
    "Child (0–12)",
    "Adolescent (13–17)",
    "Young Adult (18–24)",
    "25–34",
    "35–44",
    "45–54",
    "55–64",
    "65+"
]

def sample_demographics(goal: dict) -> dict:
    target_populations = goal.get('target_population', {})

    gender_pool = target_populations.get('gender', GENDER_BANK)
    age_pool = target_populations.get('age', AGE_BANK)
    race_pool = target_populations.get('ethnicity', RACE_BANK)

    return {
        "gender": random.choice(gender_pool),
        "age": random.choice(age_pool),
        "race": random.choice(race_pool),
    }
    

def sample_formatted_demographics(goal:dict ) -> dict: 
        demographic_dict = sample_demographics(goal)
        return f"""
GENDER: {demographic_dict['gender']}
AGE: {demographic_dict['age']}
RACE: {demographic_dict['race']}
"""

def sample_demographics_batch(goal: dict, n: int) -> list[str]:
    """Return n independently sampled demographic strings."""
    return [sample_formatted_demographics(goal) for _ in range(n)]
 

    