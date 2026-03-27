
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

def _sample_demographics(submission: dict) -> str : 
    """
    Takes a submission from an expert, samples demographics from expert-specified demographics. If expert does not specify demographics, 
    then automatically sample from a demographics bank. 

    Returns: 
    demographics = {
        race: "Black or African American", 
        gender: "female", 
        age: "Adolescent (13-17)"
    }
    """
    target_populations = submission.get('target_population', None)
    
    if not target_populations: 
        raise ValueError(f"No target population specified in benchmark submission tag in `goal.json` file.")
    
    # If experts do not specify a gender, age, or ethnicity, we retrieve from the bank 
    gender = target_populations.get('gender', GENDER_BANK)
    age = target_populations.get('age', AGE_BANK)
    race = target_populations.get('ethnicity', RACE_BANK)
    
    gender_demo = random.choice(gender)
    age_demo = random.choice(age)
    race_demo = random.choice(race)
    
    return f"""
    TARGET POPULATION: 
    
    GENDER: {gender_demo}
    AGE: {age_demo}
    RACE: {race_demo}
    
    """


    
    

    