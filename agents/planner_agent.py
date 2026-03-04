import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b"


def generate_cleaning_plan(profile: dict) -> dict:
    """
    Send dataset profile to LLaMA and get cleaning plan.
    """

    prompt = f"""
You are a professional data engineer.

Analyze the dataset profile and generate a cleaning plan.

Use:
- semantic_type
- null_percentage
- detected issues
- dataset-level issues
- dataset_health_score

to decide appropriate cleaning strategies.

Rules:
- Return STRICT JSON only
- No explanations
- Use this format:

{{
 "drop_duplicates": true or false,
 "column_actions": [
   {{"column": "column_name", "action": "median_imputation"}},
   {{"column": "column_name", "action": "normalize_text"}}
 ]
}}

Allowed actions:
- median_imputation
- mean_imputation
- mode_imputation
- normalize_text
- remove_outliers
- clip_outliers

Dataset profile:
{json.dumps(profile, indent=2)}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.2
        }
    )

    result = response.json()["response"]

    # extract JSON
    try:
        plan = json.loads(result)
    except json.JSONDecodeError:
        raise ValueError("LLM returned invalid JSON")

    return plan