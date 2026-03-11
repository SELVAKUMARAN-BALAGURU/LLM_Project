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
- If any issues are detected in a column, include that column in the plan with appropriate action.
- If outliers are detected, suggest "clip_outliers" for that column.
- If dtype is numeric and nulls > 20%, suggest "median_imputation".
- If dtype is categorical or string and nulls > 20%, suggest "mode_imputation".
- If semantic_type is datetime then suggest "datetime_imputation" for nulls.
- If there is datetime issue detected, suggest "datetime_imputation".
- Use this format:

{{
 "drop_duplicates": true or false,
 "column_actions": [
   {{"column": "column_name", "action": "appropriate_action"}},
   {{"column": "column_name", "action": "appropriate_action"}},
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