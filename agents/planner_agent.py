import json
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Load environment variables (like GOOGLE_API_KEY) from .env file
load_dotenv()

def generate_cleaning_plan(profile: dict, feedback: str = None) -> dict:
    """
    Send dataset profile to Gemini 1.5 Flash and get cleaning plan using LangChain.
    """
    # Switch from Ollama to Google's cloud API for extreme speed and context window
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    base_prompt = """
Act as a Data Engineer. Generate a JSON cleaning plan based on this Profile.

### Logic Rules:
- drop_duplicates: true if 'duplicate_rows' in dataset_level_issues.
- Action Selection (You may apply multiple actions to the same column sequentially):
    - business logic: If a column is 'age' or 'total_spent', sequence "replace_negatives_with_null" FIRST so negative anomalies turn into nulls and can be imputed.
    - semantic_type: numeric & nulls > 5% -> median_imputation
    - semantic_type: numeric & nulls < 5% -> mean_imputation
    - semantic_type: categorical & nulls > 0 -> fill_unknown (Do NOT use mode_imputation for personal attributes like Gender/City, use "fill_unknown" instead).
    - detected_issues: "invalid_dates" -> coerce_datetime
- IMPORTANT: Order matters! E.g. ["replace_negatives_with_null", "median_imputation"]. Output multiple actions if needed.
- Note: Outlier clipping and text normalization are handled completely automatically, DO NOT issue commands for them.

### Dataset Profile:
{profile}
"""

    if feedback:
        base_prompt += """
### USER FEEDBACK:
The user reviewed the previous plan and requested the following modifications:
{feedback}
Please ensure you adjust the plan according to this feedback.
"""

    base_prompt += """
### Response Format (Strict JSON):
{{
 "drop_duplicates": bool,
 "column_actions": [
   {{"column": "name", "actions": ["action_type_1", "action_type_2"]}}
 ]
}}
"""

    prompt = PromptTemplate.from_template(base_prompt)
    
    chain = prompt | llm
    
    inputs = {"profile": json.dumps(profile, indent=1)}
    if feedback:
        inputs["feedback"] = feedback
        
    response = chain.invoke(inputs)
    
    content = response.content
    
    # Strip Markdown blocks
    if "```json" in content:
        content = content.split("```json")[-1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[-1].split("```")[0].strip()
        
    try:
        plan = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Failed to decode LLM response. Raw Content: {content}")
        raise e

    return plan