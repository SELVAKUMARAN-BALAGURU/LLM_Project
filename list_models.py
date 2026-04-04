import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

try:
    print("Available Gemini Models:")
    models = genai.list_models()
    count = 0
    for m in models:
        print(f"- {m.name} (Methods: {m.supported_generation_methods})")
        count += 1
    print(f"Total models found: {count}")
except Exception as e:
    print("Error calling list_models:", e)
