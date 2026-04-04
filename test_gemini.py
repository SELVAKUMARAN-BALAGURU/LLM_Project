import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
print("API Key loaded:", bool(os.getenv("GOOGLE_API_KEY")))

try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    resp = llm.invoke("Hello world")
    print("Success:", resp.content)
except Exception as e:
    import traceback
    traceback.print_exc()
