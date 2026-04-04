import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    resp = llm.invoke("Hello world")
    print("Success flash-latest:", resp.content)
except Exception as e:
    print("Failed flash-latest:", e)

try:
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    resp = llm.invoke("Hello world")
    print("Success pro:", resp.content)
except Exception as e:
    print("Failed pro:", e)
