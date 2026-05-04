import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def get_llm():
    """
    Returns a configured LLM instance
    """

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "API_KEY is not set. Please set it before running."
        )

    return ChatOpenAI(
        model="google/gemini-2.0-flash-001",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=0,        
        max_tokens=1000,      # prevent over long responses
        timeout=30,
        verbose=True
    )