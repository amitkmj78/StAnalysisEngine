# services/llm_setup.py

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import os


def init_llms():
    """
    Initialize OpenAI + Groq safely.
    Returns:
        llm_openai, llm_groq, llm_labels
    """
    llm_openai = None
    llm_groq = None
    llm_labels = []

    # ----------------------------
    # TRY INITIALIZING OPENAI
    # ----------------------------
    openai_key = os.getenv("OPENAI_API_KEY")

    if openai_key:
        try:
            llm_openai = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,
                timeout=20,
            )
            llm_labels.append("GPT-4o mini")
        except Exception as e:
            print("OpenAI init failed:", e)
    else:
        print("⚠️ OPENAI_API_KEY not found. Skipping OpenAI models.")

    # ----------------------------
    # TRY INITIALIZING GROQ
    # ----------------------------
    groq_key = os.getenv("GROQ_API_KEY")

    if groq_key:
        try:
            llm_groq = ChatGroq(
                model="openai/gpt-oss-120b",   # SUPPORTED MODEL
                temperature=0.2,
                timeout=30,
            )
            llm_labels.append("Groq Model")
        except Exception as e:
                print(f"Groq model failed: {e}")
                llm_groq = None

    else:
        print("⚠️ GROQ_API_KEY not found. Skipping Groq models.")

    # ----------------------------
    # FAILSAFE: If NOTHING loads
    # ----------------------------
    if not llm_labels:
        print("❌ No LLMs initialized. Check API keys.")
        return None, None, []

    return llm_openai, llm_groq, llm_labels