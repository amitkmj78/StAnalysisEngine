# services/llm_setup.py

import subprocess
import os

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama


def get_local_ollama_models():
    """Return a list of locally available Ollama model names."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.splitlines()[1:]  # skip header
        return [line.split()[0] for line in lines if line.strip()]
    except Exception as e:
        print("⚠️ Failed to list Ollama models:", e)
        return []


def init_llms():
    """
    Initialize OpenAI + Groq + Ollama safely.
    Returns:
        llm_openai, llm_groq, llm_ollama, llm_labels
    """
    llm_openai = None
    llm_groq = None
    llm_ollama = None
    llm_labels = []

    # ----------------------------
    # OPENAI
    # ----------------------------
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            llm_openai = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,
                timeout=20,
            )
            llm_labels.append("OpenAI · GPT-4o mini")
        except Exception as e:
            print("❌ OpenAI init failed:", e)
    else:
        print("⚠️ OPENAI_API_KEY not found. Skipping OpenAI models.")

    # ----------------------------
    # GROQ
    # ----------------------------
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            llm_groq = ChatGroq(
                model="openai/gpt-oss-120b",
                temperature=0.2,
                timeout=30,
            )
            llm_labels.append("Groq · GPT-OSS-120B")
        except Exception as e:
            print("❌ Groq init failed:", e)
            llm_groq = None
    else:
        print("⚠️ GROQ_API_KEY not found. Skipping Groq models.")

    # ----------------------------
    # OLLAMA (LOCAL)
    # ----------------------------
    ollama_models = get_local_ollama_models()

    if ollama_models:
        for model_name in ollama_models:
            label = f"Local · {model_name} (Ollama)"
            llm_labels.append(label)

        # pick ONE default local model (no logic change)
        try:
            llm_ollama = ChatOllama(
                model=ollama_models[0],  # first available model
                temperature=0.3,
            )
        except Exception as e:
            print("⚠️ Ollama init failed:", e)
            llm_ollama = None
    else:
        print("⚠️ No local Ollama models found.")

    # ----------------------------
    # FAILSAFE
    # ----------------------------
    if not llm_labels:
        print("❌ No LLMs initialized. Check API keys / Ollama.")
        return None, None, None, []

    return llm_openai, llm_groq, llm_ollama, llm_labels
