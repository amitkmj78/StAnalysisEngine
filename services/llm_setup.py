# services/llm_setup.py

import logging
import subprocess
import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic

logger = logging.getLogger(__name__)


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
        print("WARNING: Failed to list Ollama models:", e)
        return []


def init_llms():
    """
    Initialize OpenAI + Groq + Claude + Ollama safely.
    Returns:
        llm_openai, llm_groq, llm_claude, llm_ollama, llm_labels
    """
    llm_openai = None
    llm_groq = None
    llm_claude = None
    llm_ollama = None
    llm_labels = []

    # ----------------------------
    # OPENAI
    # ----------------------------
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            llm_openai = ChatOpenAI(
                model="gpt-4o",
                temperature=0.3,
                timeout=20,
            )
            llm_labels.append("OpenAI · GPT-4.0")
        except Exception as e:
            print("ERROR: OpenAI init failed:", e)
    else:
        print("WARNING: OPENAI_API_KEY not found. Skipping OpenAI models.")

    # ----------------------------
    # GROQ
    # ----------------------------
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            llm_groq = ChatGroq(
                model="openai/gpt-oss-20b",
                temperature=0.2,
                timeout=30,
            )
            llm_labels.append("Groq · gpt-oss-20b")
        except Exception as e:
            print("ERROR: Groq init failed:", e)
            llm_groq = None
    else:
        print("WARNING: GROQ_API_KEY not found. Skipping Groq models.")

    # ----------------------------
    # CLAUDE (ANTHROPIC)
    # ----------------------------
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            llm_claude = ChatAnthropic(
                model="claude-sonnet-5",
                temperature=0.3,
                timeout=30,
            )
            llm_labels.append("Claude · Sonnet 5")
        except Exception as e:
            print("ERROR: Claude init failed:", e)
            llm_claude = None
    else:
        print("WARNING: ANTHROPIC_API_KEY not found. Skipping Claude models.")

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
            print("WARNING: Ollama init failed:", e)
            llm_ollama = None
    else:
        print("WARNING: No local Ollama models found.")

    # ----------------------------
    # FAILSAFE
    # ----------------------------
    if not llm_labels:
        print("ERROR: No LLMs initialized. Check API keys / Ollama.")
        return None, None, None, None, []

    return llm_openai, llm_groq, llm_claude, llm_ollama, llm_labels


def invoke_with_fallback(llms: list, prompt: str) -> tuple[str, int]:
    """
    Tries each non-None LLM in `llms`, in order, returning (content,
    index) from the first one that succeeds — `index` is the position
    in `llms` that actually worked, so a caller reporting "which
    provider answered" can reflect reality if a fallback occurred
    rather than the one that was merely preferred.

    A single provider being down (rate limit, exhausted billing, an
    outage — e.g. OpenAI's `insufficient_quota`) no longer breaks a
    feature outright as long as another configured provider is
    healthy. Raises the last exception if every provider fails, so
    callers keep their existing "no provider available" handling for
    the genuine worst case.
    """
    last_exc: Exception | None = None
    for i, llm in enumerate(llms):
        if llm is None:
            continue
        try:
            response = llm.invoke(prompt)
            content = getattr(response, "content", str(response))
            return content, i
        except Exception as e:
            logger.warning("LLM call failed (provider %d/%d), trying next: %s", i + 1, len(llms), e)
            last_exc = e
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No LLM provider available.")
