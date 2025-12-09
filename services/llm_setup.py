from typing import List, Tuple, Optional

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from .config import GROQ_API_KEY, OPENAI_API_KEY


def init_llms() -> Tuple[Optional[ChatOpenAI], Optional[ChatGroq], List[str]]:
    """
    Initialize available LLMs based on env keys.
    Returns: (llm_openai, llm_groq, available_labels)
    """
    llm_openai = None
    llm_groq = None
    labels: List[str] = []

    if OPENAI_API_KEY:
        llm_openai = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4,
        )
        labels.append("OpenAI GPT")

    if GROQ_API_KEY:
        llm_groq = ChatGroq(
            model="llama3-70b-8192",
            temperature=0.1,
            groq_api_key=GROQ_API_KEY,
        )
        labels.append("Groq Llama3-70B")

    return llm_openai, llm_groq, labels
