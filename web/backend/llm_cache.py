from services.cache_utils import ttl_cache
from services.llm_setup import init_llms

# init_llms() does real work (constructs LLM clients, shells out to `ollama
# list`) — cache briefly so repeated requests across chat and prediction
# narrative endpoints don't each redo that independently.
cached_init_llms = ttl_cache(maxsize=1, ttl_seconds=300)(init_llms)


def resolve_llm(label: str, llm_openai, llm_groq, llm_claude, llm_ollama):
    if label.startswith("Groq"):
        return llm_groq
    if label.startswith("Claude"):
        return llm_claude
    if label.startswith("Local"):
        return llm_ollama
    return llm_openai
