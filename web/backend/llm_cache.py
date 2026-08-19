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


def ordered_llms(preferred_label, llm_openai, llm_groq, llm_claude, llm_ollama, labels: list) -> list:
    """
    Available LLM instances ordered with `preferred_label`'s resolved
    instance first (falling back to labels[0]'s if preferred_label is
    None/not given), followed by every other distinct configured
    provider — for services.llm_setup.invoke_with_fallback, so a
    feature that hits a failure on the preferred provider (rate limit,
    exhausted billing, an outage) retries the next configured one
    instead of failing outright.
    """
    preferred = None
    if preferred_label:
        preferred = resolve_llm(preferred_label, llm_openai, llm_groq, llm_claude, llm_ollama)
    if preferred is None and labels:
        preferred = resolve_llm(labels[0], llm_openai, llm_groq, llm_claude, llm_ollama)

    ordered = [preferred] if preferred is not None else []
    for llm in (llm_openai, llm_groq, llm_claude, llm_ollama):
        if llm is not None and not any(llm is o for o in ordered):
            ordered.append(llm)
    return ordered


def label_for_llm(llm_instance, llm_openai, llm_groq, llm_claude, llm_ollama, labels: list) -> str | None:
    """
    Reverse of resolve_llm — the label (as shown to the user/returned in
    an API response) for a specific already-resolved LLM instance.
    Needed after services.llm_setup.invoke_with_fallback picks a
    provider from an ordered_llms() list, to honestly report which one
    actually answered rather than just the one originally preferred
    (multiple "Local · ..." labels all resolve to the same llm_ollama
    instance — the first matching label is reported, same simplification
    resolve_llm already makes for that case).
    """
    for label in labels:
        if resolve_llm(label, llm_openai, llm_groq, llm_claude, llm_ollama) is llm_instance:
            return label
    return None
