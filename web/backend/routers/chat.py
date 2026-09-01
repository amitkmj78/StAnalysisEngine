from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from Agent.meta_agent import ask_meta_agent, build_agent

from web.backend.auth import verify_bearer_token
from web.backend.llm_cache import cached_init_llms, label_for_llm, ordered_llms
from web.backend.rate_limit import enforce_daily_quota, limiter

router = APIRouter(prefix="/api/v1/chat", tags=["chat"], dependencies=[Depends(verify_bearer_token)])


@router.get("/providers")
async def providers():
    _, _, _, _, labels = await run_in_threadpool(cached_init_llms)
    return {"providers": labels}


class ChatRequest(BaseModel):
    ticker: str
    question: str
    provider: Optional[str] = None


@router.post("/ask")
@limiter.limit("5/minute")
async def ask(request: Request, body: ChatRequest):
    # Tighter limit than every other endpoint — this is the one place real
    # per-call LLM cost is incurred, and tool-calling chains are slow.
    await enforce_daily_quota(request, "chat/ask")

    if not body.question.strip():
        raise HTTPException(422, "question must not be empty")

    llm_openai, llm_groq, llm_claude, llm_ollama, labels = await run_in_threadpool(cached_init_llms)
    if not labels:
        raise HTTPException(503, "No LLM providers are currently configured on the server.")

    provider = body.provider or labels[0]
    if provider not in labels:
        raise HTTPException(422, f"provider must be one of {labels}")

    llms = ordered_llms(provider, llm_openai, llm_groq, llm_claude, llm_ollama, labels)
    ticker = body.ticker.strip().upper()

    # Each provider needs its own agent (bind_tools is provider-specific,
    # can't reuse one agent across models) — try the requested provider
    # first, and if it fails (down, rate-limited, exhausted billing —
    # ask_meta_agent reports that as a "❌ Meta-agent crashed" string
    # rather than raising), fall through to the next configured one so a
    # single provider outage doesn't take down chat entirely. Same
    # treatment for the "empty message" fallback (ask_meta_agent's own
    # last resort when a model returns no usable text at all) — that's
    # not a real answer either, and another configured provider is worth
    # trying before showing the user a bare warning.
    answer = "No LLM provider was available to answer."
    actual_llm = None
    for candidate in llms:
        agent = await run_in_threadpool(build_agent, candidate)
        answer = await run_in_threadpool(ask_meta_agent, agent, ticker, body.question)
        if not answer.startswith("❌ Meta-agent crashed:") and not answer.startswith("⚠️ Meta-agent responded"):
            actual_llm = candidate
            break

    actual_provider = provider
    if actual_llm is not None:
        actual_provider = label_for_llm(actual_llm, llm_openai, llm_groq, llm_claude, llm_ollama, labels) or provider

    return {"ticker": ticker, "provider": actual_provider, "answer": answer}
