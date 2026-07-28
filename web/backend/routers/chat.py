from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from Agent.meta_agent import ask_meta_agent, build_agent

from web.backend.auth import verify_bearer_token
from web.backend.llm_cache import cached_init_llms, resolve_llm
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

    llm = resolve_llm(provider, llm_openai, llm_groq, llm_claude, llm_ollama)

    agent = await run_in_threadpool(build_agent, llm)
    ticker = body.ticker.strip().upper()
    answer = await run_in_threadpool(ask_meta_agent, agent, ticker, body.question)

    return {"ticker": ticker, "provider": provider, "answer": answer}
