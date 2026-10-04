from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from Agent.meta_agent import ask_meta_agent, build_agent

from services.cited_analyst_service import answer_cited
from services.general_assistant_service import answer_general_question
from services.portfolio_health_service import compute_portfolio_risk_metrics
from services.portfolio_review_service import answer_portfolio_question, compute_sectors
from web.backend.auth import verify_bearer_token
from web.backend.db import user_conn
from web.backend.llm_cache import cached_init_llms, label_for_llm, ordered_llms
from web.backend.rate_limit import enforce_daily_quota, limiter
from web.backend.routers.portfolio import _resolve_portfolio_id

router = APIRouter(prefix="/api/v1/chat", tags=["chat"], dependencies=[Depends(verify_bearer_token)])


@router.get("/providers")
async def providers():
    _, _, _, _, labels = await run_in_threadpool(cached_init_llms)
    return {"providers": labels}


class ChatRequest(BaseModel):
    scope: Literal["ticker", "portfolio", "general"] = "ticker"
    ticker: Optional[str] = None
    portfolio_id: Optional[int] = None
    question: str
    provider: Optional[str] = None
    # DIF-8: answer only from cited stored sources, saying 'No clear cause found' when they don't cover it.
    cited: bool = False


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

    sources: list[dict] = []
    if body.cited and body.scope != "general":
        user_id = request.state.user["id"]
        if body.scope == "portfolio":
            async with user_conn(user_id) as conn:
                resolved_id = await _resolve_portfolio_id(conn, user_id, body.portfolio_id)
            answer, actual_llm, sources = await answer_cited(
                body.question.strip(), llms, user_id=user_id, portfolio_id=resolved_id,
            )
            result_ticker = "PORTFOLIO"
        else:
            if not body.ticker or not body.ticker.strip():
                raise HTTPException(422, "ticker is required when scope is 'ticker'")
            result_ticker = body.ticker.strip().upper()
            answer, actual_llm, sources = await answer_cited(body.question.strip(), llms, ticker=result_ticker)
    elif body.scope == "general":
        # No ticker or portfolio: answered from retrieved stored research (services/general_assistant_service.py).
        answer, actual_llm, sources = await answer_general_question(body.question.strip(), llms)
        result_ticker = "GENERAL"
    elif body.scope == "portfolio":
        answer, actual_llm = await _ask_portfolio(request, body, llms)
        result_ticker = "PORTFOLIO"
    else:
        if not body.ticker or not body.ticker.strip():
            raise HTTPException(422, "ticker is required when scope is 'ticker'")
        answer, actual_llm = await _ask_ticker(body, llms)
        result_ticker = body.ticker.strip().upper()

    actual_provider = provider
    if actual_llm is not None:
        actual_provider = label_for_llm(actual_llm, llm_openai, llm_groq, llm_claude, llm_ollama, labels) or provider

    return {"ticker": result_ticker, "provider": actual_provider, "answer": answer, "sources": sources}


async def _ask_ticker(body: ChatRequest, llms: list) -> tuple[str, object]:
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

    return answer, actual_llm


async def _ask_portfolio(request: Request, body: ChatRequest, llms: list) -> tuple[str, object]:
    """ASK-1: answer a question about the user's whole portfolio, from data
    this app has already computed -- see services.portfolio_review_service.
    answer_portfolio_question for the scope boundary (holdings + sectors +
    compute_portfolio_risk_metrics only; anything beyond that correctly
    comes back as "I don't know" via that function's own guardrail)."""
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        resolved_id = await _resolve_portfolio_id(conn, user_id, body.portfolio_id)
        records = await conn.fetch(
            "SELECT ticker, shares, current_price FROM portfolio_positions "
            "WHERE user_id = $1::uuid AND portfolio_id = $2",
            user_id, resolved_id,
        )

    positions = [
        {"ticker": r["ticker"], "shares": r["shares"], "market_value": (r["shares"] or 0) * (r["current_price"] or 0)}
        for r in records if r["ticker"]
    ]
    if not positions:
        return "Your portfolio has no positions yet — there's nothing to answer questions about.", None

    def _compute():
        sectors = compute_sectors([p["ticker"] for p in positions])
        for p in positions:
            p["sector"] = sectors.get(p["ticker"])
        risk = compute_portfolio_risk_metrics(positions, "1y")
        return answer_portfolio_question(llms, positions, risk, body.question)

    answer = await run_in_threadpool(_compute)
    return answer or "No LLM provider was available to answer.", None
