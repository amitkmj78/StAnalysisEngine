"""DIF-6: portfolio impact preview for a hypothetical trade. Nothing is placed or saved."""

import re
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from services.data_service import get_effective_price
from services.portfolio_health_service import compute_portfolio_beta
from services.portfolio_review_service import compute_sectors
from services.trade_impact_service import apply_trade, compare
from web.backend.auth import verify_bearer_token
from web.backend.db import service_conn, user_conn
from web.backend.rate_limit import enforce_daily_quota, limiter
from web.backend.routers.portfolio import _resolve_portfolio_id

router = APIRouter(prefix="/api/v1/portfolio", tags=["portfolio"], dependencies=[Depends(verify_bearer_token)])

TICKER_RE = re.compile(r"^[A-Z][A-Z.\-]{0,9}$")
SCORE_UNIVERSE = "All"


class TradeImpactRequest(BaseModel):
    portfolio_id: Optional[int] = None
    ticker: str
    side: Literal["buy", "sell"]
    shares: float = Field(gt=0, le=1_000_000)


def _beta(positions: list[dict]) -> Optional[float]:
    if not positions or sum(p["market_value"] for p in positions) <= 0:
        return None
    try:
        return compute_portfolio_beta(positions, "SPY", "1y").get("beta")
    except Exception:
        return None


@router.post("/trade-impact")
@limiter.limit("30/minute")
async def trade_impact(request: Request, body: TradeImpactRequest):
    await enforce_daily_quota(request, "portfolio/trade-impact")
    ticker = body.ticker.strip().upper()
    if not TICKER_RE.match(ticker):
        raise HTTPException(422, "That is not a valid ticker symbol.")

    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        portfolio_id = await _resolve_portfolio_id(conn, user_id, body.portfolio_id)
        rows = await conn.fetch(
            "SELECT ticker, shares, current_price FROM portfolio_positions WHERE portfolio_id = $1",
            portfolio_id,
        )

    tickers = sorted({r["ticker"] for r in rows} | {ticker})
    sectors = await run_in_threadpool(compute_sectors, tickers)
    async with service_conn() as conn:
        score_rows = await conn.fetch(
            """
            SELECT DISTINCT ON (ticker) ticker, short_score FROM stock_scores
            WHERE universe_id = $1 AND ticker = ANY($2::text[]) AND short_score IS NOT NULL
            ORDER BY ticker, as_of_date DESC
            """,
            SCORE_UNIVERSE, tickers,
        )
    scores = {r["ticker"]: float(r["short_score"]) for r in score_rows}

    before = []
    price_by_ticker: dict[str, float] = {}
    for r in rows:
        shares = float(r["shares"] or 0)
        price = float(r["current_price"] or 0)
        if price <= 0:
            continue  # no price, so it cannot be valued; reported through coverage below
        price_by_ticker[r["ticker"]] = price
        before.append({
            "ticker": r["ticker"], "shares": shares, "market_value": shares * price,
            "sector": sectors.get(r["ticker"]), "short_score": scores.get(r["ticker"]),
        })

    trade_price = price_by_ticker.get(ticker)
    if trade_price is None:
        trade_price = await run_in_threadpool(get_effective_price, ticker)
        if not trade_price or trade_price <= 0:
            raise HTTPException(422, f"No current price for {ticker}, so the preview cannot be calculated.")
    held = next((h for h in before if h["ticker"] == ticker), None)
    if body.side == "sell" and held is None:
        raise HTTPException(422, f"You do not hold {ticker} in this portfolio.")

    try:
        after = apply_trade(
            before, ticker, body.side, body.shares, float(trade_price),
            sectors.get(ticker), scores.get(ticker),
        )
    except ValueError as e:
        raise HTTPException(422, str(e))

    beta_before = await run_in_threadpool(_beta, [{"ticker": h["ticker"], "market_value": h["market_value"]} for h in before])
    beta_after = await run_in_threadpool(_beta, [{"ticker": h["ticker"], "market_value": h["market_value"]} for h in after])
    result = compare(before, after, beta_before, beta_after)
    result["ticker"] = ticker
    result["side"] = body.side
    result["shares"] = body.shares
    result["trade_price"] = round(float(trade_price), 4)
    result["note"] = (
        "Hypothetical preview. Nothing has been placed or saved. Scores are the app's model outputs, not advice. "
        "Holdings without a current price are left out of these numbers."
    )
    return result
