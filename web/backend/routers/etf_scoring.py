"""COV-2: ETF scoring endpoints. See services/etf_scoring_service.py's
module docstring for the two disclosed data-source limits (top-N holdings
only, tracking error only for funds with a clean benchmark index ticker)."""

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from services.etf_scoring_service import (
    compute_holdings_weighted_score,
    compute_portfolio_overlap,
    compute_tracking_error,
    etf_universe,
    fetch_etf_holdings,
    find_candidate,
    get_expense_ratios,
)
from web.backend.auth import verify_bearer_token
from web.backend.db import service_conn, user_conn

router = APIRouter(prefix="/api/v1/etf", tags=["etf-scoring"], dependencies=[Depends(verify_bearer_token)])


async def _scores_for(symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (ticker) ticker, short_score FROM stock_scores
            WHERE ticker = ANY($1::text[]) ORDER BY ticker, as_of_date DESC
            """,
            symbols,
        )
    return {r["ticker"]: float(r["short_score"]) for r in rows if r["short_score"] is not None}


@router.get("/list")
async def list_etfs():
    """A screener-style table across the whole curated ETF universe:
    expense ratio (reused from the Index Fund Screener) + holdings-
    weighted score for each. Tracking error and portfolio overlap are
    per-ticker only (see GET /{ticker}) -- computing both for all 59
    funds on every page load would mean 59 extra yfinance round trips."""
    tickers = etf_universe()
    expense_ratios = await run_in_threadpool(get_expense_ratios)

    rows = []
    for ticker in tickers:
        holdings = await run_in_threadpool(fetch_etf_holdings, ticker)
        scores = await _scores_for([h["symbol"] for h in holdings])
        weighted = compute_holdings_weighted_score(holdings, scores)
        candidate = find_candidate(ticker)
        rows.append({
            "ticker": ticker,
            "name": candidate.name if candidate else ticker,
            "category": candidate.category if candidate else None,
            "expense_ratio_pct": expense_ratios.get(ticker),
            **weighted,
        })
    return {"etfs": rows}


@router.get("/{ticker}")
async def get_etf(request: Request, ticker: str):
    ticker = ticker.upper()
    candidate = find_candidate(ticker)
    if candidate is None:
        raise HTTPException(404, f"{ticker} is not in this app's scored ETF universe.")

    holdings = await run_in_threadpool(fetch_etf_holdings, ticker)
    scores = await _scores_for([h["symbol"] for h in holdings])
    weighted = compute_holdings_weighted_score(holdings, scores)
    tracking_error = await run_in_threadpool(compute_tracking_error, ticker)
    expense_ratios = await run_in_threadpool(get_expense_ratios)

    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        position_rows = await conn.fetch("SELECT DISTINCT ticker FROM portfolio_positions WHERE user_id = $1::uuid", user_id)
    overlap = compute_portfolio_overlap(holdings, [r["ticker"] for r in position_rows])

    return {
        "ticker": ticker,
        "name": candidate.name,
        "category": candidate.category,
        "benchmark": candidate.benchmark,
        "expense_ratio_pct": expense_ratios.get(ticker),
        "tracking_error_vs_benchmark_pct": tracking_error,
        "tracking_error_benchmark_ticker": candidate.benchmark_index_ticker,
        "holdings": holdings,
        **weighted,
        "portfolio_overlap": overlap,
    }
