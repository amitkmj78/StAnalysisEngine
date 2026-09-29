"""
Stock detail page endpoints (docs/stock-analysis-requirements.html,
DET-1..5) backing the /stock/[ticker] frontend page. Fundamentals/
earnings/dividends/peers/signal-history are public, unauthenticated --
same posture as GET /stock-scores/{ticker} (general market research, not
user data). /position is the one user-specific route, auth-gated.
"""

import asyncio

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from services.cache_utils import ttl_cache
from services.data_service import get_latest_price
from services.ranking_utils import compute_position_concentration
from services.sentiment_service import score_ticker_sentiment
from services.stock_detail_service import (
    evaluate_signal_history,
    next_earnings_date,
    past_earnings_dates,
    recent_dividends,
    select_peers,
)
from services.stock_finder_service import _gics_sector, get_stock_finder_table
from services.yfinance_cache import get_cached_dividends, get_cached_earnings_dates, get_cached_history
from web.backend.auth import verify_bearer_token
from web.backend.db import service_conn, user_conn
from web.backend.llm_cache import cached_init_llms, ordered_llms
from web.backend.rate_limit import enforce_daily_quota, limiter
from web.backend.routers.portfolio import _resolve_portfolio_id

router = APIRouter(prefix="/api/v1/stock", tags=["stock-detail"])

# DET-1's range buttons.
PRICE_HISTORY_RANGES = {
    "1D": "1d",
    "5D": "5d",
    "1M": "1mo",
    "6M": "6mo",
    "1Y": "1y",
    "5Y": "5y",
}

# yfinance's default daily bar for period="1d" is a single row -- not a
# chart. 5-minute bars, regular trading hours only (no prepost), is the
# one genuinely new piece of plumbing here: no other chart in this app
# is intraday.
INTRADAY_RANGE = "1D"
INTRADAY_INTERVAL = "5m"


@router.get("/{ticker}/detail")
@limiter.limit("60/minute")
async def get_stock_detail(request: Request, ticker: str):
    """DET-1: key stats, fundamentals, earnings date, dividends. Current
    signals/reasons are deliberately NOT duplicated here -- the frontend
    already calls GET /stock-scores/{ticker} (Stage A) for that, one
    source of truth rather than two endpoints that could drift.

    past_earnings feeds DET-4's chart markers -- reads the same
    already-fetched earnings_dates frame next_earnings does, no second
    fetch."""
    ticker = ticker.upper()
    async with service_conn() as conn:
        fundamentals_row = await conn.fetchrow(
            """
            SELECT forward_pe, revenue_growth_pct, earnings_growth_pct, sector, as_of_date
            FROM pit_fundamentals WHERE ticker = $1 ORDER BY as_of_date DESC LIMIT 1
            """,
            ticker,
        )

    price, earnings_dates, dividends = await asyncio.gather(
        run_in_threadpool(get_latest_price, ticker),
        run_in_threadpool(get_cached_earnings_dates, ticker),
        run_in_threadpool(get_cached_dividends, ticker),
    )

    sector = _gics_sector(fundamentals_row["sector"]) if fundamentals_row and fundamentals_row["sector"] else None

    return {
        "ticker": ticker,
        "current_price": price,
        "sector": sector,
        "fundamentals": {
            "forward_pe": fundamentals_row["forward_pe"] if fundamentals_row else None,
            "revenue_growth_pct": fundamentals_row["revenue_growth_pct"] if fundamentals_row else None,
            "earnings_growth_pct": fundamentals_row["earnings_growth_pct"] if fundamentals_row else None,
            "as_of_date": str(fundamentals_row["as_of_date"]) if fundamentals_row else None,
        },
        "next_earnings": next_earnings_date(earnings_dates),
        "past_earnings": past_earnings_dates(earnings_dates),
        "recent_dividends": recent_dividends(dividends),
    }


@router.get("/{ticker}/price-history")
@limiter.limit("60/minute")
async def get_stock_price_history(request: Request, ticker: str, range: str = Query("1Y")):
    """DET-1's price chart. No existing endpoint anywhere in this app
    returns raw OHLC/close history for a single ticker on demand (every
    other chart is forecast/prediction-derived, or PIT-store closes --
    which only go back to 2026-08-05, not enough for "5Y") -- this is
    genuinely new plumbing, wrapping the shared yfinance history cache
    rather than a fresh fetch layer. "1D" fetches 5-minute bars instead
    of the usual daily bar (also new -- no chart in this app is intraday
    anywhere else), so `date` carries a full timestamp for that range
    instead of just a calendar date."""
    ticker = ticker.upper()
    range = range.upper()
    period = PRICE_HISTORY_RANGES.get(range)
    if period is None:
        raise HTTPException(status_code=400, detail=f"range must be one of {sorted(PRICE_HISTORY_RANGES)}")

    intraday = range == INTRADAY_RANGE
    interval = INTRADAY_INTERVAL if intraday else None
    history = await run_in_threadpool(get_cached_history, ticker, period, True, interval)
    return {
        "ticker": ticker,
        "range": range,
        "history": [
            {"date": ts.isoformat() if intraday else ts.date().isoformat(), "close": round(float(row["Close"]), 2)}
            for ts, row in history.iterrows()
        ],
    }


@ttl_cache(maxsize=512, ttl_seconds=86400)
def _cached_ticker_sentiment(ticker: str) -> dict:
    """Keyed on ticker only (not the llms list -- unhashable) so this
    caches per stock per day (NFR-6), independent of which LLM provider
    happens to be configured when it's first requested."""
    llm_openai, llm_groq, llm_claude, llm_ollama, labels = cached_init_llms()
    if not labels:
        return {"label": None, "reasoning": None}
    llms = ordered_llms(None, llm_openai, llm_groq, llm_claude, llm_ollama, labels)
    return score_ticker_sentiment(ticker, llms)


@router.get("/{ticker}/sentiment", dependencies=[Depends(verify_bearer_token)])
@limiter.limit("20/minute")
async def get_stock_sentiment(request: Request, ticker: str):
    """DET-1's news/sentiment section. Real per-call web-search + LLM
    cost (services.sentiment_service.score_ticker_sentiment), so this is
    deliberately a separate, on-demand, auth+quota-gated endpoint --
    never bundled into GET /detail's main page load, the same
    cheap-vs-expensive split /signals/quant-vs-analyst/narrative already
    uses elsewhere in this app."""
    await enforce_daily_quota(request, "stock/sentiment")
    ticker = ticker.upper()
    result = await run_in_threadpool(_cached_ticker_sentiment, ticker)
    return {"ticker": ticker, "label": result["label"], "reasoning": result["reasoning"]}


@router.get("/{ticker}/position", dependencies=[Depends(verify_bearer_token)])
@limiter.limit("60/minute")
async def get_stock_position(request: Request, ticker: str, portfolio_id: int | None = Query(None)):
    """DET-2: the caller's own position in this ticker, if owned --
    shares, cost basis, gain/loss, weight in their portfolio. Reuses
    services.ranking_utils.compute_position_concentration for the weight
    figure (the same function the Portfolio page's own concentration
    flagging uses) rather than reimplementing that math."""
    ticker = ticker.upper()
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        resolved_portfolio_id = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        rows = await conn.fetch(
            "SELECT ticker, shares, avg_cost, current_price FROM portfolio_positions WHERE user_id = $1::uuid AND portfolio_id = $2",
            user_id, resolved_portfolio_id,
        )

    positions = [
        {"ticker": r["ticker"], "market_value": float(r["shares"] or 0) * float(r["current_price"] or r["avg_cost"] or 0)}
        for r in rows
    ]
    weight_by_ticker = {c["ticker"]: c["weight_pct"] for c in compute_position_concentration(positions)}

    match = next((r for r in rows if r["ticker"] == ticker), None)
    if match is None:
        return {"owned": False}

    avg_cost = match["avg_cost"]
    current_price = match["current_price"]
    gain_loss_pct = round((current_price - avg_cost) / avg_cost * 100, 2) if avg_cost and current_price else None

    return {
        "owned": True,
        "shares": match["shares"],
        "avg_cost": avg_cost,
        "current_price": current_price,
        "gain_loss_pct": gain_loss_pct,
        "weight_pct": weight_by_ticker.get(ticker),
    }


@router.get("/{ticker}/signal-history")
@limiter.limit("60/minute")
async def get_stock_signal_history(request: Request, ticker: str, universe_id: str = Query("All")):
    """DET-3: this stock's own signal history from Stage A's stock_scores
    table, each row evaluated against the ticker's own realized price
    move -- see services.stock_detail_service.evaluate_signal_history for
    the exact hit/miss rules and horizons. A row too recent for its
    horizon to have elapsed yet reports outcome: null rather than
    guessing -- an honest "not matured yet" gap, not a missing feature."""
    ticker = ticker.upper()
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT as_of_date, short_score, short_signal, long_score, long_signal
            FROM stock_scores WHERE ticker = $1 AND universe_id = $2
            ORDER BY as_of_date DESC LIMIT 90
            """,
            ticker, universe_id,
        )
    history = [
        {
            "as_of_date": str(r["as_of_date"]),
            "short_score": r["short_score"],
            "short_signal": r["short_signal"],
            "long_score": r["long_score"],
            "long_signal": r["long_signal"],
        }
        for r in rows
    ]

    try:
        price_history = await run_in_threadpool(get_cached_history, ticker, "2y", True, None)
        closes = price_history["Close"] if not price_history.empty else pd.Series(dtype=float)
    except Exception:
        closes = pd.Series(dtype=float)

    return {
        "ticker": ticker,
        "history": evaluate_signal_history(history, closes),
        "note": (
            "Buy/Trim signals are marked hit or miss once their horizon elapses "
            "(10 trading days short-term, ~1 trading year long-term); Hold shows the "
            "realized return with no verdict. A null outcome means the horizon "
            "hasn't elapsed yet."
        ),
    }


@router.get("/{ticker}/peers")
@limiter.limit("60/minute")
async def get_stock_peers(request: Request, ticker: str, universe_id: str = Query("All")):
    """DET-5: 5 closest stocks by sector and size — a sort/filter over
    the same universe table the Stock Finder already builds, no new
    fetch layer."""
    ticker = ticker.upper()
    df = await run_in_threadpool(get_stock_finder_table, universe_id)
    return {"ticker": ticker, "peers": select_peers(ticker, df)}
