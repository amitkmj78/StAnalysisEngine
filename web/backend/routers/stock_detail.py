"""
Stock detail page endpoints (docs/stock-analysis-requirements.html,
DET-1..5) backing the /stock/[ticker] frontend page. Fundamentals/
earnings/dividends/peers/signal-history are public, unauthenticated --
same posture as GET /stock-scores/{ticker} (general market research, not
user data). /position is the one user-specific route, auth-gated.
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from services.data_service import get_latest_price
from services.ranking_utils import compute_position_concentration
from services.stock_detail_service import next_earnings_date, recent_dividends, select_peers
from services.stock_finder_service import _gics_sector, get_stock_finder_table
from services.yfinance_cache import get_cached_dividends, get_cached_earnings_dates, get_cached_history
from web.backend.auth import verify_bearer_token
from web.backend.db import service_conn, user_conn
from web.backend.rate_limit import limiter
from web.backend.routers.portfolio import _resolve_portfolio_id

router = APIRouter(prefix="/api/v1/stock", tags=["stock-detail"])

# DET-1's range buttons. No intraday charting exists anywhere in this app
# today (every other chart is daily-bar), so "1D" is deliberately scoped
# out for this first pass rather than adding new intraday/interval
# plumbing -- 5D is the shortest range offered.
PRICE_HISTORY_RANGES = {
    "5D": "5d",
    "1M": "1mo",
    "6M": "6mo",
    "1Y": "1y",
    "5Y": "5y",
}


@router.get("/{ticker}/detail")
@limiter.limit("60/minute")
async def get_stock_detail(request: Request, ticker: str):
    """DET-1: key stats, fundamentals, earnings date, dividends. Current
    signals/reasons are deliberately NOT duplicated here -- the frontend
    already calls GET /stock-scores/{ticker} (Stage A) for that, one
    source of truth rather than two endpoints that could drift."""
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
    rather than a fresh fetch layer."""
    ticker = ticker.upper()
    range = range.upper()
    period = PRICE_HISTORY_RANGES.get(range)
    if period is None:
        raise HTTPException(status_code=400, detail=f"range must be one of {sorted(PRICE_HISTORY_RANGES)}")

    history = await run_in_threadpool(get_cached_history, ticker, period, True)
    return {
        "ticker": ticker,
        "range": range,
        "history": [
            {"date": ts.date().isoformat(), "close": round(float(row["Close"]), 2)}
            for ts, row in history.iterrows()
        ],
    }


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
    """DET-3: this stock's own history from Stage A's stock_scores table.
    Hit/miss evaluation isn't tracked for this scoring system yet (it's
    new -- see Stage A), so this is raw signal history only, honestly
    labeled rather than implying an evaluated track record that doesn't
    exist yet."""
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
    return {
        "ticker": ticker,
        "history": [
            {
                "as_of_date": str(r["as_of_date"]),
                "short_score": r["short_score"],
                "short_signal": r["short_signal"],
                "long_score": r["long_score"],
                "long_signal": r["long_signal"],
            }
            for r in rows
        ],
        "note": "Hit/miss evaluation isn't tracked yet for this scoring system — shown as raw signal history only.",
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
