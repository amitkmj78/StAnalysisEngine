from fastapi import APIRouter, Depends, Query, Request
from starlette.concurrency import run_in_threadpool

from services.data_service import get_latest_price
from services.ticker_search_service import search_tickers

from web.backend.auth import verify_bearer_token
from web.backend.rate_limit import limiter

router = APIRouter(prefix="/api/v1/search", tags=["search"], dependencies=[Depends(verify_bearer_token)])


@router.get("/tickers")
@limiter.limit("30/minute")
async def search(request: Request, q: str = Query(..., min_length=1)):
    results = await run_in_threadpool(search_tickers, q)
    return {"results": results}


@router.get("/price")
@limiter.limit("60/minute")
async def price(request: Request, ticker: str = Query(..., min_length=1)):
    # get_latest_price is ttl_cache'd (services/data_service.py, 60s), so a
    # user re-checking the same ticker while typing doesn't refetch from
    # yfinance every keystroke.
    value = await run_in_threadpool(get_latest_price, ticker.strip().upper())
    return {"ticker": ticker.strip().upper(), "price": value}
