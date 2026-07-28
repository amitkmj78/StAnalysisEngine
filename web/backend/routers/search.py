from fastapi import APIRouter, Depends, Query, Request
from starlette.concurrency import run_in_threadpool

from services.ticker_search_service import search_tickers

from web.backend.auth import verify_bearer_token
from web.backend.rate_limit import limiter

router = APIRouter(prefix="/api/v1/search", tags=["search"], dependencies=[Depends(verify_bearer_token)])


@router.get("/tickers")
@limiter.limit("30/minute")
async def search(request: Request, q: str = Query(..., min_length=1)):
    results = await run_in_threadpool(search_tickers, q)
    return {"results": results}
