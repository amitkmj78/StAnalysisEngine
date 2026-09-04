from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from services.market_news_service import get_hot_market_news
from web.backend.auth import verify_bearer_token

router = APIRouter(prefix="/api/v1/news", tags=["news"], dependencies=[Depends(verify_bearer_token)])


@router.get("/hot")
async def hot_market_news():
    """
    Broad market headlines for the scrolling ticker at the top of
    /portfolio — server-cached 5 minutes (services.market_news_service),
    so this is cheap regardless of how many users/tabs have it open.
    """
    return await run_in_threadpool(get_hot_market_news)
