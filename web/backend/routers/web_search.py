from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from services.web_search import search as run_web_search
from web.backend.auth import verify_bearer_token
from web.backend.rate_limit import enforce_daily_quota, limiter

router = APIRouter(
    prefix="/api/v1/websearch",
    tags=["websearch"],
    dependencies=[Depends(verify_bearer_token)],
)


class WebSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=400)
    max_results: int = Field(default=5, ge=1, le=20)
    include_raw_content: bool = False


@router.post("/search")
@limiter.limit("20/minute")
async def web_search(request: Request, body: WebSearchRequest):
    """
    Self-hosted search: DuckDuckGo for candidate URLs, real content
    extraction for each result (services/web_search) — built to replace
    Tavily's shape closely enough to be a genuine drop-in for the same
    request pattern, not just "similar." Quota-gated like every other
    real-cost outbound-fetch endpoint in this app (each call does a live
    search plus up to `max_results` live page fetches).
    """
    await enforce_daily_quota(request, "websearch/search")

    query = body.query.strip()
    if not query:
        raise HTTPException(422, "query must not be blank.")

    result = await run_in_threadpool(run_web_search, query, body.max_results, body.include_raw_content)
    return {
        "query": result.query,
        "results": [
            {
                "title": r.title,
                "url": r.url,
                "content": r.content,
                "score": r.score,
                "raw_content": r.raw_content,
            }
            for r in result.results
        ],
        "response_time_ms": result.response_time_ms,
    }
