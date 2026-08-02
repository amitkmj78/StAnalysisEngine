import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from web.backend.db import close_pools, init_pools
from web.backend.rate_limit import limiter
from web.backend.scheduler import start_scheduler, stop_scheduler
from web.backend.routers import (
    admin_activity,
    admin_settings,
    admin_users,
    auth,
    aws_deploy,
    chat,
    entry_strategy,
    index_fund,
    momentum,
    monthly_plan,
    portfolio,
    prediction,
    search,
    stock_finder,
    strategies,
    trade_journal,
    watchlist,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_pools()
    start_scheduler()
    yield
    stop_scheduler()
    await close_pools()


app = FastAPI(title="StAnalysisEngine API", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(prediction.router)
app.include_router(stock_finder.router)
app.include_router(index_fund.router)
app.include_router(entry_strategy.router)
app.include_router(monthly_plan.router)
app.include_router(strategies.router)
app.include_router(trade_journal.router)
app.include_router(portfolio.router)
app.include_router(chat.router)
app.include_router(aws_deploy.router)
app.include_router(search.router)
app.include_router(watchlist.router)
app.include_router(momentum.router)
app.include_router(admin_users.router)
app.include_router(admin_activity.router)
app.include_router(admin_settings.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
