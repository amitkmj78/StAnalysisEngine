"""BEG-2: Learning Paths progress tracking. Lesson content itself is
static and ships with the frontend (web/frontend/lib/lessons.ts) -- this
only tracks which lessons a user has completed and their quiz score.
BEG-3's paper-trading-first gate reads this table too, for the
"risk-and-drawdown" lesson specifically (see services/
paper_trading_readiness.py)."""

from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends, Request

from web.backend.auth import verify_bearer_token
from web.backend.db import user_conn
from web.backend.rate_limit import limiter

router = APIRouter(prefix="/api/v1/learning", tags=["learning"], dependencies=[Depends(verify_bearer_token)])


@router.get("/progress")
async def get_progress(request: Request):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        rows = await conn.fetch(
            "SELECT lesson_id, completed_at, quiz_score, quiz_total FROM lesson_progress WHERE user_id = $1::uuid",
            user_id,
        )
    return {"lessons": [dict(r) for r in rows]}


class RecordProgressRequest(BaseModel):
    lesson_id: str = Field(min_length=1, max_length=100)
    score: int = Field(ge=0)
    total: int = Field(gt=0)


@router.post("/progress")
@limiter.limit("30/minute")
async def record_progress(request: Request, body: RecordProgressRequest):
    user_id = request.state.user["id"]
    score = min(body.score, body.total)
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO lesson_progress (user_id, lesson_id, quiz_score, quiz_total)
            VALUES ($1::uuid, $2, $3, $4)
            ON CONFLICT (user_id, lesson_id) DO UPDATE SET
                completed_at = now(), quiz_score = EXCLUDED.quiz_score, quiz_total = EXCLUDED.quiz_total
            RETURNING lesson_id, completed_at, quiz_score, quiz_total
            """,
            user_id, body.lesson_id, score, body.total,
        )
    return dict(row)
