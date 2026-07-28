from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from services.data_service import TIMEFRAME_MAPPING, get_latest_price
from services.prediction_narrative_service import build_prediction_narrative
from services.prediction_service import (
    compute_backtest_metrics,
    generate_trading_signal,
    predict_backtest_prices,
    predict_future_prices,
    predict_next_price,
)
from services.prediction_verification_service import verify_prediction

from web.backend.auth import verify_bearer_token
from web.backend.db import user_conn
from web.backend.llm_cache import cached_init_llms, resolve_llm
from web.backend.rate_limit import enforce_daily_quota, limiter
from web.backend.schemas import (
    BacktestOut,
    ForecastOut,
    MetricsOut,
    PredictionNarrativeOut,
    PredictionSummary,
    SignalOut,
)

router = APIRouter(prefix="/api/v1/predict", tags=["predict"], dependencies=[Depends(verify_bearer_token)])

ALLOWED_PERIODS = set(TIMEFRAME_MAPPING.values())


def _validate_period(period: str) -> None:
    if period not in ALLOWED_PERIODS:
        raise HTTPException(422, f"period must be one of {sorted(ALLOWED_PERIODS)}")


def _forecast_out(future_df) -> ForecastOut:
    return ForecastOut(
        dates=[d.strftime("%Y-%m-%d") for d in future_df.index],
        predicted=future_df["Predicted"].round(4).tolist(),
        lower_ci=future_df["Lower_CI"].round(4).tolist(),
        upper_ci=future_df["Upper_CI"].round(4).tolist(),
    )


def _backtest_out(backtest_df) -> BacktestOut:
    return BacktestOut(
        dates=[d.strftime("%Y-%m-%d") for d in backtest_df.index],
        actual=backtest_df["Actual"].round(4).tolist(),
        predicted=backtest_df["Predicted"].round(4).tolist(),
        naive=backtest_df["Naive"].round(4).tolist(),
    )


@router.get("/summary", response_model=PredictionSummary)
@limiter.limit("20/minute")
async def predict_summary(
    request: Request,
    ticker: str = Query(..., min_length=1),
    period: str = Query("1y"),
    tune: bool = Query(False),
    days_ahead: int = Query(10, ge=1, le=60),
    days_back: int = Query(30, ge=5, le=120),
):
    await enforce_daily_quota(request, "predict/summary")

    ticker = ticker.strip().upper()
    _validate_period(period)

    warnings: list[str] = []

    last_close = await run_in_threadpool(get_latest_price, ticker)
    if last_close is None:
        return PredictionSummary(ticker=ticker, period=period, warnings=["no_price_data"])

    next_price = await run_in_threadpool(predict_next_price, ticker, period, tune)

    future_df = await run_in_threadpool(predict_future_prices, ticker, period, days_ahead, tune)
    forecast = None
    if future_df is None or future_df.empty:
        warnings.append("insufficient_forecast_history")
    else:
        forecast = _forecast_out(future_df)

    signal = None
    if future_df is not None and not future_df.empty:
        sig = generate_trading_signal(last_close, future_df)
        if "target_price" in sig:
            signal = SignalOut(
                signal=sig["signal"],
                expected_return_pct=sig["expected_return_pct"],
                target_price=sig["target_price"],
            )

    backtest_df = await run_in_threadpool(predict_backtest_prices, ticker, period, days_back, tune)
    backtest = None
    metrics = None
    if backtest_df is None or backtest_df.empty:
        warnings.append("insufficient_backtest_history")
    else:
        backtest = _backtest_out(backtest_df)
        metrics = MetricsOut(**compute_backtest_metrics(backtest_df))

    return PredictionSummary(
        ticker=ticker,
        period=period,
        last_close=last_close,
        next_price=next_price,
        signal=signal,
        forecast=forecast,
        backtest=backtest,
        metrics=metrics,
        warnings=warnings,
    )


# --- Granular endpoints (not used by the MVP frontend, kept cheap for future clients) ---

@router.get("/next-price")
@limiter.limit("20/minute")
async def next_price_endpoint(
    request: Request,
    ticker: str = Query(..., min_length=1),
    period: str = Query("1y"),
    tune: bool = Query(False),
):
    await enforce_daily_quota(request, "predict/next-price")
    ticker = ticker.strip().upper()
    _validate_period(period)
    price = await run_in_threadpool(predict_next_price, ticker, period, tune)
    return {"ticker": ticker, "period": period, "next_price": price}


@router.get("/forecast")
@limiter.limit("20/minute")
async def forecast_endpoint(
    request: Request,
    ticker: str = Query(..., min_length=1),
    period: str = Query("1y"),
    days_ahead: int = Query(10, ge=1, le=60),
    tune: bool = Query(False),
):
    await enforce_daily_quota(request, "predict/forecast")
    ticker = ticker.strip().upper()
    _validate_period(period)
    future_df = await run_in_threadpool(predict_future_prices, ticker, period, days_ahead, tune)
    if future_df is None or future_df.empty:
        return {"ticker": ticker, "period": period, "forecast": None, "warnings": ["insufficient_forecast_history"]}
    return {"ticker": ticker, "period": period, "forecast": _forecast_out(future_df), "warnings": []}


@router.get("/backtest")
@limiter.limit("10/minute")
async def backtest_endpoint(
    request: Request,
    ticker: str = Query(..., min_length=1),
    period: str = Query("1y"),
    days_back: int = Query(30, ge=5, le=120),
    tune: bool = Query(False),
):
    await enforce_daily_quota(request, "predict/backtest")
    ticker = ticker.strip().upper()
    _validate_period(period)
    backtest_df = await run_in_threadpool(predict_backtest_prices, ticker, period, days_back, tune)
    if backtest_df is None or backtest_df.empty:
        return {"ticker": ticker, "period": period, "backtest": None, "metrics": None, "warnings": ["insufficient_backtest_history"]}
    return {
        "ticker": ticker,
        "period": period,
        "backtest": _backtest_out(backtest_df),
        "metrics": MetricsOut(**compute_backtest_metrics(backtest_df)),
        "warnings": [],
    }


# --- Optional LLM layer: narrative + sentiment context for the *live* prediction only.
# Never touches predict_backtest_prices' training loop — see prediction_narrative_service.py
# for why (no dated historical sentiment source exists, so it can't be backtested fairly).

@router.get("/narrative", response_model=PredictionNarrativeOut)
@limiter.limit("5/minute")
async def predict_narrative(
    request: Request,
    ticker: str = Query(..., min_length=1),
    period: str = Query("1y"),
    provider: str | None = Query(None),
):
    await enforce_daily_quota(request, "predict/narrative")

    ticker = ticker.strip().upper()
    _validate_period(period)

    last_close = await run_in_threadpool(get_latest_price, ticker)
    if last_close is None:
        raise HTTPException(422, "No price data available for that ticker.")

    next_price = await run_in_threadpool(predict_next_price, ticker, period, False)

    future_df = await run_in_threadpool(predict_future_prices, ticker, period, 10, False)
    signal = None
    if future_df is not None and not future_df.empty:
        sig = generate_trading_signal(last_close, future_df)
        if "target_price" in sig:
            signal = sig

    backtest_df = await run_in_threadpool(predict_backtest_prices, ticker, period, 30, False)
    metrics = None
    if backtest_df is not None and not backtest_df.empty:
        metrics = compute_backtest_metrics(backtest_df)

    llm_openai, llm_groq, llm_claude, llm_ollama, labels = await run_in_threadpool(cached_init_llms)
    if not labels:
        raise HTTPException(503, "No LLM providers are currently configured on the server.")

    provider = provider or labels[0]
    if provider not in labels:
        raise HTTPException(422, f"provider must be one of {labels}")
    llm = resolve_llm(provider, llm_openai, llm_groq, llm_claude, llm_ollama)

    context = {
        "last_close": last_close,
        "next_price": next_price,
        "signal": signal,
        "metrics": metrics,
    }
    result = await run_in_threadpool(build_prediction_narrative, llm, ticker, context)

    return PredictionNarrativeOut(
        ticker=ticker,
        provider=provider,
        narrative=result["narrative"],
        sentiment_context=result["sentiment_context"],
    )


# --- Save a prediction now, auto-verify it against real outcomes later.
# Distinct from the backtest above: that's a historical simulation, this is
# a real forward-looking prediction checked against what actually happened.

def _record_to_dict(record) -> dict:
    return {k: record[k] for k in record.keys()}


class SavePredictionRequest(BaseModel):
    ticker: str
    period: str = "1y"


@router.post("/save")
@limiter.limit("10/minute")
async def save_prediction(request: Request, body: SavePredictionRequest):
    await enforce_daily_quota(request, "predict/save")
    user_id = request.state.user["id"]

    ticker = body.ticker.strip().upper()
    _validate_period(body.period)

    last_close = await run_in_threadpool(get_latest_price, ticker)
    if last_close is None:
        raise HTTPException(422, "No price data available for that ticker.")

    next_price = await run_in_threadpool(predict_next_price, ticker, body.period, False)
    future_df = await run_in_threadpool(predict_future_prices, ticker, body.period, 10, False)
    if future_df is None or future_df.empty:
        raise HTTPException(422, "Not enough history to forecast this ticker yet.")

    sig = generate_trading_signal(last_close, future_df)
    signal = sig.get("signal")
    expected_return_pct = sig.get("expected_return_pct")
    target_price = sig.get("target_price")
    target_date = future_df.index[-1].to_pydatetime()

    async with user_conn(user_id) as conn:
        record = await conn.fetchrow(
            """
            INSERT INTO saved_predictions (
                user_id, ticker, period, last_close, next_price,
                signal, expected_return_pct, target_price, target_date
            ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING *
            """,
            user_id, ticker, body.period, last_close, next_price,
            signal, expected_return_pct, target_price, target_date,
        )
    return {"prediction": _record_to_dict(record)}


@router.get("/history")
async def prediction_history(request: Request, ticker: str | None = Query(None)):
    user_id = request.state.user["id"]

    async with user_conn(user_id) as conn:
        if ticker:
            records = await conn.fetch(
                "SELECT * FROM saved_predictions WHERE ticker = $1 ORDER BY predicted_at DESC",
                ticker.strip().upper(),
            )
        else:
            records = await conn.fetch("SELECT * FROM saved_predictions ORDER BY predicted_at DESC")

        rows = [_record_to_dict(r) for r in records]

        # Auto-verify on read: anything whose next-day/target date has
        # passed since it was saved gets checked against a live price
        # lookup right now, no separate scheduler needed.
        for row in rows:
            updates = await run_in_threadpool(verify_prediction, row)
            if not updates:
                continue
            set_cols = list(updates.keys())
            set_clause = ", ".join(f"{col} = ${i + 2}" for i, col in enumerate(set_cols))
            values = [updates[col] for col in set_cols]
            await conn.execute(
                f"UPDATE saved_predictions SET {set_clause} WHERE id = $1",
                row["id"], *values,
            )
            row.update(updates)

    return {"predictions": rows}
