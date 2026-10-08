from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from services.signal_publication_service import (
    DEFAULT_HORIZON_DAYS,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_PREDICT_PERIOD,
    DEFAULT_UNIVERSE,
    MODEL_PORTFOLIO_COST_BPS_ONE_WAY,
    PREDICT_COMPARE_HORIZONS,
    attach_excess_vs_spy,
    build_model_portfolio_series,
    build_spy_comparison_series,
    compute_avg_excess_vs_spy,
    compute_calibration,
    compute_outcome_metrics,
    compute_outcome_metrics_by_model_version,
    compute_outcome_metrics_by_regime,
    compute_outcome_metrics_by_signal,
    compute_predict_algo_comparison,
    compute_spy_returns_for_dates,
    fetch_spy_close_series,
    worst_misses,
)
from services.confidence_calibration_service import validate_calibration_out_of_sample
from services.quant_signal_narrative_service import build_quant_signal_narrative
from services.quant_signal_outcome_service import summarize_quant_signal_outcomes
from services.subscription_access_service import compute_free_tier_target_date, is_active_paid_subscriber
from web.backend.admin import require_admin
from web.backend.app_settings import (
    FREE_TIER_LAG_DAYS_DEFAULT,
    FREE_TIER_LAG_DAYS_KEY,
    HORIZON1_SUBSCRIPTIONS_ENABLED_KEY,
    PUBLISH_SIGNALS_ENABLED_KEY,
    get_setting_bool,
    get_setting_int,
)
from web.backend.auth import verify_bearer_token, verify_bearer_token_optional
from web.backend.db import service_conn
from web.backend.llm_cache import cached_init_llms, ordered_llms
from web.backend.pit_prices import QUANT_SIGNAL_HORIZON_DAYS, UNSTABLE_FLIP_THRESHOLD, evaluate_due_quant_signal_outcomes
from web.backend.rate_limit import enforce_daily_quota, limiter
from web.backend.scheduler import check_publication_alert
from services.stock_detail_service import DET3_SHORT_HORIZON_DAYS
from web.backend.signal_publication import (
    evaluate_due_signal_outcomes,
    evaluate_due_stock_page_signal_outcomes,
    publish_daily_signals,
)

router = APIRouter(prefix="/api/v1/signals", tags=["signals"])


def _record_to_dict(record) -> dict:
    return {k: record[k] for k in record.keys()}


@router.get("/published")
@limiter.limit("60/minute")
async def list_published_signals(
    request: Request,
    target_date: date | None = Query(None),
    universe_id: str = Query(DEFAULT_UNIVERSE),
    lookback_days: int = Query(DEFAULT_LOOKBACK_DAYS),
    user: dict | None = Depends(verify_bearer_token_optional),
):
    """
    TR-5 groundwork: the public track record, unauthenticated by design —
    this is the whole point of "published". Defaults to the most recent
    publication for the given universe/lookback when no date is given.

    RS-1/RS-2 (Horizon 1, off by default via HORIZON1_SUBSCRIPTIONS_ENABLED_KEY):
    when enabled, an anonymous or non-active-paid caller is capped to
    `latest_date - free_tier_lag_days`; an active paid subscriber sees the
    true latest date. This is the ONLY subscriber attribute this check is
    allowed to look at (RS-1's impersonality constraint) — never
    portfolio, positions, or anything else about who's asking. When the
    flag is off (today's default), behavior is byte-identical to before
    Horizon 1 existed.
    """
    horizon1_enabled = await get_setting_bool(HORIZON1_SUBSCRIPTIONS_ENABLED_KEY, default=False)
    is_paid = (
        horizon1_enabled and user is not None and await is_active_paid_subscriber(user["id"])
    )
    tier = "paid" if is_paid else "free"

    async with service_conn() as conn:
        summary = await conn.fetchrow(
            """
            SELECT min(target_date) AS record_start_date,
                   max(target_date) AS latest_date,
                   count(DISTINCT target_date) AS days_published
            FROM published_signals
            WHERE universe_id = $1 AND lookback_days = $2 AND reason_code IS NULL
            """,
            universe_id, lookback_days,
        )
        record_start_date = summary["record_start_date"] if summary else None
        latest_date = summary["latest_date"] if summary else None
        days_published = summary["days_published"] if summary else 0

        is_lagged = False
        if horizon1_enabled and not is_paid:
            lag_days = await get_setting_int(FREE_TIER_LAG_DAYS_KEY, default=FREE_TIER_LAG_DAYS_DEFAULT)
            target_date, is_lagged = compute_free_tier_target_date(
                target_date, latest_date, record_start_date, lag_days
            )
        elif target_date is None:
            target_date = latest_date

        if target_date is None:
            return {
                "target_date": None,
                "universe_id": universe_id,
                "lookback_days": lookback_days,
                "signals": [],
                "record_start_date": None,
                "days_published": 0,
                "tier": tier,
                "is_lagged": is_lagged,
            }

        rows = await conn.fetch(
            """
            SELECT * FROM published_signals
            WHERE target_date = $1 AND universe_id = $2 AND lookback_days = $3 AND reason_code IS NULL
            ORDER BY rank ASC
            """,
            target_date, universe_id, lookback_days,
        )

    return {
        "target_date": str(target_date),
        "universe_id": universe_id,
        "lookback_days": lookback_days,
        "signals": [_record_to_dict(r) for r in rows],
        "record_start_date": str(record_start_date) if record_start_date else None,
        "days_published": days_published,
        "tier": tier,
        "is_lagged": is_lagged,
    }


@router.get("/quant-vs-analyst")
@limiter.limit("30/minute")
async def quant_vs_analyst(
    request: Request,
    as_of_date: date | None = Query(None),
):
    """
    Internal Quant Signal (pit_quant_signal) and real Wall Street
    Analyst Rating (pit_analyst_rating) side by side, for every ticker
    captured that day — both are daily point-in-time snapshots of the
    full S&P 500, already bulk-captured, so this just reads and joins
    them rather than doing ~500 live per-ticker fetches. Defaults to
    the most recent date the quant signal was captured; analyst data is
    left-joined for that same date and may be null for a ticker if that
    day's analyst-rating capture missed it (yfinance coverage gap), not
    a code bug — shown as "no analyst data" rather than hidden.

    Also carries each ticker's signal_flip_count over the trailing 30
    captured days (as of as_of_date) and a signal_unstable flag once that
    exceeds UNSTABLE_FLIP_THRESHOLD — one batch query, not a per-ticker
    lookup, since this already covers the whole universe at once.
    """
    async with service_conn() as conn:
        if as_of_date is None:
            as_of_date = await conn.fetchval("SELECT max(as_of_date) FROM pit_quant_signal")
            if as_of_date is None:
                return {"as_of_date": None, "ticker_count": 0, "rows": []}

        rows = await conn.fetch(
            """
            WITH windowed AS (
                SELECT ticker, as_of_date, signal,
                       LAG(signal) OVER (PARTITION BY ticker ORDER BY as_of_date) AS prev_signal
                FROM pit_quant_signal
                WHERE as_of_date <= $1 AND as_of_date >= $1 - 30
            ),
            flips AS (
                SELECT ticker,
                       count(*) FILTER (WHERE prev_signal IS NOT NULL AND signal <> prev_signal) AS flip_count,
                       count(*) AS days_captured
                FROM windowed
                GROUP BY ticker
            )
            SELECT
                q.ticker,
                q.signal AS quant_signal,
                q.expected_return_pct AS quant_expected_return_pct,
                q.target_price AS quant_target_price,
                q.last_close,
                a.consensus AS analyst_consensus,
                a.analyst_count,
                a.buy_pct AS analyst_buy_pct,
                a.target_mean AS analyst_target_mean,
                a.target_high AS analyst_target_high,
                a.target_low AS analyst_target_low,
                COALESCE(f.flip_count, 0) AS signal_flip_count,
                COALESCE(f.days_captured, 1) AS signal_days_captured,
                COALESCE(f.flip_count, 0) >= $2 AS signal_unstable
            FROM pit_quant_signal q
            LEFT JOIN pit_analyst_rating a ON a.ticker = q.ticker AND a.as_of_date = q.as_of_date
            LEFT JOIN flips f ON f.ticker = q.ticker
            WHERE q.as_of_date = $1
            ORDER BY q.expected_return_pct DESC
            """,
            as_of_date,
            UNSTABLE_FLIP_THRESHOLD,
        )

    return {
        "as_of_date": str(as_of_date),
        "ticker_count": len(rows),
        "rows": [_record_to_dict(r) for r in rows],
    }


@router.get("/quant-vs-analyst/history", dependencies=[Depends(verify_bearer_token)])
@limiter.limit("30/minute")
async def quant_signal_history(
    request: Request,
    ticker: str = Query(...),
    days: int = Query(30, ge=1, le=90),
):
    """
    One ticker's real captured Quant Signal history (pit_quant_signal) —
    same trailing window signal_flip_count above is computed from, just
    returned row by row instead of collapsed to a count. Lets the UI show
    *when* a signal last changed and what expected_return_pct/target_price
    actually did around that date, rather than just "it flipped N times."
    No LLM involved — this is the real captured data, not an inference.
    """
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT as_of_date, signal, expected_return_pct, target_price, last_close
            FROM pit_quant_signal
            WHERE ticker = $1 AND as_of_date >= (SELECT max(as_of_date) FROM pit_quant_signal) - $2::integer
            ORDER BY as_of_date ASC
            """,
            ticker.strip().upper(), days,
        )
    return {"ticker": ticker.strip().upper(), "history": [_record_to_dict(r) for r in rows]}


@router.get("/quant-vs-analyst/narrative", dependencies=[Depends(verify_bearer_token)])
@limiter.limit("20/minute")
async def quant_signal_narrative(
    request: Request,
    ticker: str = Query(...),
    signal: str = Query(...),
    expected_return_pct: float = Query(...),
    target_price: float = Query(...),
    last_close: float = Query(...),
    current_price: float | None = Query(None),
):
    """
    On-demand AI explanation of an already-known Quant Signal (the row
    the caller already has from /quant-vs-analyst) against current
    technicals — deliberately scoped to the quant/technical picture
    only, never analyst opinions or news, per explicit product scope.
    Real per-call LLM cost, so this is auth-required and quota-gated
    like /predict/narrative, not exposed on the public /quant-vs-analyst
    endpoint itself.

    `current_price` is optional — the caller (the UI's own "Load" button
    on the Current Price column) fetches it separately via /search/price
    and passes it through here so the explanation can address whether
    the move since the signal was captured is tracking the model's call.
    Omitted, this behaves exactly as before.
    """
    await enforce_daily_quota(request, "signals/quant-vs-analyst/narrative")

    llm_openai, llm_groq, llm_claude, llm_ollama, labels = await run_in_threadpool(cached_init_llms)
    if not labels:
        raise HTTPException(502, "No LLM provider is configured on the server.")
    llms = ordered_llms(None, llm_openai, llm_groq, llm_claude, llm_ollama, labels)

    narrative = await run_in_threadpool(
        build_quant_signal_narrative,
        llms, ticker, signal, expected_return_pct, target_price, last_close, current_price,
    )
    if narrative is None:
        raise HTTPException(502, "Failed to generate a narrative for this ticker.")
    return {"ticker": ticker, "narrative": narrative["technical"], "plain_english": narrative["plain_english"]}


@router.get("/quant-vs-analyst/outcomes")
@limiter.limit("30/minute")
async def quant_signal_outcomes(
    request: Request,
    horizon_days: int = Query(QUANT_SIGNAL_HORIZON_DAYS),
):
    """
    The real, live track record behind the Quant Signal column: for every
    already-captured BUY/HOLD/SELL call old enough to have a real,
    already-captured exit price (horizon_days trading days later), was it
    actually right? Pooled by signal type, not per-ticker — most tickers
    see very few BUY/SELL calls (they're the extreme tail of the
    forecast), so a per-ticker win rate would mostly be noise from a
    handful of samples. Public, unauthenticated, same rationale as
    /signals/outcomes: this is the honest answer to "should I trust
    this," not something to gate behind a login.
    """
    async with service_conn() as conn:
        rows = await conn.fetch(
            "SELECT signal, correct FROM quant_signal_outcomes WHERE horizon_days = $1",
            horizon_days,
        )
    summary = summarize_quant_signal_outcomes([dict(r) for r in rows])
    return {"horizon_days": horizon_days, "summary": summary}


@router.post("/quant-vs-analyst/evaluate-now", dependencies=[Depends(require_admin)])
async def quant_signal_evaluate_now(
    horizon_days: int = Query(QUANT_SIGNAL_HORIZON_DAYS),
):
    """Manual trigger for the same outcome evaluation the scheduler runs
    daily — for verifying the pipeline and catching up on the backlog of
    already-due calls, not routine use. No enable-gate, same reasoning as
    /signals/evaluate-now: evaluating already-captured PIT data isn't
    itself a new disclosure."""
    evaluated = await evaluate_due_quant_signal_outcomes(horizon_days=horizon_days)
    return {"evaluated": evaluated}


@router.get("/published/compare-to-predict-algo")
@limiter.limit("10/minute")
async def compare_to_predict_algo(
    request: Request,
    universe_id: str = Query(DEFAULT_UNIVERSE),
    lookback_days: int = Query(DEFAULT_LOOKBACK_DAYS),
    days_ahead: int = Query(DEFAULT_LOOKBACK_DAYS),
):
    """
    What the separate, trained Predict-page model currently says about
    today's published momentum picks — a different algorithm, not a
    validation of this one. Deliberately only ever compares against the
    *latest* publication (no target_date param): running today's model
    against an older published date would mix in price data the model
    couldn't have had at that original date, since there's no point-in-time
    store yet to prevent that honestly.

    days_ahead is the Predict-algo forecast horizon, restricted to
    PREDICT_COMPARE_HORIZONS (1/5/10/30) so short reads (does the algorithm
    agree over the next day or week?) are possible too, not just the
    30-day window matching the momentum lookback.
    """
    if days_ahead not in PREDICT_COMPARE_HORIZONS:
        raise HTTPException(422, f"days_ahead must be one of {PREDICT_COMPARE_HORIZONS}")
    predict_days_ahead = days_ahead
    async with service_conn() as conn:
        latest_date = await conn.fetchval(
            """
            SELECT max(target_date) FROM published_signals
            WHERE universe_id = $1 AND lookback_days = $2 AND reason_code IS NULL
            """,
            universe_id, lookback_days,
        )
        if latest_date is None:
            return {"target_date": None, "comparisons": []}

        rows = await conn.fetch(
            """
            SELECT ticker, rank, trailing_return_pct FROM published_signals
            WHERE target_date = $1 AND universe_id = $2 AND lookback_days = $3 AND reason_code IS NULL
            ORDER BY rank ASC
            """,
            latest_date, universe_id, lookback_days,
        )

    tickers = [r["ticker"] for r in rows]
    comparison = await run_in_threadpool(
        compute_predict_algo_comparison, tickers, DEFAULT_PREDICT_PERIOD, predict_days_ahead
    )
    comparison_by_ticker = {c["ticker"]: c for c in comparison}

    return {
        "target_date": str(latest_date),
        "predict_period": DEFAULT_PREDICT_PERIOD,
        "predict_days_ahead": predict_days_ahead,
        "comparisons": [
            {
                "rank": r["rank"],
                "ticker": r["ticker"],
                "trailing_return_pct": r["trailing_return_pct"],
                **comparison_by_ticker.get(r["ticker"], {}),
            }
            for r in rows
        ],
    }


@router.post("/publish-now", dependencies=[Depends(require_admin)])
async def publish_now(
    universe_id: str = Query(DEFAULT_UNIVERSE),
    lookback_days: int = Query(DEFAULT_LOOKBACK_DAYS),
    force: bool = Query(False, description="Bypass the publish_signals_enabled gate for a controlled test."),
):
    """Manual trigger for the same publication the scheduler runs daily —
    for verifying the pipeline and catching up a missed day, not routine use.
    Respects the same off-by-default gate as the scheduled job unless
    force=true is passed explicitly, so a routine test call can't
    accidentally become the record's real first publication."""
    if not force and not await get_setting_bool(PUBLISH_SIGNALS_ENABLED_KEY, default=False):
        raise HTTPException(
            409,
            "Publishing is currently disabled (publish_signals_enabled=false). "
            "Enable it via /admin/settings, or pass force=true for a one-off test.",
        )
    published = await publish_daily_signals(universe_id=universe_id, lookback_days=lookback_days)
    return {"published": published}


@router.get("/outcomes")
@limiter.limit("60/minute")
async def list_signal_outcomes(
    request: Request,
    universe_id: str = Query(DEFAULT_UNIVERSE),
    lookback_days: int = Query(DEFAULT_LOOKBACK_DAYS),
    horizon_days: int = Query(DEFAULT_HORIZON_DAYS),
):
    """
    TR-4: live performance to date — public, unauthenticated, same as
    /published. Aggregate metrics (hit rate, information coefficient,
    quintile spread) plus the full per-date history, computed only from
    signal_outcomes (never blended with anything backtest-derived, per
    TR-4's hard requirement).
    """
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT target_date, ticker, rank, entry_price, exit_price,
                   realized_return_pct, benchmark_return_pct, beat_benchmark
            FROM signal_outcomes
            WHERE universe_id = $1 AND lookback_days = $2 AND horizon_days = $3
            ORDER BY target_date ASC, rank ASC
            """,
            universe_id, lookback_days, horizon_days,
        )

    row_dicts = [_record_to_dict(r) for r in rows]
    metrics = compute_outcome_metrics(row_dicts)

    return {
        "universe_id": universe_id,
        "lookback_days": lookback_days,
        "horizon_days": horizon_days,
        **metrics,
        "outcomes": [
            {**r, "target_date": str(r["target_date"])} for r in row_dicts
        ],
    }


@router.get("/track-record")
@limiter.limit("60/minute")
async def get_track_record(
    request: Request,
    universe_id: str = Query(DEFAULT_UNIVERSE),
    lookback_days: int = Query(DEFAULT_LOOKBACK_DAYS),
    horizon_days: int = Query(DEFAULT_HORIZON_DAYS),
    regime: str | None = Query(
        None,
        description="REG-2: filter to only outcomes whose target_date fell in this market "
        "regime (e.g. 'Risk-Off', or 'unknown' for dates before market_regime_daily has a "
        "reading). Every metric/list in the response is scoped to this filter when set; "
        "metrics_by_regime is always the full breakdown regardless, so a caller can see "
        "every regime's stats before picking one to filter to.",
    ),
):
    """
    FND-3: the public track record now scores every ticker's own
    stock-page SCR-1 short-term Buy/Trim signal (stock_scores.
    short_signal, via the nightly evaluate_due_stock_page_signal_outcomes
    job -> stock_signal_outcomes) against its OWN realized move, not a
    top-N trailing-return momentum rank — that was the literal gap this
    endpoint used to have (confirmed: signal_outcomes/published_signals
    is a pure momentum ranking, the same rule shown on /top-performers,
    with no Buy/Hold/Trim concept in its schema at all). The old
    pipeline's tables and nightly job are left running, not deleted —
    just no longer what this endpoint reads — in case a future "momentum
    picks" comparison view wants them; see the tracker note for FND-3.

    `universe_id`/`lookback_days` are accepted for backward API
    compatibility but don't change anything here: the nightly evaluation
    job only ever scores stock_scores' "All" universe (every ticker gets
    a real signal, there's no separate per-universe momentum rank to
    select), and there's no "lookback window" concept for a signal that
    isn't a ranking in the first place. `horizon_days` still matters —
    10 is the short-term (SCR-1) horizon and the only one evaluated so
    far; the long-term (SCR-2, ~1-year) horizon is explicitly not
    evaluated yet (needs far more elapsed history to mature at all).

    TRK-2/3/5/6 (docs/stock-analysis-requirements.html): average return,
    excess-vs-SPY, a weights-version, signal, and regime breakdown,
    calibration, worst misses, and a model-portfolio-vs-SPY growth
    chart — same public/unauthenticated posture as /outcomes and
    /published. Confidence is now the REAL stated confidence captured
    at signal time (stock_scores.short_confidence_score/label via
    services.portfolio_compare_service.derive_confidence), not a rank-
    stability proxy — see FND-4 for the calibration fit built on top of
    this. worst_misses/metrics_by_signal now genuinely include both Buy
    and Trim (the old pipeline's one real limitation). REG-2: `regime`
    (from market_regime_daily, joined on as_of_date) is attached to
    every row and, when the `regime` query param is set, filters the
    whole response — see services/market_regime_service.py for why this
    ships despite a failed validation gate.
    """
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT so.as_of_date AS target_date, so.ticker, so.signal, so.entry_price, so.exit_price,
                   so.realized_return_pct, so.benchmark_return_pct, so.beat_benchmark,
                   so.confidence_score, so.confidence_label, so.weights_version AS model_version_hash,
                   mr.regime_confirmed AS regime
            FROM stock_signal_outcomes so
            LEFT JOIN market_regime_daily mr ON mr.as_of_date = so.as_of_date
            WHERE so.horizon_days = $1
            ORDER BY so.as_of_date ASC, so.ticker ASC
            """,
            horizon_days,
        )
    outcome_rows = [{**_record_to_dict(r), "rank": None} for r in rows]
    # Computed from the FULL (unfiltered) set, before `regime` narrows
    # outcome_rows below -- this breakdown should always show every
    # regime's stats so a caller can see them all before picking one to
    # filter to, regardless of whether a filter is currently active.
    metrics_by_regime = compute_outcome_metrics_by_regime(outcome_rows)
    if regime:
        outcome_rows = [r for r in outcome_rows if (r.get("regime") or "unknown") == regime]

    empty_response = {
        "universe_id": universe_id,
        "lookback_days": lookback_days,
        "horizon_days": horizon_days,
        "regime_filter": regime,
        "metrics": compute_outcome_metrics([]),
        "metrics_by_model_version": {},
        "metrics_by_signal": {},
        "metrics_by_regime": metrics_by_regime,
        "avg_excess_vs_spy_pct": None,
        "calibration": compute_calibration([]),
        "confidence_calibration": validate_calibration_out_of_sample([]),
        "worst_misses": [],
        "model_portfolio_series": [],
        "spy_portfolio_series": [],
        "trim_note": "Trim signals are included alongside Buy -- worst misses rank a Trim's biggest unrealized gain the same way a Buy's biggest loss ranks, per TRK-5.",
        "signal_note": None,
        "model_portfolio_cost_bps_one_way": MODEL_PORTFOLIO_COST_BPS_ONE_WAY,
    }
    if not outcome_rows:
        return empty_response

    target_dates = sorted({r["target_date"] for r in outcome_rows})

    spy_close = await run_in_threadpool(fetch_spy_close_series)
    spy_return_by_date = compute_spy_returns_for_dates(spy_close, target_dates, horizon_days)
    with_excess = attach_excess_vs_spy(outcome_rows, spy_return_by_date)

    # FND-4: confidence_score/confidence_label are already the real,
    # stated-at-signal-time values from stock_signal_outcomes (the SQL
    # above selects them straight off the row) -- no separate proxy
    # computation needed, unlike the old rank-stability-proxy this
    # endpoint used before (confidence_for_outcome, still used elsewhere
    # for the momentum-rank pipeline's own views).
    return {
        **empty_response,
        "metrics": compute_outcome_metrics(outcome_rows),
        "metrics_by_model_version": compute_outcome_metrics_by_model_version(outcome_rows),
        "metrics_by_signal": compute_outcome_metrics_by_signal(outcome_rows),
        "avg_excess_vs_spy_pct": compute_avg_excess_vs_spy(with_excess),
        "calibration": compute_calibration(with_excess),
        # FND-4: a real fit/holdout calibration check of the stated
        # confidence captured at signal time, bucketed by its exact
        # discrete value (0/25/50/75/100) rather than CONFIDENCE_BUCKETS'
        # continuous ranges above (built for the old rank-stability
        # proxy) -- see confidence_calibration_service's module
        # docstring for what this does and doesn't change.
        "confidence_calibration": validate_calibration_out_of_sample(outcome_rows),
        "worst_misses": [
            {**m, "target_date": str(m["target_date"])} for m in worst_misses(outcome_rows, top_n=10)
        ],
        "model_portfolio_series": build_model_portfolio_series(
            [r for r in outcome_rows if r.get("signal", "Buy") != "Trim"], horizon_days
        ),
        "spy_portfolio_series": build_spy_comparison_series(outcome_rows, spy_return_by_date, horizon_days),
    }


@router.post("/evaluate-now", dependencies=[Depends(require_admin)])
async def evaluate_now(
    universe_id: str = Query(DEFAULT_UNIVERSE),
    lookback_days: int = Query(DEFAULT_LOOKBACK_DAYS),
    horizon_days: int = Query(DEFAULT_HORIZON_DAYS),
):
    """Manual trigger for the same outcome evaluation the scheduler runs
    daily — for verifying the pipeline and catching up, not routine use.
    Unlike publish-now, no enable-gate: evaluating already-published,
    already-public history isn't itself a new disclosure."""
    evaluated = await evaluate_due_signal_outcomes(
        universe_id=universe_id, lookback_days=lookback_days, horizon_days=horizon_days
    )
    return {"evaluated": evaluated}


@router.post("/evaluate-stock-page-now", dependencies=[Depends(require_admin)])
async def evaluate_stock_page_now(horizon_days: int = Query(DET3_SHORT_HORIZON_DAYS)):
    """FND-3: the same manual trigger as /evaluate-now above, for the
    real-signal (stock_signal_outcomes) pipeline /track-record now reads
    from. No enable-gate, same reasoning as /evaluate-now."""
    evaluated = await evaluate_due_stock_page_signal_outcomes(horizon_days=horizon_days)
    return {"evaluated": evaluated}


@router.post("/check-publication-alert", dependencies=[Depends(require_admin)])
async def check_publication_alert_now(
    checkpoint: str = Query("nfr01", description="'nfr01' (60-min) or 'nfr02' (2-hour)"),
    force: bool = Query(False, description="Send a real test email even if publication already succeeded today."),
):
    """NFR-01/02: manual trigger for the same alert check the scheduler
    runs at 5pm/6pm ET — for verifying the email actually arrives, not
    routine use. With force=true, bypasses both the enabled-check and the
    already-published-check purely to confirm mail delivery works."""
    if checkpoint == "nfr01":
        label, deadline = "delayed (NFR-01, 60-min check)", "60 minutes"
    elif checkpoint == "nfr02":
        label, deadline = "missing (NFR-02, 2-hour check)", "2 hours"
    else:
        raise HTTPException(422, "checkpoint must be 'nfr01' or 'nfr02'")

    return await check_publication_alert(label, deadline, force=force)
