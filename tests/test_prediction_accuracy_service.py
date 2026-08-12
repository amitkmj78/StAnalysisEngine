from services.prediction_accuracy_service import compute_prediction_accuracy


def _pred(ticker, signal_correct=None, next_err=None, target_err=None):
    return {
        "ticker": ticker,
        "signal_correct": signal_correct,
        "next_price_error_pct": next_err,
        "target_price_error_pct": target_err,
    }


def test_empty_input_returns_empty_leaderboard_and_no_suggestion():
    result = compute_prediction_accuracy([])
    assert result["tickers"] == []
    assert result["suggested_ticker"] is None
    assert result["suggested_reason"] is None


def test_unverified_predictions_get_no_rank_but_still_appear():
    preds = [_pred("AAPL"), _pred("AAPL"), _pred("MSFT")]
    result = compute_prediction_accuracy(preds)
    by_ticker = {r["ticker"]: r for r in result["tickers"]}
    assert by_ticker["AAPL"]["total_predictions"] == 2
    assert by_ticker["AAPL"]["verified_count"] == 0
    assert by_ticker["AAPL"]["win_rate"] is None
    assert by_ticker["AAPL"]["rank"] is None
    assert result["suggested_ticker"] is None


def test_win_rate_and_rank_ordering():
    preds = (
        [_pred("AAPL", True)] * 3 + [_pred("AAPL", False)]  # 3/4 = 75%
        + [_pred("MSFT", True)] * 4  # 4/4 = 100%
        + [_pred("TSLA", False)] * 2  # 0/2 = 0%
    )
    result = compute_prediction_accuracy(preds, min_verified_for_recommendation=3)
    by_ticker = {r["ticker"]: r for r in result["tickers"]}

    assert by_ticker["MSFT"]["win_rate"] == 1.0
    assert by_ticker["MSFT"]["rank"] == 1
    assert by_ticker["AAPL"]["win_rate"] == 0.75
    assert by_ticker["AAPL"]["rank"] == 2
    assert by_ticker["TSLA"]["win_rate"] == 0.0
    assert by_ticker["TSLA"]["rank"] == 3


def test_tie_broken_by_verified_count():
    # Both 100% win rate, MSFT has more evidence -> ranks first.
    preds = [_pred("AAPL", True)] * 3 + [_pred("MSFT", True)] * 5
    result = compute_prediction_accuracy(preds, min_verified_for_recommendation=3)
    by_ticker = {r["ticker"]: r for r in result["tickers"]}
    assert by_ticker["MSFT"]["rank"] == 1
    assert by_ticker["AAPL"]["rank"] == 2


def test_recommendation_requires_minimum_verified_count():
    # 100% win rate but only 2 verified predictions -> not eligible.
    preds = [_pred("AAPL", True)] * 2
    result = compute_prediction_accuracy(preds, min_verified_for_recommendation=3)
    by_ticker = {r["ticker"]: r for r in result["tickers"]}
    assert by_ticker["AAPL"]["eligible_for_recommendation"] is False
    assert result["suggested_ticker"] is None


def test_recommendation_picks_top_eligible_ticker():
    preds = (
        [_pred("AAPL", True)] * 3 + [_pred("AAPL", False)]  # 75%, eligible
        + [_pred("MSFT", True)] * 2  # 100%, NOT eligible (only 2 verified)
    )
    result = compute_prediction_accuracy(preds, min_verified_for_recommendation=3)
    assert result["suggested_ticker"] == "AAPL"
    assert "75%" in result["suggested_reason"]
    assert "4 verified" in result["suggested_reason"]


def test_avg_error_ignores_missing_values_and_uses_absolute_value():
    preds = [
        _pred("AAPL", True, next_err=-2.0, target_err=5.0),
        _pred("AAPL", True, next_err=4.0, target_err=None),
        _pred("AAPL", False, next_err=None, target_err=None),
    ]
    result = compute_prediction_accuracy(preds)
    row = result["tickers"][0]
    assert row["avg_next_price_error_pct"] == 3.0  # mean(|-2|, |4|)
    assert row["avg_target_price_error_pct"] == 5.0  # only one value present
