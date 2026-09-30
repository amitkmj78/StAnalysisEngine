import services.alert_engine_service as alert_engine_service
from services.alert_engine_service import CONDITION_TYPES, evaluate_alert


def test_condition_types_includes_price_and_score():
    assert CONDITION_TYPES == {"price_above", "price_below", "score_above", "score_below"}


def test_price_above_triggers_when_price_meets_threshold(monkeypatch):
    monkeypatch.setattr(alert_engine_service, "get_latest_price", lambda ticker: 105.0)
    assert evaluate_alert("AAPL", "price_above", 100.0) == 105.0


def test_price_above_does_not_trigger_below_threshold(monkeypatch):
    monkeypatch.setattr(alert_engine_service, "get_latest_price", lambda ticker: 95.0)
    assert evaluate_alert("AAPL", "price_above", 100.0) is None


def test_price_below_triggers_when_price_meets_threshold(monkeypatch):
    monkeypatch.setattr(alert_engine_service, "get_latest_price", lambda ticker: 95.0)
    assert evaluate_alert("AAPL", "price_below", 100.0) == 95.0


def test_price_condition_none_when_price_unavailable(monkeypatch):
    monkeypatch.setattr(alert_engine_service, "get_latest_price", lambda ticker: None)
    assert evaluate_alert("AAPL", "price_above", 100.0) is None


def test_score_above_triggers_from_caller_supplied_score(monkeypatch):
    # Never touches get_latest_price -- score_* conditions must not fetch
    # a live price at all, only use the caller-supplied latest_short_score.
    monkeypatch.setattr(
        alert_engine_service, "get_latest_price",
        lambda ticker: (_ for _ in ()).throw(AssertionError("should not fetch a live price for a score condition")),
    )
    assert evaluate_alert("AAPL", "score_above", 70.0, latest_short_score=75.0) == 75.0


def test_score_above_does_not_trigger_below_threshold():
    assert evaluate_alert("AAPL", "score_above", 70.0, latest_short_score=65.0) is None


def test_score_below_triggers_from_caller_supplied_score():
    assert evaluate_alert("AAPL", "score_below", 30.0, latest_short_score=25.0) == 25.0


def test_score_condition_none_when_no_score_supplied():
    # The caller's batch query found no stock_scores row for this ticker.
    assert evaluate_alert("AAPL", "score_above", 70.0, latest_short_score=None) is None


def test_unknown_condition_type_returns_none():
    assert evaluate_alert("AAPL", "rsi_above", 50.0, latest_short_score=80.0) is None
