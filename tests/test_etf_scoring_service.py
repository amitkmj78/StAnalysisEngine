"""services/etf_scoring_service.py's pure functions (COV-2). Network calls
(fetch_etf_holdings, compute_tracking_error) are mocked or exercised only
through their pure counterparts here -- never a real yfinance call."""

import services.etf_scoring_service as etf_mod
from services.etf_scoring_service import compute_holdings_weighted_score, compute_portfolio_overlap


def _holdings(*pairs):
    return [{"symbol": s, "name": s, "weight": w} for s, w in pairs]


# ---------------------------------------------------------------- compute_holdings_weighted_score

def test_weighted_score_full_coverage():
    holdings = _holdings(("AAPL", 0.5), ("MSFT", 0.5))
    scores = {"AAPL": 80.0, "MSFT": 60.0}
    result = compute_holdings_weighted_score(holdings, scores)
    assert result["weighted_score"] == 70.0
    assert result["coverage_pct"] == 100.0
    assert result["covered_holdings"] == 2
    assert result["total_holdings"] == 2


def test_weighted_score_renormalizes_over_covered_holdings_only():
    # NVDA uncovered (0.3 weight) -- weighted score computed only from the
    # covered 0.7, renormalized, not diluted by the uncovered slice.
    holdings = _holdings(("AAPL", 0.4), ("MSFT", 0.3), ("NVDA", 0.3))
    scores = {"AAPL": 80.0, "MSFT": 60.0}
    result = compute_holdings_weighted_score(holdings, scores)
    expected = (80.0 * 0.4 + 60.0 * 0.3) / 0.7
    assert result["weighted_score"] == round(expected, 1)
    assert result["coverage_pct"] == 70.0
    assert result["covered_holdings"] == 2
    assert result["total_holdings"] == 3


def test_weighted_score_none_when_coverage_too_thin():
    holdings = _holdings(("AAPL", 0.2), ("MSFT", 0.3), ("NVDA", 0.5))
    scores = {"AAPL": 80.0}  # only 20% covered -- below the 50% floor
    result = compute_holdings_weighted_score(holdings, scores)
    assert result["weighted_score"] is None
    assert result["coverage_pct"] == 20.0


def test_weighted_score_handles_no_holdings():
    result = compute_holdings_weighted_score([], {})
    assert result == {"weighted_score": None, "coverage_pct": 0.0, "covered_holdings": 0, "total_holdings": 0}


# ---------------------------------------------------------------- compute_portfolio_overlap

def test_portfolio_overlap_matches_held_tickers():
    holdings = _holdings(("AAPL", 0.3), ("MSFT", 0.2), ("NVDA", 0.5))
    result = compute_portfolio_overlap(holdings, ["AAPL", "TSLA"])
    assert result["overlap_pct"] == 30.0
    assert result["overlapping_tickers"] == ["AAPL"]


def test_portfolio_overlap_zero_when_nothing_held():
    holdings = _holdings(("AAPL", 0.3), ("MSFT", 0.7))
    result = compute_portfolio_overlap(holdings, ["TSLA"])
    assert result["overlap_pct"] == 0.0
    assert result["overlapping_tickers"] == []


def test_portfolio_overlap_handles_no_holdings():
    assert compute_portfolio_overlap([], ["AAPL"]) == {"overlap_pct": 0.0, "overlapping_tickers": []}


# ---------------------------------------------------------------- find_candidate / etf_universe

def test_etf_universe_is_nonempty_and_matches_index_fund_service():
    from services.index_fund_service import INDEX_FUND_UNIVERSE
    assert etf_mod.etf_universe() == [f.ticker for f in INDEX_FUND_UNIVERSE]


def test_find_candidate_is_case_insensitive():
    assert etf_mod.find_candidate("spy") is not None
    assert etf_mod.find_candidate("spy").ticker == "SPY"


def test_find_candidate_none_for_unknown_ticker():
    assert etf_mod.find_candidate("NOT_A_REAL_ETF") is None


# ---------------------------------------------------------------- compute_tracking_error

def test_tracking_error_none_without_a_mapped_benchmark(monkeypatch):
    # VEA (International Developed) has no benchmark_index_ticker in INDEX_FUND_UNIVERSE.
    assert etf_mod.compute_tracking_error("VEA") is None


def test_tracking_error_none_for_unknown_ticker():
    assert etf_mod.compute_tracking_error("NOT_A_REAL_ETF") is None
