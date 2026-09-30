"""
Diversified Basket rebuild (DI-01 through DI-09 + the approved scope
additions: sub-industry correlation capping, sector-weighting choice,
risk preview). No live network -- every test builds its own synthetic
data and monkeypatches the module's data-fetching functions where the
function under test would otherwise call them.
"""

import numpy as np
import pandas as pd
import pytest

import services.stock_finder_service as sfs
from services.stock_finder_service import (
    GICS_SECTORS_ORDER,
    SECTOR_WEIGHTING_MODES,
    _gics_sector,
    _select_sector_picks,
    _trim_to_max_stocks,
    assemble_sector_picks,
    check_concentration_warning,
    compute_basket_risk_preview,
    compute_sp500_sector_mix,
    get_basket_candidates,
    replace_basket_ticker,
    size_basket_positions,
)


def _prices(values, start="2023-01-01", freq="B"):
    index = pd.date_range(start, periods=len(values), freq=freq)
    return pd.Series(values, index=index, dtype=float)


# ---------------------------------------------------------------------------
# _gics_sector
# ---------------------------------------------------------------------------


def test_gics_sector_renames_yahoo_names():
    assert _gics_sector("Technology") == "Information Technology"
    assert _gics_sector("Financial Services") == "Financials"
    assert _gics_sector("Consumer Cyclical") == "Consumer Discretionary"
    assert _gics_sector("Healthcare") == "Health Care"
    assert _gics_sector("Consumer Defensive") == "Consumer Staples"
    assert _gics_sector("Basic Materials") == "Materials"


def test_gics_sector_passthrough_and_unknown():
    assert _gics_sector("Energy") == "Energy"  # already identical in both systems
    assert _gics_sector(None) == "Unknown"
    assert _gics_sector("") == "Unknown"


# ---------------------------------------------------------------------------
# _select_sector_picks / assemble_sector_picks (DI-02 + sub-industry cap)
# ---------------------------------------------------------------------------


def _sector_df(rows):
    df = pd.DataFrame(rows)
    return df.sort_values(["Score"], ascending=False).reset_index(drop=True)


def test_select_sector_picks_caps_one_per_industry():
    df = _sector_df([
        {"Ticker": "A", "GICS Sector": "Energy", "Industry": "Oil & Gas", "Score": 90.0},
        {"Ticker": "B", "GICS Sector": "Energy", "Industry": "Oil & Gas", "Score": 85.0},  # same industry as A
        {"Ticker": "C", "GICS Sector": "Energy", "Industry": "Coal", "Score": 80.0},
    ])
    picked, note = _select_sector_picks(df, 2)
    assert list(picked["Ticker"]) == ["A", "C"]  # B skipped: same industry as A, first pass
    assert note is None


def test_select_sector_picks_relaxes_when_too_few_industries():
    df = _sector_df([
        {"Ticker": "A", "GICS Sector": "Energy", "Industry": "Oil & Gas", "Score": 90.0},
        {"Ticker": "B", "GICS Sector": "Energy", "Industry": "Oil & Gas", "Score": 85.0},
    ])
    picked, note = _select_sector_picks(df, 2)
    assert list(picked["Ticker"]) == ["A", "B"]  # only one industry available -- relax, still return 2
    assert note is not None and "sub-industries" in note


def test_select_sector_picks_takes_all_when_fewer_eligible_than_n():
    df = _sector_df([{"Ticker": "A", "GICS Sector": "Utilities", "Industry": "Electric", "Score": 90.0}])
    picked, note = _select_sector_picks(df, 2)
    assert list(picked["Ticker"]) == ["A"]
    assert note == "Utilities: only 1 eligible stock"


def test_select_sector_picks_empty_is_a_noop():
    picked, note = _select_sector_picks(pd.DataFrame(), 2)
    assert picked.empty
    assert note is None


def test_assemble_sector_picks_skips_and_notes_zero_eligible_sectors():
    df = pd.DataFrame([
        {"Ticker": "A", "GICS Sector": "Energy", "Industry": "Oil & Gas", "Score": 90.0},
        {"Ticker": "B", "GICS Sector": "Utilities", "Industry": "Electric", "Score": 80.0},
    ])
    basket, notes = assemble_sector_picks(df, 2)
    assert set(basket["Ticker"]) == {"A", "B"}
    zero_note = [n for n in notes if n.startswith("0 eligible")]
    assert len(zero_note) == 1
    # every GICS sector except Energy/Utilities should show up as skipped
    for sector in GICS_SECTORS_ORDER:
        if sector not in ("Energy", "Utilities"):
            assert sector in zero_note[0]


# ---------------------------------------------------------------------------
# _trim_to_max_stocks (DI-04 round-robin-by-score fix)
# ---------------------------------------------------------------------------


def test_trim_to_max_stocks_matches_spec_test_case():
    # 11 sectors, 2 picks/sector, hand-chosen scores so the 4 highest
    # #2-picks are NOT the 4 alphabetically-first sectors -- this is
    # exactly the case a fixed-alphabetical-order round-robin gets wrong.
    rows = []
    for i, sector in enumerate(GICS_SECTORS_ORDER):
        rows.append({"Ticker": f"{sector}-1", "GICS Sector": sector, "Score": 100.0 - i})
        # #2 picks scored in REVERSE sector order, so the strongest #2
        # picks belong to sectors that sort LAST alphabetically/by-order.
        rows.append({"Ticker": f"{sector}-2", "GICS Sector": sector, "Score": 50.0 + i})
    basket = pd.DataFrame(rows)

    trimmed, notes = _trim_to_max_stocks(basket, 15, sector_col="GICS Sector")
    assert len(trimmed) == 15
    assert notes == []

    # All 11 #1 picks must be present.
    firsts = [t for t in trimmed["Ticker"] if t.endswith("-1")]
    assert len(firsts) == 11

    # The 4 #2 picks present must be the 4 highest-scoring ones (i=7..10),
    # not the 4 alphabetically/order-first sectors (i=0..3).
    seconds = sorted(t for t in trimmed["Ticker"] if t.endswith("-2"))
    expected = sorted(f"{GICS_SECTORS_ORDER[i]}-2" for i in (7, 8, 9, 10))
    assert seconds == expected


def test_trim_to_max_stocks_notes_left_out_sectors_when_cap_below_sector_count():
    rows = [{"Ticker": f"S{i}", "GICS Sector": f"Sector{i}", "Score": float(i)} for i in range(5)]
    basket = pd.DataFrame(rows)
    trimmed, notes = _trim_to_max_stocks(basket, 3, sector_col="GICS Sector")
    assert len(trimmed) == 3
    assert len(notes) == 1
    assert "left out entirely" in notes[0]
    # the 2 lowest-scored sectors (Sector0, Sector1) should be the ones left out
    assert "Sector0" in notes[0] and "Sector1" in notes[0]


# ---------------------------------------------------------------------------
# size_basket_positions (DI-05 + sector-weighting enhancement)
# ---------------------------------------------------------------------------


def _flat_basket(n, price=50.0, sector="Information Technology", cap=10.0):
    return pd.DataFrame([
        {"Ticker": f"T{i}", "GICS Sector": sector, "Price": price, "Market Cap ($B)": cap}
        for i in range(n)
    ])


def test_size_basket_positions_equal_dollar_target_matches_spec_example():
    basket = _flat_basket(15, price=1.0)  # price=1 so Target $ IS the per-share dollar figure
    sized, totals, warnings = size_basket_positions(basket, 10_000, fractional_shares=True, sector_weighting="equal_dollar")
    assert sized["Target $"].iloc[0] == pytest.approx(666.666666, rel=1e-6)
    assert warnings == []


def test_size_basket_positions_whole_share_floors_and_leftover_cash():
    basket = _flat_basket(3, price=300.0)
    sized, totals, warnings = size_basket_positions(basket, 1000, fractional_shares=False, sector_weighting="equal_dollar")
    # target = 333.33/position; floor(333.33/300) = 1 share each
    assert (sized["Shares"] == 1).all()
    assert totals["invested"] == pytest.approx(900.0)
    assert totals["leftover_cash"] == pytest.approx(100.0)
    assert warnings == []


def test_size_basket_positions_fractional_mode_is_exact():
    basket = _flat_basket(4, price=37.0)
    sized, totals, warnings = size_basket_positions(basket, 1000, fractional_shares=True, sector_weighting="equal_dollar")
    assert sized["Shares"].iloc[0] == pytest.approx(round(250.0 / 37.0, 4))
    assert totals["leftover_cash"] == pytest.approx(0.0, abs=0.01)


def test_size_basket_positions_warns_when_target_below_price():
    basket = pd.DataFrame([
        {"Ticker": "CHEAP", "GICS Sector": "Energy", "Price": 10.0, "Market Cap ($B)": 5.0},
        {"Ticker": "PRICEY", "GICS Sector": "Energy", "Price": 5000.0, "Market Cap ($B)": 5.0},
    ])
    sized, totals, warnings = size_basket_positions(basket, 100, fractional_shares=False, sector_weighting="equal_dollar")
    assert len(warnings) == 1
    assert "PRICEY" in warnings[0]
    assert "fractional shares" in warnings[0]


def test_size_basket_positions_market_cap_by_sector_splits_proportionally():
    basket = pd.DataFrame([
        {"Ticker": "A", "GICS Sector": "Energy", "Price": 1.0, "Market Cap ($B)": 90.0},
        {"Ticker": "B", "GICS Sector": "Utilities", "Price": 1.0, "Market Cap ($B)": 10.0},
    ])
    sized, totals, warnings = size_basket_positions(basket, 1000, fractional_shares=True, sector_weighting="market_cap_by_sector")
    energy_target = sized.loc[sized["Ticker"] == "A", "Target $"].iloc[0]
    utilities_target = sized.loc[sized["Ticker"] == "B", "Target $"].iloc[0]
    assert energy_target == pytest.approx(900.0)
    assert utilities_target == pytest.approx(100.0)


def test_size_basket_positions_empty_basket_is_a_noop():
    sized, totals, warnings = size_basket_positions(pd.DataFrame(), 1000, False, "equal_dollar")
    assert sized.empty
    assert totals == {"invested": 0.0, "leftover_cash": 1000, "holding_count": 0}
    assert warnings == []


def test_size_basket_positions_rejects_unknown_weighting_mode():
    with pytest.raises(ValueError):
        size_basket_positions(_flat_basket(2), 1000, False, "not_a_real_mode")


# ---------------------------------------------------------------------------
# compute_sp500_sector_mix
# ---------------------------------------------------------------------------


def test_compute_sp500_sector_mix_matches_hand_computed_percentages(monkeypatch):
    # compute_sp500_sector_mix takes no args, so its @ttl_cache caches
    # under one fixed key across the whole test session -- clear it first
    # so each test actually re-invokes the monkeypatched fetch.
    #
    # Uses get_peer_lookup_table (DET-5's lightweight, .info-only scan),
    # not get_stock_finder_table -- confirmed live that the heavy,
    # 3-year-price-history version can take minutes to build on a cold
    # cache across ~500 S&P 500 tickers, which HLT-1's Portfolio Health
    # Check page (the first synchronous, page-load-blocking caller of
    # this function) exposed directly. GICS Sector is already normalized
    # in get_peer_lookup_table's own output, so no separate _gics_sector
    # mapping step is needed here.
    compute_sp500_sector_mix.cache.clear()
    df = pd.DataFrame([
        {"Ticker": "A", "GICS Sector": "Information Technology", "Market Cap ($B)": 300.0},
        {"Ticker": "B", "GICS Sector": "Energy", "Market Cap ($B)": 100.0},
    ])
    monkeypatch.setattr(sfs, "get_peer_lookup_table", lambda universe_key: df)
    mix = compute_sp500_sector_mix()
    assert mix == {"Information Technology": 75.0, "Energy": 25.0}


def test_compute_sp500_sector_mix_empty_is_a_noop(monkeypatch):
    compute_sp500_sector_mix.cache.clear()
    monkeypatch.setattr(sfs, "get_peer_lookup_table", lambda universe_key: pd.DataFrame())
    assert compute_sp500_sector_mix() == {}


# ---------------------------------------------------------------------------
# check_concentration_warning (DI-09)
# ---------------------------------------------------------------------------


def test_check_concentration_warning_few_sectors():
    warning = check_concentration_warning({"Energy": 100.0}, universe_sector_count=3)
    assert warning is not None and "3 sector" in warning


def test_check_concentration_warning_one_sector_dominates():
    warning = check_concentration_warning(
        {"Energy": 45.0, "Utilities": 15.0, "Financials": 10.0, "Health Care": 10.0, "Materials": 10.0, "Real Estate": 10.0},
        universe_sector_count=6,
    )
    assert warning is not None and "Energy" in warning


def test_check_concentration_warning_none_when_well_spread():
    warning = check_concentration_warning(
        {"Energy": 20.0, "Utilities": 20.0, "Financials": 20.0, "Health Care": 20.0, "Materials": 20.0},
        universe_sector_count=5,
    )
    assert warning is None


# ---------------------------------------------------------------------------
# get_basket_candidates (DI-03)
# ---------------------------------------------------------------------------


def test_get_basket_candidates_classifies_each_exclusion_reason(monkeypatch):
    ranked = pd.DataFrame([
        {"Ticker": "GOOD", "GICS Sector": "Energy", "Industry": "Oil", "Score": 80.0,
         "return_3y_annualized": 12.0, "return_1y": 10.0, "sharpe_3y": 1.0, "max_drawdown_3y": 20.0,
         "revenue_growth": 5.0, "earnings_growth": 5.0, "forward_pe": 15.0,
         "Last Close Date": "2026-09-25"},
        {"Ticker": "NODATA", "GICS Sector": "Energy", "Industry": "Oil", "Score": 0.0,
         "return_3y_annualized": None, "return_1y": None, "sharpe_3y": None, "max_drawdown_3y": None,
         "revenue_growth": None, "earnings_growth": None, "forward_pe": None,
         "Last Close Date": "2026-09-25"},
        {"Ticker": "SHORTHIST", "GICS Sector": "Energy", "Industry": "Oil", "Score": 40.0,
         "return_3y_annualized": None, "return_1y": 8.0, "sharpe_3y": None, "max_drawdown_3y": 10.0,
         "revenue_growth": 3.0, "earnings_growth": 3.0, "forward_pe": 12.0,
         "Last Close Date": "2026-09-25"},
        {"Ticker": "STALE", "GICS Sector": "Energy", "Industry": "Oil", "Score": 60.0,
         "return_3y_annualized": 9.0, "return_1y": 7.0, "sharpe_3y": 0.8, "max_drawdown_3y": 15.0,
         "revenue_growth": 2.0, "earnings_growth": 2.0, "forward_pe": 18.0,
         "Last Close Date": "2026-09-01"},  # >9 calendar days before as_of_date
    ])
    monkeypatch.setattr(sfs, "rank_stocks", lambda goal, universe_key: ranked)
    monkeypatch.setattr(sfs, "_universe_tickers", lambda universe_key: ["GOOD", "NODATA", "SHORTHIST", "STALE", "NEVERFETCHED"])

    eligible, exclusions, as_of_date = get_basket_candidates("Long Term", "S&P 500")

    assert as_of_date == "2026-09-25"
    assert list(eligible["Ticker"]) == ["GOOD"]

    reasons = {e["ticker"]: e["reason"] for e in exclusions}
    assert "no usable price history" in reasons["NEVERFETCHED"]
    assert "no computable score" in reasons["NODATA"]
    assert "insufficient price history" in reasons["SHORTHIST"]
    assert "no price in the last 5 trading days" in reasons["STALE"]


def test_get_basket_candidates_empty_universe(monkeypatch):
    monkeypatch.setattr(sfs, "rank_stocks", lambda goal, universe_key: pd.DataFrame())
    monkeypatch.setattr(sfs, "_universe_tickers", lambda universe_key: ["A", "B"])
    eligible, exclusions, as_of_date = get_basket_candidates("Long Term", "All")
    assert eligible.empty
    assert {e["ticker"] for e in exclusions} == {"A", "B"}
    assert as_of_date == ""


# ---------------------------------------------------------------------------
# replace_basket_ticker (DI-06)
# ---------------------------------------------------------------------------


def test_replace_basket_ticker_returns_next_ranked_same_sector():
    eligible = pd.DataFrame([
        {"Ticker": "A", "GICS Sector": "Energy", "Score": 90.0},
        {"Ticker": "B", "GICS Sector": "Energy", "Score": 80.0},
        {"Ticker": "C", "GICS Sector": "Energy", "Score": 70.0},
        {"Ticker": "D", "GICS Sector": "Utilities", "Score": 60.0},
    ])
    replacement = replace_basket_ticker(eligible, current_tickers=["A", "D"], removed_ticker="A")
    assert replacement["Ticker"] == "B"


def test_replace_basket_ticker_none_when_sector_exhausted():
    eligible = pd.DataFrame([{"Ticker": "A", "GICS Sector": "Energy", "Score": 90.0}])
    assert replace_basket_ticker(eligible, current_tickers=["A"], removed_ticker="A") is None


def test_replace_basket_ticker_none_when_removed_ticker_unknown():
    eligible = pd.DataFrame([{"Ticker": "A", "GICS Sector": "Energy", "Score": 90.0}])
    assert replace_basket_ticker(eligible, current_tickers=["A"], removed_ticker="ZZZ") is None


# ---------------------------------------------------------------------------
# compute_basket_risk_preview (Scope E)
# ---------------------------------------------------------------------------


def test_compute_basket_risk_preview_beta_matches_construction(monkeypatch):
    # SPY moves by `spy_moves` each day; the basket's one holding moves at
    # exactly 1.5x SPY's daily moves by construction -- beta must come
    # back ~1.5.
    spy_moves = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.008, -0.012, 0.005, 0.01] * 5
    spy_prices = [100.0]
    for m in spy_moves:
        spy_prices.append(spy_prices[-1] * (1 + m))
    stock_prices = [50.0]
    for m in spy_moves:
        stock_prices.append(stock_prices[-1] * (1 + 1.5 * m))

    def fake_history(ticker, lookback, auto_adjust=True):
        if ticker == "SPY":
            return pd.DataFrame({"Close": _prices(spy_prices)})
        return pd.DataFrame({"Close": _prices(stock_prices)})

    monkeypatch.setattr(sfs, "get_cached_history", fake_history)

    basket = pd.DataFrame([{"Ticker": "LEVERED", "GICS Sector": "Energy", "Weight_pct": 100.0}])
    result = compute_basket_risk_preview(basket)
    assert result["beta_to_spy"] == pytest.approx(1.5, abs=0.05)
    assert result["excluded_from_risk"] == []
    assert result["largest_single_stock_weight_pct"] == 100.0
    assert result["largest_single_sector_weight_pct"] == 100.0


def test_compute_basket_risk_preview_empty_basket():
    result = compute_basket_risk_preview(pd.DataFrame())
    assert result["beta_to_spy"] is None
    assert result["annualized_volatility_pct"] is None


def test_compute_basket_risk_preview_missing_spy_data_degrades_gracefully(monkeypatch):
    monkeypatch.setattr(sfs, "get_cached_history", lambda ticker, lookback, auto_adjust=True: pd.DataFrame())
    basket = pd.DataFrame([{"Ticker": "A", "GICS Sector": "Energy", "Weight_pct": 100.0}])
    result = compute_basket_risk_preview(basket)
    assert result["beta_to_spy"] is None
    assert "A" in result["excluded_from_risk"]
