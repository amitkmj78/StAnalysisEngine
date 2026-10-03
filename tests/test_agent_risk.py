import numpy as np
import pandas as pd
import pytest

from services.agent.config import CONFIG, config_version
from services.agent.indicators import annualized_volatility_pct, atr, average_dollar_volume, sma
from services.agent.risk import (
    CIRCUIT_BREAKER,
    NORMAL,
    REDUCE_ONLY,
    Candidate,
    Holding,
    Order,
    filter_candidates,
    plan_orders,
    preflight,
    regime_cap_pct,
    risk_state,
    scale_for_vol_target,
    size_positions,
)


def _cand(ticker="AAA", sector="Tech", price=110.0, signal="Buy", sma200=100.0,
          adv=100_000_000.0, vol=20.0, blackout=False):
    return Candidate(ticker, sector, price, signal, sma200, adv, vol, blackout)


# --- regime caps (AGT-15) ---

@pytest.mark.parametrize("label,expected", [("Risk-On", 100.0), ("Neutral", 80.0), ("Cautious", 50.0), ("Risk-Off", 20.0)])
def test_regime_caps_match_spec(label, expected):
    cap, _ = regime_cap_pct(label)
    assert cap == expected


def test_missing_regime_uses_conservative_cap_and_says_so():
    cap, reason = regime_cap_pct(None)
    assert cap == 50.0
    assert "No regime reading" in reason


# --- risk state (AGT-17, AGT-18) ---

def test_daily_loss_of_two_percent_goes_reduce_only():
    st = risk_state(equity=97_900.0, last_equity=100_000.0, peak_equity=100_000.0,
                    breaker_latched=False, regime_cap=100.0)
    assert st.state == REDUCE_ONLY
    assert st.buys_allowed is False
    assert "daily loss limit" in st.reasons[0]


def test_small_daily_move_stays_normal():
    st = risk_state(99_000.0, 100_000.0, 100_000.0, False, 100.0)
    assert st.state == NORMAL and st.buys_allowed


def test_ten_percent_below_peak_trips_breaker_and_caps_exposure():
    st = risk_state(equity=89_000.0, last_equity=89_500.0, peak_equity=100_000.0,
                    breaker_latched=False, regime_cap=100.0)
    assert st.state == CIRCUIT_BREAKER
    assert st.exposure_cap_pct == 20.0
    assert st.buys_allowed is False


def test_latched_breaker_persists_even_after_recovery():
    st = risk_state(equity=100_000.0, last_equity=100_000.0, peak_equity=100_000.0,
                    breaker_latched=True, regime_cap=100.0)
    assert st.state == CIRCUIT_BREAKER
    assert any("manual reset" in r for r in st.reasons)


def test_breaker_exposure_cap_never_exceeds_regime_cap():
    st = risk_state(89_000.0, None, 100_000.0, False, regime_cap=10.0)
    assert st.exposure_cap_pct == 10.0


# --- candidate filters (AGT-6, 7, 8) ---

def test_filters_reject_non_buy_below_trend_illiquid_and_blackout():
    cands = [
        _cand("GOOD"),
        _cand("HOLD", signal="Hold"),
        _cand("BELOW", price=90.0),
        _cand("ILLIQ", adv=10_000_000.0),
        _cand("EARN", blackout=True),
    ]
    passed, rejected = filter_candidates(cands)
    assert [c.ticker for c in passed] == ["GOOD"]
    reasons = dict(rejected)
    assert "not Buy" in reasons["HOLD"]
    assert "200-day average" in reasons["BELOW"]
    assert "dollar volume" in reasons["ILLIQ"]
    assert "Earnings within" in reasons["EARN"]


def test_no_200dma_history_is_rejected_not_passed():
    passed, rejected = filter_candidates([_cand("NEW", sma200=None)])
    assert passed == []
    assert "200-day" in rejected[0][1]


# --- sizing (AGT-11, 12, 13) ---

def test_inverse_volatility_gives_calmer_name_bigger_weight():
    cands = [_cand("CALM", sector="A", vol=10.0), _cand("WILD", sector="B", vol=40.0)]
    sized = {s["ticker"]: s for s in size_positions(cands, 100_000.0, 4.0)}
    assert sized["CALM"]["target_value"] > sized["WILD"]["target_value"]
    assert sized["CALM"]["volatility_pct"] == 10.0  # vol is recorded with the weight


def test_single_name_cap_is_five_percent_of_equity():
    sized = size_positions([_cand("ONLY")], 100_000.0, 100.0)
    assert sized[0]["target_value"] == pytest.approx(5_000.0)


def test_sector_cap_scales_down_concentrated_sector():
    cands = [_cand(f"T{i}", sector="Tech", vol=20.0) for i in range(8)]
    sized = size_positions(cands, 100_000.0, 100.0)
    sector_total = sum(s["target_value"] for s in sized)
    assert sector_total <= 25_000.0 + 1e-6


def test_position_count_cap_limits_sized_names():
    cands = [_cand(f"N{i}", sector=f"S{i}", vol=20.0) for i in range(CONFIG.max_positions + 5)]
    assert len(size_positions(cands, 100_000.0, 100.0)) == CONFIG.max_positions


def test_exposure_never_exceeds_cap_for_regime():
    cands = [_cand(f"N{i}", sector=f"S{i}", vol=20.0) for i in range(10)]
    sized = size_positions(cands, 100_000.0, 50.0)
    assert sum(s["target_value"] for s in sized) <= 50_000.0 + 1e-6


# --- plan & sequencing (AGT-9, AGT-10, AGT-20) ---

def test_exits_are_sold_before_buys():
    holdings = [Holding("OLD", "Tech", 100, 50.0, "Trim", 40.0)]
    sized = [{"ticker": "NEW", "sector": "Energy", "target_value": 5_000.0, "weight_pct": 5.0,
              "volatility_pct": 20.0, "price": 100.0}]
    st = risk_state(100_000.0, 100_000.0, 100_000.0, False, 100.0)
    orders, _ = plan_orders(holdings, sized, 100_000.0, st)
    assert [o.side for o in orders] == ["sell", "buy"]
    assert orders[0].trigger == "signal_trim"
    assert "Trim" in orders[0].reason


def test_holding_that_leaves_top_set_is_sold_with_reason():
    holdings = [Holding("GONE", "Tech", 10, 100.0, "Buy", 90.0)]
    st = risk_state(100_000.0, 100_000.0, 100_000.0, False, 100.0)
    orders, _ = plan_orders(holdings, [], 100_000.0, st)
    assert orders[0].side == "sell" and orders[0].trigger == "left_top_set"


def test_small_drift_inside_band_produces_no_order():
    holdings = [Holding("HOLD", "Tech", 50, 100.0, "Buy", 90.0)]  # $5,000 = 5%
    sized = [{"ticker": "HOLD", "sector": "Tech", "target_value": 5_400.0, "weight_pct": 5.4,
              "volatility_pct": 20.0, "price": 100.0}]
    st = risk_state(100_000.0, 100_000.0, 100_000.0, False, 100.0)
    orders, skipped = plan_orders(holdings, sized, 100_000.0, st)
    assert orders == []
    assert "within 1%" in skipped[0][1]


def test_reduce_only_keeps_sells_but_blocks_buys():
    holdings = [Holding("OLD", "Tech", 100, 50.0, "Trim", 40.0)]
    sized = [{"ticker": "NEW", "sector": "Energy", "target_value": 5_000.0, "weight_pct": 5.0,
              "volatility_pct": 20.0, "price": 100.0}]
    st = risk_state(97_000.0, 100_000.0, 100_000.0, False, 100.0)
    orders, skipped = plan_orders(holdings, sized, 97_000.0, st)
    assert [o.side for o in orders] == ["sell"]
    assert any("NEW" == t for t, _ in skipped)


# --- pre-trade checks (AGT-19) ---

def _pf(order, holdings=(), cash=100_000.0, equity=100_000.0, state=None, cap=100.0, sectors=None):
    return preflight(
        order, holdings=list(holdings), cash=cash, equity=equity,
        state=state or risk_state(equity, equity, equity, False, cap), exposure_cap_pct=cap,
        sector_by_ticker=sectors or {},
    )


def test_no_shorting_blocks_sell_larger_than_holding():
    h = [Holding("AAA", "Tech", 10, 100.0, "Buy", 90.0)]
    reason = _pf(Order("AAA", "sell", 11, 100.0, 1100.0, "signal_trim", "x"), holdings=h)
    assert "no shorting" in reason


def test_buy_above_cash_is_rejected_with_reason():
    reason = _pf(Order("AAA", "buy", 500, 100.0, 50_000.0, "new_entry", "x"), cash=10_000.0)
    assert "exceeds available cash" in reason


def test_buy_over_single_name_cap_is_rejected():
    reason = _pf(Order("AAA", "buy", 60, 100.0, 6_000.0, "new_entry", "x"))
    assert "single-stock cap" in reason


def test_buy_over_sector_cap_is_rejected():
    h = [Holding("T1", "Tech", 220, 100.0, "Buy", 90.0)]  # $22k Tech, already held
    reason = _pf(Order("T2", "buy", 40, 100.0, 4_000.0, "new_entry", "x"),
                 holdings=h, sectors={"T1": "Tech", "T2": "Tech"})
    assert "sector cap" in reason


def test_buy_over_regime_exposure_cap_is_rejected():
    h = [Holding("A", "X", 480, 100.0, "Buy", 90.0)]  # $48k invested
    reason = _pf(Order("B", "buy", 40, 100.0, 4_000.0, "new_entry", "x"),
                 holdings=h, cap=50.0, sectors={"A": "X", "B": "Y"})
    assert "exposure" in reason and "50%" in reason


def test_buys_blocked_in_reduce_only_state():
    st = risk_state(97_000.0, 100_000.0, 100_000.0, False, 100.0)
    reason = _pf(Order("AAA", "buy", 1, 100.0, 100.0, "new_entry", "x"), state=st)
    assert "daily loss limit" in reason


def test_position_count_cap_blocks_new_name():
    holdings = [Holding(f"H{i}", f"S{i}", 1, 100.0, "Buy", 90.0) for i in range(CONFIG.max_positions)]
    reason = _pf(Order("NEW", "buy", 1, 100.0, 100.0, "new_entry", "x"), holdings=holdings,
                 sectors={h.ticker: h.sector for h in holdings})
    assert "position cap" in reason


def test_valid_buy_passes_preflight():
    assert _pf(Order("AAA", "buy", 10, 100.0, 1_000.0, "new_entry", "x")) is None


# --- vol target (AGT-14) ---

def test_vol_target_scales_down_only_when_estimate_exceeds_target():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0, 0.03, size=(120, 2)), columns=["A", "B"])
    scale, est = scale_for_vol_target({"A": 0.8, "B": 0.8}, returns, target_annual_pct=15.0)
    assert est > 15.0 and scale == pytest.approx(15.0 / est, rel=1e-3)
    scale_ok, _ = scale_for_vol_target({"A": 0.01}, returns, target_annual_pct=15.0)
    assert scale_ok == 1.0


# --- config version (AGT-33) ---

def test_config_version_is_stable_and_changes_with_limits():
    from dataclasses import replace
    assert config_version() == config_version()
    assert config_version(replace(CONFIG, max_positions=19)) != config_version()


# --- indicators ---

def test_indicators_on_known_series():
    closes = pd.Series(np.arange(1.0, 301.0))
    assert sma(closes, 200) == pytest.approx(closes.tail(200).mean())
    assert sma(closes.head(100), 200) is None
    vols = pd.Series(np.full(300, 1_000_000.0))
    assert average_dollar_volume(closes, vols, 20) == pytest.approx(closes.tail(20).mean() * 1_000_000.0)
    highs = closes + 1.0
    lows = closes - 1.0
    assert atr(highs, lows, closes, 14) == pytest.approx(2.0, rel=0.2)


def test_annualized_volatility_needs_enough_history():
    assert annualized_volatility_pct(pd.Series(np.linspace(1, 2, 30)), 63) is None
    flat = pd.Series(np.full(100, 50.0))
    assert annualized_volatility_pct(flat, 63) == 0.0
