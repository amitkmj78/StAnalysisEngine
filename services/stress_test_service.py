"""
Pure-compute module for Scenario and Stress Tests (docs/stock-analysis-
requirements.html, STR-1..3). Mirrors services/portfolio_health_
service.py's convention: no FastAPI/DB imports, takes positions:
[{"ticker", "market_value"}] (the same shape GET /health/risk already
builds from portfolio_positions).

STR-3: every result this module produces carries a `method` field --
the plain-English disclosure of how the number was computed, rendered
directly under the result on the frontend (same convention as
TOP10_DISCLOSURE/WASH_SALE_DISCLOSURE). This is not optional/decorative
copy; it's the condition STR-3 sets for shipping any scenario result at
all, so every code path below that returns a numeric impact also
returns the sentence explaining it.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from services.portfolio_health_service import MAX_PARALLEL_HEALTH_FETCHES, compute_portfolio_beta
from services.yfinance_cache import get_cached_history_range

# STR-1's four preset shocks. Each benchmark_ticker is the real-world
# proxy whose historical beta estimates the portfolio's sensitivity to
# the stated shock -- see each method string for exactly how and why
# that proxy was chosen, including its known limitations.
SHOCK_PRESETS: dict[str, dict] = {
    "market_down_10": {
        "label": "Market -10%",
        "benchmark_ticker": "SPY",
        "shock_pct": -10.0,
        "method": "Beta-based estimate: your portfolio's beta to SPY × -10%.",
    },
    "tech_down_20": {
        "label": "Tech -20%",
        "benchmark_ticker": "XLK",
        "shock_pct": -20.0,
        "method": (
            "Beta-based estimate: your portfolio's beta to XLK (Technology Select "
            "Sector SPDR) × -20%."
        ),
    },
    "oil_up_30": {
        "label": "Oil +30%",
        "benchmark_ticker": "USO",
        "shock_pct": 30.0,
        "method": (
            "Beta-based estimate using USO (United States Oil Fund), which tracks the "
            "spot price of crude oil futures directly. Your portfolio's estimated "
            "impact = your portfolio's beta to USO × +30%. Caveat: USO's own price "
            "can diverge from spot crude over time because of futures-roll costs "
            "(contango/backwardation), so its historical beta is an imperfect -- but "
            "the most literal available -- proxy for a pure oil-price move."
        ),
    },
    "rates_up_1pct": {
        "label": "Rates +1%",
        "benchmark_ticker": "TLT",
        "shock_pct": -17.0,
        "method": (
            "Beta-based estimate. A +1 percentage point parallel rise in long-term "
            "interest rates is approximated as a -17% move in TLT (iShares 20+ Year "
            "Treasury Bond ETF), using TLT's commonly-cited approximate effective "
            "duration of ~17 years (Price Change ≈ -Duration × Rate Change). "
            "Your portfolio's estimated impact = your portfolio's beta to TLT × "
            "-17%. TLT's actual duration moves with the rate environment -- it is not "
            "a fixed constant -- so treat this figure as an order-of-magnitude "
            "estimate, not a precise one."
        ),
    },
}

# STR-1's three historical replays. start/end are the exact window each
# holding's own real return is measured over -- STR-3 requires the
# *stated* window to be the *actual* window used, so these are the
# single source of truth for both the computation and the method text.
HISTORICAL_REPLAYS: dict[str, dict] = {
    "2008_gfc": {
        "label": "2008 Financial Crisis",
        "start": "2008-09-01",
        "end": "2009-03-09",
        "method": (
            "Historical replay: each holding's own actual total return from "
            "2008-09-01 (Lehman Brothers' collapse) through 2009-03-09 (the market "
            "bottom), applied to its current market value. This is realized history "
            "for each ticker, not a model."
        ),
    },
    "2020_covid": {
        "label": "2020 COVID Crash",
        "start": "2020-02-19",
        "end": "2020-03-23",
        "method": (
            "Historical replay: each holding's own actual total return from "
            "2020-02-19 (the pre-crash peak) through 2020-03-23 (the crash bottom), "
            "applied to its current market value. This is realized history for each "
            "ticker, not a model."
        ),
    },
    "2022_bear": {
        "label": "2022 Bear Market",
        "start": "2022-01-03",
        "end": "2022-10-12",
        "method": (
            "Historical replay: each holding's own actual total return from "
            "2022-01-03 (the market's 2022 peak) through 2022-10-12 (the 2022 "
            "bottom), applied to its current market value. This is realized history "
            "for each ticker, not a model."
        ),
    },
}


def run_preset_shock(positions: list[dict], preset_key: str, period: str = "1y") -> dict:
    """STR-1: portfolio-aggregate $ and % impact of one preset shock, via
    the portfolio's blended beta to that shock's benchmark proxy
    (services/portfolio_health_service.py::compute_portfolio_beta --
    same math HLT-2's risk page already ships). Deliberately aggregate-
    only, no per-holding breakdown: a single-ticker beta over a 1-year
    daily series is materially noisier than the blended-portfolio beta
    this app already uses elsewhere, and STR-1's literal acceptance
    criterion doesn't require a per-holding table."""
    if preset_key not in SHOCK_PRESETS:
        raise KeyError(f"Unknown preset: {preset_key}")
    preset = SHOCK_PRESETS[preset_key]
    beta_result = compute_portfolio_beta(positions, preset["benchmark_ticker"], period)
    total_market_value = round(sum(p.get("market_value") or 0.0 for p in positions), 2)

    beta = beta_result["beta"]
    if beta is None or not total_market_value:
        estimated_pct_impact = None
        estimated_dollar_impact = None
    else:
        estimated_pct_impact = round(beta * preset["shock_pct"], 2)
        estimated_dollar_impact = round(total_market_value * estimated_pct_impact / 100.0, 2)

    return {
        "kind": "preset",
        "preset_key": preset_key,
        "label": preset["label"],
        "method": preset["method"],
        "benchmark_ticker": preset["benchmark_ticker"],
        "shock_pct": preset["shock_pct"],
        "beta": beta,
        "period": beta_result["period"],
        "data_start": beta_result["data_start"],
        "data_end": beta_result["data_end"],
        "total_market_value": total_market_value,
        "estimated_pct_impact": estimated_pct_impact,
        "estimated_dollar_impact": estimated_dollar_impact,
        "excluded_from_beta": beta_result["excluded_from_benchmark"],
    }


def _ticker_window_return(ticker: str, start: str, end: str) -> float | None:
    """None means "no usable price history for this ticker over this
    exact window" (fetch failed, empty, or fewer than 2 rows to compute
    a return from) -- the caller excludes it with disclosure rather than
    estimating it, per this feature's "historical replay = real history,
    not a second model" design decision."""
    try:
        hist = get_cached_history_range(ticker, start, end, auto_adjust=True)
    except Exception:
        return None
    if hist.empty or len(hist) < 2:
        return None
    first, last = hist["Close"].iloc[0], hist["Close"].iloc[-1]
    if not first:
        return None
    return float(last / first - 1.0) * 100.0


def run_historical_replay(positions: list[dict], replay_key: str) -> dict:
    """STR-1: each holding's own actual realized return over a fixed
    historical window, applied to its current market value -- a real
    per-holding breakdown by construction (unlike run_preset_shock's
    modeled, aggregate-only result), since a replay's whole point is
    "what actually happened to your actual holdings." A holding with no
    price history that far back (e.g. IPO'd after the window started) is
    excluded, not beta-proxied via a second ticker -- that would quietly
    layer an unstated second model under what's supposed to be realized
    history; see the disclosure sentence for exactly who's excluded."""
    if replay_key not in HISTORICAL_REPLAYS:
        raise KeyError(f"Unknown replay: {replay_key}")
    replay = HISTORICAL_REPLAYS[replay_key]
    total_market_value = round(sum(p.get("market_value") or 0.0 for p in positions), 2)

    tickers = [p["ticker"] for p in positions]
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_HEALTH_FETCHES) as executor:
        returns = list(
            executor.map(lambda t: _ticker_window_return(t, replay["start"], replay["end"]), tickers)
        )

    holdings = []
    total_dollar_impact = 0.0
    excluded_holdings = []
    for p, pct in zip(positions, returns):
        market_value = p.get("market_value") or 0.0
        if pct is None:
            holdings.append({
                "ticker": p["ticker"], "market_value": round(market_value, 2),
                "estimated_pct_impact": None, "estimated_dollar_impact": None,
                "method_used": "excluded_no_history_for_window",
            })
            excluded_holdings.append(p["ticker"])
            continue
        dollar_impact = round(market_value * pct / 100.0, 2)
        total_dollar_impact += dollar_impact
        holdings.append({
            "ticker": p["ticker"], "market_value": round(market_value, 2),
            "estimated_pct_impact": round(pct, 2), "estimated_dollar_impact": dollar_impact,
            "method_used": "actual",
        })

    total_pct_impact = (
        round(total_dollar_impact / total_market_value * 100.0, 2) if total_market_value else None
    )
    disclosure = replay["method"]
    if excluded_holdings:
        disclosure += (
            f" Excluded (no price history over this window, likely IPO'd after it "
            f"started): {', '.join(excluded_holdings)}. The total below does not "
            f"reflect these holdings and may understate the real impact."
        )

    return {
        "kind": "replay",
        "replay_key": replay_key,
        "label": replay["label"],
        "method": disclosure,
        "window_start": replay["start"],
        "window_end": replay["end"],
        "total_market_value": total_market_value,
        "estimated_pct_impact": total_pct_impact,
        "estimated_dollar_impact": round(total_dollar_impact, 2) if total_market_value else None,
        "excluded_holdings": excluded_holdings,
        "holdings": holdings,
    }


def run_custom_scenario(
    positions: list[dict], sector_by_ticker: dict[str, str], components: list[dict], period: str = "1y"
) -> dict:
    """STR-2: components is a list of either
    {"kind": "sector", "sector": "Technology", "shock_pct": -15.0, "label": "..."}
    or {"kind": "factor", "benchmark_ticker": "SPY", "shock_pct": -10.0, "label": "..."}.

    Sector components apply shock_pct DIRECTLY (1:1, no beta, no network
    call) to the market value of holdings classified in that sector only
    -- honest and exact, since "Technology -15%" literally means
    "stocks classified Technology move -15%." Factor components apply
    via the portfolio's blended beta to that benchmark (same math as
    run_preset_shock) -- portfolio-aggregate, not per-holding, for the
    same single-ticker-beta-noise reason STR-1's presets avoid it.
    Components combine by simple (naive) addition -- no cross-factor
    correlation modeled, explicitly disclosed in the result's own
    method field."""
    total_market_value = round(sum(p.get("market_value") or 0.0 for p in positions), 2)
    component_results = []
    total_dollar_impact = 0.0

    for c in components:
        if c["kind"] == "sector":
            sector_value = sum(
                p.get("market_value") or 0.0
                for p in positions
                if sector_by_ticker.get(p["ticker"]) == c["sector"]
            )
            dollar_impact = round(sector_value * c["shock_pct"] / 100.0, 2)
            method = (
                f"Direct: holdings classified {c['sector']} (${round(sector_value, 2):,.2f}) move "
                f"{c['shock_pct']:+.1f}%. Holdings outside {c['sector']} are unaffected by this component."
            )
        elif c["kind"] == "factor":
            beta_result = compute_portfolio_beta(positions, c["benchmark_ticker"], period)
            beta = beta_result["beta"]
            if beta is None or not total_market_value:
                dollar_impact = None
                method = (
                    f"Beta-based: portfolio's beta to {c['benchmark_ticker']} could not be "
                    f"computed (insufficient data)."
                )
            else:
                pct = beta * c["shock_pct"]
                dollar_impact = round(total_market_value * pct / 100.0, 2)
                method = (
                    f"Beta-based: portfolio's beta to {c['benchmark_ticker']} ({beta:.2f}) "
                    f"× {c['shock_pct']:+.1f}%."
                )
        else:
            raise ValueError(f"Unknown component kind: {c['kind']}")

        if dollar_impact is not None:
            total_dollar_impact += dollar_impact
        component_results.append({
            "label": c.get("label") or c["kind"],
            "kind": c["kind"],
            "method": method,
            "estimated_dollar_impact": dollar_impact,
            "estimated_pct_impact": (
                round(dollar_impact / total_market_value * 100.0, 2)
                if dollar_impact is not None and total_market_value else None
            ),
        })

    return {
        "kind": "custom",
        "components": component_results,
        "total_market_value": total_market_value,
        "total_estimated_dollar_impact": round(total_dollar_impact, 2) if total_market_value else None,
        "total_estimated_pct_impact": (
            round(total_dollar_impact / total_market_value * 100.0, 2) if total_market_value else None
        ),
        "method": (
            "Combination of the components below. Sector components apply directly to "
            "holdings in that sector only; factor components apply via the portfolio's "
            "beta to that benchmark. Impacts are summed (simple addition) -- this does "
            "not model correlation between components, so it is a simplification, not a "
            "joint simulation."
        ),
    }
