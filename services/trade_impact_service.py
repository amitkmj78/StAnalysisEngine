"""DIF-6: what a hypothetical trade would do to a portfolio, before anything is placed.

Pure functions over holdings, so each measure can be checked by hand. Beta is
computed by the caller (portfolio_health_service.compute_portfolio_beta) and
passed in, which keeps this module free of market-data imports.

A holding is {"ticker", "shares", "market_value", "sector", "short_score"}.
  - sector: raw sector name or None (funds and ETFs often have none)
  - short_score: the stored short-term score or None when the app has none

Measures:
  - largest and top-5 position weight, as % of total market value
  - sector weights, as % of total market value (unclassified holdings stay in
    the denominator, the same convention as portfolio_health_service)
  - portfolio score: market-value-weighted short-term score over the holdings
    that have one, with the share of value it covers, so a gap is visible
"""

from typing import Optional

TOP_N = 5


def _total(holdings: list[dict]) -> float:
    return sum(max(h.get("market_value") or 0.0, 0.0) for h in holdings)


def concentration(holdings: list[dict]) -> dict:
    total = _total(holdings)
    if total <= 0:
        return {"largest_position_pct": None, "top5_pct": None, "holdings": 0}
    values = sorted((max(h.get("market_value") or 0.0, 0.0) for h in holdings), reverse=True)
    return {
        "largest_position_pct": round(values[0] / total * 100, 2),
        "top5_pct": round(sum(values[:TOP_N]) / total * 100, 2),
        "holdings": len(holdings),
    }


def sector_weights(holdings: list[dict]) -> dict[str, float]:
    total = _total(holdings)
    if total <= 0:
        return {}
    out: dict[str, float] = {}
    for h in holdings:
        sector = h.get("sector")
        if not sector:
            continue
        out[sector] = out.get(sector, 0.0) + max(h.get("market_value") or 0.0, 0.0)
    return {s: round(v / total * 100, 2) for s, v in sorted(out.items(), key=lambda kv: -kv[1])}


def portfolio_score(holdings: list[dict]) -> dict:
    total = _total(holdings)
    scored = [h for h in holdings if h.get("short_score") is not None and (h.get("market_value") or 0) > 0]
    scored_value = sum(h["market_value"] for h in scored)
    if total <= 0 or scored_value <= 0:
        return {"score": None, "coverage_pct": 0.0}
    score = sum(h["market_value"] * h["short_score"] for h in scored) / scored_value
    return {"score": round(score, 2), "coverage_pct": round(scored_value / total * 100, 2)}


def apply_trade(
    holdings: list[dict],
    ticker: str,
    side: str,
    shares: float,
    price: float,
    sector: Optional[str],
    short_score: Optional[float],
) -> list[dict]:
    """Returns the holdings after the trade. Raises ValueError for a sell larger than the holding."""
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    if shares <= 0 or price <= 0:
        raise ValueError("shares and price must be positive")
    out = [dict(h) for h in holdings]
    existing = next((h for h in out if h["ticker"] == ticker), None)
    held = float(existing["shares"]) if existing else 0.0
    if side == "sell":
        if shares > held + 1e-9:
            raise ValueError(f"cannot sell {shares:g} shares of {ticker}; the portfolio holds {held:g}")
        new_shares = held - shares
        new_value = new_shares * price
        if existing is None:
            return out
        if new_shares <= 1e-9:
            return [h for h in out if h["ticker"] != ticker]
        existing.update(shares=new_shares, market_value=new_value)
        return out
    new_shares = held + shares
    if existing is None:
        out.append({
            "ticker": ticker, "shares": new_shares, "market_value": new_shares * price,
            "sector": sector, "short_score": short_score,
        })
    else:
        existing.update(shares=new_shares, market_value=new_shares * price)
    return out


def _measures(holdings: list[dict]) -> dict:
    return {
        "concentration": concentration(holdings),
        "sector_weights": sector_weights(holdings),
        "portfolio_score": portfolio_score(holdings),
        "total_value": round(_total(holdings), 2),
    }


def compare(before: list[dict], after: list[dict], beta_before: Optional[float], beta_after: Optional[float]) -> dict:
    b, a = _measures(before), _measures(after)
    sectors = sorted(set(b["sector_weights"]) | set(a["sector_weights"]))
    sector_rows = [
        {
            "sector": s,
            "before_pct": b["sector_weights"].get(s, 0.0),
            "after_pct": a["sector_weights"].get(s, 0.0),
        }
        for s in sectors
    ]
    for row in sector_rows:
        row["change_pct"] = round(row["after_pct"] - row["before_pct"], 2)

    def delta(x, y):
        return None if x is None or y is None else round(y - x, 2)

    return {
        "before": {**b, "beta": beta_before},
        "after": {**a, "beta": beta_after},
        "changes": {
            "largest_position_pct": delta(b["concentration"]["largest_position_pct"], a["concentration"]["largest_position_pct"]),
            "top5_pct": delta(b["concentration"]["top5_pct"], a["concentration"]["top5_pct"]),
            "beta": delta(beta_before, beta_after),
            "portfolio_score": delta(b["portfolio_score"]["score"], a["portfolio_score"]["score"]),
        },
        "sectors": sector_rows,
    }
