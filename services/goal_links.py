"""STRAT-10: the target mix of a goal's linked holdings, drift against it, and where the next contributions should go.

Targets come from the time horizon: a long horizon allows more stock, and the stock share falls as the goal nears. The
linked holdings are compared as their own sleeve, because bonds usually sit outside this portfolio. Individual stock
picks are capped (10% under seven years, 20% beyond), and the rest of the stock share is core broad funds.

Steering suggests where the next contribution should go to move the sleeve back toward target. It buys with new money
only, so no shares are sold, which avoids capital-gains tax in a taxable account. Whole shares only; leftovers are cash.
"""

import math

DRIFT_FLAG_POINTS = 5.0
# (years to goal, stock share of the goal %), interpolated linearly and held flat outside these points.
_STOCK_SHARE_CURVE = [(3.0, 40.0), (5.0, 60.0), (10.0, 75.0), (15.0, 90.0)]


def target_mix(years: float) -> dict:
    """Stock share of the goal, its bonds share, and the stock-picks cap. The core share is the rest of the stock share."""
    points = _STOCK_SHARE_CURVE
    if years <= points[0][0]:
        stock = points[0][1]
    elif years >= points[-1][0]:
        stock = points[-1][1]
    else:
        stock = points[0][1]
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if x0 <= years <= x1:
                stock = y0 + (y1 - y0) * (years - x0) / (x1 - x0)
                break
    picks_cap = 10.0 if years < 7 else 20.0
    picks = min(picks_cap, stock)
    return {"stock_pct": round(stock, 1), "bonds_pct": round(100 - stock, 1), "picks_pct": round(picks, 1),
            "core_pct": round(stock - picks, 1), "picks_cap_pct": picks_cap}


def sleeve_drift(holdings: list[dict], targets: dict) -> dict:
    """holdings: [{ticker, role, value}]. Compares each role's share of the linked sleeve with its target share of stock."""
    total = sum(h["value"] for h in holdings)
    stock = targets["stock_pct"] or 0.0
    if total <= 0 or stock <= 0:
        return {"total": round(total, 2), "rows": [], "flags": []}
    rows, flags = [], []
    for role, target_of_goal in (("core", targets["core_pct"]), ("pick", targets["picks_pct"])):
        value = sum(h["value"] for h in holdings if h["role"] == role)
        actual = value / total * 100
        target = target_of_goal / stock * 100
        drift = actual - target
        rows.append({"role": role, "actual_pct": round(actual, 1), "target_pct": round(target, 1),
                     "drift_points": round(drift, 1)})
        if abs(drift) > DRIFT_FLAG_POINTS:
            label = "Core funds" if role == "core" else "Stock picks"
            flags.append(f"{label} are {abs(drift):.1f} points {'above' if drift > 0 else 'below'} their target share.")
    return {"total": round(total, 2), "rows": rows, "flags": flags}


def steering(monthly_amount: float, holdings: list[dict], targets: dict, prices: dict[str, float]) -> dict:
    """Where the next monthly contribution should go, to move the sleeve back toward target. Buys only. Returns each
    holding's share of the money and the whole shares it buys at its price, plus any leftover cash."""
    stock_share = targets["stock_pct"] / 100
    money = monthly_amount * stock_share
    total = sum(h["value"] for h in holdings)
    after = total + money
    picks_value = sum(h["value"] for h in holdings if h["role"] == "pick")
    core_value = sum(h["value"] for h in holdings if h["role"] == "core")
    stock = targets["stock_pct"] or 1.0
    picks_goal = after * (targets["picks_pct"] / stock)
    core_goal = after * (targets["core_pct"] / stock)
    need_picks = max(0.0, picks_goal - picks_value)
    need_core = max(0.0, core_goal - core_value)
    need = need_picks + need_core
    if need > 0:
        picks_amt = money * need_picks / need
        core_amt = money * need_core / need
    else:
        picks_amt = money * (targets["picks_pct"] / stock)
        core_amt = money - picks_amt

    buys, cash = [], 0.0
    for role, amount in (("pick", picks_amt), ("core", core_amt)):
        members = [h for h in holdings if h["role"] == role and prices.get(h["ticker"], 0) > 0]
        if not members:
            cash += amount
            continue
        per = amount / len(members)
        for h in members:
            price = prices[h["ticker"]]
            shares = math.floor(per / price)
            spent = shares * price
            buys.append({"ticker": h["ticker"], "role": role, "amount": round(per, 2), "shares": shares,
                         "price": price, "spent": round(spent, 2)})
            cash += per - spent
    return {"monthly": round(monthly_amount, 2), "into_stocks": round(money, 2),
            "toward_bonds_outside_portfolio": round(monthly_amount - money, 2),
            "buys": buys, "cash": round(cash, 2)}
