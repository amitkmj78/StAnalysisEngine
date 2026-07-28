from typing import Optional


def _entry_reference(trade: dict) -> Optional[float]:
    """Best-known entry price: the real fill if the trade has triggered,
    otherwise the midpoint of the planned entry range."""
    if trade.get("entry_price") is not None:
        return trade["entry_price"]
    entry_low = trade.get("entry_low")
    entry_high = trade.get("entry_high")
    if entry_low is not None and entry_high is not None:
        return (entry_low + entry_high) / 2.0
    return None


def compute_trade_strategy(trade: dict) -> dict:
    """
    Standard risk/reward math from data already on the trade row plus a live
    current price — no new user input, not a recommendation engine. Two
    well-known, explainable rules for the trailing-stop suggestion: move to
    breakeven once up 1R, trail to lock in 1R of profit once up 2R.
    """
    direction = (trade.get("direction") or "LONG").upper()
    is_short = direction == "SHORT"

    entry_ref = _entry_reference(trade)
    stop_loss = trade.get("stop_loss")
    target = trade.get("target")
    current_price = trade.get("current_price")
    status = trade.get("status")

    risk_reward_ratio = None
    if entry_ref is not None and stop_loss is not None and target is not None:
        risk = abs(entry_ref - stop_loss)
        reward = abs(target - entry_ref)
        if risk > 0:
            risk_reward_ratio = round(reward / risk, 2)

    unrealized_pnl_pct = None
    suggested_stop = None
    strategy_note = None

    if status == "ACTIVE" and trade.get("entry_price") is not None and current_price is not None:
        entry_price = trade["entry_price"]
        if is_short:
            unrealized_pnl_pct = round((entry_price - current_price) / entry_price * 100.0, 2)
            gained = entry_price - current_price
        else:
            unrealized_pnl_pct = round((current_price - entry_price) / entry_price * 100.0, 2)
            gained = current_price - entry_price

        risk_amount = abs(entry_price - stop_loss) if stop_loss is not None else None
        if risk_amount and risk_amount > 0 and gained > 0:
            gain_in_r = gained / risk_amount
            if gain_in_r >= 2:
                suggested_stop = round(entry_price + risk_amount, 2) if not is_short else round(entry_price - risk_amount, 2)
                strategy_note = f"Up {gain_in_r:.1f}R — consider trailing stop to ${suggested_stop:.2f} to lock in 1R of profit."
            elif gain_in_r >= 1:
                suggested_stop = round(entry_price, 2)
                strategy_note = f"Up {gain_in_r:.1f}R — consider moving stop to breakeven (${entry_price:.2f})."

        if suggested_stop is not None and stop_loss is not None:
            is_improvement = (suggested_stop > stop_loss) if not is_short else (suggested_stop < stop_loss)
            if not is_improvement:
                suggested_stop = None
                strategy_note = None

    return {
        "risk_reward_ratio": risk_reward_ratio,
        "unrealized_pnl_pct": unrealized_pnl_pct,
        "suggested_stop": suggested_stop,
        "strategy_note": strategy_note,
    }
