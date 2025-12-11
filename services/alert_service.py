def alert_breakout(current_price: float, reference_price: float) -> str:
    """
    Alerts if the current price is breaking above/below a reference level.
    """
    if current_price > reference_price:
        return f"📈 Breakout detected: Price ${current_price:.2f} is above reference ${reference_price:.2f}"
    elif current_price < reference_price:
        return f"📉 Breakdown: Price ${current_price:.2f} is below reference ${reference_price:.2f}"
    else:
        return "➡ Price is at the reference level."