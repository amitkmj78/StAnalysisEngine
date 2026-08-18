from typing import Any, Dict

from .sentiment_service import get_sentiment_summary

# Kept out of prediction_service.py deliberately — that module's "no LLM,
# no news, no sentiment" claim (shown to users on /predict) stays literally
# true for the actual forecast/backtest; this is a separate, optional layer.


def _build_prompt(ticker: str, context: Dict[str, Any], sentiment_text: str) -> str:
    lines = [
        f"You are a plain-English markets assistant. A quantitative model just produced the "
        f"following forecast for {ticker}. Write a short (4-6 sentence) narrative summary for a "
        f"retail investor. State the model's view, then weigh it against the news/earnings context "
        f"below, and explicitly say whether they agree or conflict. Specifically call out any recent "
        f"earnings report, guidance change, or news catalyst that plausibly explains the stock's "
        f"recent price action (including any pre/post-market move noted below) — this is usually the "
        f"real reason for a large single-day move, not the quant model. Do not invent numbers not "
        f"given here. Do not give financial advice beyond describing what the data shows.",
        "",
        f"Last close: {context.get('last_close')}",
    ]
    if context.get("next_price") is not None:
        lines.append(f"Predicted next close: {context['next_price']}")
    extended = context.get("extended_hours")
    if extended:
        label = "After-hours" if extended["state"] == "POST" else "Pre-market"
        lines.append(
            f"{label} price right now: {extended['price']} "
            f"({'+' if (extended.get('change_pct') or 0) >= 0 else ''}{extended.get('change_pct')}% vs regular close) "
            f"— this is a real-time move worth explaining if the news/earnings context below accounts for it."
        )
    signal = context.get("signal")
    if signal:
        days_ahead = context.get("days_ahead", 10)
        lines.append(
            f"{days_ahead}-day signal: {signal['signal']} (expected return {signal['expected_return_pct']}%, "
            f"target price {signal['target_price']})"
        )
    metrics = context.get("metrics")
    if metrics and metrics.get("beats_naive") is not None:
        lines.append(
            f"Backtest: this model {'beats' if metrics['beats_naive'] else 'does not beat'} a naive "
            f"no-change baseline for this ticker recently (RMSE {metrics.get('rmse')} vs naive "
            f"{metrics.get('naive_rmse')})."
        )
    lines.append("")
    lines.append(f"News & earnings context:\n{sentiment_text}")
    return "\n".join(lines)


def build_prediction_narrative(llm, ticker: str, context: Dict[str, Any]) -> Dict[str, str]:
    sentiment_text = get_sentiment_summary(ticker, llm=llm)
    prompt = _build_prompt(ticker, context, sentiment_text)
    response = llm.invoke(prompt)
    narrative = getattr(response, "content", str(response))
    return {"narrative": narrative, "sentiment_context": sentiment_text}
