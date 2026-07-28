from typing import Optional, Dict
import numpy as np
import pandas as pd

from .data_service import get_stock_data
from .feature_service import FEATURE_COLUMNS, build_feature_frame, next_step_features
from .model_service import RETURN_SHRINKAGE, get_price_model, train_model_on_frame


# ============================================
#   CORE PREDICT FUNCTIONS
# ============================================

def predict_next_price(ticker: str, period: str, tune: bool = False) -> Optional[float]:
    """
    Predict the next-day closing price.
    """
    data = get_stock_data(ticker, period)
    if data.empty or len(data) < 20:
        return None

    df = build_feature_frame(data)
    if df.empty:
        return None

    model = get_price_model(ticker, period, tune=tune)
    if model is None:
        return None
    X_new = next_step_features(df)
    pred_return = float(model.predict(X_new)[0]) * RETURN_SHRINKAGE
    last_close = float(df["Close"].iloc[-1])
    return last_close * (1 + pred_return)


def predict_backtest_prices(
    ticker: str,
    period: str,
    days_back: int = 30,
    tune: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Walk-forward backtest:
    For each of the last `days_back` points, train on data before that day,
    predict that day, and compare to actual.
    Returns a DataFrame indexed by date with columns: Actual, Predicted.
    """
    data = get_stock_data(ticker, period)
    if data.empty or len(data) < days_back + 26:
        return None

    df = build_feature_frame(data)
    if df.empty:
        return None

    actual_list = []
    predicted_list = []
    naive_list = []
    dates_list = []

    n = len(df)

    for i in range(n - days_back, n):
        sample = df.iloc[:i]

        # Exclude the sample's own last row from training: its Target is
        # the return from day i-1 to day i, i.e. exactly the value this
        # step is trying to predict. Precomputing Target over the whole
        # series and then slicing a prefix would otherwise leak that
        # answer into training, even though the row's *features* (used
        # below as the prediction input) are legitimately known already.
        train_sample = sample.iloc[:-1]
        if len(train_sample) < 25:
            continue

        model = train_model_on_frame(train_sample, tune=tune)
        X_next = next_step_features(sample)
        pred_return = float(model.predict(X_next)[0]) * RETURN_SHRINKAGE

        last_close = float(sample["Close"].iloc[-1])
        pred = last_close * (1 + pred_return)

        # Naive baseline: "tomorrow's price = today's close" (no-change
        # random-walk forecast). A model that can't beat this isn't adding
        # predictive value, however good its raw error numbers look alone.
        naive_pred = last_close

        actual = df["Close"].iloc[i]
        actual_list.append(actual)
        predicted_list.append(pred)
        naive_list.append(naive_pred)
        dates_list.append(df.index[i])

    if not dates_list:
        return None

    backtest_df = pd.DataFrame(
        {
            "Date": dates_list,
            "Actual": actual_list,
            "Predicted": predicted_list,
            "Naive": naive_list,
        }
    ).set_index("Date")

    return backtest_df


def predict_future_prices(
    ticker: str,
    period: str,
    days_ahead: int = 10,
    tune: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Multi-step recursive forecast: Predict the next `days_ahead` future closing prices.
    Also returns 95% confidence intervals based on in-sample residuals.
    """
    data = get_stock_data(ticker, period)
    if data.empty or len(data) < 40:
        return None

    df = build_feature_frame(data)
    if df.empty:
        return None

    model = get_price_model(ticker, period, tune=tune)
    if model is None:
        return None

    # Compute residual std for CI, converted back to price space: the
    # model predicts a return, so in-sample error has to be measured as
    # (predicted next price) vs (actual next price), not on the raw
    # return values.
    trainable = df.dropna(subset=["Target"])
    X_full = trainable[FEATURE_COLUMNS].values
    prev_close = trainable["Close"].values
    y_hat_return = model.predict(X_full) * RETURN_SHRINKAGE
    y_hat_price = prev_close * (1 + y_hat_return)
    actual_next_price = prev_close * (1 + trainable["Target"].values)
    residuals = y_hat_price - actual_next_price
    sigma = np.std(residuals, ddof=1) if len(residuals) > 1 else 0.0

    # Future dates: business days
    last_date = df.index[-1]
    future_dates = pd.date_range(
        start=last_date,
        periods=days_ahead + 1,
        freq="B",
    )[1:]  # skip the current last_date

    rolling_closes = list(df["Close"].values)

    # RSI/MACD/Bollinger move slowly relative to a one-day step, so they're
    # held at their last observed value for the whole horizon rather than
    # recursively re-simulated — a known simplification, not a true
    # multi-day forecast of oscillator state.
    last_row = df.iloc[-1]
    fixed_rsi = last_row["RSI"]
    fixed_macd = last_row["MACD"]
    fixed_macd_signal = last_row["MACD_signal"]
    fixed_bb_pctb = last_row["BB_pctB"]

    preds = []
    lower_ci = []
    upper_ci = []

    z = 1.96  # 95% CI

    for _ in range(days_ahead):
        lag1 = rolling_closes[-1]
        ma5 = np.mean(rolling_closes[-5:])
        ma10 = np.mean(rolling_closes[-10:]) if len(rolling_closes) >= 10 else ma5
        return1 = (rolling_closes[-1] - rolling_closes[-2]) / rolling_closes[-2] if len(rolling_closes) >= 2 else 0.0

        X_new = np.array([[lag1, ma5, ma10, return1, fixed_rsi, fixed_macd, fixed_macd_signal, fixed_bb_pctb]])
        pred_return = float(model.predict(X_new)[0]) * RETURN_SHRINKAGE
        pred = rolling_closes[-1] * (1 + pred_return)

        preds.append(pred)

        if sigma > 0:
            lower_ci.append(pred - z * sigma)
            upper_ci.append(pred + z * sigma)
        else:
            lower_ci.append(pred)
            upper_ci.append(pred)

        rolling_closes.append(pred)

    future_df = pd.DataFrame(
        {
            "Date": future_dates,
            "Predicted": preds,
            "Lower_CI": lower_ci,
            "Upper_CI": upper_ci,
        }
    ).set_index("Date")

    return future_df


# ============================================
#   METRICS & SIGNALS
# ============================================

def _error_stats(predicted: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    errors = predicted - actual
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    # Avoid division by zero for MAPE
    non_zero_mask = actual != 0
    if np.any(non_zero_mask):
        mape = float(
            np.mean(
                np.abs((predicted[non_zero_mask] - actual[non_zero_mask]) / actual[non_zero_mask])
            )
            * 100.0
        )
    else:
        mape = float("nan")

    return {"rmse": rmse, "mae": mae, "mape": mape}


def compute_backtest_metrics(backtest_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute RMSE, MAE, and MAPE for the model's predictions, and — when a
    "Naive" column is present — for a naive "no price change" baseline too,
    so the model's accuracy is judged against doing nothing rather than in
    isolation. Adds `beats_naive` (True if the model's RMSE is lower).
    """
    actual = backtest_df["Actual"].values
    predicted = backtest_df["Predicted"].values

    metrics = _error_stats(predicted, actual)

    if "Naive" in backtest_df.columns:
        naive_stats = _error_stats(backtest_df["Naive"].values, actual)
        metrics["naive_rmse"] = naive_stats["rmse"]
        metrics["naive_mae"] = naive_stats["mae"]
        metrics["naive_mape"] = naive_stats["mape"]
        metrics["beats_naive"] = metrics["rmse"] < naive_stats["rmse"]

    return metrics


def generate_trading_signal(
    last_close: float,
    future_df: pd.DataFrame,
    buy_threshold: float = 0.05,
    sell_threshold: float = -0.05,
) -> Dict[str, float | str]:
    """
    Generate a simple Buy/Hold/Sell signal based on expected return to the
    last forecasted price.

    buy_threshold / sell_threshold are in decimal form (0.05 = 5%).
    """
    if last_close is None or np.isnan(last_close):
        return {
            "signal": "UNKNOWN",
            "expected_return_pct": 0.0,
        }

    final_price = float(future_df["Predicted"].iloc[-1])
    expected_return = (final_price - last_close) / last_close

    if expected_return >= buy_threshold:
        signal = "BUY"
    elif expected_return <= sell_threshold:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "signal": signal,
        "expected_return_pct": round(expected_return * 100.0, 2),
        "last_close": float(last_close),
        "target_price": round(final_price, 2),
    }
def predict_long_term_prices(ticker: str, period: str, years: int = 5):
    """
    Project long-term price using GBM:
    S(t) = S0 * exp((mu - 0.5*σ²)t + σ*sqrt(t)*Z)
    """
    data = get_stock_data(ticker, period)
    if data.empty: return None

    log_returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()
    mu = log_returns.mean() * 252
    sigma = log_returns.std() * 252**0.5

    S0 = data["Close"].iloc[-1]

    forecast = []
    days = years * 252

    for t in range(1, days + 1):
        Z = np.random.normal()
        price = S0 * np.exp((mu - 0.5 * sigma**2) * (t/252) + sigma * ((t/252)**0.5) * Z)
        forecast.append(price)

    future_index = pd.date_range(data.index[-1], periods=days+1, freq="B")[1:]
    return pd.DataFrame({"Forecast": forecast}, index=future_index)
