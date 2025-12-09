from typing import Optional, Dict
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from .data_service import get_stock_data


# ============================================
#   INTERNAL MODEL TRAINING UTILITIES
# ============================================

def _base_gbm() -> GradientBoostingRegressor:
    """Base GBM config used as a starting point."""
    return GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.01,
        max_depth=4,
        random_state=42,
    )


def _train_price_model(df: pd.DataFrame, tune: bool = False) -> GradientBoostingRegressor:
    """
    Train a GBM model on engineered features.
    If tune=True, run a very small GridSearchCV to auto-tune hyperparams.
    """
    features = ["Index", "Lag1", "MA5", "MA10"]
    X = df[features].values
    y = df["Close"].values

    if not tune:
        model = _base_gbm()
        model.fit(X, y)
        return model

    # Very small grid to avoid being too slow
    param_grid = {
        "n_estimators": [200, 400],
        "learning_rate": [0.01, 0.02],
        "max_depth": [3, 4],
    }

    grid = GridSearchCV(
        _base_gbm(),
        param_grid,
        cv=3,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
    )
    grid.fit(X, y)
    return grid.best_estimator_


def _prepare_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    """
    Take raw OHLC data and add features needed for the model.
    """
    df = data.copy()
    df["Index"] = np.arange(len(df))
    df["Lag1"] = df["Close"].shift(1)
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df = df.dropna()
    return df


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

    df = _prepare_feature_frame(data)
    if df.empty:
        return None

    model = _train_price_model(df, tune=tune)

    last = df.iloc[-1]
    X_new = np.array([[last["Index"] + 1, last["Close"], last["MA5"], last["MA10"]]])
    return float(model.predict(X_new)[0])


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
    if data.empty or len(data) < days_back + 25:
        return None

    df = _prepare_feature_frame(data)
    if df.empty:
        return None

    actual_list = []
    predicted_list = []
    dates_list = []

    n = len(df)

    for i in range(n - days_back, n):
        sample = df.iloc[:i]
        if len(sample) < 25:
            continue

        model = _train_price_model(sample, tune=tune)

        last_sample = sample.iloc[-1]
        X_next = np.array(
            [[last_sample["Index"] + 1, last_sample["Close"], last_sample["MA5"], last_sample["MA10"]]]
        )
        pred = float(model.predict(X_next)[0])

        actual = df["Close"].iloc[i]
        actual_list.append(actual)
        predicted_list.append(pred)
        dates_list.append(df.index[i])

    if not dates_list:
        return None

    backtest_df = pd.DataFrame(
        {
            "Date": dates_list,
            "Actual": actual_list,
            "Predicted": predicted_list,
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

    df = _prepare_feature_frame(data)
    if df.empty:
        return None

    model = _train_price_model(df, tune=tune)

    # Compute residual std for CI
    features = ["Index", "Lag1", "MA5", "MA10"]
    X_full = df[features].values
    y_full = df["Close"].values
    y_hat = model.predict(X_full)
    residuals = y_full - y_hat
    sigma = np.std(residuals, ddof=1) if len(residuals) > 1 else 0.0

    # Future dates: business days
    last_date = df.index[-1]
    future_dates = pd.date_range(
        start=last_date,
        periods=days_ahead + 1,
        freq="B",
    )[1:]  # skip the current last_date

    last_index = df["Index"].iloc[-1]
    rolling_closes = list(df["Close"].values)

    preds = []
    lower_ci = []
    upper_ci = []

    z = 1.96  # 95% CI

    for i in range(1, days_ahead + 1):
        new_index = last_index + i
        lag1 = rolling_closes[-1]
        ma5 = np.mean(rolling_closes[-5:])
        ma10 = np.mean(rolling_closes[-10:]) if len(rolling_closes) >= 10 else ma5

        X_new = np.array([[new_index, lag1, ma5, ma10]])
        pred = float(model.predict(X_new)[0])

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

def compute_backtest_metrics(backtest_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute RMSE, MAE, and MAPE from a backtest DataFrame.
    """
    actual = backtest_df["Actual"].values
    predicted = backtest_df["Predicted"].values

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
