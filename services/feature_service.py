import numpy as np
import pandas as pd

from .technical_service import add_indicators

# Deliberately no raw row-index / time feature: a model given "how many
# rows in" as an input can just learn to extrapolate the trend line
# instead of learning price behavior, which inflates backtest accuracy
# without adding real predictive signal.
FEATURE_COLUMNS = [
    "Lag1",
    "MA5",
    "MA10",
    "Return1",
    "RSI",
    "MACD",
    "MACD_signal",
    "BB_pctB",
]


def build_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    """
    Turn raw OHLC data into the model-ready feature + target frame.

    The target ("Target") is the next day's *return*, not the next day's
    raw price level. Regressing the price level lets a model win mostly by
    copying yesterday's close through the Lag1 feature, which is why it
    rarely beat a naive no-change baseline in backtesting. The final row's
    Target is NaN (its next-day return hasn't happened yet) and is left in
    place rather than dropped here, so that row's features remain usable
    for a next-step prediction; callers that train on this frame drop the
    NaN target themselves.
    """
    df = add_indicators(data)

    df["Lag1"] = df["Close"].shift(1)
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["Return1"] = df["Close"].pct_change()

    band_width = df["BB_high"] - df["BB_low"]
    df["BB_pctB"] = np.where(band_width > 0, (df["Close"] - df["BB_low"]) / band_width, 0.5)

    df = df.dropna()
    df["Target"] = df["Close"].pct_change().shift(-1)
    return df


def next_step_features(df_features: pd.DataFrame) -> np.ndarray:
    """
    Build the input row used to predict the next (not-yet-seen) close.

    Only Lag1 needs to roll forward explicitly (tomorrow's Lag1 is today's
    close); the other indicators are carried forward from their latest
    known value as a same-day approximation, since they move slowly
    relative to a one-day-ahead horizon.
    """
    last = df_features.iloc[-1]
    row = {
        "Lag1": last["Close"],
        "MA5": last["MA5"],
        "MA10": last["MA10"],
        "Return1": last["Return1"],
        "RSI": last["RSI"],
        "MACD": last["MACD"],
        "MACD_signal": last["MACD_signal"],
        "BB_pctB": last["BB_pctB"],
    }
    return np.array([[row[col] for col in FEATURE_COLUMNS]])
