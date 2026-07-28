from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from .cache_utils import ttl_cache
from .data_service import get_stock_data
from .feature_service import FEATURE_COLUMNS, build_feature_frame

# How long a trained model is kept before Streamlit silently retrains it.
# The model targets next-day price movement from *daily* bars, which only
# change once per trading day, so an hourly-scale TTL is plenty fresh
# without retraining a GBM on every widget interaction.
MODEL_TTL_SECONDS = 6 * 60 * 60

# Empirically found via walk-forward backtesting across a 6-ticker basket
# (AAPL/MSFT/TSLA/SPY/AMD/JNJ, 30- and 60-day windows): the raw GBM's
# predicted return is noisy enough that using it at full strength loses to
# a naive "no change" baseline 100% of the time (avg RMSE 9.5% worse than
# naive). Damping it toward zero before use consistently reduces RMSE/MAE
# and raises the beat-naive rate to ~33% (avg RMSE within 0.5% of naive) —
# a real, measured improvement. Smaller shrink values keep helping RMSE in
# the limit as they approach pure naive, which is itself an honest signal
# that the raw model carries little directional edge beyond noise; 0.2 is
# chosen to keep a real backtested improvement without collapsing the
# forecast to a flat line.
RETURN_SHRINKAGE = 0.2


def _base_gbm() -> GradientBoostingRegressor:
    """Base GBM config used as a starting point."""
    return GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.01,
        max_depth=4,
        random_state=42,
    )


def _fit(X, y, tune: bool) -> GradientBoostingRegressor:
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
    # TimeSeriesSplit, not plain K-Fold: default cv shuffles rows, which
    # for a time series means training folds can contain data from
    # *after* the validation fold — the tuned hyperparameters would be
    # picked using information that wouldn't actually be available yet.
    grid = GridSearchCV(
        _base_gbm(),
        param_grid,
        cv=TimeSeriesSplit(n_splits=3),
        n_jobs=-1,
        scoring="neg_mean_squared_error",
    )
    grid.fit(X, y)
    return grid.best_estimator_


def train_model_on_frame(df, tune: bool = False) -> GradientBoostingRegressor:
    """
    Train a GBM directly on an already-built feature frame, predicting
    next-day return (df["Target"]).

    Used by the walk-forward backtest, which trains a fresh model per
    historical cutoff by design and must never be cached — each call
    trains on a different slice of history, not the current live data.
    """
    trainable = df.dropna(subset=["Target"])
    X = trainable[FEATURE_COLUMNS].values
    y = trainable["Target"].values
    return _fit(X, y, tune)


@ttl_cache(maxsize=128, ttl_seconds=MODEL_TTL_SECONDS)
def get_price_model(ticker: str, period: str, tune: bool = False):
    """
    Return the current live model for a ticker/period, training it once
    and reusing it for MODEL_TTL_SECONDS instead of retraining on every
    Streamlit rerun or API request.
    """
    data = get_stock_data(ticker, period)
    if data.empty:
        return None
    df = build_feature_frame(data)
    if df.empty:
        return None
    return train_model_on_frame(df, tune=tune)
