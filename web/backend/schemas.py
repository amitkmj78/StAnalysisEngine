from typing import List, Optional

from pydantic import BaseModel


class SignalOut(BaseModel):
    signal: str
    expected_return_pct: float
    target_price: float
    # Day-over-day flip stats from the pit_quant_signal PIT history — None
    # when there isn't enough captured history yet for this ticker to say
    # anything (e.g. a ticker outside the default capture universe).
    signal_flip_count: int | None = None
    signal_days_captured: int | None = None
    signal_unstable: bool | None = None


class ForecastOut(BaseModel):
    dates: List[str]
    predicted: List[float]
    lower_ci: List[float]
    upper_ci: List[float]


class BacktestOut(BaseModel):
    dates: List[str]
    actual: List[float]
    predicted: List[float]
    naive: List[float]


class MetricsOut(BaseModel):
    rmse: float
    mae: float
    mape: float
    naive_rmse: Optional[float] = None
    naive_mae: Optional[float] = None
    naive_mape: Optional[float] = None
    beats_naive: Optional[bool] = None


class PredictionSummary(BaseModel):
    ticker: str
    period: str
    last_close: Optional[float] = None
    next_price: Optional[float] = None
    signal: Optional[SignalOut] = None
    forecast: Optional[ForecastOut] = None
    backtest: Optional[BacktestOut] = None
    metrics: Optional[MetricsOut] = None
    warnings: List[str] = []


class PredictionNarrativeOut(BaseModel):
    ticker: str
    provider: str
    narrative: str
    sentiment_context: str
