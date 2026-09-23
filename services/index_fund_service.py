from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from services.cache_utils import ttl_cache
from services.yfinance_cache import get_cached_history, get_cached_info

# Was 10 — even with fetch_with_backoff's per-call pacing, 10 concurrent
# workers each making 3 calls was still a real, observed trigger for
# sustained Yahoo rate limiting.
MAX_PARALLEL_FETCHES = 4


@dataclass(frozen=True)
class IndexFundCandidate:
    ticker: str
    name: str
    benchmark: str
    category: str
    # Only populated for the handful of broad equity benchmarks with an
    # unambiguous, free Yahoo index ticker -- Dow-Jones-branded indices,
    # MSCI/FTSE international indices, and every bond index have no clean
    # free Yahoo ticker, so Tracking Difference is "N/A" for those funds
    # rather than a guessed/wrong number. See index_fund_service's own
    # module docstring-equivalent comment above TRACKING_DIFFERENCE below.
    benchmark_index_ticker: Optional[str] = None


INDEX_FUND_UNIVERSE: List[IndexFundCandidate] = [
    # US Large Blend
    IndexFundCandidate("VOO", "Vanguard S&P 500 ETF", "S&P 500", "US Large Blend", "^GSPC"),
    IndexFundCandidate("IVV", "iShares Core S&P 500 ETF", "S&P 500", "US Large Blend", "^GSPC"),
    IndexFundCandidate("SPLG", "SPDR Portfolio S&P 500 ETF", "S&P 500", "US Large Blend", "^GSPC"),
    IndexFundCandidate("SPY", "SPDR S&P 500 ETF Trust", "S&P 500", "US Large Blend", "^GSPC"),
    # US Total Market
    IndexFundCandidate("VTI", "Vanguard Total Stock Market ETF", "CRSP US Total Market", "US Total Market"),
    IndexFundCandidate("ITOT", "iShares Core S&P Total US Stock Market ETF", "S&P Total US Stock Market", "US Total Market"),
    IndexFundCandidate("SCHB", "Schwab US Broad Market ETF", "Dow Jones US Broad Stock Market", "US Total Market"),
    # US Large Growth
    IndexFundCandidate("QQQ", "Invesco QQQ Trust", "Nasdaq-100", "US Large Growth", "^NDX"),
    IndexFundCandidate("VUG", "Vanguard Growth ETF", "CRSP US Large Cap Growth", "US Large Growth"),
    IndexFundCandidate("IWF", "iShares Russell 1000 Growth ETF", "Russell 1000 Growth", "US Large Growth"),
    IndexFundCandidate("SCHG", "Schwab US Large-Cap Growth ETF", "Dow Jones US Large-Cap Growth", "US Large Growth"),
    # US Large Value
    IndexFundCandidate("VTV", "Vanguard Value ETF", "CRSP US Large Cap Value", "US Large Value"),
    IndexFundCandidate("IWD", "iShares Russell 1000 Value ETF", "Russell 1000 Value", "US Large Value"),
    IndexFundCandidate("SCHV", "Schwab US Large-Cap Value ETF", "Dow Jones US Large-Cap Value", "US Large Value"),
    # US Mid Cap
    IndexFundCandidate("VO", "Vanguard Mid-Cap ETF", "CRSP US Mid Cap", "US Mid Cap"),
    IndexFundCandidate("IJH", "iShares Core S&P Mid-Cap ETF", "S&P MidCap 400", "US Mid Cap", "^MID"),
    IndexFundCandidate("SCHM", "Schwab US Mid-Cap ETF", "Dow Jones US Mid-Cap", "US Mid Cap"),
    # US Small Cap
    IndexFundCandidate("IWM", "iShares Russell 2000 ETF", "Russell 2000", "US Small Cap", "^RUT"),
    IndexFundCandidate("VB", "Vanguard Small-Cap ETF", "CRSP US Small Cap", "US Small Cap"),
    IndexFundCandidate("IJR", "iShares Core S&P Small-Cap ETF", "S&P SmallCap 600", "US Small Cap", "^SML"),
    IndexFundCandidate("SCHA", "Schwab US Small-Cap ETF", "Dow Jones US Small-Cap", "US Small Cap"),
    # International Developed
    IndexFundCandidate("VEA", "Vanguard FTSE Developed Markets ETF", "FTSE Developed All Cap ex US", "International Developed"),
    IndexFundCandidate("SCHF", "Schwab International Equity ETF", "FTSE Developed ex US", "International Developed"),
    IndexFundCandidate("IEFA", "iShares Core MSCI EAFE ETF", "MSCI EAFE IMI", "International Developed"),
    # International Total
    IndexFundCandidate("VXUS", "Vanguard Total International Stock ETF", "FTSE Global All Cap ex US", "International Total"),
    IndexFundCandidate("IXUS", "iShares Core MSCI Total International Stock ETF", "MSCI ACWI ex USA IMI", "International Total"),
    # Emerging Markets
    IndexFundCandidate("VWO", "Vanguard FTSE Emerging Markets ETF", "FTSE Emerging Markets All Cap China A Inclusion", "Emerging Markets"),
    IndexFundCandidate("IEMG", "iShares Core MSCI Emerging Markets ETF", "MSCI Emerging Markets IMI", "Emerging Markets"),
    IndexFundCandidate("SCHE", "Schwab Emerging Markets Equity ETF", "FTSE Emerging", "Emerging Markets"),
    # Bond — Total Market
    IndexFundCandidate("BND", "Vanguard Total Bond Market ETF", "Bloomberg US Aggregate Float Adjusted", "Bond — Total Market"),
    IndexFundCandidate("AGG", "iShares Core US Aggregate Bond ETF", "Bloomberg US Aggregate Bond", "Bond — Total Market"),
    # Bond — Short-Term
    IndexFundCandidate("BSV", "Vanguard Short-Term Bond ETF", "Bloomberg US 1-5yr Government/Credit Float Adjusted", "Bond — Short-Term"),
    IndexFundCandidate("SCHO", "Schwab Short-Term US Treasury ETF", "Bloomberg US Treasury 1-3 Year", "Bond — Short-Term"),
    IndexFundCandidate("VGSH", "Vanguard Short-Term Treasury ETF", "Bloomberg US Treasury 1-3 Year", "Bond — Short-Term"),
    # Bond — Long-Term/Treasury
    IndexFundCandidate("TLT", "iShares 20+ Year Treasury Bond ETF", "ICE US Treasury 20+ Year", "Bond — Long-Term/Treasury"),
    IndexFundCandidate("VGLT", "Vanguard Long-Term Treasury ETF", "Bloomberg US Long Treasury", "Bond — Long-Term/Treasury"),
    IndexFundCandidate("SPTL", "SPDR Portfolio Long Term Treasury ETF", "Bloomberg US Long Treasury", "Bond — Long-Term/Treasury"),
    # Bond — Corporate
    IndexFundCandidate("LQD", "iShares iBoxx Investment Grade Corporate Bond ETF", "Markit iBoxx USD Liquid Investment Grade", "Bond — Corporate"),
    IndexFundCandidate("VCIT", "Vanguard Intermediate-Term Corporate Bond ETF", "Bloomberg US 5-10yr Corporate", "Bond — Corporate"),
    # Bond — High Yield
    IndexFundCandidate("HYG", "iShares iBoxx High Yield Corporate Bond ETF", "Markit iBoxx USD Liquid High Yield", "Bond — High Yield"),
    IndexFundCandidate("JNK", "SPDR Bloomberg High Yield Bond ETF", "Bloomberg Very Liquid High Yield", "Bond — High Yield"),
    # Bond — TIPS
    IndexFundCandidate("TIP", "iShares TIPS Bond ETF", "Bloomberg US TIPS", "Bond — TIPS"),
    IndexFundCandidate("SCHP", "Schwab US TIPS ETF", "Bloomberg US Treasury Inflation Protected Securities", "Bond — TIPS"),
    # Dividend/Income
    IndexFundCandidate("VYM", "Vanguard High Dividend Yield ETF", "FTSE High Dividend Yield", "Dividend/Income"),
    IndexFundCandidate("SCHD", "Schwab US Dividend Equity ETF", "Dow Jones US Dividend 100", "Dividend/Income"),
    IndexFundCandidate("DVY", "iShares Select Dividend ETF", "Dow Jones US Select Dividend", "Dividend/Income"),
    IndexFundCandidate("VIG", "Vanguard Dividend Appreciation ETF", "S&P US Dividend Growers", "Dividend/Income"),
    # Real Estate
    IndexFundCandidate("VNQ", "Vanguard Real Estate ETF", "MSCI US Investable Market Real Estate 25/50", "Real Estate"),
    IndexFundCandidate("SCHH", "Schwab US REIT ETF", "Dow Jones US Select REIT", "Real Estate"),
    # Sector
    IndexFundCandidate("XLK", "Technology Select Sector SPDR Fund", "Technology Select Sector", "Sector"),
    IndexFundCandidate("VGT", "Vanguard Information Technology ETF", "MSCI US Investable Market Information Technology 25/50", "Sector"),
    IndexFundCandidate("XLF", "Financial Select Sector SPDR Fund", "Financial Select Sector", "Sector"),
    IndexFundCandidate("VFH", "Vanguard Financials ETF", "MSCI US Investable Market Financials 25/50", "Sector"),
    IndexFundCandidate("XLV", "Health Care Select Sector SPDR Fund", "Health Care Select Sector", "Sector"),
    IndexFundCandidate("VHT", "Vanguard Health Care ETF", "MSCI US Investable Market Health Care 25/50", "Sector"),
    IndexFundCandidate("XLE", "Energy Select Sector SPDR Fund", "Energy Select Sector", "Sector"),
    IndexFundCandidate("VDE", "Vanguard Energy ETF", "MSCI US Investable Market Energy 25/50", "Sector"),
    IndexFundCandidate("XLY", "Consumer Discretionary Select Sector SPDR Fund", "Consumer Discretionary Select Sector", "Sector"),
    IndexFundCandidate("XLP", "Consumer Staples Select Sector SPDR Fund", "Consumer Staples Select Sector", "Sector"),
]


# The four preset goals keep their exact original metric/weight definitions
# -- not redefined to match any external example, since that would silently
# change what an existing goal means to someone who already picked it.
GOAL_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Balanced Core": {
        "return_1y": 0.35,
        "return_3y_annualized": 0.25,
        "expense_ratio": 0.20,
        "volatility_1y": 0.10,
        "max_drawdown_3y": 0.10,
    },
    "Lowest Cost": {
        "expense_ratio": 0.65,
        "return_3y_annualized": 0.20,
        "volatility_1y": 0.10,
        "assets_billions": 0.05,
    },
    # Deliberately NOT just "Balanced Core with return weighted higher" —
    # that made the two goals pick nearly identical top funds in practice
    # (both were return-dominated, and the same few funds led on every
    # return window). This instead measures recent momentum (30/60/90d)
    # rather than the 1Y/3Y windows Balanced Core already covers, so the
    # two goals actually answer different questions.
    "Best Growth": {
        "return_30d": 0.30,
        "return_60d": 0.25,
        "return_90d": 0.20,
        "return_1y": 0.15,
        "volatility_1y": 0.10,
    },
    "Most Stable": {
        "volatility_1y": 0.45,
        "max_drawdown_3y": 0.30,
        "expense_ratio": 0.15,
        "return_3y_annualized": 0.10,
    },
}

# Metrics a "Custom" goal's sliders may weight -- broader than the four
# presets above (adds the new window-based/liquidity metrics), but every
# preset above only ever uses a subset of this same set, so scoring logic
# never has to special-case "preset vs custom."
CUSTOM_WEIGHTABLE_METRICS = [
    "return_1y", "return_3y_annualized", "return_30d", "return_60d", "return_90d", "cagr_window",
    "expense_ratio", "volatility_1y", "max_drawdown_3y", "std_dev_window", "sharpe_window", "sortino_window",
    "assets_billions", "avg_daily_volume", "bid_ask_spread_pct",
]

LOWER_IS_BETTER = {
    "expense_ratio", "volatility_1y", "max_drawdown_3y", "std_dev_window", "bid_ask_spread_pct",
}

METRIC_LABELS: Dict[str, str] = {
    "return_1y": "1-Year Return",
    "return_3y_annualized": "3-Year Annualized Return",
    "return_30d": "30-Day Return",
    "return_60d": "60-Day Return",
    "return_90d": "90-Day Return",
    "cagr_window": "CAGR (selected window)",
    "expense_ratio": "Expense Ratio",
    "volatility_1y": "1-Year Volatility",
    "max_drawdown_3y": "3-Year Max Drawdown",
    "std_dev_window": "Std. Dev. (selected window)",
    "sharpe_window": "Sharpe Ratio (selected window)",
    "sortino_window": "Sortino Ratio (selected window)",
    "assets_billions": "Fund Assets (AUM)",
    "avg_daily_volume": "Avg. Daily Volume",
    "bid_ask_spread_pct": "Bid/Ask Spread (live)",
}

METRIC_UNITS: Dict[str, str] = {
    "return_1y": "%", "return_3y_annualized": "%", "return_30d": "%", "return_60d": "%", "return_90d": "%",
    "cagr_window": "%", "expense_ratio": "%", "volatility_1y": "%", "max_drawdown_3y": "%", "std_dev_window": "%",
    "sharpe_window": "", "sortino_window": "", "assets_billions": "$B", "avg_daily_volume": "sh", "bid_ask_spread_pct": "%",
}

# Which of the four FS-5 display buckets each metric's sub-score rolls up
# into. A bucket is only shown for a given goal if that goal actually
# weights at least one metric in it -- "Most Stable" never weights
# assets_billions/avg_daily_volume/bid_ask_spread_pct, so it never shows a
# fake empty Liquidity bucket.
METRIC_BUCKET: Dict[str, str] = {
    "return_1y": "Return", "return_3y_annualized": "Return", "return_30d": "Return",
    "return_60d": "Return", "return_90d": "Return", "cagr_window": "Return",
    "volatility_1y": "Risk", "max_drawdown_3y": "Risk", "std_dev_window": "Risk",
    "sharpe_window": "Risk", "sortino_window": "Risk",
    "expense_ratio": "Cost",
    "assets_billions": "Liquidity", "avg_daily_volume": "Liquidity", "bid_ask_spread_pct": "Liquidity",
}

WINDOW_DAYS: Dict[str, int] = {"1y": 365, "3y": 3 * 365, "5y": 5 * 365, "10y": 10 * 365}
VALID_WINDOWS = set(WINDOW_DAYS) | {"max_common"}


class InvalidCustomWeights(ValueError):
    """Raised by normalize_custom_weights on any validation failure. Kept as
    a plain ValueError (not an HTTPException) so this module has no FastAPI
    dependency -- callers like web/backend/routers/index_fund.py catch this
    and translate it into a 422."""


def normalize_custom_weights(raw: Dict[str, object]) -> Dict[str, float]:
    """
    Validates a raw metric->weight mapping (as parsed from the API's
    `weights` JSON query param) and normalizes it to sum to 1.0, so the
    Custom goal's sliders can send any positive relative values without
    keeping their own normalization in sync with the server's.
    """
    if not raw:
        raise InvalidCustomWeights("weights must be a non-empty object of metric -> weight.")

    unknown = set(raw) - set(CUSTOM_WEIGHTABLE_METRICS)
    if unknown:
        raise InvalidCustomWeights(f"Unknown weight metric(s): {sorted(unknown)}. Allowed: {CUSTOM_WEIGHTABLE_METRICS}")

    total = 0.0
    cleaned: Dict[str, float] = {}
    for metric, value in raw.items():
        try:
            v = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise InvalidCustomWeights(f"weights[{metric}] must be a number.")
        if v < 0:
            raise InvalidCustomWeights(f"weights[{metric}] must be non-negative.")
        cleaned[metric] = v
        total += v

    if total <= 0:
        raise InvalidCustomWeights("At least one weight must be greater than 0.")
    return {metric: value / total for metric, value in cleaned.items()}


def _coerce_percent(value: Optional[float]) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value) * 100 if abs(value) <= 1 else float(value)


def _max_drawdown(prices: pd.Series) -> Optional[float]:
    if prices.empty:
        return None
    running_max = prices.cummax()
    drawdown = (prices / running_max) - 1
    return abs(float(drawdown.min())) * 100


def _annualized_return(prices: pd.Series, trading_days: int = 252) -> Optional[float]:
    if prices.empty or len(prices) < 2:
        return None
    total_return = prices.iloc[-1] / prices.iloc[0]
    years = len(prices) / trading_days
    if years <= 0:
        return None
    return (float(total_return) ** (1 / years) - 1) * 100


def _lookback_return(prices: pd.Series, trading_days: int) -> Optional[float]:
    """Trailing % return over the last `trading_days` bars — same semantics
    as stock_finder_service._pct_return, kept local since funds and stocks
    build their rows from separate history fetches."""
    if prices.empty or len(prices) <= trading_days:
        return None
    start = float(prices.iloc[-trading_days - 1])
    end = float(prices.iloc[-1])
    if start == 0:
        return None
    return (end / start - 1.0) * 100


@dataclass(frozen=True)
class FundRawData:
    info: dict
    prices: pd.Series  # full "max"-period close history, auto_adjusted (dividends reinvested)


def _fetch_raw(ticker_symbol: str) -> Optional[FundRawData]:
    try:
        history = get_cached_history(ticker_symbol, "max", auto_adjust=True)
        info = get_cached_info(ticker_symbol)
        if history.empty:
            return None
        return FundRawData(info=info, prices=history["Close"].dropna())
    except Exception:
        return None


@ttl_cache(maxsize=8, ttl_seconds=86400)
def _get_raw_fund_data() -> Dict[str, FundRawData]:
    """
    One yfinance info+max-history fetch per fund in INDEX_FUND_UNIVERSE,
    plus one per unique benchmark index ticker (deduped -- several funds
    share e.g. ^GSPC). Fetched in parallel, same pattern as before. Cached
    24h: this is the only network-bound step: every per-window statistic
    (CAGR/drawdown/Sharpe/etc. for whatever Window the user picks) is
    computed by slicing this already-fetched "max" series, never by a new
    yfinance call -- so changing the Window control re-scores instantly
    (FS-1/FS-2), it never re-fetches.
    """
    tickers = {fund.ticker for fund in INDEX_FUND_UNIVERSE}
    tickers |= {fund.benchmark_index_ticker for fund in INDEX_FUND_UNIVERSE if fund.benchmark_index_ticker}

    raw: Dict[str, FundRawData] = {}
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_FETCHES) as executor:
        futures = {executor.submit(_fetch_raw, t): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            data = future.result()
            if data is not None:
                raw[ticker] = data
    return raw


def _window_bounds(window: str, price_series: Dict[str, pd.Series]) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[str]]:
    """
    Returns (start, end, error). `end` is always the earliest of each
    series' own last date, so every fund in the set is compared over
    literally identical trading days even if one fund's data happens to be
    a day staler than another's.

    Fixed windows (1y/3y/5y/10y): start = end - N calendar years.
    "max_common": start = the LATEST of each series' own first date -- the
    longest range every fund in the set actually has data for (FS-2's own
    wording: "the longest window where every fund in the result set has
    data").
    """
    series = [s for s in price_series.values() if not s.empty]
    if not series:
        return None, None, "No price history available for this selection."

    end = min(s.index[-1] for s in series)

    if window == "max_common":
        start = max(s.index[0] for s in series)
        if start >= end:
            return None, None, "These funds have no overlapping history."
        return start, end, None

    if window not in WINDOW_DAYS:
        return None, None, f"window must be one of {sorted(VALID_WINDOWS)}"

    start = end - pd.Timedelta(days=WINDOW_DAYS[window])
    return start, end, None


def _slice(prices: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return prices[(prices.index >= start) & (prices.index <= end)]


def _stats_for_window(prices: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, Optional[float]]:
    """CAGR/max drawdown/std dev/Sharpe/Sortino over one fund's price slice
    for the resolved window. Risk-free rate is treated as 0% -- the same
    simplifying convention services/momentum_backtest_service.py already
    uses for its own risk_free_rate_annual, not a new assumption. Returns
    every field as None (not a misleading number) when the slice has fewer
    than ~20 trading days — too little data for any of these to mean
    anything."""
    window_prices = _slice(prices, start, end)
    if len(window_prices) < 20:
        return {"cagr_window": None, "max_drawdown_window": None, "std_dev_window": None, "sharpe_window": None, "sortino_window": None}

    daily_returns = window_prices.pct_change().dropna()
    cagr = _annualized_return(window_prices)
    max_dd = _max_drawdown(window_prices)

    if daily_returns.empty or daily_returns.std() == 0 or pd.isna(daily_returns.std()):
        std_dev = float(daily_returns.std() * np.sqrt(252) * 100) if not daily_returns.empty else None
        sharpe = None
    else:
        std_dev = float(daily_returns.std() * np.sqrt(252) * 100)
        mean_annual = float(daily_returns.mean() * 252)
        std_annual = float(daily_returns.std() * np.sqrt(252))
        sharpe = mean_annual / std_annual if std_annual else None

    downside = daily_returns[daily_returns < 0]
    if not downside.empty and downside.std() and not pd.isna(downside.std()):
        downside_annual = float(downside.std() * np.sqrt(252))
        mean_annual = float(daily_returns.mean() * 252)
        sortino = mean_annual / downside_annual if downside_annual else None
    else:
        sortino = None

    return {
        "cagr_window": cagr,
        "max_drawdown_window": max_dd,
        "std_dev_window": std_dev,
        "sharpe_window": sharpe,
        "sortino_window": sortino,
    }


def _tracking_difference(fund_prices: pd.Series, benchmark_prices: Optional[pd.Series], start: pd.Timestamp, end: pd.Timestamp) -> Optional[float]:
    """Fund CAGR minus benchmark index CAGR over the same window -- only
    computed when this fund has a confidently-mapped benchmark_index_ticker
    (see IndexFundCandidate.benchmark_index_ticker's own comment); None
    otherwise, shown as "N/A" rather than guessed."""
    if benchmark_prices is None:
        return None
    fund_cagr = _annualized_return(_slice(fund_prices, start, end))
    bench_cagr = _annualized_return(_slice(benchmark_prices, start, end))
    if fund_cagr is None or bench_cagr is None:
        return None
    return fund_cagr - bench_cagr


def _build_fund_row(fund: IndexFundCandidate, raw: FundRawData, start: pd.Timestamp, end: pd.Timestamp, benchmark_raw: Optional[FundRawData]) -> Dict[str, object]:
    info = raw.info
    prices = raw.prices

    close_1y = _slice(prices, end - pd.Timedelta(days=365), end)
    daily_returns_1y = close_1y.pct_change().dropna()
    latest_price = float(prices.iloc[-1])
    return_1y = _lookback_return(prices, min(len(close_1y) - 1, 252)) if len(close_1y) > 1 else None
    volatility_1y = float(daily_returns_1y.std() * np.sqrt(252) * 100) if not daily_returns_1y.empty else None
    return_3y_annualized = _annualized_return(_slice(prices, end - pd.Timedelta(days=3 * 365), end))
    max_drawdown_3y = _max_drawdown(_slice(prices, end - pd.Timedelta(days=3 * 365), end))

    return_10d = _lookback_return(prices, 10)
    return_30d = _lookback_return(prices, 30)
    return_60d = _lookback_return(prices, 60)
    return_90d = _lookback_return(prices, 90)

    window_stats = _stats_for_window(prices, start, end)

    # netExpenseRatio is the only one of these four keys this yfinance
    # version actually populates (confirmed live: annualReportExpenseRatio/
    # expenseRatio/totalExpenseRatio are None for every fund in the
    # universe) -- and unlike the other three, it already arrives as a
    # percentage-point value (0.03 means 0.03%, matching the fund's real
    # prospectus rate), not a fraction of 1. Passing it through
    # _coerce_percent's fraction heuristic silently inflated every fund's
    # expense ratio 100x (0.03% shown as 3.0%). The fallback keys are kept
    # in case a future yfinance version populates them in the older,
    # fraction-of-1 convention -- only they go through _coerce_percent.
    net_expense_ratio = info.get("netExpenseRatio")
    if net_expense_ratio is not None:
        expense_ratio = float(net_expense_ratio)
    else:
        expense_ratio = _coerce_percent(
            info.get("annualReportExpenseRatio") or info.get("expenseRatio") or info.get("totalExpenseRatio")
        )
    assets = info.get("totalAssets")
    avg_daily_volume = info.get("averageDailyVolume3Month") or info.get("averageVolume")
    bid = info.get("bid")
    ask = info.get("ask")
    bid_ask_spread_pct = None
    if bid and ask and bid > 0 and ask > 0 and ask >= bid:
        mid = (bid + ask) / 2
        bid_ask_spread_pct = ((ask - bid) / mid) * 100 if mid else None
    distribution_yield = _coerce_percent(info.get("yield") or info.get("trailingAnnualDividendYield"))
    # annualHoldingsTurnover is confirmed missing even for SPY/BND (the two
    # largest, most-disclosed funds in this universe) -- shipped as N/A
    # rather than silently omitted, so the gap is visible, not hidden.
    turnover_pct = _coerce_percent(info.get("annualHoldingsTurnover"))

    # longName first, not shortName: Yahoo's own shortName field is
    # genuinely truncated at the source for several sector SPDRs (e.g. XLY
    # comes back as "Consumer Discretio") -- longName is the real full
    # name, and our own curated INDEX_FUND_UNIVERSE name is a reliable
    # second fallback; shortName (when neither of those exist) is last.
    name = info.get("longName") or fund.name or info.get("shortName")
    category = info.get("category") or fund.category

    inception_ts = info.get("fundInceptionDate")
    inception_date = (
        datetime.fromtimestamp(inception_ts, tz=timezone.utc).strftime("%Y-%m-%d") if inception_ts else None
    )

    tracking_difference = _tracking_difference(prices, benchmark_raw.prices if benchmark_raw else None, start, end)

    return {
        "Ticker": fund.ticker,
        "Fund": name,
        "Benchmark": fund.benchmark,
        "Category": category,
        "Price": latest_price,
        "Expense Ratio %": expense_ratio,
        "Tracking Difference %": tracking_difference,
        "Assets ($B)": (float(assets) / 1_000_000_000) if assets else None,
        "Avg Daily Volume": float(avg_daily_volume) if avg_daily_volume else None,
        "Bid/Ask Spread %": bid_ask_spread_pct,
        "1Y Return %": return_1y,
        "3Y Annualized %": return_3y_annualized,
        "1Y Volatility %": volatility_1y,
        "3Y Max Drawdown %": max_drawdown_3y,
        "Return 10D %": return_10d,
        "Return 30D %": return_30d,
        "Return 60D %": return_60d,
        "Return 90D %": return_90d,
        "CAGR (Window) %": window_stats["cagr_window"],
        "Max Drawdown (Window) %": window_stats["max_drawdown_window"],
        "Std Dev (Window) %": window_stats["std_dev_window"],
        "Sharpe (Window)": window_stats["sharpe_window"],
        "Sortino (Window)": window_stats["sortino_window"],
        "Distribution Yield %": distribution_yield,
        "Turnover %": turnover_pct,
        "Inception Date": inception_date,
        # Internal columns feeding scoring -- stripped or kept depending on
        # caller; mirrored 1:1 onto the display columns above so scoring
        # and display never drift out of sync.
        "expense_ratio": expense_ratio,
        "return_1y": return_1y,
        "return_3y_annualized": return_3y_annualized,
        "volatility_1y": volatility_1y,
        "max_drawdown_3y": max_drawdown_3y,
        "assets_billions": (float(assets) / 1_000_000_000) if assets else None,
        "return_30d": return_30d,
        "return_60d": return_60d,
        "return_90d": return_90d,
        "cagr_window": window_stats["cagr_window"],
        "std_dev_window": window_stats["std_dev_window"],
        "sharpe_window": window_stats["sharpe_window"],
        "sortino_window": window_stats["sortino_window"],
        "avg_daily_volume": float(avg_daily_volume) if avg_daily_volume else None,
        "bid_ask_spread_pct": bid_ask_spread_pct,
    }


def _zscore_series(series: pd.Series, lower_is_better: bool) -> pd.Series:
    """Cross-sectional z-score within whatever group `series` already is
    (the caller is responsible for having grouped by Category first) --
    (x - group_mean) / group_std, sign-flipped when lower is better. 0.0
    when the group's std is 0/NaN (every fund tied, or all missing) rather
    than a divide-by-zero or a misleadingly large score."""
    numeric = pd.to_numeric(series, errors="coerce")
    mean = numeric.mean()
    std = numeric.std()
    if pd.isna(std) or std == 0:
        z = pd.Series([0.0] * len(series), index=series.index)
    else:
        z = (numeric - mean) / std
    if lower_is_better:
        z = -z
    return z.fillna(0.0)


def _score_group(group: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    group = group.copy()
    score = pd.Series([0.0] * len(group), index=group.index)
    breakdown_by_row: Dict[object, Dict[str, dict]] = {idx: {} for idx in group.index}

    for metric, weight in weights.items():
        if metric not in group.columns:
            continue
        z = _zscore_series(group[metric], metric in LOWER_IS_BETTER)
        contribution = z * weight
        score += contribution
        bucket = METRIC_BUCKET.get(metric, "Other")
        for idx in group.index:
            bucket_entry = breakdown_by_row[idx].setdefault(bucket, {"sub_score": 0.0, "metrics": []})
            bucket_entry["sub_score"] += float(contribution.loc[idx])
            bucket_entry["metrics"].append(
                {
                    "key": metric,
                    "label": METRIC_LABELS.get(metric, metric),
                    "unit": METRIC_UNITS.get(metric, ""),
                    "raw_value": None if pd.isna(group.loc[idx, metric]) else float(group.loc[idx, metric]),
                    "weight": weight,
                }
            )

    group["Score"] = (score * 100).round(1)
    group["_breakdown"] = [breakdown_by_row[idx] for idx in group.index]
    return group


def _apply_peer_group_scores(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """Peer-group (FS-3) scoring: every metric is z-scored against its own
    Category group's mean/std before weighting, never against the whole
    result set. When `category != "All"` upstream already filtered to one
    category, so this groupby naturally has exactly one group -- there is
    no separate "single category" code path.

    Deliberately iterates the groupby and concatenates rather than using
    groupby(...).apply(...): pandas changed .apply()'s default behavior
    across the 2.x -> 3.x line to silently exclude the grouping column
    (Category) from what's passed to the function and from the
    reconstructed result -- caught in production (pandas 3.0.5) via a
    KeyError on "Category" downstream, while the local/CI pandas (2.3.3)
    only warned. Manual iteration's `group` is always the real DataFrame
    slice, Category column included, on every pandas version."""
    if df.empty:
        return df
    scored_groups = [_score_group(group, weights) for _, group in df.groupby("Category", sort=False)]
    return pd.concat(scored_groups, ignore_index=False)


def rank_index_funds(goal: str, category: str, window: str = "5y", custom_weights: Optional[Dict[str, float]] = None) -> tuple[pd.DataFrame, dict]:
    """
    Returns (ranked_df, window_meta). window_meta always carries
    {"window": ..., "start": "YYYY-MM-DD"|None, "end": "YYYY-MM-DD"|None,
    "error": str|None} so the frontend can show the resolved date range
    (FS-2) even when nothing failed.

    Deliberately NOT cached (unlike _get_raw_fund_data, which is the one
    real network-bound step and stays cached 24h): this only slices/scores
    already-cached price series -- cheap pandas arithmetic over at most a
    few dozen rows -- and custom_weights is a plain dict, which cachetools'
    default key function can't hash anyway. Recomputing on every call is
    both correct and fast enough that a cache would only add complexity.
    """
    raw = _get_raw_fund_data()
    candidates = [f for f in INDEX_FUND_UNIVERSE if (category == "All" or f.category == category) and f.ticker in raw]
    if not candidates:
        return pd.DataFrame(), {"window": window, "start": None, "end": None, "error": "No funds matched this selection."}

    price_series = {f.ticker: raw[f.ticker].prices for f in candidates}
    start, end, error = _window_bounds(window, price_series)
    window_meta = {
        "window": window,
        "start": str(start.date()) if start is not None else None,
        "end": str(end.date()) if end is not None else None,
        "error": error,
    }
    if start is None or end is None:
        return pd.DataFrame(), window_meta

    rows = [
        _build_fund_row(fund, raw[fund.ticker], start, end, raw.get(fund.benchmark_index_ticker) if fund.benchmark_index_ticker else None)
        for fund in candidates
    ]
    df = pd.DataFrame(rows)

    weights = custom_weights if goal == "Custom" else GOAL_WEIGHTS[goal]
    df = _apply_peer_group_scores(df, weights)
    df = df.sort_values(["Category", "Score", "1Y Return %", "Assets ($B)"], ascending=[True, False, False, False]).reset_index(drop=True)
    return df, window_meta


def rank_funds_overall(df: pd.DataFrame) -> pd.DataFrame:
    """
    rank_index_funds's own return is sorted Category-first (so a category=
    "All" caller can group its display by category, per FS-3) -- Score is
    only a secondary tiebreaker *within* whichever category happens to sort
    alphabetically first. A caller that wants "the single best fund" or
    "the top N funds overall" (get_top_fund, get_diverse_strategy_picks, the
    legacy Streamlit page, etc.) must re-sort by Score first, or it silently
    returns whichever category's funds come first alphabetically -- not the
    best-scoring ones. This was a real, live bug: two funds with a
    peer-group Score of 0.0 (the correct z-score result for a single-member
    category, which has no peers to compare against) were still being
    returned as the "top picks" for every goal, because their category name
    sorted before every other category's.
    """
    if df.empty:
        return df
    return df.sort_values(["Score", "1Y Return %", "Assets ($B)"], ascending=[False, False, False]).reset_index(drop=True)


def score_fund_ticker(goal: str, ticker_symbol: str, window: str = "5y", custom_weights: Optional[Dict[str, float]] = None) -> tuple[pd.DataFrame, dict]:
    """
    Scores one arbitrary ticker (FS-2's "Score one fund" mode) against the
    peer-group statistics of the matching category in the main universe --
    same z-score mechanics as rank_index_funds, just applied to a group of
    one candidate row plus the reference universe's own rows for the same
    category, so the standalone fund is judged against real peers rather
    than a baseline of itself (z of a lone row is always 0/meaningless).
    """
    cleaned = ticker_symbol.strip().upper()
    if not cleaned:
        return pd.DataFrame(), {"window": window, "start": None, "end": None, "error": "No ticker given."}

    raw_data = _fetch_raw(cleaned)
    if raw_data is None:
        return pd.DataFrame(), {"window": window, "start": None, "end": None, "error": f"No price history found for {cleaned}."}

    detected_category = raw_data.info.get("category") or "Custom"
    universe_raw = _get_raw_fund_data()
    peers = [f for f in INDEX_FUND_UNIVERSE if f.category == detected_category and f.ticker in universe_raw]

    price_series = {cleaned: raw_data.prices, **{f.ticker: universe_raw[f.ticker].prices for f in peers}}
    start, end, error = _window_bounds(window, price_series)
    window_meta = {
        "window": window,
        "start": str(start.date()) if start is not None else None,
        "end": str(end.date()) if end is not None else None,
        "error": error,
    }
    if start is None or end is None:
        return pd.DataFrame(), window_meta

    candidate = IndexFundCandidate(cleaned, cleaned, "Custom", detected_category)
    rows = [_build_fund_row(candidate, raw_data, start, end, None)]
    rows += [_build_fund_row(f, universe_raw[f.ticker], start, end, universe_raw.get(f.benchmark_index_ticker) if f.benchmark_index_ticker else None) for f in peers]
    df = pd.DataFrame(rows)

    weights = custom_weights if goal == "Custom" else GOAL_WEIGHTS[goal]
    df = _apply_peer_group_scores(df, weights)
    # Only the requested ticker's own row is returned -- the peers were
    # only fetched to give it something real to be scored against.
    return df[df["Ticker"] == cleaned].reset_index(drop=True), window_meta


@ttl_cache(maxsize=8, ttl_seconds=86400)
def get_index_fund_table() -> pd.DataFrame:
    """Kept for any other caller expecting the old flat, unscored table
    shape (e.g. momentum.py's /top-performers, which only reads Ticker/
    Name/Price/return columns, never Score) -- rebuilt from the same
    5Y-windowed row-builder as everything else in this module, so it's one
    consistent source of truth rather than a second, divergent fetch path."""
    df, _ = rank_index_funds("Balanced Core", "All", "5y")
    return df
