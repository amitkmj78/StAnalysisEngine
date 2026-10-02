import type { ColumnInfo } from "@/components/InfoModal";

/**
 * LRN-1: the single shared home for every "what is X?" metric/factor
 * definition in the app, consolidated from what used to be 9 separate
 * per-page `COLUMN_INFO` objects (~80 definitions, copy-pasted rather than
 * shared). Looked up by components/MetricLabel.tsx, keyed by the label
 * text shown on screen -- the same convention those per-page objects
 * already used, so most entries below are a direct migration of existing,
 * already-reviewed copy, not a rewrite.
 *
 * Two labels meant genuinely different things on different pages (e.g.
 * "Signal" was both an entry-timing label on /entry and a BUY/SELL/HOLD
 * rank on /portfolio) -- those get distinct, explicitly-scoped keys below
 * rather than one silently overwriting the other.
 */
export const GLOSSARY: Record<string, ColumnInfo> = {
  // --- Stock Finder (app/stock-finder/page.tsx) ---
  Ticker: {
    title: "Ticker",
    body: ["The stock's exchange symbol — click Forecast or Watchlist on any row to act on it without retyping."],
  },
  Name: {
    title: "Name",
    body: ["The company or fund's short name, as reported by the data provider."],
  },
  Sector: {
    title: "Sector",
    body: ["The GICS sector this ticker is classified under. Use the Sector filter to narrow results to one or more sectors."],
  },
  Price: {
    title: "Price",
    body: ["The latest available trade price at the time this scan ran (cached up to an hour — see the note above the table)."],
  },
  Score: {
    title: "Score",
    body: [
      "A 0–100 blend of several metrics, each normalized against the other tickers in this result set (the best value in the current list scores highest on that metric, the worst scores lowest) — it's a relative ranking within this run, not an absolute grade. Re-running with a different universe can change a ticker's score even if nothing about the ticker itself changed.",
      "The metrics and their weights depend on the Goal you picked — see the weights strip above the table for the exact breakdown of whichever Goal is active.",
      "Filters below narrow which rows are shown, but never change how Score is computed — scores stay comparable across different filter selections since they're calculated before filtering.",
    ],
  },
  "Quant Signal": {
    title: "Quant Signal",
    body: [
      "The same BUY/HOLD/SELL signal shown on the Predict page — a gradient-boosted model trained on this ticker's own recent price/technical history, forecasting 10 days ahead. BUY means the forecast implies at least +5% expected return; SELL means -5% or worse; HOLD is in between.",
      "A completely different, independent computation from Score above — Score is a relative rank against this result set's other tickers; Quant Signal is a standalone per-ticker forecast, the same one you'd get analyzing this ticker alone on /predict.",
      "Loaded on demand per row (click \"Load\") rather than for the whole scan, since it means training a fresh model per ticker — too slow to run automatically across a large universe.",
    ],
  },
  "Analyst Rating": {
    title: "Analyst Rating",
    body: [
      "Real, third-party Wall Street analyst consensus and price targets — straight from the data provider, nothing computed or modeled by this app. Shows the consensus rating (e.g. Buy, Hold, Sell), how many analysts contributed, a buy% derived from Yahoo's 1 (Strong Buy) to 5 (Strong Sell) consensus scale, and the mean/high/low 12-month price targets.",
      "Not available for every ticker — small caps, ETFs, and funds often have no analyst coverage at all, which shows as \"No coverage\" rather than a guessed value.",
      "Loaded on demand per row, same as Quant Signal — a separate network call per ticker.",
    ],
  },
  "Market Cap ($B)": {
    title: "Market Cap ($B)",
    body: ["Market capitalization in billions of dollars — share price × shares outstanding, as reported by the data provider."],
  },
  "Forward PE": {
    title: "Forward P/E",
    body: [
      "Price divided by analysts' consensus estimate of next year's earnings per share — a lower number generally means the stock is cheaper relative to its expected earnings.",
      "Only used in the \"Long Term\" Score (8%, lower is better). Not meaningful for companies expected to have negative earnings.",
    ],
  },
  "Dividend Yield %": {
    title: "Dividend Yield %",
    body: ["Trailing 12-month dividend payments as a percent of the current price, as reported by the data provider. Shows \"N/A\" when the provider has no dividend data for this ticker (typically non-dividend-payers). Not used in either Score — display/filter only."],
  },
  "Revenue Growth %": {
    title: "Revenue Growth %",
    body: ["Year-over-year revenue growth, as reported by the data provider. Used in the \"Long Term\" Score (12%)."],
  },
  "Earnings Growth %": {
    title: "Earnings Growth %",
    body: ["Year-over-year earnings growth, as reported by the data provider. Used in the \"Long Term\" Score (10%)."],
  },
  "1M Return %": {
    title: "1-Month Return",
    body: ["Price change over the trailing ~21 trading days. Used in the \"Short Term\" Score (25%)."],
  },
  "3M Return %": {
    title: "3-Month Return",
    body: ["Price change over the trailing ~63 trading days. Used in the \"Short Term\" Score (30%, the single largest weight in that goal)."],
  },
  "6M Return %": {
    title: "6-Month Return",
    body: ["Price change over the trailing ~126 trading days. Used in the \"Long Term\" Score (12%)."],
  },
  "1Y Return %": {
    title: "1-Year Return",
    body: ["Price change over the trailing ~252 trading days. Used in the \"Long Term\" Score (28%, the single largest weight in that goal)."],
  },
  "3Y Annualized %": {
    title: "3-Year Annualized Return",
    body: ["Total 3-year return converted to an annualized (per-year) rate. Used in the \"Long Term\" Score (20%)."],
  },
  "Return 10D %": {
    title: "10-Day Return",
    body: ["Literal trailing 10-trading-day price change. Display-only — not part of either Score, and computed separately from the 1M/3M/6M/1Y columns above to keep this reading distinct from that composite."],
  },
  "Return 30D %": {
    title: "30-Day Return",
    body: ["Literal trailing 30-trading-day price change. Display-only — not part of either Score."],
  },
  "Return 60D %": {
    title: "60-Day Return",
    body: ["Literal trailing 60-trading-day price change. Display-only — not part of either Score."],
  },
  "Return 90D %": {
    title: "90-Day Return",
    body: ["Literal trailing 90-trading-day price change. Display-only — not part of either Score."],
  },
  RSI: {
    title: "RSI — Relative Strength Index",
    body: [
      "Measures how fast and how much a stock's price has moved recently, on a 0–100 scale, based on the ratio of average recent gains to average recent losses.",
      "Above 70 is often considered overbought (may be due for a pullback). Below 30 is often considered oversold (may be due for a bounce). Around 50 is neutral momentum.",
      "This app computes it over a standard 14-day window.",
      "It doesn't just reward high RSI: the ranking score prefers RSI near 55 — strong momentum without being overheated — and penalizes distance from 55 in either direction. A stock at RSI 90 scores worse than one at RSI 55, same as a weak stock sitting at RSI 20.",
      "It's only used for the \"Short Term\" goal (15% of that score). \"Long Term\" ranking doesn't use RSI at all — it weights fundamentals and multi-year returns instead.",
    ],
  },
  "RSI Balance": {
    title: "RSI Balance",
    body: ["The raw RSI value transformed into the 0–100 score actually used in Score: 100 minus 3× the distance from RSI 55, floored at 0 — see the RSI column's own explanation for why 55 (not 100) is the target."],
  },
  "MACD Strength": {
    title: "MACD Strength",
    body: ["The gap between the MACD line and its signal line, scaled — positive means the MACD is above its signal line (often read as bullish momentum). Used in the \"Short Term\" Score (15%)."],
  },
  "Volume Strength %": {
    title: "Volume Strength %",
    body: ["Today's trading volume vs. its own trailing 20-day average, as a percent change — positive means unusually high volume. Used in the \"Short Term\" Score (10%)."],
  },
  "6M Volatility %": {
    title: "6-Month Volatility",
    body: ["Annualized standard deviation of daily returns over the trailing 6 months — higher means choppier price action. Used in the \"Short Term\" Score (5%, lower is better)."],
  },
  "1Y Max Drawdown %": {
    title: "1-Year Max Drawdown",
    body: ["The largest peak-to-trough decline over the trailing year. Used in the \"Long Term\" Score (10%, lower is better)."],
  },
  "Spark 90D": {
    title: "90-Day Sparkline",
    body: [
      "The trailing 90 trading days' closing price, min–max normalized to fit a small inline shape — a quick visual of the recent trend, not a chart with axes or exact values.",
      "Colored by this row's 1-Month Return: green if positive, red if negative. Not sortable and not used in either Score — display only.",
    ],
  },
  "Short-Term Score": {
    title: "Short-Term Score",
    body: [
      "The 0–100 short-term score computed nightly by this app's two-score ranking engine (momentum, short-term reversal, earnings surprise, earnings revisions) — the same number shown on a ticker's own Score page.",
      "A completely different computation from the Score column above: Score is a live, goal-weighted rank against just this result set; Short-Term Score is Stage A's daily composite, comparable across every day and every screen. \"N/A\" means this ticker hasn't been scored yet.",
    ],
  },
  "Short-Term Signal": {
    title: "Short-Term Signal",
    body: ["Buy/Hold/Trim derived from the Short-Term Score's percentile against the rest of the universe — see the Short-Term Score column."],
  },
  "Long-Term Score": {
    title: "Long-Term Score",
    body: [
      "The 0–100 long-term score computed nightly by this app's two-score ranking engine (value, growth, low volatility, quality) — the same number shown on a ticker's own Score page.",
      "Same disambiguation as Short-Term Score: a different, daily-computed number from the goal-weighted Score column above.",
    ],
  },
  "Long-Term Signal": {
    title: "Long-Term Signal",
    body: ["Buy/Hold/Trim derived from the Long-Term Score's percentile against the rest of the universe — see the Long-Term Score column."],
  },
  Owned: {
    title: "Owned",
    body: ["Whether this ticker is a position in any of your portfolios right now."],
  },
  Watchlisted: {
    title: "Watchlisted",
    body: ["Whether you have an active price alert on this ticker that you set up yourself. Excludes alerts this app auto-creates when you add a position to a portfolio, so this stays a distinct signal from Owned."],
  },

  // --- Portfolio (app/portfolio/page.tsx) ---
  // "Position Ticker"/"Portfolio Signal" are deliberately distinct from
  // Stock Finder's generic "Ticker"/entry's "Entry Signal" above -- same
  // label text, different pages, genuinely different explanations (this
  // one talks about the concentration badge / composite BUY-SELL-HOLD
  // read specific to a held position, not a screener row).
  "Position Ticker": {
    title: "Ticker",
    body: [
      "The position's stock/fund symbol. A percentage badge next to it means this position is concentrated — it makes up a large enough share of your portfolio's total value that it's driving most of the swings.",
    ],
  },
  "Portfolio Signal": {
    title: "Signal",
    body: [
      "BUY, SELL, or HOLD for this ticker, from the same composite ranking used on the Stock Screener — a relative read against other tickers in its universe, not a standalone prediction.",
      "Blank means the signal hasn't loaded yet or isn't available for this ticker.",
    ],
  },
  "Momentum Rank": {
    title: "Momentum Rank",
    body: [
      "Where this ticker ranks by trailing return within its universe (e.g. \"#3 of 24\") — lower is stronger recent momentum relative to its peers.",
      "Not shown for tickers outside the app's covered universes.",
    ],
  },
  "1-Day Forecast": {
    title: "1-Day Forecast",
    body: [
      "The Predict-page model's projected price 1 trading day out, and the implied percent change from today's price — the same underlying forecast as Signal, read at its earliest point rather than a second prediction.",
      "A standalone, per-ticker statistical projection — not a guarantee, and not the same thing as Momentum Rank's relative comparison against other tickers.",
    ],
  },
  "5-Day Forecast": {
    title: "5-Day Forecast",
    body: [
      "The Predict-page model's projected price 5 trading days out, and the implied percent change from today's price — the same underlying forecast as Signal, read at an earlier point on its curve rather than a second prediction.",
      "A standalone, per-ticker statistical projection — not a guarantee, and not the same thing as Momentum Rank's relative comparison against other tickers.",
    ],
  },
  "10-Day Forecast": {
    title: "10-Day Forecast",
    body: [
      "The Predict-page model's projected price 10 trading days out, and the implied percent change from today's price. Signal (BUY/SELL/HOLD) is derived from this same 10-day figure.",
    ],
  },
  Shares: {
    title: "Shares",
    body: ["The quantity you hold, as entered manually or imported from your CSV — not adjusted for any splits since import."],
  },
  "Price Now": {
    title: "Price Now",
    body: [
      "The latest trade price used for this row's value/gain figures. When the market is in pre-market or after-hours and a quote is available, this is that session's price, not the regular session's stale close — a badge marks it, and the regular-session price is shown underneath for reference.",
    ],
  },
  "Market Value": {
    title: "Market Value",
    body: [
      "What this position is worth right now: Shares × Price Now. Summed across every holding, this is the same number shown in the Total Value figure above.",
    ],
  },
  Today: {
    title: "Today",
    body: [
      "Today's dollar and percent gain/loss versus yesterday's regular-session close: (Price Now − Previous Close) × Shares — the standard \"day P&L\" figure most brokerages show.",
      "While the market is in pre-market or after-hours, this uses that session's price, so it reflects the after-hours move too, not just the regular session's.",
    ],
  },
  "Price 30D Ago": {
    title: "Price 30D Ago",
    body: ["The closing price approximately 30 calendar days back — the reference point for the 30D Diff column."],
  },
  "30D Diff": {
    title: "30D Diff",
    body: [
      "Dollar and percent change in this position's value over the last 30 days: (Price Now − Price 30D Ago) × Shares.",
      "This is about recent price movement, not your original purchase — see Gain vs. Paid for that.",
    ],
  },
  "Avg Cost Paid": {
    title: "Avg Cost Paid",
    body: ["Your average cost basis per share, as entered manually or computed from your imported CSV activity."],
  },
  "Gain vs. Paid": {
    title: "Gain vs. Paid",
    body: [
      "Dollar and percent gain/loss versus what you actually paid: (Price Now − Avg Cost Paid) × Shares.",
      "Unlike 30D Diff, this reflects your entire holding period, not just the last 30 days.",
    ],
  },

  // --- Index Fund (app/index-fund/page.tsx) ---
  // "Fund Score" is deliberately distinct from Stock Finder's "Score" --
  // same word, different scoring mechanics (category-relative z-score vs.
  // a 0-100 blend), see the inline comment at the call site.
  "Fund Score": {
    title: "Score",
    body: [
      "A relative rank within this fund's own category: every metric is z-scored against the other funds in the same Category before weighting, so a fund is only ever compared to real peers — a bond fund is never scored against an equity fund's volatility.",
      "It isn't a 0–100 grade. 0 means \"about average for its category\" on the metrics that matter to this Goal; positive means better than its peers, negative means worse — and the further from 0, the bigger the gap. A very small peer group (a handful of nearly-identical funds plus one real outlier) can push a Score well beyond ±100.",
      "The metrics and weights depend on the Goal you picked — see the weights strip above the table for the exact breakdown of whichever Goal is active.",
      "Expand a row (the ▸ on the left) to see the Return/Risk/Cost/Liquidity sub-scores and the raw metric behind each.",
    ],
  },
  "Expense Ratio %": {
    title: "Expense Ratio",
    body: [
      "The fund's annual operating fee, as a percentage of your invested assets — pulled live from Yahoo Finance's fund data for each ticker.",
      "It's deducted automatically from the fund's returns over the year, so a higher expense ratio quietly eats into your net return every year you hold it, compounding over time. Lower is better.",
    ],
  },
  "Tracking Difference %": {
    title: "Tracking Difference",
    body: [
      "This fund's CAGR minus its benchmark index's own CAGR over the selected window — how much the fund gave up (or gained) versus the index it tracks, beyond the stated expense ratio.",
      "Only computed for funds mapped to a benchmark with an unambiguous, free index ticker (the major S&P/Nasdaq/Russell indices). Everything else — Dow-Jones-branded, MSCI/FTSE international, and every bond index — shows N/A rather than a guessed number.",
    ],
  },
  "Assets ($B)": {
    title: "Fund Assets (AUM)",
    body: ["Total net assets under management, in billions — a rough proxy for how liquid and established a fund is."],
  },
  "Avg Daily Volume": {
    title: "Average Daily Volume",
    body: ["Shares traded per day on average (3-month average where available) — higher volume generally means tighter spreads and easier entry/exit at the quoted price."],
  },
  "Bid/Ask Spread %": {
    title: "Bid/Ask Spread (live)",
    body: [
      "A live snapshot of (ask − bid) / midpoint, taken at the time the data was last refreshed — not a historical median, since no historical bid/ask series exists via this data source.",
      "Smaller is better: it's roughly what you give up in one round-trip just from the spread, separate from any commission.",
    ],
  },
  "CAGR (Window) %": {
    title: "CAGR (selected window)",
    body: ["Compound annual growth rate over the currently selected Window, using total return with dividends reinvested — not a simple average of yearly returns."],
  },
  "Max Drawdown (Window) %": {
    title: "Max Drawdown (selected window)",
    body: ["The largest peak-to-trough decline within the selected Window — how much this fund lost from its best point before recovering, expressed as a positive percentage."],
  },
  "Std Dev (Window) %": {
    title: "Std. Dev. (selected window)",
    body: ["Annualized standard deviation of daily returns over the selected Window — a measure of how bumpy the ride was, not of long-run direction."],
  },
  "Sharpe (Window)": {
    title: "Sharpe Ratio (selected window)",
    body: ["Annualized return divided by annualized volatility over the selected Window, assuming a 0% risk-free rate. Higher means more return per unit of risk taken."],
  },
  "Sortino (Window)": {
    title: "Sortino Ratio (selected window)",
    body: ["Like Sharpe, but only penalizes downside volatility (losing days), not all volatility — a fund that's volatile only on the way up scores better here than on Sharpe."],
  },
  "Distribution Yield %": {
    title: "Distribution Yield",
    body: ["The fund's trailing distribution yield, pulled live from Yahoo Finance."],
  },
  "Turnover %": {
    title: "Turnover",
    body: [
      "Annual holdings turnover ratio, as disclosed by the fund. This figure is not reliably available via this data source for most funds — including large, well-known funds like SPY and BND — so it shows N/A far more often than not. Shown anyway rather than hidden, so the gap is visible.",
    ],
  },

  // --- Monthly Plan (app/monthly-plan/page.tsx) ---
  // Same ranking engine (and same weights) as /stock-finder and /index-fund
  // -- this page's "Score" is that Score, not a separate computation. Kept
  // as its own entry (not reusing "Score"/"Fund Score" above) because this
  // page has no weights strip to point to, so it spells the weights out
  // directly instead.
  "Monthly Plan: Stock Score": {
    title: "Score",
    body: [
      "A 0–100 blend of several metrics, each normalized against the other tickers in the current universe (the best value in the current list scores highest on that metric, the worst scores lowest) — it's a relative ranking within this run, not an absolute grade. Re-running with a different universe can change a ticker's score even if nothing about the ticker itself changed.",
      "The metrics and their weights depend on the Goal you picked:",
      "\"Short Term\": 3-month return (30%), 1-month return (25%), RSI balance (15%), MACD signal strength (15%), volume strength (10%), 6-month volatility (5%, lower is better).",
      "\"Long Term\": 1-year return (28%), 3-year annualized return (20%), 6-month return (12%), revenue growth (12%), earnings growth (10%), forward P/E (8%, lower is better), 1-year max drawdown (10%, lower is better).",
    ],
  },
  "Monthly Plan: Fund Score": {
    title: "Score",
    body: [
      "A 0–100 blend of several metrics, each normalized against the other funds in the current category (the best value in the current list scores highest on that metric, the worst scores lowest) — it's a relative ranking within this run, not an absolute grade. Re-running with a different category can change a fund's score even if nothing about the fund itself changed.",
      "The metrics and their weights depend on the Goal you picked:",
      "\"Balanced Core\": 1-year return (35%), 3-year annualized return (25%), expense ratio (20%, lower is better), 1-year volatility (10%, lower is better), 3-year max drawdown (10%, lower is better).",
      "\"Lowest Cost\": expense ratio (65%, lower is better), 3-year annualized return (20%), 1-year volatility (10%, lower is better), fund assets (5%).",
      "\"Best Growth\": 1-year return (50%), 3-year annualized return (35%), 1-year volatility (10%, lower is better), expense ratio (5%, lower is better).",
      "\"Most Stable\": 1-year volatility (45%, lower is better), 3-year max drawdown (30%, lower is better), expense ratio (15%, lower is better), 3-year annualized return (10%).",
    ],
  },

  // --- Entry (app/entry/page.tsx) ---
  // "Entry Signal"/"Entry Quant Signal" are deliberately distinct from
  // "Portfolio Signal"/"Quant Signal" above -- confirmed real conflicts,
  // not just duplication: this page's "Signal" is an entry-timing label
  // ("Buy Now"/"Buy on Pullback"/etc.), not a BUY/SELL/HOLD rank, and its
  // "Quant Signal" explanation covers this page's own daily-capture-vs-
  // fresh-lookup loading behavior, which differs from Stock Finder's.
  "Entry Score": {
    title: "Entry Score",
    body: [
      "A score built from this ticker's own technical setup right now — not a percentile rank against other tickers like the Screener's Score, so it can be compared across different scans and doesn't shift just because the universe changed. 100 is a strong, well-rounded setup; a genuinely exceptional one (strong on every factor at once) can score above it — there's no artificial ceiling hiding real differences between setups.",
      "Signal strength — up to 90 points: the Signal label (Wait = 0, Wait for Pullback = 1, Watch for Reversal = 2, Breakout Entry = 3, Buy on Pullback = 4, Buy Now = 5) times 18.",
      "RSI closeness to 52 — up to 20 points: full 20 at RSI exactly 52 (strong momentum without being overheated), losing a point per unit away, reaching 0 once RSI is 20+ points from 52 in either direction.",
      "Bullish short-term momentum — +14 if present, except for \"Buy Now\"/\"Breakout Entry\" where it's already required to earn that label (counted once via signal strength, not twice).",
      "Short-term uptrend — +14 if present, except for \"Buy Now\"/\"Wait for Pullback\" where it's already required to earn that label.",
      "Long-term uptrend — +10 if present, except for \"Buy on Pullback\" where it's already required to earn that label.",
      "Proximity to a level — near 20-day support: +12 (except for \"Buy on Pullback\", already required). Otherwise, near a breakout level: +8 (except for \"Breakout Entry\", already required). Only one of these ever applies.",
      "Above-average volume — up to +12: 0 at today's volume equal to its 20-day average, scaling up to the full 12 points once volume is 60%+ above that average.",
      "The exceptions above matter: a stock's signal label already implies certain conditions (e.g. \"Buy Now\" requires an uptrend with momentum), so re-awarding those same points on top would double-count the same evidence. The remaining points only come from genuine extra strength beyond what the label already guarantees — nothing here is capped, so two stocks with the same signal can still show meaningfully different scores.",
    ],
  },
  "Entry Signal": {
    title: "Signal — what each label means",
    body: [
      "\"Buy Now\" — short-term uptrend with supportive momentum, and price isn't overextended (RSI below 70). The most straightforward setup.",
      "\"Buy on Pullback\" — the longer-term trend is still intact, and price has pulled back near recent support.",
      "\"Breakout Entry\" — price is pressing against recent resistance with supportive momentum, close to breaking out.",
      "\"Watch for Reversal\" — the stock looks oversold (RSI 35 or below). Washed out, but wait for confirmation before entering.",
      "\"Wait for Pullback\" — the trend is healthy, but price looks stretched (RSI 70+). Healthy trend, risky entry point right now.",
      "\"Wait\" — no clear edge either way; the setup is mixed.",
      "Ranked strongest to weakest for scan ordering: Buy Now → Buy on Pullback → Breakout Entry → Watch for Reversal → Wait for Pullback → Wait.",
    ],
  },
  "Entry Quant Signal": {
    title: "Quant Signal — a second, independent opinion",
    body: [
      "BUY / HOLD / SELL from this app's own forecasting model — the same signal shown on the Predict page and the Stock Screener. Completely separate from the \"Signal\" column: that one reads the current technical setup (trend, RSI, support/resistance); this one is a 10-day price forecast.",
      "When they agree, that's two independent methods pointing the same direction. When they disagree, that's worth a closer look, not a reason to distrust one or the other.",
      "In a scan, this comes from the most recent daily capture (fast to look up for many tickers at once) rather than being recomputed live, so it can be a few hours old. Checking a single ticker always computes it fresh.",
      "Shown as \"—\" when no capture exists yet for that ticker (a capture gap) rather than hidden.",
    ],
  },

  // --- Signal Comparison (app/signal-comparison/page.tsx) ---
  // Keys here are the page's own original snake_case field names (not
  // Title Case label text) -- kept as-is during migration since they're
  // already unique (case-sensitive, so "quant_signal" never collides with
  // "Quant Signal" above) and every call site already passes `term`
  // explicitly, so renaming would only add churn with no real benefit.
  // "quant_signal" is yet a fourth distinct Quant Signal explanation
  // (after Stock Finder's, Entry's, and Entry's "Entry Quant Signal") --
  // this one emphasizes the +5%/-5% damped-threshold mechanics, and gets
  // a real-track-record paragraph appended live via MetricLabel's
  // extraBody once outcome data loads (see quantSignalExtraBody).
  quant_signal: {
    title: "Quant Signal",
    body: [
      "The app's own model's BUY/HOLD/SELL call, based on a 10-trading-day price forecast.",
      "BUY requires an expected return of at least +5%, SELL at most -5%, both measured after the raw " +
        "model output is deliberately damped toward zero — the model's raw predictions lost to a simple " +
        "\"no change\" forecast in backtesting, so the damping and wide neutral band are intentional, not a bug. " +
        "Most tickers land in HOLD by design.",
    ],
  },
  quant_expected_return_pct: {
    title: "Quant Expected Return",
    body: ["The model's forecasted return over the next 10 trading days, after damping. This is what the BUY/SELL threshold is measured against."],
  },
  quant_target_price: {
    title: "Quant Target Price",
    body: ["The model's forecasted price 10 trading days out, implied by the expected return above."],
  },
  analyst_consensus: {
    title: "Analyst Consensus",
    body: ["Real, third-party Wall Street analyst consensus rating for this ticker (e.g. Buy, Hold, Sell), sourced from Yahoo Finance — independent of this app's own model."],
  },
  analyst_buy_pct: {
    title: "Analyst Buy %",
    body: ["Share of covering analysts rating this ticker a Buy or Strong Buy."],
  },
  analyst_target_mean: {
    title: "Analyst Target (Mean)",
    body: ["The average of covering analysts' individual price targets."],
  },
  signal_flip_count: {
    title: "Flips",
    body: [
      "How many times this ticker's Quant Signal has changed (BUY/HOLD/SELL) over its trailing captured history (up to 30 days).",
      "\"Unstable\" means it's flipped 3+ times — treat today's signal with less confidence, since it hasn't settled on a view.",
      "Click \"Why?\" next to any flip count to see exactly when the signal last changed and what moved — the model's expected return, target price, and the actual last close around that date. Pulled from the real captured history, not an AI guess.",
    ],
  },
  current_price: {
    title: "Current Price",
    body: [
      "A live quote, fetched on demand — distinct from Last Close, which is the price at the moment this signal was captured (as of the date shown at the top of the page).",
      "The % shown next to it is the real move since the signal was captured, so you can see at a glance whether the price is actually tracking the model's call or has gone the other way.",
      "Click \"Get AI Context\" after loading this to have the explanation address that move directly, alongside the usual technical picture.",
    ],
  },

  // --- Predict (app/predict/page.tsx) ---
  // "Signal" here is genuinely dynamic (its title/body change with the
  // selected forecast horizon) and stays as a function (getSignalInfo),
  // passed via MetricLabel's `info` override rather than living here.
  RMSE: {
    title: "RMSE — Root Mean Squared Error",
    body: [
      "The typical size of the model's miss, in dollars, across the walk-forward backtest window.",
      "Squares each day's error before averaging, then square-roots the result — which weights big misses more heavily than small ones. A model that's usually close but occasionally way off will show a higher RMSE than its MAE.",
      "Shown next to the naive baseline's own RMSE below, so you can see whether the model's typical miss is bigger or smaller than just assuming no price change.",
    ],
  },
  MAE: {
    title: "MAE — Mean Absolute Error",
    body: [
      "The average size of the model's miss, in dollars, treating every day's error equally regardless of whether it was a small miss or a large one (unlike RMSE, which penalizes large misses more).",
      "If RMSE is noticeably higher than MAE, that's a sign the model has a few bad days that are much worse than its typical miss, rather than being uniformly a little off.",
    ],
  },
  MAPE: {
    title: "MAPE — Mean Absolute Percentage Error",
    body: [
      "The average miss expressed as a percentage of the actual price, rather than in dollars — this is what makes it comparable across tickers at very different price levels (a $5 miss means very different things for a $20 stock vs. a $500 stock).",
    ],
  },

  // --- Strategies (app/strategies/page.tsx) ---
  // "Ranking score" there is genuinely per-pick dynamic (it lists that
  // specific pick's own factor breakdown) and stays as a function
  // (scoreInfo), passed via MetricLabel's `info` override.
  "Historic Annualized Return": {
    title: "Historic Annualized Return",
    body: [
      "This is the pick's own trailing 3-year annualized return (CAGR), computed straight from price history — how much it actually grew per year, on average, over the last 3 years.",
      "It is a raw historical fact about that one ticker, not the weighted Ranking Score below it — a pick can have a huge historic return but a middling score if it scores poorly on the other factors (cost, valuation, drawdown, etc).",
      "Past performance like this does not guarantee future results, especially for a single stock rather than a diversified fund.",
    ],
  },

  // --- SafeBaselineBand (components/SafeBaselineBand.tsx, used on /predict
  // and /entry) ---
  "Safe Baseline Price Band": {
    title: "Safe Baseline Price Band — what this is",
    body: [
      "A report card on the past, not a prediction of the future. It looks at years of this ticker's real price history and asks: every time someone bought at a given price and held for N trading days, what actually happened — how far did it typically dip, and how far did it typically run?",
      "Floor and Ceiling: historically, price has rarely moved outside this range within the horizon you picked. Accumulation Zone and Distribution Zone: a \"typical\" dip and a \"typical\" rally — where a normal pullback has bottomed, or a normal run has topped out. Median Path is simply today's price, unmodified — an anchor, not a forecast.",
      "How people use it: if the price falls into the Accumulation Zone, that's historically a normal-to-deep pullback, not usually a sign something's broken. If it's already above the Distribution Zone, it's already had a historically strong run, with typically less room left before a pause.",
      "The trust-check numbers below the band tell you how much history this is built on (\"Samples\") and how well-calibrated it's been (\"Breach Rate\" — how often price has actually broken the floor vs. how often the math expected it to).",
      "The most important caveat: this only looks at price history. It knows nothing about earnings, news, or what's happening with the company right now — treat it as one input, not the whole picture.",
      "Different from Predict's forecast confidence interval (model-based) or Entry Signals' stop/target (an ATR heuristic) — each uses a different method, so don't expect the numbers to match across pages.",
    ],
  },

  // --- Portfolio Health (app/portfolio/health/page.tsx) ---
  // Reuses "Ticker"/"Sector"/"Shares" above as-is (same meaning on this
  // page); "Avg Cost"/"Expense Ratio" are wired via term overrides to the
  // existing "Avg Cost Paid"/"Expense Ratio %" entries above, same reuse.
  Weight: {
    title: "Weight",
    body: ["This position's share of your total portfolio value right now."],
  },
  Concentrated: {
    title: "Concentrated",
    body: [
      "Flagged when a single position makes up 25% or more of your total portfolio value — a share large enough that its own moves drive most of the portfolio's swings.",
    ],
  },
  "Your Portfolio": {
    title: "Your Portfolio",
    body: ["This sector's share of your total portfolio value."],
  },
  "S&P 500": {
    title: "S&P 500 (approximation)",
    body: [
      "This sector's approximate share of the S&P 500 — a market-cap-weighted estimate across the index, not real fund weights, so treat it as directional rather than exact.",
    ],
  },
  Gap: {
    title: "Gap",
    body: [
      "Your Portfolio's sector weight minus the S&P 500's. Positive means you're overweight that sector relative to the index; negative means underweight.",
    ],
  },
  Volatility: {
    title: "Volatility",
    body: ["Annualized standard deviation of the portfolio's daily returns over this window — how bumpy the ride has been, not which direction it went."],
  },
  "Beta (vs. SPY)": {
    title: "Beta (vs. SPY)",
    body: [
      "How much the portfolio has historically moved for every 1% move in SPY (the S&P 500 ETF). 1.0 means it's tracked the index roughly 1:1; above 1.0 means more volatile than the index; below 1.0 means less.",
    ],
  },
  "Correlation (vs. SPY)": {
    title: "Correlation (vs. SPY)",
    body: [
      "How closely the portfolio's day-to-day moves have tracked SPY's, from -1 (perfectly opposite) to +1 (perfectly in lockstep).",
      "High correlation with a high beta means the portfolio amplifies the market's moves; low correlation means it's been moving somewhat independently of it.",
    ],
  },
  "Max drawdown": {
    title: "Max Drawdown",
    body: [
      "The largest peak-to-trough decline in the portfolio's value within this window — how much it fell from its best point before recovering, expressed as a positive percentage.",
    ],
  },
  Direct: {
    title: "Direct",
    body: ["This ticker's value held directly in your portfolio, not through any fund."],
  },
  "Via Funds": {
    title: "Via Funds",
    body: [
      "This ticker's estimated value held indirectly, through funds in your portfolio that disclose it among their top-10 holdings — see the fund coverage note above the table.",
    ],
  },
  Combined: {
    title: "Combined",
    body: ["Direct + Via Funds — your total real exposure to this ticker once fund look-through is accounted for."],
  },
  "Combined Weight": {
    title: "Weight (look-through-adjusted)",
    body: [
      "This ticker's Combined value as a share of total portfolio value — distinct from the plain \"Weight\" column on the Concentration table above, which doesn't account for fund look-through.",
    ],
  },
  Trailing: {
    title: "Trailing (last 12 months)",
    body: ["Dividend income actually received over the last 12 months, from real payment history."],
  },
  Projected: {
    title: "Projected (annual)",
    body: [
      "Dividend income expected over the next 12 months, projected from each holding's current trailing yield and share count — not a guarantee, since yields and holdings can change.",
    ],
  },
  Fund: {
    title: "Fund",
    body: ["The fund ticker this fee row is for."],
  },
  Value: {
    title: "Value",
    body: ["This fund's current market value in your portfolio — what the expense ratio below is applied against to estimate the annual drag."],
  },
  "Annual Drag": {
    title: "Annual Drag",
    body: [
      "This fund's expense ratio applied to its current market value — roughly what it costs you per year just for holding it, deducted automatically from the fund's own returns rather than billed separately.",
    ],
  },
  "Current Price": {
    title: "Current Price",
    body: ["The latest available trade price for this ticker."],
  },
  Loss: {
    title: "Loss",
    body: ["Unrealized loss on this position versus your average cost basis — the percent and dollar amount you'd realize if you sold today, before any tax effect."],
  },

  // --- Portfolio Stress Test (app/portfolio/stress-test/page.tsx) ---
  // Reuses "Ticker" above. "Market Value" here means something subtly
  // different from the Portfolio page's entry (a historical snapshot at
  // the start of the replay window, not today's live value) -- kept as
  // its own "Replay Market Value" key rather than silently reusing text
  // that would be inaccurate here.
  "Beta to Benchmark": {
    title: "Beta",
    body: [
      "Your portfolio's blended beta to the shock's benchmark ticker (e.g. SPY for a broad market shock, XLK for a tech-sector shock) — the same beta calculation used on the Portfolio Health risk page.",
      "The estimated dollar/percent impact above is this beta multiplied by the shock's stated move, applied to your portfolio's total value.",
    ],
  },
  "Replay Market Value": {
    title: "Market Value",
    body: ["This holding's value at the start of the historical replay window — what the replay's real return is applied against."],
  },
  "Replay %": {
    title: "% (historical replay)",
    body: ["This holding's own actual price return over the real historical window being replayed — not a model estimate."],
  },
  "Replay $": {
    title: "$ (historical replay)",
    body: ["This holding's estimated dollar impact: its Market Value at the start of the window × its own actual % return over that window."],
  },

  // --- Stock Detail (app/stock/[ticker]/page.tsx) ---
  // Reuses "Sector"/"Shares" as-is, and "Forward PE"/"Revenue Growth %"/
  // "Earnings Growth %"/"Avg Cost Paid"/"Weight" above via term overrides
  // (this page's visible labels are worded slightly differently --
  // "Forward P/E" vs "Forward PE", "Portfolio Weight" vs "Weight" -- but
  // mean the same thing, so they reuse rather than duplicate the copy).
  "Gain/Loss": {
    title: "Gain/Loss",
    body: ["Your total unrealized percent gain or loss on this position versus your average cost basis, as of the latest price."],
  },
  "EPS Est.": {
    title: "EPS Estimate",
    body: ["The consensus analyst estimate for earnings per share for this quarter, as of just before the report — sourced from the data provider, not computed by this app."],
  },
  "EPS Actual": {
    title: "EPS Actual",
    body: ["The company's actual reported earnings per share for this quarter."],
  },
  "Beat / Miss": {
    title: "Beat / Miss",
    body: ["Whether EPS Actual came in above (Beat) or below (Miss) the EPS Estimate, and by how much — the surprise percentage shown next to it."],
  },
  "Earnings Revenue": {
    title: "Revenue",
    body: [
      "Shown as \"Not available\" for past quarters — no data source this app uses has a historical record of what revenue was estimated to be at the time, so a beat/miss read here would be invented, not sourced.",
    ],
  },
  "Next-Day Move": {
    title: "Next-Day Move",
    body: [
      "The stock's price change the trading day after this earnings report, with a BMO (before market open) or AMC (after market close) tag inferred from the report's timestamp — not a confirmed flag from the data provider.",
    ],
  },

  // --- Two-score factors (Top Drivers/Drags, "Why these scores", weekly
  // change) --- the fixed 8-factor vocabulary behind services/stock_score_
  // service.py's SHORT_TERM_WEIGHTS (momentum/reversal/earnings_surprise/
  // earnings_revisions) and LONG_TERM_WEIGHTS (value/growth/low_vol/
  // quality). Keyed on the raw snake_case field name (same convention as
  // signal-comparison's quant_signal etc.), since that's what the API
  // actually returns and what every call site already has in hand.
  momentum: {
    title: "Momentum",
    body: [
      "Trailing ~3-month (63 trading day) price return, ranked against the rest of the universe — stronger recent momentum scores higher.",
      "Weighted 35% of the Short-Term score.",
    ],
  },
  reversal: {
    title: "Reversal",
    body: [
      "14-day RSI, ranked against the rest of the universe with a lower RSI scoring higher — this factor rewards a stock that looks oversold (a potential bounce), the opposite of a pure momentum read.",
      "Weighted 25% of the Short-Term score.",
    ],
  },
  earnings_surprise: {
    title: "Earnings Surprise",
    body: [
      "How far the company's most recently reported quarter's actual EPS came in above or below the analyst consensus estimate, as a percent — a positive surprise scores higher.",
      "Weighted 20% of the Short-Term score.",
    ],
  },
  earnings_revisions: {
    title: "Earnings Revisions",
    body: [
      "How much analysts' current-quarter EPS estimate has moved over the trailing 30 days — estimates trending up (analysts turning more optimistic) scores higher.",
      "Weighted 20% of the Short-Term score.",
    ],
  },
  value: {
    title: "Value",
    body: [
      "Forward P/E (price ÷ next year's consensus EPS estimate), ranked against the rest of the universe with a lower (cheaper) multiple scoring higher.",
      "Weighted 30% of the Long-Term score.",
    ],
  },
  growth: {
    title: "Growth",
    body: [
      "Average of trailing revenue growth % and earnings growth % (whichever is available) — faster growth scores higher.",
      "Weighted 25% of the Long-Term score.",
    ],
  },
  low_vol: {
    title: "Low Volatility",
    body: [
      "Annualized volatility of daily returns over the trailing year, ranked with lower volatility scoring higher — this rewards a steadier stock, not necessarily a stronger one.",
      "Weighted 20% of the Long-Term score.",
    ],
  },
  quality: {
    title: "Quality",
    body: [
      "Average of return on equity % and profit margin % (whichever is available) — stronger profitability scores higher.",
      "Weighted 25% of the Long-Term score.",
    ],
  },
};
