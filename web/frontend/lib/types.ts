export interface SignalOut {
  signal: string;
  expected_return_pct: number;
  target_price: number;
  signal_flip_count: number | null;
  signal_days_captured: number | null;
  signal_unstable: boolean | null;
}

export interface ForecastOut {
  dates: string[];
  predicted: number[];
  lower_ci: number[];
  upper_ci: number[];
}

export interface BacktestOut {
  dates: string[];
  actual: number[];
  predicted: number[];
  naive: number[];
}

export interface MetricsOut {
  rmse: number;
  mae: number;
  mape: number;
  naive_rmse?: number | null;
  naive_mae?: number | null;
  naive_mape?: number | null;
  beats_naive?: boolean | null;
}

export interface PredictionSummary {
  ticker: string;
  period: string;
  last_close: number | null;
  next_price: number | null;
  signal: SignalOut | null;
  forecast: ForecastOut | null;
  backtest: BacktestOut | null;
  metrics: MetricsOut | null;
  warnings: string[];
}

export interface PredictionNarrative {
  ticker: string;
  provider: string;
  narrative: string;
  sentiment_context: string;
}

export interface SavedNarrative {
  id: number;
  ticker: string;
  provider: string;
  period: string;
  days_ahead: number;
  narrative: string;
  sentiment_context: string;
  saved_at: string;
}

export interface PredictionActivity {
  ticker: string;
  latest_volume: number | null;
  avg_volume_10d: number | null;
  insider_buys: number | null;
  insider_sells: number | null;
  insider_period: string;
  institutional_increased: number | null;
  institutional_decreased: number | null;
  institutional_unchanged: number | null;
  institutional_holder_count: number | null;
  institutional_as_of: string | null;
}

export interface SavedPrediction {
  id: number;
  ticker: string;
  period: string;
  predicted_at: string;
  last_close: number | null;
  next_price: number | null;
  signal: string | null;
  expected_return_pct: number | null;
  target_price: number | null;
  target_date: string | null;
  actual_next_price: number | null;
  actual_target_price: number | null;
  actual_target_open: number | null;
  next_price_error_pct: number | null;
  target_price_error_pct: number | null;
  signal_correct: boolean | null;
  verified_at: string | null;
}

export interface TopFund {
  ticker: string;
  name: string;
}

export interface PredictionComparisonRow {
  prediction_id: number;
  ticker: string;
  predicted_at: string;
  signal: string | null;
  predicted_return_pct: number | null;
  actual_return_pct: number | null;
  stock_return_since_saved_pct: number | null;
  fund_return_since_saved_pct: number | null;
  signal_correct: boolean | null;
}

export interface PredictionCompareResponse {
  top_fund: TopFund | null;
  fund_current_price: number | null;
  comparisons: PredictionComparisonRow[];
}

export interface PredictionAccuracyRow {
  ticker: string;
  total_predictions: number;
  verified_count: number;
  win_rate: number | null;
  avg_next_price_error_pct: number | null;
  avg_target_price_error_pct: number | null;
  eligible_for_recommendation: boolean;
  rank: number | null;
}

export interface PredictionAccuracyLeaderboard {
  tickers: PredictionAccuracyRow[];
  suggested_ticker: string | null;
  suggested_reason: string | null;
  min_verified_for_recommendation: number;
}

export interface StockRankRow {
  Ticker: string;
  Name: string;
  Sector: string;
  Price: number;
  Score: number;
  [key: string]: string | number | null | boolean | number[];
}

export interface AnalystRatingSummary {
  ticker: string;
  consensus: string;
  analyst_count: number | null;
  buy_pct: number | null;
  target_mean: number | null;
  target_high: number | null;
  target_low: number | null;
  current_price: number | null;
}

export interface StockRankResponse {
  results: StockRankRow[];
}

export interface StockScoreResponse {
  result: StockRankRow | null;
}

export interface UniversesResponse {
  universes: string[];
}

export interface ScreenSnapshotRow {
  Ticker: string;
  Score: number;
  Price: number;
}

export interface SavedScreen {
  id: number;
  name: string;
  goal: string;
  universe: string;
  filters: Record<string, unknown>;
  visible_columns: string[];
  sort_keys: { column: string; direction: "asc" | "desc" }[];
  snapshot_top10: ScreenSnapshotRow[];
  saved_at: string;
}

export interface SavedScreenAlert {
  id: number;
  screen_id: number;
  check_date: string;
  entered: string[];
  left_tickers: string[];
  membership: string[];
  emailed_at: string | null;
}

export interface PresetScreen {
  key: string;
  name: string;
  rules: string;
  goal: string;
  universe: string;
  filters: Record<string, unknown>;
}

export interface TickerSearchResult {
  symbol: string;
  name: string;
  exchange: string;
  type: string;
}

export interface ExtendedHoursPrice {
  state: "PRE" | "POST";
  price: number;
  change_pct: number | null;
}

export interface CurrentPriceResponse {
  ticker: string;
  price: number | null;
  extended_hours: ExtendedHoursPrice | null;
}

// Index Fund Finder
export interface ScoreBreakdownMetric {
  key: string;
  label: string;
  unit: string;
  raw_value: number | null;
  weight: number;
}

export interface ScoreBreakdownBucket {
  sub_score: number;
  metrics: ScoreBreakdownMetric[];
}

export interface FundRankRow {
  Ticker: string;
  Fund: string;
  Benchmark: string;
  Category: string;
  Price: number;
  Score: number;
  _breakdown?: Record<string, ScoreBreakdownBucket>;
  [key: string]: string | number | null | undefined | Record<string, ScoreBreakdownBucket>;
}

export interface FundWindowMeta {
  window: string;
  start: string | null;
  end: string | null;
  error: string | null;
}

export interface FundRankResponse extends FundWindowMeta {
  results: FundRankRow[];
}

export interface FundScoreResponse extends FundWindowMeta {
  result: FundRankRow | null;
}

export interface FundGoalWeight {
  metric: string;
  label: string;
  weight: number | null;
  lower_is_better: boolean;
}

export interface FundGoal {
  name: string;
  description: string | null;
  weights: FundGoalWeight[];
}

export interface FundGoalsResponse {
  goals: FundGoal[];
}

export interface FundReturnSince {
  ticker: string;
  since: string;
  days: number;
  price_then: number;
  price_now: number;
  return_pct: number | null;
}

// Best To Enter Now
export interface EntryScanRow {
  Ticker: string;
  Signal: string;
  "Entry Score": number;
  "Current Price": number;
  "Quant Signal"?: string | null;
  "Quant Expected Return %"?: number | null;
  "Quant Target Price"?: number | null;
  [key: string]: string | number | null | undefined;
}

export interface EntryPlan {
  ticker: string;
  current_price: number;
  signal: string;
  summary: string;
  rsi: number | null;
  atr: number | null;
  macd: number | null;
  macd_signal: number | null;
  sma20: number | null;
  sma50: number | null;
  sma200: number | null;
  support_20: number;
  support_60: number;
  resistance_20: number;
  resistance_60: number;
  ideal_entry_low: number;
  ideal_entry_high: number;
  breakout_entry: number;
  stop_loss: number;
  first_target: number;
  avg_volume_20: number | null;
  latest_volume: number | null;
  trend_up: boolean;
  long_term_up: boolean;
  entry_score: number;
  quant_signal: string | null;
  quant_expected_return_pct: number | null;
  quant_target_price: number | null;
}

export interface EntryHistory {
  dates: string[];
  close: number[];
}

// Safe Baseline Price Band
export interface BaselineBand {
  ticker: string;
  as_of: string;
  last_price: number;
  horizon_days: number;
  confidence: number;
  method: "empirical" | "sqrt";
  floor: number;
  floor_pct: number;
  accumulation_zone_hi: number;
  accumulation_zone_hi_pct: number;
  median_path: number;
  median_path_pct: number;
  distribution_zone_lo: number;
  distribution_zone_lo_pct: number;
  ceiling: number;
  ceiling_pct: number;
  rr_ratio: number | null;
  skew: number;
  upside_first_rate: number;
  samples: number;
  effective_samples: number;
  breach_rate: number;
  breach_rate_full: number;
  breach_rate_recent: number;
  expected_breach: number;
  calibration_warning: boolean;
}

export interface SavedBaselineSnapshot {
  id: number;
  ticker: string;
  horizon_days: number;
  confidence: number;
  method: string;
  as_of: string;
  last_price: number;
  floor: number;
  floor_pct: number;
  accumulation_zone_hi: number;
  accumulation_zone_hi_pct: number;
  median_path: number;
  distribution_zone_lo: number;
  distribution_zone_lo_pct: number;
  ceiling: number;
  ceiling_pct: number;
  samples: number;
  effective_samples: number;
  breach_rate_full: number;
  saved_at: string;
}

// Monthly Investing Plan
export interface MonthlyRecommendation {
  ticker: string;
  name: string;
  score: number;
  asset_type: string;
  expected_return_pct: number | null;
}

export interface MonthlyHistory {
  dates: string[];
  contribution: number[];
  price: number[];
  shares_bought: number[];
  total_invested: number[];
  portfolio_value: number[];
}

export interface MonthlyPlanSummaryData {
  months: number;
  total_invested: number;
  ending_value: number;
  gain: number;
  gain_pct: number;
  latest_price: number;
}

export interface MonthlyPlanResponse {
  recommendation: MonthlyRecommendation | null;
  history: MonthlyHistory | null;
  summary: MonthlyPlanSummaryData | null;
  projected_value: number | null;
}

export interface SavedMonthlyPlan {
  id: number;
  name: string;
  monthly_amount: number;
  years: number;
  fund_goal: string;
  fund_category: string;
  stock_goal: string;
  stock_universe: string;
  created_at: string;
}

// Strategies
export interface ScoreFactor {
  metric: string;
  weight_pct: number;
  lower_is_better: boolean;
  value: number | null;
  unit: string;
}

export interface StrategyPickRow {
  label: string;
  ticker: string;
  name: string;
  annual_return_pct: number | null;
  score: number;
  asset_type: string;
  score_basis: ScoreFactor[];
}

export type SolveMode = "required_return" | "required_contribution" | "time_to_goal" | "achievable_amount";
export type DollarsMode = "today" | "future";
export type AccountType = "Taxable" | "Traditional" | "Roth";
export type FeasibilityLevel = "ok" | "warning" | "blocked";

export interface GoalPlanFix {
  type: "more_time" | "more_contribution" | "lower_target";
  label: string;
  years_needed?: number | null;
  monthly_contribution_needed?: number | null;
  achievable_target_future_dollars?: number;
  achievable_target_today_dollars?: number;
}

export interface GoalPlan {
  mode: SolveMode;
  target_today_dollars: number;
  target_future_dollars: number;
  years: number;
  starting_capital: number;
  monthly_contribution: number;
  annual_contribution_increase_pct: number;
  account_type: AccountType;
  tax_drag_pct: number;
  inflation_pct: number;
  solved_value: number | null;
  solved_field_label: string;
  gross_return_pct: number | null;
  net_return_pct: number | null;
  feasibility_level: FeasibilityLevel;
  feasibility_message: string | null;
  fixes: GoalPlanFix[] | null;
  return_assumption_table: ReturnAssumptionRow[] | null;
  realistic_return_pct?: number;
  sp500_long_run_pct?: number;
  reach_today_dollars?: number | null;
  reach_future_dollars?: number | null;
  target_vs_reach_ratio?: number | null;
  horizon_warnings: string[];
  monte_carlo: MonteCarloResult | null;
}

export interface ReturnAssumptionRow {
  annual_return_pct: number;
  monthly_contribution_needed: number | null;
}

export interface MonteCarloPercentileBand {
  year: number;
  p10: number;
  p50: number;
  p90: number;
}

export interface MonteCarloAssumptions {
  return_distribution_method: string;
  num_paths: number;
  sequence_of_returns_modeled: boolean;
  rebalancing_frequency: string;
  sleeve_correlation_model: string;
}

export interface MonteCarloResult {
  probability_of_success_pct: number | null;
  median_ending_balance: number;
  p10_ending_balance: number;
  p90_ending_balance: number;
  percentile_bands: MonteCarloPercentileBand[];
  assumptions: MonteCarloAssumptions;
}

// STRAT-10: a goal's linked holdings and their drift and steering.
export interface GoalLinkRow {
  id: number;
  portfolio_id: number;
  ticker: string;
  share_pct: number;
  role: "core" | "pick";
  value: number;
  baseline_value: number;
  gain: number;
}

export interface GoalLinksResponse {
  plan_id: number;
  targets: { stock_pct: number; bonds_pct: number; picks_pct: number; core_pct: number; picks_cap_pct: number };
  links: GoalLinkRow[];
  linked_value: number;
  linked_baseline: number;
  drift: { total: number; rows: { role: string; actual_pct: number; target_pct: number; drift_points: number }[]; flags: string[] };
  steering: {
    monthly: number;
    into_stocks: number;
    toward_bonds_outside_portfolio: number;
    buys: { ticker: string; role: string; amount: number; shares: number; price: number; spent: number }[];
    cash: number;
  } | null;
  assumption_note: string;
  months_elapsed: number;
  progress: {
    months_elapsed: number;
    expected_value: number;
    actual_value: number;
    diff: number;
    diff_pct: number | null;
    on_track: boolean;
  } | null;
  alerts: { kind: "over_cap" | "drift" | "yearly_review"; ticker: string | null; message: string }[];
}

export interface StrategiesSummaryResponse {
  plan: GoalPlan;
  picks: StrategyPickRow[] | null;
  // Set when today's rankings were unavailable and these are the last picks saved for this choice.
  picks_as_of?: string | null;
}

export interface StrategyPlanProgress {
  months_elapsed: number;
  expected_value: number;
  actual_value: number;
  diff: number;
  diff_pct: number | null;
  on_track: boolean;
}

export interface SavedStrategyPlan {
  success_pct?: number | null;
  id: number;
  name: string | null;
  target_amount: number;
  years: number;
  starting_capital: number;
  annual_return_pct: number;
  monthly_contribution: number;
  annual_contribution_increase_pct: number;
  account_type: AccountType;
  inflation_pct: number;
  created_at: string;
  progress: StrategyPlanProgress;
}

// Trade Journal
export interface Trade {
  trade_id: string;
  ticker: string;
  direction: string;
  strategy_type: string;
  created_at: string;
  entry_low: number | null;
  entry_high: number | null;
  stop_loss: number | null;
  target: number | null;
  context: string | null;
  risk_profile: string | null;
  risk_factor: number | null;
  status: string;
  entry_price: number | null;
  entry_date: string | null;
  exit_price: number | null;
  exit_date: string | null;
  max_runup_pct: number | null;
  max_drawdown_pct: number | null;
  realized_pnl_pct: number | null;
  days_in_trade: number | null;
  current_price: number | null;
  risk_reward_ratio: number | null;
  unrealized_pnl_pct: number | null;
  suggested_stop: number | null;
  strategy_note: string | null;
}

export interface TradeCreateInput {
  ticker: string;
  entry_low: number;
  entry_high: number;
  stop_loss: number;
  target: number;
  direction: string;
  strategy_type: string;
  context: string;
  risk_profile: string;
  risk_factor: number | null;
}

// Portfolio
export interface Portfolio {
  id: number;
  name: string;
  created_at: string;
  margin_balance: number;
  cash_balance: number;
  account_type: AccountType;
  position_count: number;
}

export interface PortfolioListResponse {
  portfolios: Portfolio[];
}

export interface PortfolioPosition {
  id: number;
  ticker: string;
  name: string;
  shares: number | null;
  avg_cost: number | null;
  current_price: number | null;
  unrealized_pnl_pct: number | null;
  source: string;
  created_at: string;
  acquired_at: string | null;
}

export interface PortfolioStrategyRow {
  id: number;
  ticker: string;
  shares: number | null;
  avg_cost: number | null;
  current_price: number | null;
  unrealized_pnl_pct: number | null;
  short_term_plan: string;
  long_term_plan: string;
  risk_profile: string;
  risk_factor: number;
  created_at: string;
  // Non-null only for a position synced from a linked Alpaca paper-trading
  // account (see web/backend/paper_order_sync.py) -- used to tag the row
  // "Paper" in the Holdings table so it's never confused with a real
  // manual/CSV/Plaid-synced holding.
  alpaca_paper_account_id?: number | null;
}

export interface PortfolioSummary {
  total_positions: number;
  total_value: number;
  total_pnl_pct: number;
  cash_balance: number;
}

export interface PortfolioInsight {
  ticker: string;
  signal: "BUY" | "SELL" | "HOLD" | null;
  expected_return_pct: number | null;
  target_price: number | null;
  expected_return_pct_1d: number | null;
  target_price_1d: number | null;
  expected_return_pct_5d: number | null;
  target_price_5d: number | null;
  expected_return_pct_30d: number | null;
  target_price_30d: number | null;
  rank: number | null;
  universe_size: number | null;
  trailing_return_pct: number | null;
  weight_pct: number | null;
  concentrated: boolean;
}

export interface Portfolio1yForecast {
  ticker: string;
  expected_return_pct: number | null;
  target_price: number | null;
}

export interface PortfolioInsightsResponse {
  positions: PortfolioInsight[];
  concentration_threshold_pct?: number;
  predict_period?: string;
  predict_days_ahead?: number;
  lookback_days?: number;
  as_of_date?: string;
  updated_at?: string | null;
}

export interface TickerSentiment {
  label: "Bullish" | "Neutral" | "Bearish" | null;
  reasoning: string | null;
}

export interface PortfolioSentimentResponse {
  sentiment: Record<string, TickerSentiment>;
  as_of_date: string;
}

export interface FlaggedPosition {
  ticker: string;
  signal: "BUY" | "SELL" | "HOLD" | null;
  weight_pct: number | null;
  sentiment_label: "Bullish" | "Neutral" | "Bearish" | null;
  market_value: number | null;
  sector: string | null;
  reasons: string[];
}

export interface PortfolioReviewResponse {
  summary: string | null;
  flagged: FlaggedPosition[];
  as_of_date: string | null;
}

export interface PortfolioPerformanceRow {
  ticker: string;
  shares: number;
  avg_cost: number | null;
  acquired_at: string | null;
  cost_basis: number | null;
  price_now: number | null;
  price_now_regular: number | null;
  price_30d_ago: number | null;
  value_now: number | null;
  value_30d_ago: number | null;
  diff: number | null;
  diff_pct: number | null;
  gain_vs_cost: number | null;
  gain_vs_cost_pct: number | null;
  price_unavailable: boolean;
  extended_hours: ExtendedHoursPrice | null;
  used_extended_hours: boolean;
  day_gain: number | null;
  day_gain_pct: number | null;
}

export interface PortfolioPerformance {
  lookback_days: number;
  rows: PortfolioPerformanceRow[];
  total_value_now: number;
  total_value_30d_ago: number;
  value_diff: number;
  value_diff_pct: number | null;
  total_cost_basis: number;
  total_gain_vs_cost: number;
  total_gain_vs_cost_pct: number | null;
  total_day_gain: number | null;
  total_day_gain_pct: number | null;
  margin_balance: number;
  cash_balance: number;
  net_equity: number;
}

export interface BenchmarkWorstPosition {
  ticker: string;
  gain_vs_cost_pct: number;
  gain_vs_cost: number;
  value_now: number;
}

export interface PortfolioBenchmarkComparison {
  benchmark_ticker: string;
  portfolio_return_pct: number | null;
  benchmark_return_pct: number | null;
  benchmark_today_pct: number | null;
  gap_pct: number | null;
  underperforming: boolean;
  worst_positions: BenchmarkWorstPosition[];
  suggestion: string | null;
}

export interface DualBenchmarkComparison {
  spy: PortfolioBenchmarkComparison;
  rsp: PortfolioBenchmarkComparison;
}

// Build a Diversified Basket
export interface BasketHolding {
  Ticker: string;
  Name: string;
  "GICS Sector": string;
  Industry: string;
  Score: number;
  Price: number;
  Shares: number;
  Amount: number;
  Weight_pct: number;
}

export interface BasketSectorSummary {
  Sector: string;
  Count: number;
  Weight_pct: number;
  Spy_Approx_Weight_pct: number;
}

export interface BasketExclusion {
  ticker: string;
  reason: string;
}

export interface BasketRiskPreview {
  annualized_volatility_pct: number | null;
  beta_to_spy: number | null;
  max_drawdown_pct: number | null;
  largest_single_stock_weight_pct: number | null;
  largest_single_sector_weight_pct: number | null;
  lookback: string;
  excluded_from_risk: string[];
}

export interface BasketTotals {
  invested: number;
  leftover_cash: number;
  holding_count: number;
}

export interface DiversifiedBasketPreview {
  as_of_date: string;
  holdings: BasketHolding[];
  sector_summary: BasketSectorSummary[];
  excluded: BasketExclusion[];
  sector_notes: string[];
  trim_notes: string[];
  totals: BasketTotals;
  warnings: string[];
  concentration_warning: string | null;
  risk_preview: BasketRiskPreview;
}

export type SectorWeighting = "equal_dollar" | "market_cap_by_sector";

export interface DiversifiedBasketRequest {
  goal: string;
  universe: string;
  picks_per_sector: number;
  max_stocks: number | null;
  total_amount: number;
  fractional_shares: boolean;
  sector_weighting: SectorWeighting;
  excluded_tickers: string[];
}

export type RebalanceFrequency = "none" | "monthly" | "quarterly";

export interface SaveDiversifiedBasketRequest {
  name: string;
  goal: string;
  universe: string;
  picks_per_sector: number;
  max_stocks: number | null;
  total_amount: number;
  fractional_shares: boolean;
  sector_weighting: SectorWeighting;
  as_of_date: string | null;
  holdings: BasketHolding[];
  rebalance_frequency: RebalanceFrequency;
  drift_threshold_pct: number;
}

export interface UniverseDetail {
  key: string;
  description: string;
  stock_count: number;
  sector_counts: Record<string, number>;
  as_of_date: string | null;
}

export interface UniverseDetailResponse {
  universes: UniverseDetail[];
}

export interface BasketRebalanceDriftRow {
  ticker: string;
  target_weight_pct: number;
  current_weight_pct: number;
  drift_pct: number;
}

export interface BasketRebalanceSwap {
  sell_ticker: string;
  buy_ticker: string | null;
  buy_name: string | null;
  reason: string;
}

export interface BasketRebalanceAlert {
  id: number;
  portfolio_id: number;
  check_date: string;
  score_as_of: string;
  drift_summary: BasketRebalanceDriftRow[];
  suggested_swaps: BasketRebalanceSwap[];
  target_weights: Record<string, number>;
  max_drift_pct: number;
  status: "pending" | "applied" | "dismissed";
  applied_at: string | null;
  seen_at: string | null;
  created_at: string;
  updated_at: string | null;
}

export interface ManualPositionInput {
  name: string;
  ticker: string;
  shares: number;
  current_price: number;
  avg_cost: number;
  total_return_pct?: number | null;
  acquired_at?: string | null;
}

export interface PortfolioSubmitResponse {
  positions: PortfolioPosition[];
  strategies: PortfolioStrategyRow[];
  summary: PortfolioSummary;
  watchlist_alerts_created: number;
}

// Portfolio vs. Top Picks compare page
export type CompareWindowCode = "10D" | "30D" | "60D" | "90D" | "1Y";
export type CompareSignalAction = "buy" | "hold" | "trim" | null;
export type CompareSignalLabel = "high" | "medium" | "low" | "unknown";

export interface CompareWindow {
  code: CompareWindowCode;
  start: string;
  end: string;
  trading_days: number;
}

export interface CompareGoal {
  code: string;
  label: string;
  description: string | null;
}

export interface CompareSignal {
  action: CompareSignalAction;
  label: CompareSignalLabel;
  score: number | null;
  as_of: string | null;
}

export type CompareSeriesPoint = [string, number];

export interface CompareHolding {
  ticker: string;
  weight_pct: number;
  return_pct: number | null;
  contribution_pts: number;
  spark: number[];
  signal: CompareSignal;
  since: string | null;
}

export interface CompareTopFund {
  rank: number;
  ticker: string;
  name: string;
  score: number;
  reason: string;
  return_pct: number | null;
  expense_ratio_pct: number | null;
  volatility_pct: number | null;
  series: CompareSeriesPoint[] | null;
}

export interface CompareTopStock {
  rank: number;
  ticker: string;
  name: string;
  sector: string | null;
  return_pct: number | null;
  owned: boolean;
  spark: number[];
  signal: CompareSignal | null;
  expected_return_pct: number | null;
  target_price: number | null;
}

export interface CompareGapDriver {
  ticker: string;
  kind: "lead" | "drag";
  contribution_pts: number;
}

// Portfolio Health Check (HLT-1..4)

export interface PortfolioHealthPosition {
  ticker: string;
  weight_pct: number;
  concentrated: boolean;
}

export interface SectorComparisonRow {
  sector: string;
  portfolio_weight_pct: number;
  sp500_weight_pct: number;
  gap_pct: number;
}

export interface PortfolioHealthConcentrationResponse {
  largest_positions: PortfolioHealthPosition[];
  sector_comparison: SectorComparisonRow[];
  as_of_date: string;
}

export interface PortfolioRiskWindow {
  period: string;
  data_start: string | null;
  data_end: string | null;
  volatility_pct: number | null;
  beta_to_spy: number | null;
  correlation_to_spy: number | null;
  max_drawdown_pct: number | null;
  excluded_from_risk: string[];
}

export interface PortfolioHealthRiskResponse {
  as_of: string;
  windows: { "1Y": PortfolioRiskWindow; "3Y": PortfolioRiskWindow };
}

// STR-1/2/3 — Scenario and Stress Tests. Every result carries its own
// `method` string (STR-3): rendered directly under the result, same
// amber-box convention as TOP10_DISCLOSURE/WASH_SALE_DISCLOSURE.
export interface StressTestPreset {
  preset_key: string;
  label: string;
  benchmark_ticker: string;
  shock_pct: number;
  method: string;
}

export interface StressTestReplay {
  replay_key: string;
  label: string;
  window_start: string;
  window_end: string;
  method: string;
}

export interface StressTestPresetsResponse {
  presets: StressTestPreset[];
  replays: StressTestReplay[];
}

export interface StressTestPresetResult {
  kind: "preset";
  preset_key: string;
  label: string;
  method: string;
  benchmark_ticker: string;
  shock_pct: number;
  beta: number | null;
  period: string;
  data_start: string | null;
  data_end: string | null;
  total_market_value: number;
  estimated_pct_impact: number | null;
  estimated_dollar_impact: number | null;
  excluded_from_beta: string[];
}

export interface StressTestReplayHolding {
  ticker: string;
  market_value: number;
  estimated_pct_impact: number | null;
  estimated_dollar_impact: number | null;
  method_used: "actual" | "excluded_no_history_for_window";
}

export interface StressTestReplayResult {
  kind: "replay";
  replay_key: string;
  label: string;
  method: string;
  window_start: string;
  window_end: string;
  total_market_value: number;
  estimated_pct_impact: number | null;
  estimated_dollar_impact: number | null;
  excluded_holdings: string[];
  holdings: StressTestReplayHolding[];
}

export interface StressTestRunResponse {
  as_of: string;
  result: StressTestPresetResult | StressTestReplayResult;
}

export type ScenarioComponentKind = "sector" | "factor";

export interface ScenarioComponent {
  kind: ScenarioComponentKind;
  label: string;
  sector?: string | null;
  benchmark_ticker?: string | null;
  shock_pct: number;
}

export interface CustomScenarioComponentResult {
  label: string;
  kind: ScenarioComponentKind;
  method: string;
  estimated_dollar_impact: number | null;
  estimated_pct_impact: number | null;
}

export interface CustomScenarioResult {
  kind: "custom";
  components: CustomScenarioComponentResult[];
  total_market_value: number;
  total_estimated_dollar_impact: number | null;
  total_estimated_pct_impact: number | null;
  method: string;
}

export interface CustomScenarioRunResponse {
  as_of: string;
  result: CustomScenarioResult;
}

export interface SavedStressScenario {
  id: number;
  user_id: string;
  name: string;
  shock_config: ScenarioComponent[];
  saved_at: string;
}

export interface CombinedExposureRow {
  ticker: string;
  direct_value: number;
  look_through_value: number;
  combined_value: number;
  combined_weight_pct: number;
  via_funds: { fund_ticker: string; dollars: number }[];
}

export interface PortfolioHealthOverlapResponse {
  combined_exposure: CombinedExposureRow[];
  fund_coverage_pct: Record<string, number>;
  sector_comparison: SectorComparisonRow[];
  disclosure: string;
  as_of_date: string;
}

export interface DividendIncomeRow {
  ticker: string;
  trailing_per_share: number | null;
  trailing_income: number | null;
  projected_per_share: number | null;
  projected_income: number | null;
}

export interface FeeDragRow {
  ticker: string;
  market_value: number;
  expense_ratio_pct: number | null;
  annual_fee_drag_dollars: number | null;
}

export interface PortfolioHealthIncomeFeesResponse {
  as_of_date: string;
  dividends: {
    by_ticker: DividendIncomeRow[];
    total_trailing_income: number | null;
    total_projected_income: number | null;
  };
  fee_drag: {
    by_fund: FeeDragRow[];
    total_annual_fee_drag_dollars: number | null;
  };
}

export interface TaxLossHarvestCandidate {
  ticker: string;
  shares: number;
  avg_cost: number;
  current_price: number;
  unrealized_loss_pct: number;
  unrealized_loss_dollars: number;
  wash_sale_note: string;
}

export interface PortfolioHealthTaxLossHarvestingResponse {
  as_of_date: string;
  account_type: AccountType;
  eligible: boolean;
  reason: string | null;
  candidates: TaxLossHarvestCandidate[];
}

export interface PortfolioCompareResponse {
  as_of: string;
  window: CompareWindow;
  goal: CompareGoal;
  portfolio: {
    id: number;
    name: string;
    holdings_count: number;
    return_pct: number | null;
    volatility_pct: number | null;
    max_drawdown_pct: number | null;
    signals: { buy: number; hold: number; trim: number };
    series: CompareSeriesPoint[];
  };
  benchmark: {
    ticker: string;
    return_pct: number | null;
    volatility_pct: number | null;
    expense_ratio_pct: number | null;
    series: CompareSeriesPoint[];
  };
  top_funds: CompareTopFund[];
  holdings: CompareHolding[];
  gap_drivers: CompareGapDriver[];
  top_stocks: CompareTopStock[];
  headline: string;
}

// Plaid brokerage connection
export interface PlaidItem {
  id: number;
  institution_id: string | null;
  institution_name: string | null;
  status: "active" | "login_required" | "error" | "revoked";
  last_sync_at: string | null;
  last_sync_error: string | null;
  created_at: string;
}

export interface PlaidLinkTokenResponse {
  link_token: string;
}

export interface PlaidSyncResult {
  status: "success" | "login_required" | "error" | "not_found";
  positions_upserted: number;
}

export interface PlaidExchangeResponse {
  item: PlaidItem;
  sync: PlaidSyncResult;
}

// Paper trading (Alpaca), Stage 1: no real money, no live orders -- see
// docs/live-trading-requirements.html and the paper-trading plan doc.
export interface PaperAccount {
  id: number;
  portfolio_id: number;
  api_key_id: string;
  alpaca_account_id: string | null;
  account_number: string | null;
  status: "active" | "invalid_key" | "disabled";
  last_sync_at: string | null;
  last_sync_error: string | null;
  disclosure_accepted_at: string | null;
  created_at: string;
}

export interface PaperClock {
  is_open: boolean;
  next_open: string;
  next_close: string;
}

export type PaperOrderStatus =
  | "DRAFT"
  | "SUBMITTING"
  | "OPEN"
  | "PARTIALLY_FILLED"
  | "FILLED"
  | "CANCELLED"
  | "REJECTED"
  | "UNKNOWN";

export interface PaperOrder {
  id: number;
  alpaca_paper_account_id: number;
  client_order_id: string;
  alpaca_order_id: string | null;
  ticker: string;
  side: "buy" | "sell";
  order_type: "market" | "limit";
  time_in_force: "day" | "gtc";
  qty: number;
  limit_price: number | null;
  status: PaperOrderStatus;
  filled_qty: number;
  filled_avg_price: number | null;
  reject_reason: string | null;
  submitted_at: string | null;
  created_at: string;
}

export interface PaperOrderCheckFailure {
  code: string;
  message: string;
}

// Phase 1 ("Trust") two-score system -- docs/stock-analysis-requirements.html.
// Named "TwoScore*" (not "Stock*") to avoid colliding with the pre-existing
// Stock Finder StockScoreResponse/getStockScore(goal, ticker) above, a
// different, unrelated concept.
export type TwoScoreSignal = "Buy" | "Hold" | "Trim";

export interface TwoScoreConfidence {
  score: number | null;
  label: "high" | "medium" | "low" | "unknown";
}

export interface TwoScoreFactor {
  raw: number | null;
  raw_revenue?: number | null;
  raw_earnings?: number | null;
  raw_roe?: number | null;
  raw_margin?: number | null;
  source: "pit" | "live";
  percentile: number | null;
  contribution: number | null;
}

export interface TwoScoreFactorDetail {
  momentum: TwoScoreFactor;
  reversal: TwoScoreFactor;
  earnings_surprise: TwoScoreFactor;
  earnings_revisions: TwoScoreFactor;
  value: TwoScoreFactor;
  growth: TwoScoreFactor;
  low_vol: TwoScoreFactor;
  quality: TwoScoreFactor;
}

export interface TwoScoreDriver {
  factor: string;
  contribution: number;
}

export interface TwoScoreResponse {
  ticker: string;
  as_of_date: string;
  universe_id: string;
  // REG-3: the market regime as of this row's own as_of_date (not
  // necessarily "today") -- see services/market_regime_service.py.
  regime: string | null;
  short_score: number | null;
  short_signal: TwoScoreSignal;
  short_confidence: TwoScoreConfidence;
  long_score: number | null;
  long_signal: TwoScoreSignal;
  long_confidence: TwoScoreConfidence;
  sector_key: string;
  short_sector_percentile: number | null;
  long_sector_percentile: number | null;
  short_universe_percentile: number | null;
  long_universe_percentile: number | null;
  short_sector_rank: { rank: number; of: number } | null;
  long_sector_rank: { rank: number; of: number } | null;
  factor_detail: TwoScoreFactorDetail;
  explanations: { drivers: TwoScoreDriver[]; drags: TwoScoreDriver[] };
  sentences: {
    momentum: string;
    reversal: string;
    earnings_surprise: string;
    earnings_revisions: string;
    value: string;
    growth: string;
    low_vol: string;
    quality: string;
  };
}

export interface TwoScoreTrend {
  weekly_series: [string, number][];
  flagged: boolean;
  change_pts: number | null;
}

export interface TwoScoreHistoryResponse {
  ticker: string;
  short_term: TwoScoreTrend;
  long_term: TwoScoreTrend;
}

export interface TwoScoreWeeklyChangeResponse {
  ticker: string;
  as_of_date: string;
  compared_to: string | null;
  change: { factor: string; delta_contribution: number } | null;
}

export type TwoScoreFactorKey =
  | "momentum"
  | "reversal"
  | "earnings_surprise"
  | "earnings_revisions"
  | "value"
  | "growth"
  | "low_vol"
  | "quality";

export interface TwoScoreFactorHistoryResponse {
  ticker: string;
  factor: TwoScoreFactorKey;
  sector_key: string;
  history: { as_of_date: string; raw: number | null; percentile: number | null; sector_median: number | null }[];
}

// Stock detail page (docs/stock-analysis-requirements.html DET-1..5).
// Scores/signals/explanations are deliberately NOT part of this response
// family -- see TwoScore* above, the single source of truth for those.
export interface EarningsCalendarEntry {
  ticker: string;
  owned: boolean;
  watchlisted: boolean;
  date: string;
  market_timing: "before_market" | "after_market";
  eps_estimate: number | null;
}

export interface EarningsCalendarResponse {
  as_of: string;
  window_days: number;
  entries: EarningsCalendarEntry[];
}

// REG-1/2/3 — see services/market_regime_service.py's module docstring
// for why this ships despite a failed validation gate; `disclosure` must
// always be rendered, not treated as optional copy.
export interface MarketRegimeComponents {
  breadth?: { pct_above_50dma: number; change_5d_pct: number | null };
  volatility?: { vix: number; vix3m: number; term_spread: number; inverted: boolean };
  trend?: { spy_close: number; spy_50dma: number | null; above_50dma: boolean | null };
  risk_appetite?: { momentum_pct: number | null };
}

export interface MarketRegimeResponse {
  available: boolean;
  reason?: string;
  as_of_date?: string;
  regime?: string | null;
  regime_raw?: string | null;
  mds?: number | null;
  internals_score?: number | null;
  data_completeness?: number;
  conflict_flag?: boolean;
  components?: MarketRegimeComponents;
  dimensions?: MarketRegimeDimensions;
  takeaways?: string[];
  disclosure?: string;
  methodology?: string[];
}

export interface RegimeReading {
  score?: number | null;
  text: string;
}

export interface MarketRegimeDimensions {
  rates?: RegimeReading;
  credit?: RegimeReading;
  breadth?: RegimeReading;
  divergence?: RegimeReading & { flag?: boolean | null };
  risk_appetite?: RegimeReading;
}

export interface StockDetailResponse {
  ticker: string;
  current_price: number | null;
  sector: string | null;
  fundamentals: {
    forward_pe: number | null;
    revenue_growth_pct: number | null;
    earnings_growth_pct: number | null;
    as_of_date: string | null;
  };
  next_earnings: { date: string; eps_estimate: number | null } | null;
  past_earnings: {
    date: string;
    reported_eps: number | null;
    eps_estimate: number | null;
    eps_beat: boolean | null;
    surprise_pct: number | null;
    revenue_beat: null; // always null -- yfinance has no historical revenue-estimate data, an honest gap
  }[];
  earnings_moves: { date: string; market_timing: "before_market" | "after_market"; move_pct: number | null }[];
  typical_earnings_move: { avg_abs_move_pct: number; quarters_counted: number } | null;
  recent_dividends: { date: string; amount: number }[];
}

export type StockPriceHistoryRange = "1D" | "5D" | "1M" | "6M" | "1Y" | "5Y";

// Signal explanation step 1: a stored SEC 8-K filing for a ticker. filed_on is a date; the filing has no time of day.
export interface StockNewsItem {
  id: number;
  title: string;
  url: string;
  publisher: string;
  filed_on: string;
  event_type: string;
  item_codes: string[];
  // Present only for an earnings filing that has a stored press-release summary.
  summary: string | null;
  summary_method: string | null;
}

export interface StockNewsResponse {
  ticker: string;
  days: number;
  items: StockNewsItem[];
}

// CHT-6: bar size. Intraday sizes are capped by yfinance (1 min: 7 days, 5 and 15 min: 60 days, 1 hour: 2 years).
export type StockPriceHistoryInterval = "1m" | "5m" | "15m" | "1h" | "1D" | "1W" | "1M";

export interface StockPriceHistoryRow {
  date: string;
  close: number;
  open: number | null;
  high: number | null;
  low: number | null;
  volume: number | null;
}

// CHT-3: one array per indicator, aligned to `history` rows. null = warm-up or missing.
export interface StockChartIndicators {
  sma_20: (number | null)[];
  sma_50: (number | null)[];
  sma_200: (number | null)[];
  ema_20: (number | null)[];
  bollinger: { upper: (number | null)[]; mid: (number | null)[]; lower: (number | null)[] };
  vwap: (number | null)[];
  rsi_14: (number | null)[];
  macd: { macd: (number | null)[]; signal: (number | null)[]; histogram: (number | null)[] };
  atr_14: (number | null)[];
  obv: (number | null)[];
}

export interface StockPriceHistoryResponse {
  ticker: string;
  range: StockPriceHistoryRange;
  history: StockPriceHistoryRow[];
  indicators: StockChartIndicators | null;
}

export interface StockPositionResponse {
  owned: boolean;
  shares?: number;
  avg_cost?: number;
  current_price?: number;
  gain_loss_pct?: number | null;
  weight_pct?: number | null;
}

export interface SignalOutcome {
  entry_date: string;
  exit_date: string;
  realized_return_pct: number;
  outcome: "hit" | "miss" | null;
}

export interface StockSignalHistoryResponse {
  ticker: string;
  history: {
    as_of_date: string;
    short_score: number | null;
    short_signal: TwoScoreSignal;
    long_score: number | null;
    long_signal: TwoScoreSignal;
    short_outcome: SignalOutcome | null;
    long_outcome: SignalOutcome | null;
    // DIF-1: stored confidence and short-term reasons for that day (absent on older responses).
    short_confidence?: { label: string; score: number | null } | null;
    short_reasons?: { drivers: { factor: string; contribution: number }[]; drags: { factor: string; contribution: number }[] } | null;
  }[];
  note: string;
}

export interface StockPeer {
  ticker: string;
  name: string | null;
  market_cap_b: number;
  short_score: number | null;
  short_signal: TwoScoreSignal | null;
  long_score: number | null;
  long_signal: TwoScoreSignal | null;
}

export interface StockPeersResponse {
  ticker: string;
  peers: StockPeer[];
  reason?: "data_unavailable" | "no_same_sector_peers" | null;
  message?: string | null;
}

export interface StockSentimentResponse {
  ticker: string;
  label: "Bullish" | "Neutral" | "Bearish" | null;
  reasoning: string | null;
}

// Meta-Agent Chat
export interface ChatProvidersResponse {
  providers: string[];
}

export interface ChatAskResponse {
  ticker: string;
  provider: string;
  answer: string;
  // General questions only: the stored passages the answer was built from, numbered as in the answer.
  sources?: { number: number; source: string; text: string }[];
}

export interface ChatAskParams {
  scope: "ticker" | "portfolio" | "general";
  ticker?: string;
  portfolio_id?: number;
  question: string;
  provider?: string;
  cited?: boolean;
}

export interface FilingSummary {
  form_type: string;
  filing_date: string;
  report_date: string | null;
  document_url: string;
  compared_to_prior_filing: boolean;
  summary: string;
  method: string;
}

export interface FilingSummariesResponse {
  ticker: string;
  filings: FilingSummary[];
}

export interface EarningsReleaseSummary {
  filing_date: string;
  report_date: string | null;
  document_url: string;
  summary: string;
  method: string;
}

export interface EarningsReleaseSummaryResponse {
  ticker: string;
  release: EarningsReleaseSummary | null;
}

// Admin — user approvals
export interface AdminUser {
  id: string;
  email: string;
  approved: boolean;
  is_active: boolean;
  created_at: string;
  last_login_at: string | null;
  last_login_ip: string | null;
  portfolio_count: number;
  position_count: number;
}

export interface AdminUserPortfolio {
  id: number;
  name: string;
  is_active: boolean;
  drop_alerts_enabled: boolean;
  created_at: string;
  position_count: number;
}

export interface AdminActivityRow {
  id: number;
  email: string;
  endpoint: string;
  created_at: string;
}

export type AlertConditionType = "price_above" | "price_below" | "score_above" | "score_below";

// ALR-2: one normalized row per triggered alert across every source
// table -- see web/backend/routers/alerts_inbox.py.
export interface AlertInboxItem {
  source: "watchlist" | "portfolio_drop" | "signal_change" | "earnings" | "cost_drop";
  id: number;
  ticker: string;
  alert_type: string;
  summary: string;
  created_at: string;
  event_at: string;
  seen_at: string | null;
  link: string | null;
}

// ALR-1: matches services/notification_dispatcher.py's alert_type strings.
export type AlertPreferenceType = "signal_change" | "earnings" | "cost_drop";

export interface AlertPreferenceOverride {
  id: number;
  user_id: string;
  ticker: string | null;
  alert_type: AlertPreferenceType;
  enabled: boolean;
  channel_email: boolean;
  channel_inapp: boolean;
  created_at: string;
  updated_at: string;
}

export interface AlertPreferencesResponse {
  default: { enabled: boolean; channel_email: boolean; channel_inapp: boolean };
  overrides: AlertPreferenceOverride[];
}

export interface AlertNotificationSettings {
  quiet_hours_start: string | null;
  quiet_hours_end: string | null;
  digest_enabled: boolean;
  digest_time: string;
  webhook_enabled: boolean;
  webhook_url: string | null;
  has_webhook_secret: boolean;
  // Present only in the one PUT response where it was just generated --
  // never returned again afterward (GET never includes it).
  webhook_secret?: string;
}

export interface WatchlistAlert {
  id: number;
  ticker: string;
  condition_type: AlertConditionType;
  threshold: number;
  created_at: string;
  active: boolean;
  triggered_at: string | null;
  triggered_price: number | null;
  seen_at: string | null;
  source: string | null;
}

export interface AdminSettings {
  verify_predictions_enabled: boolean;
  publish_signals_enabled: boolean;
  password_policy_enabled: boolean;
  pit_price_capture_enabled: boolean;
  pit_analyst_rating_capture_enabled: boolean;
  pit_quant_signal_capture_enabled: boolean;
  portfolio_drop_alerts_enabled: boolean;
  portfolio_drop_threshold_pct: number;
  daily_quota: number;
  db_backup_enabled: boolean;
  horizon1_subscriptions_enabled: boolean;
  free_tier_lag_days: number;
  price_data_provider: "yahoo" | "alpaca";
  basket_rebalance_enabled: boolean;
  saved_screen_alerts_enabled: boolean;
  stock_finder_cache_prewarm_enabled: boolean;
  market_regime_enabled: boolean;
  filing_summaries_enabled: boolean;
  news_8k_enabled: boolean;
  earnings_release_summaries_enabled: boolean;
  evening_recap_enabled: boolean;
  morning_brief_enabled: boolean;
  paper_account_equity_capture_enabled: boolean;
  challenge_notifications_enabled: boolean;
  stock_score_compute_enabled: boolean;
}

export interface BackupRun {
  id: number;
  started_at_utc: string;
  s3_key: string | null;
  size_bytes: number | null;
  tables_verified: string[] | null;
  structural_check_passed: boolean;
  restore_test_run: boolean;
  restore_test_passed: boolean | null;
  restore_test_row_counts: Record<string, { restored: number; live: number; match: boolean }> | null;
  error: string | null;
}

export interface BackupStatus {
  recent_runs: BackupRun[];
  backup_tables: string[];
}

export interface PortfolioDropAlert {
  id: number;
  ticker: string;
  alert_date: string;
  prev_close: number;
  price_at_check: number;
  pct_change: number;
  sentiment_summary: string | null;
  predicted_signal: string | null;
  predicted_expected_return_pct: number | null;
  predicted_target_price: number | null;
  recommended_action: string | null;
  created_at: string;
  seen_at: string | null;
  updated_at: string | null;
}

export interface DropAlertThreshold {
  threshold_pct: number;
  is_custom: boolean;
  default_pct: number;
}

export interface GoalPlanAllocationRow {
  ticker: string;
  signal: "BUY" | "SELL" | "HOLD" | null;
  expected_return_pct: number | null;
  annualized_return_pct: number | null;
  current_value: number;
  weight_pct: number;
  monthly_amount: number;
}

export interface GoalPlanResponse {
  portfolio_id: number;
  months_remaining: number;
  target_amount: number;
  target_date: string;
  current_value: number;
  current_holdings_annualized_return_pct: number | null;
  contribution_annualized_return_pct: number | null;
  allocation: GoalPlanAllocationRow[];
  best_stock_comparison: BestStockComparison | null;
  warnings: string[];
  future_value_of_current_holdings: number;
  required_monthly_contribution: number | null;
  projected_value_with_given_contribution?: number;
  gap_vs_target?: number;
}

export interface BestStockComparison {
  ticker: string;
  name: string;
  annualized_return_pct: number;
  universe: string;
  goal: string;
  future_value_of_current_holdings: number;
  required_monthly_contribution: number | null;
  projected_value_with_given_contribution?: number;
  gap_vs_target?: number;
}

export interface SavedGoal {
  id: number;
  portfolio_id: number;
  name: string;
  target_amount: number;
  target_date: string;
  monthly_amount: number | null;
  compare_universe: string | null;
  created_at: string;
}

export interface PitCaptureStats {
  row_count: number;
  days_captured: number;
  earliest_date: string | null;
  latest_date: string | null;
  last_captured_at_utc: string | null;
}

export interface PitPricesStats extends PitCaptureStats {
  ticker_count: number;
}

export interface PitUniverseMembershipStats extends PitCaptureStats {
  universe_count: number;
}

export interface PitFundamentalsStats extends PitCaptureStats {
  ticker_count: number;
}

export interface PitQuantSignalStats extends PitCaptureStats {
  ticker_count: number;
}

export interface PitAnalystRatingStats extends PitCaptureStats {
  ticker_count: number;
}

export interface PitPriceStatus {
  universe_id: string;
  prices: PitPricesStats;
  universe_membership: PitUniverseMembershipStats;
  fundamentals: PitFundamentalsStats;
  quant_signal: PitQuantSignalStats;
  analyst_rating: PitAnalystRatingStats;
}

export interface QuantVsAnalystRow {
  ticker: string;
  quant_signal: "BUY" | "SELL" | "HOLD" | "UNKNOWN";
  quant_expected_return_pct: number;
  quant_target_price: number;
  last_close: number;
  analyst_consensus: string | null;
  analyst_count: number | null;
  analyst_buy_pct: number | null;
  analyst_target_mean: number | null;
  analyst_target_high: number | null;
  analyst_target_low: number | null;
  signal_flip_count: number;
  signal_days_captured: number;
  signal_unstable: boolean;
}

export interface QuantVsAnalystResponse {
  as_of_date: string | null;
  ticker_count: number;
  rows: QuantVsAnalystRow[];
}

export interface QuantSignalOutcomeSummaryEntry {
  count: number;
  win_rate_pct: number | null;
}

export interface QuantSignalOutcomesResponse {
  horizon_days: number;
  summary: {
    BUY: QuantSignalOutcomeSummaryEntry;
    SELL: QuantSignalOutcomeSummaryEntry;
    HOLD: QuantSignalOutcomeSummaryEntry;
  };
}

export interface QuantSignalHistoryPoint {
  as_of_date: string;
  signal: "BUY" | "SELL" | "HOLD" | "UNKNOWN";
  expected_return_pct: number;
  target_price: number;
  last_close: number;
}

export interface PitReconciliationReport {
  target_date: string;
  universe_id: string;
  lookback_days: number;
  pit_trading_days_available: number;
  pit_trading_days_required: number;
  published_count: number;
  reconstructed_count: number;
  matches: number;
  mismatches: { ticker: string; published_rank: number; reconstructed_rank: number }[];
  missing_from_pit_history: { ticker: string; published_rank: number }[];
  byte_identical: boolean;
}

export interface SignalStabilityTicker {
  ticker: string;
  days_captured: number;
  flip_count: number;
  current_signal: "BUY" | "SELL" | "HOLD" | "UNKNOWN";
  current_streak_days: number;
  last_flip_date: string | null;
}

export interface SignalStabilityFlip {
  ticker: string;
  prev_date: string;
  prev_signal: "BUY" | "SELL" | "HOLD" | "UNKNOWN";
  date: string;
  signal: "BUY" | "SELL" | "HOLD" | "UNKNOWN";
  prev_expected_return_pct: number | null;
  expected_return_pct: number | null;
  prev_close: number | null;
  last_close: number | null;
  price_move_pct: number | null;
  classification: "boundary" | "chase" | "model_shift";
}

export interface SignalStabilityReport {
  buy_threshold_pct: number;
  sell_threshold_pct: number;
  lookback_days: number;
  tickers: SignalStabilityTicker[];
  recent_flips: SignalStabilityFlip[];
}

export interface PublishedSignalRow {
  id: number;
  published_at_utc: string;
  model_version_hash: string;
  as_of_data_timestamp: string;
  target_date: string;
  universe_id: string;
  lookback_days: number;
  rank: number;
  ticker: string;
  trailing_return_pct: number;
  data_source: "pit" | "live";
  reason_code: string | null;
  corrects_id: number | null;
}

export interface PublishedSignalsResponse {
  target_date: string | null;
  universe_id: string;
  lookback_days: number;
  signals: PublishedSignalRow[];
  record_start_date: string | null;
  days_published: number;
  tier: "free" | "paid";
  is_lagged: boolean;
}

export interface SignalOutcomeRow {
  target_date: string;
  ticker: string;
  rank: number;
  entry_price: number;
  exit_price: number;
  realized_return_pct: number;
  benchmark_return_pct: number;
  beat_benchmark: boolean;
}

export interface SignalOutcomesResponse {
  universe_id: string;
  lookback_days: number;
  horizon_days: number;
  num_evaluated_dates: number;
  num_evaluated_picks: number;
  hit_rate_pct: number | null;
  avg_return_pct: number | null;
  information_coefficient: number | null;
  quintile_spread_pct: number | null;
  outcomes: SignalOutcomeRow[];
}

// TRK-2/3/5/6 (docs/stock-analysis-requirements.html) -- the enhanced
// public track record, built from the same signal_outcomes record above.
export interface TrackRecordMetrics {
  num_evaluated_dates: number;
  num_evaluated_picks: number;
  hit_rate_pct: number | null;
  avg_return_pct: number | null;
  information_coefficient: number | null;
  quintile_spread_pct: number | null;
}

export interface TrackRecordCalibrationBucket {
  bucket_label: string;
  hit_rate_pct: number | null;
  sample_size: number;
}

export interface TrackRecordWorstMiss {
  target_date: string;
  ticker: string;
  rank: number;
  entry_price: number;
  exit_price: number;
  realized_return_pct: number;
  benchmark_return_pct: number;
  beat_benchmark: boolean;
  model_version_hash: string | null;
}

export interface TrackRecordResponse {
  universe_id: string;
  lookback_days: number;
  horizon_days: number;
  regime_filter: string | null;
  metrics: TrackRecordMetrics;
  metrics_by_model_version: Record<string, TrackRecordMetrics>;
  metrics_by_signal: Record<string, TrackRecordMetrics>;
  metrics_by_regime: Record<string, TrackRecordMetrics>;
  avg_excess_vs_spy_pct: number | null;
  calibration: TrackRecordCalibrationBucket[];
  worst_misses: TrackRecordWorstMiss[];
  model_portfolio_series: [string, number][];
  spy_portfolio_series: [string, number][];
  trim_note: string;
  signal_note: string;
  model_portfolio_cost_bps_one_way: number;
}

export interface PredictAlgoComparisonRow {
  rank: number;
  ticker: string;
  trailing_return_pct: number;
  predict_signal: string | null;
  predict_expected_return_pct: number | null;
  predict_target_price: number | null;
}

export interface PredictAlgoComparisonResponse {
  target_date: string | null;
  predict_period: string;
  predict_days_ahead: number;
  comparisons: PredictAlgoComparisonRow[];
}

export interface TopPerformerRow {
  ticker: string;
  name: string;
  price: number | null;
  return_pct: number;
}

export interface TopPerformersResponse {
  results: TopPerformerRow[];
  window: number;
  asset_type: string;
}

export interface MomentumOptions {
  windows: number[];
  stock_universes: string[];
  fund_categories: string[];
}

export interface MomentumBacktestPeriod {
  date: string;
  picks: string[];
  strategy_return_pct: number | null;
  strategy_return_gross_pct: number | null;
  benchmark_return_pct: number | null;
  turnover_pct: number;
}

export interface MomentumBacktestResponse {
  run_id: number;
  asset_type: string;
  universe: string;
  lookback_days: number;
  top_n: number;
  years: number;
  horizon_days: number;
  slippage_bps: number;
  commission_bps: number;
  borrow_cost_bps_annual: number;
  borrow_cost_drag_pct: number;
  risk_free_rate_annual: number;
  num_periods: number;
  hit_rate_pct: number | null;
  strategy_cumulative_return_pct: number | null;
  benchmark_cumulative_return_pct: number | null;
  avg_strategy_period_return_pct: number | null;
  avg_benchmark_period_return_pct: number | null;
  cagr_pct: number | null;
  volatility_pct: number | null;
  sharpe_ratio: number | null;
  sortino_ratio: number | null;
  max_drawdown_pct: number | null;
  avg_turnover_pct: number | null;
  capacity_estimate_usd: number | null;
  periods: MomentumBacktestPeriod[];
}

// Admin — SQL runner
export interface AdminSqlColumn {
  name: string;
  type: string;
}

export interface AdminSqlTable {
  table_name: string;
  approx_row_count: number | null;
  columns: AdminSqlColumn[];
}

export interface AdminSqlTablesResponse {
  tables: AdminSqlTable[];
}

export interface AdminSqlQueryResponse {
  columns: string[];
  rows: unknown[][];
  row_count: number;
  truncated: boolean;
}

export interface AdminIntegration {
  key: string;
  name: string;
  category: string;
  configured: boolean;
  note: string;
}

export interface AdminIntegrationTestResult {
  ok: boolean;
  detail: string;
  latency_ms: number | null;
}

export interface CrawlSearchDomainStats {
  domain: string;
  pages_crawled: number;
  pages_skipped_robots: number;
  pages_skipped_filter: number;
  pages_failed: number;
  skipped_domain: boolean;
  disabled: boolean;
  cancelled: boolean;
}

export interface CrawlSearchStatus {
  running: boolean;
  requested_stop: boolean;
  started_at: number | null;
  finished_at: number | null;
  requested_domains: string[] | null;
  current_domain: string | null;
  current_domain_pages: number;
  pages_crawled: number;
  completed: CrawlSearchDomainStats[];
  error: string | null;
  cancelled: boolean;
}

export interface CrawlSearchDomainConfig {
  domain: string;
  enabled: boolean;
  max_pages: number;
  max_depth: number;
  min_delay_seconds: number;
  boost: number;
  use_sitemap: boolean;
  include: string[];
  exclude: string[];
  seeds: string[];
  tags: string[];
  overridden_keys: string[];
  in_yaml: boolean;
  pages: number;
  last_crawled: string | null;
}

export interface CrawlSearchDomainsResponse {
  domains: CrawlSearchDomainConfig[];
  all_tags: string[];
  fields: string[];
}

export interface CrawlSearchDomainSaveResult {
  domain: string;
  overrides: Record<string, unknown>;
  effective: CrawlSearchDomainConfig | null;
}

export interface CrawlSearchDomainResetResult {
  domain: string;
  result: "reset_to_yaml" | "removed";
}

export interface CrawlSearchStats {
  indexed_pages: number;
  domains_with_pages: number;
  last_crawled_at: string | null;
  pages_last_24h: number;
  urls_seen: number;
  configured_domains: number;
  enabled_domains: number;
  retention_days: number;
  per_domain: { domain: string; pages: number; last_crawled: string | null }[];
}

// Horizon 1 — Impersonal Research Subscription (built, kept off; see
// docs/signal-licensing-whitelabel-requirements.md.pdf)
export interface MySubscription {
  tier: "free" | "paid";
  status: "active" | "canceled" | "past_due" | "incomplete" | null;
  current_period_end: string | null;
  created_at?: string;
  canceled_at?: string | null;
}

export interface CohortRetention {
  window: "1_month" | "3_month" | "6_month";
  cohort_size: number;
  retained: number;
  retention_rate: number | null;
}

export interface EnquiryTypeCount {
  enquiry_type: string;
  count: number;
}

export interface DemandReport {
  ever_paid_subscribers: number;
  currently_active_subscribers: number;
  canceled_total: number;
  monthly_churn_rate: number | null;
  checkout_started: number;
  checkout_completed: number;
  checkout_conversion_rate: number | null;
  cohort_retention: CohortRetention[];
  enquiries_by_type: EnquiryTypeCount[];
}

export interface AuditLogEntry {
  id: number;
  actor_user_id: string | null;
  event_type: string;
  resource: string | null;
  metadata: Record<string, unknown> | null;
  created_at: string;
}

export interface AuditLogResponse {
  events: AuditLogEntry[];
  limit: number;
  offset: number;
}

export interface WebSearchResult {
  title: string;
  url: string;
  content: string;
  score: number;
  raw_content: string | null;
}

export interface WebSearchResponse {
  query: string;
  results: WebSearchResult[];
  response_time_ms: number;
}

export interface MarketNewsItem {
  title: string;
  url: string;
  source: string;
  published_at: string | null;
}

export interface MarketNewsResponse {
  items: MarketNewsItem[];
  source: "yahoo" | "duckduckgo";
}

export interface Challenge {
  id: number;
  name: string;
  start_date: string;
  end_date: string;
  member_count: number;
}

export interface ChallengeDetail {
  id: number;
  name: string;
  join_code: string;
  start_date: string;
  end_date: string;
  members: string[];
}

export interface ChallengeBadge {
  badge: string;
  detail: string;
}

export interface ChallengeLeaderboardEntry {
  member: string;
  has_paper_account: boolean;
  is_model: boolean;
  badges: ChallengeBadge[];
  vs_spy_pct: number | null;
  score: number | null;
  sharpe: number | null;
  sortino: number | null;
  calmar: number | null;
  return_pct: number | null;
  max_drawdown_pct: number | null;
  annualized_volatility_pct: number | null;
  days_of_data: number;
}

export interface ChallengeLeaderboardResponse {
  start_date: string;
  end_date: string;
  scoring: string;
  scoring_label: string;
  spy_return_pct: number | null;
  ended: boolean;
  entries: ChallengeLeaderboardEntry[];
}

export interface ChallengeInvite {
  id: number;
  challenge_id: number;
  challenge_name: string;
  invited_by_label: string;
  created_at: string;
}

export interface DiscoverableUser {
  id: string;
  label: string;
}

export interface AgentEvent {
  event_type: string;
  ticker: string | null;
  side: string | null;
  qty: number | null;
  est_value: number | null;
  trigger: string | null;
  reason: string;
  alpaca_order_id?: string | null;
  created_at: string;
}

export interface AgentRunSummary {
  id: number;
  mode: string;
  status: string;
  regime: string | null;
  exposure_cap_pct: number | null;
  equity: number | null;
  risk_state: string | null;
  config_version: string;
  reason: string | null;
  created_at: string;
  events: AgentEvent[];
}

export interface TradingAgentStatus {
  enabled: boolean;
  mode: "plan" | "paper" | "live";
  kill_engaged: boolean;
  global_enabled: boolean;
  global_kill_engaged: boolean;
  breaker_latched: boolean;
  peak_equity: number | null;
  config_version: string;
  limits: {
    max_position_pct: number;
    max_sector_pct: number;
    max_positions: number;
    daily_loss_limit_pct: number;
    drawdown_breaker_pct: number;
  };
  regime: { label: string | null; exposure_cap_pct: number; reason: string; disclosure: string };
  live: { allowed: boolean; reason: string };
  disclosure: string;
  broker: { positions: { ticker: string; qty: number; market_value: number; stop: { qty: string; trail_percent: string } | null }[] } | { error: string } | null;
  performance_paper: {
    days_of_data: number;
    note?: string;
    return_pct?: number | null;
    annualized_volatility_pct?: number | null;
    max_drawdown_pct?: number | null;
    sharpe?: number | null;
    spy_return_pct?: number | null;
    costs_paid?: string;
    worst_month?: string;
  } | null;
  latest_run: { run: AgentRunSummary; events: AgentEvent[] } | null;
}

export interface EquityPoint {
  date: string;
  value: number;
}

export interface ChallengeEquityCurves {
  start_date: string;
  end_date: string;
  spy: EquityPoint[];
  members: { member: string; points: EquityPoint[] }[];
}

// DIF-6: hypothetical trade preview. Nothing is placed or saved.
export interface TradeImpactMeasures {
  concentration: { largest_position_pct: number | null; top5_pct: number | null; holdings: number };
  sector_weights: Record<string, number>;
  portfolio_score: { score: number | null; coverage_pct: number };
  total_value: number;
  beta: number | null;
}

export interface TradeImpactResponse {
  ticker: string;
  side: "buy" | "sell";
  shares: number;
  trade_price: number;
  before: TradeImpactMeasures;
  after: TradeImpactMeasures;
  changes: {
    largest_position_pct: number | null;
    top5_pct: number | null;
    beta: number | null;
    portfolio_score: number | null;
  };
  sectors: { sector: string; before_pct: number; after_pct: number; change_pct: number }[];
  note: string;
}

// DIF-9: stored regime label per trading day, for shading the price chart. Condition labels, not recommendations.
export interface RegimeHistoryResponse {
  available: boolean;
  reason?: string;
  history: { date: string; regime: string }[];
  disclosure: string;
}

// DIF-7: past days with a similar stored score and regime, and the price move over the next sessions.
export interface SimilarSetupsResponse {
  ticker: string;
  available: boolean;
  reason?: string;
  current_score?: number;
  current_regime?: string | null;
  horizon_sessions?: number;
  score_band_points?: number;
  n?: number;
  median_return_pct?: number | null;
  min_return_pct?: number | null;
  max_return_pct?: number | null;
  caveat?: string | null;
  note?: string;
}

// STB-1..5: no-code strategy backtest. Past prices only; nothing is placed.
export interface StrategyRuleInput {
  field: string;
  op: string;
  value: string | number;
}

export interface StrategyMetrics {
  days?: number;
  total_return_pct?: number | null;
  cagr_pct?: number | null;
  volatility_pct?: number | null;
  max_drawdown_pct?: number | null;
  sharpe?: number | null;
  worst_month_pct?: number | null;
  turnover_pct_per_year?: number | null;
}

export interface StrategyExplanation {
  bottom_line: string;
  tested: { item: string; setting: string; plain: string }[];
  headline: string[];
  full_period: {
    label: string; total: string; cagr: string; volatility: string; max_drop: string; sharpe: string;
    worst_month: string; turnover: string; cost_drag: string;
  }[];
  checks: { label: string; status: "pass" | "caution" | "fail"; detail: string; meaning: string }[];
  in_vs_out: {
    in_sample: { cagr: string; volatility: string; max_drop: string; sharpe: string };
    out_of_sample: { cagr: string; volatility: string; max_drop: string; sharpe: string };
    note: string;
  } | null;
  problems: string[];
  next_steps: string[];
  disclaimer: string;
}

export interface ModelPortfolioSummary {
  available: boolean;
  months_of_history: number;
  reason?: string;
  note?: string;
  total_return_pct?: number | null;
  cagr_pct?: number | null;
  volatility_pct?: number | null;
  max_drawdown_pct?: number | null;
  sharpe?: number | null;
  worst_period_pct?: number | null;
}

export interface StrategyCheck {
  status: "pass" | "caution" | "fail";
  label: string;
  detail: string;
}

export interface StrategyTradeRow {
  ticker: string;
  entry_date: string;
  entry_price: number;
  exit_date: string;
  exit_price: number;
  exit_reason: string;
  holding_days: number;
  return_pct: number;
}

export interface StrategyBacktestResponse {
  period: { start: string; end: string; sessions: number };
  tickers: string[];
  costs: { cost_bps_per_side: number; slippage_bps_per_side: number };
  cooldown_sessions: number;
  protective_exit: {
    stop_loss_pct: number | null;
    trailing_stop_pct: number | null;
    atr_stop_k: number | null;
    time_stop_sessions: number | null;
    take_profit_pct: number | null;
    waived: boolean;
  };
  trades: number;
  trades_per_year: number | null;
  churn_pct: number;
  cost_drag: { total_costs_pct_of_equity: number; cagr_points: number | null };
  exposure_pct: number;
  win_rate_pct: number | null;
  avg_win_pct: number | null;
  avg_loss_pct: number | null;
  strategy: StrategyMetrics;
  basket: StrategyMetrics;
  benchmark_spy: StrategyMetrics;
  model_portfolio?: ModelPortfolioSummary;
  explanation?: StrategyExplanation | null;
  in_sample: StrategyMetrics;
  out_of_sample: StrategyMetrics;
  per_ticker_contribution: Record<string, number>;
  top_ticker: { ticker: string; share_pct: number; flag: boolean } | null;
  variants_tried: number;
  chance_sharpe_bar: number | null;
  checks: StrategyCheck[];
  verdict: {
    benchmark: "basket" | "spy";
    excess_cagr_pct: number | null;
    sharpe_vs_basket: number | null;
    sharpe_vs_spy: number | null;
    beats_benchmark_after_costs: boolean;
  };
  trade_log: StrategyTradeRow[];
  equity_curve: { dates: string[]; strategy: number[]; basket: number[] };
  state_warnings: string[];
  walk_forward?: {
    test_windows: number;
    beat_basket_windows: number;
    beat_basket_pct: number | null;
    note: string;
  };
  sensitivity?: {
    step_pct: number;
    base_sharpe: number | null;
    widest_swing: number | null;
    rows: { parameter: string; swing: number | null; cells: { factor: number; value: number; sharpe: number | null }[] }[];
    note: string;
  };
  deflated_sharpe?: { probability: number | null; variants: number; note: string };
  caveats: string[];
  selection?: { source: "hand_picked" | "random_sample"; seed: number | null; size?: number; tickers?: string[] };
  disclaimer: string;
}

export interface StrategyPresetResponse {
  kind: "sp500_sample" | "sector";
  tickers: string[];
  seed?: number;
  sector?: string;
}

export interface StrategySummary {
  total_return_pct: number | null;
  cagr_pct: number | null;
  max_drawdown_pct: number | null;
  sharpe: number | null;
  excess_cagr_vs_basket_pct: number | null;
  top_stock?: string | null;
  top_stock_share_pct?: number | null;
  excess_cagr_vs_spy_pct: number | null;
  sharpe_vs_basket: number | null;
  sharpe_vs_spy: number | null;
  trades: number | null;
  churn_pct: number | null;
  cost_drag_points: number | null;
  deflated_probability: number | null;
  checks_failed: number;
  checks_caution: number;
}

export interface StrategySavedRow {
  id: number;
  name: string;
  created_at: string;
  data_end: string | null;
  summary: StrategySummary;
}

export interface StrategySavedDetail {
  id: number;
  name: string;
  created_at: string;
  definition: Record<string, unknown>;
  result: StrategyBacktestResponse;
  share_token: string | null;
}

export interface StrategyCompareRow extends StrategySummary {
  id: number;
  name: string;
}

export interface StrategySharedResponse {
  name: string;
  created_at: string;
  definition: Record<string, unknown>;
  result: StrategyBacktestResponse;
  read_only: true;
}

// Template scan on a random S&P 500 sample. Candidates to study, not recommendations.
export interface StrategyScanCandidate {
  rank: number;
  key: string;
  name: string;
  holding_class: "short" | "swing" | "long" | "none";
  avg_hold_days: number | null;
  trades_oos: number;
  trades_per_year: number;
  win_rate_pct: number | null;
  avg_trade_net_pct: number | null;
  oos_return_after_costs_pct: number;
  oos_return_vs_basket_pct: number;
  oos_difference_pts: number;
  oos_holding_return_pct: number;
  oos_sharpe: number | null;
  oos_calmar: number | null;
  time_in_market_pct: number | null;
  exposure_adjusted_cagr_pct: number | null;
  passes_short_test: boolean;
  pass_test: "return" | "risk" | null;
  pass_label: string | null;
  oos_difference_ex_top_pts: number | null;
  walk_forward?: {
    windows: number;
    windows_won: number;
    median_difference_pts: number | null;
    worst_window_pts: number | null;
    by_regime_avg_difference_pts: Record<string, number | null>;
  };
  eligibility: "ok" | "too_few_trades" | "never_triggered";
  eligibility_reason: string | null;
  oos_cagr_vs_spy_pct: number | null;
  oos_max_drawdown_pct: number | null;
  spy_oos_max_drawdown_pct: number | null;
  deflated_probability: number | null;
  warnings: string[];
  profile: string;
}

export interface StrategyScanGroup {
  ranked_by: string;
  minimum: string;
  candidates: StrategyScanCandidate[];
  message: string | null;
}

export interface StrategyScanUniverse {
  basis: "point_in_time" | "current_members_biased";
  as_of: string;
  members_at_start: number;
  left_index_in_window: string[];
  sampled?: string[];
  skipped_no_prices?: string[];
  tested?: string[];
}

export interface StrategyScanConcentration {
  stocks: { ticker: string; oos_return_pct: number; contribution_pts: number; top_template_trades: number }[];
  median_stock_return_pct: number | null;
  holding_return_pct: number | null;
  top_contributor: string | null;
  top_contributor_share_pct: number | null;
  holding_without_top_return_pct: number | null;
  warning: string | null;
  top_template: string | null;
}

export interface StrategyScanResult {
  concentration?: StrategyScanConcentration;
  walk_forward?: {
    fit_months: number;
    test_months: number;
    step_months: number;
    rule: string;
    windows: { start: string; end: string; regime: string; spy_return_pct: number; holding_return_pct: number }[];
  };
  downturn?: { peak: string; trough: string; max_drawdown_pct: number } | null;
  sample_size: number;
  tickers: string[];
  universe?: StrategyScanUniverse;
  variants_tried: number;
  seed: number;
  period: { start: string; end: string };
  out_of_sample: { start: string; end: string; years: number };
  holding_oos_return_pct: number;
  holding: { label: string; total_return_pct: number | null; cagr_pct: number | null; volatility_pct: number | null; max_drawdown_pct: number | null; sharpe: number | null; calmar: number | null; time_in_market_pct: number; exposure_adjusted_cagr_pct: number | null };
  benchmark: { name: string; oos_cagr_pct: number | null; oos_max_drawdown_pct: number | null; full_cagr_pct: number | null; full_max_drawdown_pct: number | null };
  downturn_in_test_window: boolean;
  groups: { short_term: StrategyScanGroup; long_term: StrategyScanGroup };
  candidates: StrategyScanCandidate[];
  note: string;
}

export interface StrategyScanStatus {
  status: "running" | "done" | "error";
  done: number;
  total: number;
  seed: number;
  tickers: string[];
  error: string | null;
  result: StrategyScanResult | null;
}

// CHT-7: saved chart layouts for the chart grid.
export interface ChartGridLayout {
  tickers: string[];
  range: "1M" | "6M" | "1Y" | "5Y";
  chart_type: "line" | "candles";
  log_scale: boolean;
  linked_crosshair: boolean;
}

export interface SavedChartLayout {
  id: number;
  name: string;
  layout: ChartGridLayout;
  updated_at: string;
}

// DIF-2: this stock's short-term signal record against SPY.
export interface StockTrackRecord {
  ticker: string;
  enough_data: boolean;
  signals_evaluated: number;
  min_signals: number;
  horizon_sessions: number;
  hit_rate_pct: number | null;
  avg_excess_vs_spy_pct: number | null;
  worst_miss: { as_of_date: string; signal: "Buy" | "Trim"; realized_return_pct: number; excess_vs_spy_pct: number } | null;
  message: string | null;
}
