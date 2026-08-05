export interface SignalOut {
  signal: string;
  expected_return_pct: number;
  target_price: number;
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

export interface StockRankRow {
  Ticker: string;
  Name: string;
  Sector: string;
  Price: number;
  Score: number;
  [key: string]: string | number | null;
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
export interface FundRankRow {
  Ticker: string;
  Fund: string;
  Benchmark: string;
  Category: string;
  Price: number;
  Score: number;
  [key: string]: string | number | null;
}

export interface FundRankResponse {
  results: FundRankRow[];
}

export interface FundScoreResponse {
  result: FundRankRow | null;
}

// Best To Enter Now
export interface EntryScanRow {
  Ticker: string;
  Signal: string;
  "Entry Score": number;
  "Current Price": number;
  [key: string]: string | number | null;
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
}

export interface EntryHistory {
  dates: string[];
  close: number[];
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
  implied_monthly: number | null;
  projected_value: number | null;
  score_basis: ScoreFactor[];
}

export interface StrategyPlanRow {
  Strategy: string;
  "Annual Return %": number;
  "Required Monthly Invest": number;
  "Total Contributions": number;
  "Projected Value": number;
}

export interface StrategiesSummaryResponse {
  plan_table: StrategyPlanRow[];
  custom_monthly: number;
  picks: StrategyPickRow[];
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
}

export interface PortfolioSummary {
  total_positions: number;
  total_value: number;
  total_pnl_pct: number;
}

export interface PortfolioPerformanceRow {
  ticker: string;
  shares: number;
  avg_cost: number | null;
  cost_basis: number | null;
  price_now: number | null;
  price_30d_ago: number | null;
  value_now: number | null;
  value_30d_ago: number | null;
  diff: number | null;
  diff_pct: number | null;
  gain_vs_cost: number | null;
  gain_vs_cost_pct: number | null;
  price_unavailable: boolean;
  extended_hours: ExtendedHoursPrice | null;
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
}

export interface ManualPositionInput {
  name: string;
  ticker: string;
  shares: number;
  current_price: number;
  avg_cost: number;
  total_return_pct?: number | null;
}

export interface PortfolioSubmitResponse {
  positions: PortfolioPosition[];
  strategies: PortfolioStrategyRow[];
  summary: PortfolioSummary;
  watchlist_alerts_created: number;
}

// Meta-Agent Chat
export interface ChatProvidersResponse {
  providers: string[];
}

export interface ChatAskResponse {
  ticker: string;
  provider: string;
  answer: string;
}

// Admin — user approvals
export interface AdminUser {
  id: string;
  email: string;
  approved: boolean;
  created_at: string;
}

export interface AdminActivityRow {
  id: number;
  email: string;
  endpoint: string;
  created_at: string;
}

export type AlertConditionType = "price_above" | "price_below";

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
}

export interface PitPriceStatus {
  universe_id: string;
  row_count: number;
  ticker_count: number;
  trading_days_captured: number;
  earliest_date: string | null;
  latest_date: string | null;
  last_captured_at_utc: string | null;
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
  information_coefficient: number | null;
  quintile_spread_pct: number | null;
  outcomes: SignalOutcomeRow[];
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
  slippage_bps: number;
  commission_bps: number;
  borrow_cost_bps_annual: number;
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
