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
  [key: string]: string | number | null;
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

export interface StrategyPlanProgress {
  months_elapsed: number;
  expected_value: number;
  actual_value: number;
  diff: number;
  diff_pct: number | null;
  on_track: boolean;
}

export interface SavedStrategyPlan {
  id: number;
  name: string | null;
  target_amount: number;
  years: number;
  starting_capital: number;
  annual_return_pct: number;
  monthly_contribution: number;
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

export interface PortfolioInsight {
  ticker: string;
  signal: "BUY" | "SELL" | "HOLD" | null;
  expected_return_pct: number | null;
  target_price: number | null;
  rank: number | null;
  universe_size: number | null;
  trailing_return_pct: number | null;
  weight_pct: number | null;
  concentrated: boolean;
}

export interface PortfolioInsightsResponse {
  positions: PortfolioInsight[];
  concentration_threshold_pct?: number;
  predict_period?: string;
  predict_days_ahead?: number;
  lookback_days?: number;
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
  is_active: boolean;
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
  pit_analyst_rating_capture_enabled: boolean;
  pit_quant_signal_capture_enabled: boolean;
  portfolio_drop_alerts_enabled: boolean;
  portfolio_drop_threshold_pct: number;
  daily_quota: number;
  db_backup_enabled: boolean;
  horizon1_subscriptions_enabled: boolean;
  free_tier_lag_days: number;
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
}

export interface QuantVsAnalystResponse {
  as_of_date: string | null;
  ticker_count: number;
  rows: QuantVsAnalystRow[];
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
