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
}
