"use client";

import type {
  AdminActivityRow,
  AdminSettings,
  AdminSqlQueryResponse,
  AdminSqlTablesResponse,
  AdminUser,
  AlertConditionType,
  AnalystRatingSummary,
  AuditLogResponse,
  BackupStatus,
  BaselineBand,
  ChatAskResponse,
  ChatProvidersResponse,
  CurrentPriceResponse,
  DemandReport,
  EntryHistory,
  EntryPlan,
  EntryScanRow,
  FundRankResponse,
  FundScoreResponse,
  ManualPositionInput,
  MomentumBacktestResponse,
  MomentumOptions,
  MonthlyPlanResponse,
  MySubscription,
  PitPriceStatus,
  PitReconciliationReport,
  Portfolio,
  PortfolioDropAlert,
  PortfolioInsightsResponse,
  PortfolioListResponse,
  PortfolioPerformance,
  PortfolioPosition,
  PortfolioStrategyRow,
  PortfolioSubmitResponse,
  PortfolioSummary,
  PredictAlgoComparisonResponse,
  PredictionAccuracyLeaderboard,
  PredictionActivity,
  PredictionCompareResponse,
  PredictionNarrative,
  PredictionSummary,
  PublishedSignalsResponse,
  SavedBaselineSnapshot,
  SavedNarrative,
  SavedPrediction,
  SavedScreen,
  SavedStrategyPlan,
  SignalOutcomesResponse,
  StockRankResponse,
  StockScoreResponse,
  StrategiesSummaryResponse,
  TickerSearchResult,
  TopPerformersResponse,
  Trade,
  TradeCreateInput,
  UniversesResponse,
  WatchlistAlert,
} from "@/lib/types";

// Absolute in local dev (http://127.0.0.1:8010); empty once deployed behind
// nginx on the same origin, since every call site below already passes the
// full "/api/v1/..." path — nginx's /api/ location proxies that literal
// path straight to the backend. Do NOT set this to "/api" in production:
// that doubles the prefix into "/api/api/v1/...", a 404.
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function apiFetch<T>(path: string, params?: Record<string, string>): Promise<T> {
  const url = new URL(`${API_BASE}${path}`, window.location.origin);
  if (params) {
    Object.entries(params).forEach(([key, value]) => url.searchParams.set(key, value));
  }

  // Session lives in an httpOnly cookie now (not readable by client JS at
  // all, by design) — credentials: "include" sends it automatically instead
  // of manually attaching an Authorization header from a client-side session read.
  const res = await fetch(url.toString(), { credentials: "include" });

  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // response wasn't JSON — keep the generic message
    }
    throw new ApiError(detail, res.status);
  }

  return res.json();
}

export function getPredictionSummary(ticker: string, period: string, daysAhead = 10) {
  return apiFetch<PredictionSummary>("/api/v1/predict/summary", {
    ticker,
    period,
    days_ahead: String(daysAhead),
  });
}

export function getPredictionNarrative(ticker: string, period: string, provider?: string, daysAhead = 10) {
  const params: Record<string, string> = { ticker, period, days_ahead: String(daysAhead) };
  if (provider) params.provider = provider;
  return apiFetch<PredictionNarrative>("/api/v1/predict/narrative", params);
}

export function getPredictionActivity(ticker: string) {
  return apiFetch<PredictionActivity>("/api/v1/predict/activity", { ticker });
}

export function saveNarrative(narrative: {
  ticker: string;
  provider: string;
  period: string;
  days_ahead: number;
  narrative: string;
  sentiment_context: string;
}) {
  return apiSend<{ narrative: SavedNarrative }>("/api/v1/predict/narrative/save", "POST", narrative);
}

export function getNarrativeHistory(ticker?: string) {
  return apiFetch<{ narratives: SavedNarrative[] }>(
    "/api/v1/predict/narrative/history",
    ticker ? { ticker } : undefined,
  );
}

export function deleteNarrative(narrativeId: number) {
  return apiSend<{ ok: boolean }>(`/api/v1/predict/narrative/${narrativeId}`, "DELETE");
}

export function savePrediction(ticker: string, period: string, daysAhead = 10) {
  return apiSend<{ prediction: SavedPrediction }>("/api/v1/predict/save", "POST", {
    ticker,
    period,
    days_ahead: daysAhead,
  });
}

export function getPredictionHistory(ticker?: string) {
  return apiFetch<{ predictions: SavedPrediction[] }>(
    "/api/v1/predict/history",
    ticker ? { ticker } : undefined,
  );
}

export function getPredictionAccuracyLeaderboard() {
  return apiFetch<PredictionAccuracyLeaderboard>("/api/v1/predict/accuracy-leaderboard");
}

export function getStockUniverses() {
  return apiFetch<UniversesResponse>("/api/v1/stock-finder/universes");
}

export function searchTickers(q: string) {
  return apiFetch<{ results: TickerSearchResult[] }>("/api/v1/search/tickers", { q });
}

export function getCurrentPrice(ticker: string) {
  return apiFetch<CurrentPriceResponse>("/api/v1/search/price", { ticker });
}

export function getStockRanking(goal: string, universe: string) {
  return apiFetch<StockRankResponse>("/api/v1/stock-finder/rank", { goal, universe });
}

export function getStockScore(goal: string, ticker: string) {
  return apiFetch<StockScoreResponse>("/api/v1/stock-finder/score", { goal, ticker });
}

export function getAnalystRating(ticker: string) {
  return apiFetch<AnalystRatingSummary>("/api/v1/stock-finder/analyst", { ticker });
}

export function saveScreen(screen: {
  name: string;
  goal: string;
  universe: string;
  filters: Record<string, unknown>;
  visible_columns: string[];
  sort_keys: { column: string; direction: "asc" | "desc" }[];
  snapshot_top10: { Ticker: string; Score: number; Price: number }[];
}) {
  return apiSend<{ screen: SavedScreen }>("/api/v1/stock-finder/screens/save", "POST", screen);
}

export function getScreens() {
  return apiFetch<{ screens: SavedScreen[] }>("/api/v1/stock-finder/screens");
}

export function deleteScreen(screenId: number) {
  return apiSend<{ ok: boolean }>(`/api/v1/stock-finder/screens/${screenId}`, "DELETE");
}

// Index Fund Finder
export function getFundGoals() {
  return apiFetch<{ goals: string[] }>("/api/v1/index-fund/goals");
}

export function getFundCategories() {
  return apiFetch<{ categories: string[] }>("/api/v1/index-fund/categories");
}

export function getFundRanking(goal: string, category: string) {
  return apiFetch<FundRankResponse>("/api/v1/index-fund/rank", { goal, category });
}

export function getFundScore(goal: string, ticker: string) {
  return apiFetch<FundScoreResponse>("/api/v1/index-fund/score", { goal, ticker });
}

// Best To Enter Now
export function getEntryUniverses(assetType: string) {
  return apiFetch<{ universes: string[] }>("/api/v1/entry/universes", { asset_type: assetType });
}

export function getEntryScan(assetType: string, universe: string, topN: number) {
  return apiFetch<{ results: EntryScanRow[] }>("/api/v1/entry/scan", {
    asset_type: assetType,
    universe,
    top_n: String(topN),
  });
}

export function getEntryPlan(ticker: string) {
  return apiFetch<{ plan: EntryPlan | null; history: EntryHistory | null }>("/api/v1/entry/plan", { ticker });
}

// Safe Baseline Price Band
export function getBaselineBand(
  ticker: string,
  options?: { horizon?: number; confidence?: number; method?: "empirical" | "sqrt" },
) {
  const params: Record<string, string> = {};
  if (options?.horizon !== undefined) params.horizon = String(options.horizon);
  if (options?.confidence !== undefined) params.confidence = String(options.confidence);
  if (options?.method !== undefined) params.method = options.method;
  return apiFetch<BaselineBand>(`/api/v1/baseline/${encodeURIComponent(ticker)}`, params);
}

export function saveBaselineSnapshot(band: BaselineBand) {
  return apiSend<{ snapshot: SavedBaselineSnapshot }>("/api/v1/baseline/save", "POST", {
    ticker: band.ticker,
    horizon_days: band.horizon_days,
    confidence: band.confidence,
    method: band.method,
    as_of: band.as_of,
    last_price: band.last_price,
    floor: band.floor,
    floor_pct: band.floor_pct,
    accumulation_zone_hi: band.accumulation_zone_hi,
    accumulation_zone_hi_pct: band.accumulation_zone_hi_pct,
    median_path: band.median_path,
    distribution_zone_lo: band.distribution_zone_lo,
    distribution_zone_lo_pct: band.distribution_zone_lo_pct,
    ceiling: band.ceiling,
    ceiling_pct: band.ceiling_pct,
    samples: band.samples,
    effective_samples: band.effective_samples,
    breach_rate_full: band.breach_rate_full,
  });
}

export function getBaselineHistory(ticker?: string) {
  return apiFetch<{ snapshots: SavedBaselineSnapshot[] }>(
    "/api/v1/baseline/history",
    ticker ? { ticker } : undefined,
  );
}

export function deleteBaselineSnapshot(snapshotId: number) {
  return apiSend<{ ok: boolean }>(`/api/v1/baseline/snapshot/${snapshotId}`, "DELETE");
}

// Monthly Investing Plan
export function getMonthlyPlanOptions() {
  return apiFetch<{
    fund_goals: string[];
    fund_categories: string[];
    stock_goals: string[];
    stock_universes: string[];
  }>("/api/v1/monthly-plan/options");
}

export function getMonthlyPlanSummary(
  assetType: string,
  goal: string,
  selection: string,
  monthlyAmount: number,
  years: number,
) {
  return apiFetch<MonthlyPlanResponse>("/api/v1/monthly-plan/summary", {
    asset_type: assetType,
    goal,
    selection,
    monthly_amount: String(monthlyAmount),
    years: String(years),
  });
}

// Strategies
export function getStrategiesOptions() {
  return apiFetch<{
    fund_goals: string[];
    fund_categories: string[];
    stock_goals: string[];
    stock_universes: string[];
    defaults: { target_amount: number; years: number };
  }>("/api/v1/strategies/options");
}

export function getStrategiesSummary(params: Record<string, string>) {
  return apiFetch<StrategiesSummaryResponse>("/api/v1/strategies/summary", params);
}

export function saveStrategyPlan(body: {
  name?: string;
  target_amount: number;
  years: number;
  starting_capital: number;
  annual_return_pct: number;
}) {
  return apiSend<SavedStrategyPlan>("/api/v1/strategies/plans", "POST", body);
}

export function getStrategyPlans() {
  return apiFetch<{ plans: SavedStrategyPlan[] }>("/api/v1/strategies/plans");
}

export function deleteStrategyPlan(planId: number) {
  return apiSend<{ ok: boolean }>(`/api/v1/strategies/plans/${planId}`, "DELETE");
}

async function apiSend<T>(
  path: string,
  method: "POST" | "PUT" | "DELETE",
  body?: unknown,
): Promise<T> {
  const res = await fetch(new URL(`${API_BASE}${path}`, window.location.origin).toString(), {
    method,
    credentials: "include",
    headers: body !== undefined ? { "Content-Type": "application/json" } : {},
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });

  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const errBody = await res.json();
      detail = errBody.detail || detail;
    } catch {
      // response wasn't JSON — keep the generic message
    }
    throw new ApiError(detail, res.status);
  }

  return res.json();
}

async function apiUpload<T>(path: string, formData: FormData): Promise<T> {
  const res = await fetch(new URL(`${API_BASE}${path}`, window.location.origin).toString(), {
    method: "POST",
    credentials: "include",
    body: formData,
  });

  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const errBody = await res.json();
      detail = errBody.detail || detail;
    } catch {
      // response wasn't JSON — keep the generic message
    }
    throw new ApiError(detail, res.status);
  }

  return res.json();
}

// Trade Journal
export function listTrades() {
  return apiFetch<{ trades: Trade[] }>("/api/v1/trades");
}

export function createTrade(body: TradeCreateInput) {
  return apiSend<{ trade: Trade }>("/api/v1/trades", "POST", body);
}

export function evaluateTrades() {
  return apiSend<{ trades: Trade[] }>("/api/v1/trades/evaluate", "POST");
}

export function deleteTrade(tradeId: string) {
  return apiSend<{ deleted: string }>(`/api/v1/trades/${encodeURIComponent(tradeId)}`, "DELETE");
}

// Portfolio
export function getPortfolios() {
  return apiFetch<PortfolioListResponse>("/api/v1/portfolio/list");
}

export function createPortfolio(name: string) {
  return apiSend<Portfolio>("/api/v1/portfolio/create", "POST", { name });
}

export function submitManualPositions(
  positions: ManualPositionInput[],
  riskProfile: string,
  riskFactor: number,
  portfolioId?: number,
) {
  return apiSend<PortfolioSubmitResponse>("/api/v1/portfolio/manual", "POST", {
    positions,
    risk_profile: riskProfile,
    risk_factor: riskFactor,
    portfolio_id: portfolioId,
  });
}

export async function importPortfolioCsv(
  file: File,
  riskProfile: string,
  riskFactor: number,
  portfolioId?: number,
): Promise<PortfolioSubmitResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const params = new URLSearchParams({
    risk_profile: riskProfile,
    risk_factor: String(riskFactor),
  });
  if (portfolioId !== undefined) params.set("portfolio_id", String(portfolioId));
  return apiUpload<PortfolioSubmitResponse>(`/api/v1/portfolio/import-csv?${params.toString()}`, formData);
}

export function editPortfolioPosition(
  ticker: string,
  shares: number,
  avgCost: number,
  riskProfile: string,
  riskFactor: number,
  portfolioId?: number,
) {
  return apiSend<PortfolioSubmitResponse>(`/api/v1/portfolio/positions/${encodeURIComponent(ticker)}`, "PUT", {
    shares,
    avg_cost: avgCost,
    risk_profile: riskProfile,
    risk_factor: riskFactor,
    portfolio_id: portfolioId,
  });
}

export function deletePortfolioPosition(ticker: string, portfolioId?: number) {
  const params = portfolioId !== undefined ? `?portfolio_id=${portfolioId}` : "";
  return apiSend<{ ok: boolean }>(`/api/v1/portfolio/positions/${encodeURIComponent(ticker)}${params}`, "DELETE");
}

export function movePortfolioPosition(
  ticker: string,
  toPortfolioId: number,
  riskProfile: string,
  riskFactor: number,
  fromPortfolioId?: number,
) {
  return apiSend<PortfolioSubmitResponse>(`/api/v1/portfolio/positions/${encodeURIComponent(ticker)}/move`, "POST", {
    to_portfolio_id: toPortfolioId,
    from_portfolio_id: fromPortfolioId,
    risk_profile: riskProfile,
    risk_factor: riskFactor,
  });
}

export function refreshPortfolio(riskProfile: string, riskFactor: number, portfolioId?: number) {
  const params = new URLSearchParams({
    risk_profile: riskProfile,
    risk_factor: String(riskFactor),
  });
  if (portfolioId !== undefined) params.set("portfolio_id", String(portfolioId));
  return apiSend<PortfolioSubmitResponse>(`/api/v1/portfolio/refresh?${params.toString()}`, "POST");
}

export function getPortfolioPositions(portfolioId?: number) {
  return apiFetch<{ positions: PortfolioPosition[] }>(
    "/api/v1/portfolio/positions",
    portfolioId !== undefined ? { portfolio_id: String(portfolioId) } : undefined,
  );
}

export function getPortfolioStrategies(portfolioId?: number) {
  return apiFetch<{ strategies: PortfolioStrategyRow[] }>(
    "/api/v1/portfolio/strategies",
    portfolioId !== undefined ? { portfolio_id: String(portfolioId) } : undefined,
  );
}

export function getPortfolioSummary(portfolioId?: number) {
  return apiFetch<{ summary: PortfolioSummary }>(
    "/api/v1/portfolio/summary",
    portfolioId !== undefined ? { portfolio_id: String(portfolioId) } : undefined,
  );
}

export function getPortfolioPerformance(lookbackDays = 30, portfolioId?: number) {
  const params: Record<string, string> = { lookback_days: String(lookbackDays) };
  if (portfolioId !== undefined) params.portfolio_id = String(portfolioId);
  return apiFetch<PortfolioPerformance>("/api/v1/portfolio/performance", params);
}

export function getPortfolioInsights(portfolioId?: number) {
  return apiFetch<PortfolioInsightsResponse>(
    "/api/v1/portfolio/insights",
    portfolioId !== undefined ? { portfolio_id: String(portfolioId) } : undefined,
  );
}

export function getPortfolioDropAlerts() {
  return apiFetch<{ alerts: PortfolioDropAlert[] }>("/api/v1/portfolio/drop-alerts");
}

export function dismissPortfolioDropAlert(alertId: number) {
  return apiSend<{ ok: boolean }>(`/api/v1/portfolio/drop-alerts/${alertId}/dismiss`, "POST");
}

export function refreshPortfolioDropAlerts() {
  return apiSend<{ inserted: number }>("/api/v1/portfolio/drop-alerts/refresh", "POST");
}

export function refreshDropAlert(alertId: number) {
  return apiSend<{ alert: PortfolioDropAlert }>(`/api/v1/portfolio/drop-alerts/${alertId}/refresh`, "POST");
}

// Meta-Agent Chat
export function getChatProviders() {
  return apiFetch<ChatProvidersResponse>("/api/v1/chat/providers");
}

export function askMetaAgent(ticker: string, question: string, provider?: string) {
  return apiSend<ChatAskResponse>("/api/v1/chat/ask", "POST", { ticker, question, provider });
}

// Admin — user approvals
export function getAdminUsers() {
  return apiFetch<AdminUser[]>("/api/v1/admin/users");
}

export function approveUser(userId: string) {
  return apiSend<AdminUser>(`/api/v1/admin/users/${userId}/approve`, "POST");
}

export function rejectUser(userId: string) {
  return apiSend<{ ok: boolean }>(`/api/v1/admin/users/${userId}/reject`, "POST");
}

export function deleteUser(userId: string) {
  return apiSend<{ ok: boolean }>(`/api/v1/admin/users/${userId}`, "DELETE");
}

export function deactivateUser(userId: string) {
  return apiSend<{ id: string; email: string; is_active: boolean }>(
    `/api/v1/admin/users/${userId}/deactivate`,
    "POST",
  );
}

export function reactivateUser(userId: string) {
  return apiSend<{ id: string; email: string; is_active: boolean }>(
    `/api/v1/admin/users/${userId}/reactivate`,
    "POST",
  );
}

export function sendWelcomeEmail(userId: string) {
  return apiSend<{ ok: boolean; email: string }>(`/api/v1/admin/users/${userId}/send-welcome-email`, "POST");
}

export function getAdminActivity(limit = 200) {
  return apiFetch<AdminActivityRow[]>("/api/v1/admin/activity", { limit: String(limit) });
}

export function getWatchlistAlerts() {
  return apiFetch<WatchlistAlert[]>("/api/v1/watchlist");
}

export function createWatchlistAlert(ticker: string, condition_type: AlertConditionType, threshold: number) {
  return apiSend<WatchlistAlert>("/api/v1/watchlist", "POST", { ticker, condition_type, threshold });
}

export function deleteWatchlistAlert(alertId: number) {
  return apiSend<{ ok: boolean }>(`/api/v1/watchlist/${alertId}`, "DELETE");
}

export function dismissWatchlistAlert(alertId: number) {
  return apiSend<{ ok: boolean }>(`/api/v1/watchlist/${alertId}/dismiss`, "POST");
}

export function getAdminSettings() {
  return apiFetch<AdminSettings>("/api/v1/admin/settings");
}

export function enableVerifyPredictions() {
  return apiSend<AdminSettings>("/api/v1/admin/settings/verify-predictions/enable", "POST");
}

export function disableVerifyPredictions() {
  return apiSend<AdminSettings>("/api/v1/admin/settings/verify-predictions/disable", "POST");
}

export function enablePublishSignals() {
  return apiSend<AdminSettings>("/api/v1/admin/settings/publish-signals/enable", "POST");
}

export function disablePublishSignals() {
  return apiSend<AdminSettings>("/api/v1/admin/settings/publish-signals/disable", "POST");
}

export function enablePasswordPolicy() {
  return apiSend<AdminSettings>("/api/v1/admin/settings/password-policy/enable", "POST");
}

export function disablePasswordPolicy() {
  return apiSend<AdminSettings>("/api/v1/admin/settings/password-policy/disable", "POST");
}

export function enablePitPriceCapture() {
  return apiSend<AdminSettings>("/api/v1/admin/settings/pit-price-capture/enable", "POST");
}

export function disablePitPriceCapture() {
  return apiSend<AdminSettings>("/api/v1/admin/settings/pit-price-capture/disable", "POST");
}

export function getPitPriceStatus() {
  return apiFetch<PitPriceStatus>("/api/v1/pit-prices/status");
}

export function capturePitPricesNow() {
  return apiSend<{
    prices_inserted: number;
    universe_membership_inserted: number;
    fundamentals_inserted: number;
  }>("/api/v1/pit-prices/capture-now", "POST");
}

export function enablePitAnalystRatingCapture() {
  return apiSend<AdminSettings>("/api/v1/admin/settings/pit-analyst-rating-capture/enable", "POST");
}

export function disablePitAnalystRatingCapture() {
  return apiSend<AdminSettings>("/api/v1/admin/settings/pit-analyst-rating-capture/disable", "POST");
}

export function capturePitAnalystRatingsNow() {
  return apiSend<{ analyst_rating_inserted: number }>("/api/v1/pit-prices/capture-now-analyst-ratings", "POST");
}

export function enablePitQuantSignalCapture() {
  return apiSend<AdminSettings>("/api/v1/admin/settings/pit-quant-signal-capture/enable", "POST");
}

export function disablePitQuantSignalCapture() {
  return apiSend<AdminSettings>("/api/v1/admin/settings/pit-quant-signal-capture/disable", "POST");
}

export function capturePitQuantSignalsNow() {
  return apiSend<{ quant_signal_inserted: number }>("/api/v1/pit-prices/capture-now-quant-signals", "POST");
}

export function getPitReconciliation(targetDate: string, universeId = "All", lookbackDays = 30, topN = 25) {
  return apiFetch<PitReconciliationReport>("/api/v1/pit-prices/reconcile", {
    target_date: targetDate,
    universe_id: universeId,
    lookback_days: String(lookbackDays),
    top_n: String(topN),
  });
}

export function publishSignalsNow() {
  return apiSend<{ published: number }>("/api/v1/signals/publish-now", "POST");
}

export function checkPublicationAlertNow(checkpoint: "nfr01" | "nfr02", force: boolean) {
  return apiSend<{ alert_sent: boolean; reason: string }>(
    `/api/v1/signals/check-publication-alert?checkpoint=${checkpoint}&force=${force}`,
    "POST",
  );
}

export function enablePortfolioDropAlerts() {
  return apiSend<AdminSettings>("/api/v1/admin/settings/portfolio-drop-alerts/enable", "POST");
}

export function disablePortfolioDropAlerts() {
  return apiSend<AdminSettings>("/api/v1/admin/settings/portfolio-drop-alerts/disable", "POST");
}

export function setPortfolioDropThreshold(thresholdPct: number) {
  return apiSend<{ portfolio_drop_threshold_pct: number }>(
    "/api/v1/admin/settings/portfolio-drop-alerts/threshold",
    "POST",
    { threshold_pct: thresholdPct },
  );
}

export function scanPortfolioDropAlertsNow() {
  return apiSend<{ inserted: number }>("/api/v1/portfolio/drop-alerts/scan-now", "POST");
}

export function setDailyQuota(dailyQuota: number) {
  return apiSend<{ daily_quota: number }>("/api/v1/admin/settings/daily-quota", "POST", { daily_quota: dailyQuota });
}

export function enableDbBackup() {
  return apiSend<AdminSettings>("/api/v1/admin/settings/db-backup/enable", "POST");
}

export function disableDbBackup() {
  return apiSend<AdminSettings>("/api/v1/admin/settings/db-backup/disable", "POST");
}

export function getBackupStatus() {
  return apiFetch<BackupStatus>("/api/v1/db-backup/status");
}

export function backupNow() {
  return apiSend<{ id: number; s3_key: string | null; structural_check_passed: boolean; error: string | null }>(
    "/api/v1/db-backup/backup-now", "POST",
  );
}

export function restoreTestNow(s3Key?: string) {
  const query = s3Key ? `?s3_key=${encodeURIComponent(s3Key)}` : "";
  return apiSend<{
    restore_succeeded: boolean;
    all_match: boolean;
    row_counts: Record<string, { restored: number; live: number; match: boolean }>;
    s3_key?: string;
    error?: string;
  }>(`/api/v1/db-backup/restore-test-now${query}`, "POST");
}

export function getPublishedSignals(params?: { targetDate?: string; universeId?: string; lookbackDays?: number }) {
  const query: Record<string, string> = {};
  if (params?.targetDate) query.target_date = params.targetDate;
  if (params?.universeId) query.universe_id = params.universeId;
  if (params?.lookbackDays) query.lookback_days = String(params.lookbackDays);
  return apiFetch<PublishedSignalsResponse>("/api/v1/signals/published", query);
}

export function getPredictAlgoComparison(daysAhead = 30) {
  return apiFetch<PredictAlgoComparisonResponse>("/api/v1/signals/published/compare-to-predict-algo", {
    days_ahead: String(daysAhead),
  });
}

export function getSignalOutcomes() {
  return apiFetch<SignalOutcomesResponse>("/api/v1/signals/outcomes");
}

export function comparePredictionsToFund(fundGoal = "Balanced Core", fundCategory = "All") {
  return apiFetch<PredictionCompareResponse>("/api/v1/predict/compare", {
    fund_goal: fundGoal,
    fund_category: fundCategory,
  });
}

export function deletePrediction(predictionId: number) {
  return apiSend<{ ok: boolean }>(`/api/v1/predict/${predictionId}`, "DELETE");
}

export function getMomentumOptions() {
  return apiFetch<MomentumOptions>("/api/v1/momentum/options");
}

export function getTopPerformers(window: number, assetType: "Stock" | "Fund", universe: string, topN = 15) {
  return apiFetch<TopPerformersResponse>("/api/v1/momentum/top-performers", {
    window: String(window),
    asset_type: assetType,
    universe,
    top_n: String(topN),
  });
}

export function getMomentumBacktest(
  assetType: "Stock" | "Fund",
  universe: string,
  lookbackDays: number,
  topN = 5,
  years = 3,
  horizonDays = 30,
) {
  return apiFetch<MomentumBacktestResponse>("/api/v1/momentum/backtest", {
    asset_type: assetType,
    universe,
    lookback_days: String(lookbackDays),
    top_n: String(topN),
    years: String(years),
    horizon_days: String(horizonDays),
  });
}

export function getAdminSqlTables() {
  return apiFetch<AdminSqlTablesResponse>("/api/v1/admin/sql/tables");
}

// Horizon 1 — Impersonal Research Subscription (built, kept off)

export function enableHorizon1Subscriptions() {
  return apiSend<AdminSettings>("/api/v1/admin/settings/horizon1-subscriptions/enable", "POST");
}

export function disableHorizon1Subscriptions() {
  return apiSend<AdminSettings>("/api/v1/admin/settings/horizon1-subscriptions/disable", "POST");
}

export function setFreeTierLagDays(days: number) {
  return apiSend<{ free_tier_lag_days: number }>("/api/v1/admin/settings/free-tier-lag-days", "POST", {
    free_tier_lag_days: days,
  });
}

export function getMySubscription() {
  return apiFetch<MySubscription>("/api/v1/subscriptions/me");
}

export function startCheckout() {
  return apiSend<{ url: string }>("/api/v1/subscriptions/checkout", "POST");
}

export function openBillingPortal() {
  return apiSend<{ url: string }>("/api/v1/subscriptions/portal", "POST");
}

export function submitEnquiry(body: { enquiry_type: string; contact_email: string; message: string }) {
  return apiSend<{ ok: boolean }>("/api/v1/subscriptions/enquiry", "POST", body);
}

export function getSignalsCsvExportUrl(universeId?: string, lookbackDays?: number): string {
  const params = new URLSearchParams();
  if (universeId) params.set("universe_id", universeId);
  if (lookbackDays) params.set("lookback_days", String(lookbackDays));
  const query = params.toString();
  return `${API_BASE}/api/v1/subscriptions/export/csv${query ? `?${query}` : ""}`;
}

export function getDemandReport() {
  return apiFetch<DemandReport>("/api/v1/subscriptions/demand-report");
}

export function getAuditLog(params?: { eventType?: string; actorUserId?: string; since?: string; limit?: number; offset?: number }) {
  const query: Record<string, string> = {};
  if (params?.eventType) query.event_type = params.eventType;
  if (params?.actorUserId) query.actor_user_id = params.actorUserId;
  if (params?.since) query.since = params.since;
  if (params?.limit) query.limit = String(params.limit);
  if (params?.offset) query.offset = String(params.offset);
  return apiFetch<AuditLogResponse>("/api/v1/subscriptions/audit-log", query);
}

export function runAdminSqlQuery(sql: string) {
  return apiSend<AdminSqlQueryResponse>("/api/v1/admin/sql/query", "POST", { sql });
}
