"use client";

import type {
  AdminActivityRow,
  AdminSettings,
  AdminSqlQueryResponse,
  AdminSqlTablesResponse,
  AdminUser,
  AlertConditionType,
  ChatAskResponse,
  ChatProvidersResponse,
  CurrentPriceResponse,
  EntryHistory,
  EntryPlan,
  EntryScanRow,
  FundRankResponse,
  FundScoreResponse,
  ManualPositionInput,
  MomentumBacktestResponse,
  MomentumOptions,
  MonthlyPlanResponse,
  PitPriceStatus,
  PitReconciliationReport,
  PortfolioDropAlert,
  PortfolioPerformance,
  PortfolioPosition,
  PortfolioStrategyRow,
  PortfolioSubmitResponse,
  PortfolioSummary,
  PredictAlgoComparisonResponse,
  PredictionActivity,
  PredictionCompareResponse,
  PredictionNarrative,
  PredictionSummary,
  PublishedSignalsResponse,
  SavedPrediction,
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
export function submitManualPositions(
  positions: ManualPositionInput[],
  riskProfile: string,
  riskFactor: number,
) {
  return apiSend<PortfolioSubmitResponse>("/api/v1/portfolio/manual", "POST", {
    positions,
    risk_profile: riskProfile,
    risk_factor: riskFactor,
  });
}

export async function importPortfolioCsv(
  file: File,
  riskProfile: string,
  riskFactor: number,
): Promise<PortfolioSubmitResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const params = new URLSearchParams({
    risk_profile: riskProfile,
    risk_factor: String(riskFactor),
  });
  return apiUpload<PortfolioSubmitResponse>(`/api/v1/portfolio/import-csv?${params.toString()}`, formData);
}

export function editPortfolioPosition(
  ticker: string,
  shares: number,
  avgCost: number,
  riskProfile: string,
  riskFactor: number,
) {
  return apiSend<PortfolioSubmitResponse>(`/api/v1/portfolio/positions/${encodeURIComponent(ticker)}`, "PUT", {
    shares,
    avg_cost: avgCost,
    risk_profile: riskProfile,
    risk_factor: riskFactor,
  });
}

export function refreshPortfolio(riskProfile: string, riskFactor: number) {
  const params = new URLSearchParams({
    risk_profile: riskProfile,
    risk_factor: String(riskFactor),
  });
  return apiSend<PortfolioSubmitResponse>(`/api/v1/portfolio/refresh?${params.toString()}`, "POST");
}

export function getPortfolioPositions() {
  return apiFetch<{ positions: PortfolioPosition[] }>("/api/v1/portfolio/positions");
}

export function getPortfolioStrategies() {
  return apiFetch<{ strategies: PortfolioStrategyRow[] }>("/api/v1/portfolio/strategies");
}

export function getPortfolioSummary() {
  return apiFetch<{ summary: PortfolioSummary }>("/api/v1/portfolio/summary");
}

export function getPortfolioPerformance(lookbackDays = 30) {
  return apiFetch<PortfolioPerformance>("/api/v1/portfolio/performance", { lookback_days: String(lookbackDays) });
}

export function getPortfolioDropAlerts() {
  return apiFetch<{ alerts: PortfolioDropAlert[] }>("/api/v1/portfolio/drop-alerts");
}

export function dismissPortfolioDropAlert(alertId: number) {
  return apiSend<{ ok: boolean }>(`/api/v1/portfolio/drop-alerts/${alertId}/dismiss`, "POST");
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
) {
  return apiFetch<MomentumBacktestResponse>("/api/v1/momentum/backtest", {
    asset_type: assetType,
    universe,
    lookback_days: String(lookbackDays),
    top_n: String(topN),
    years: String(years),
  });
}

export function getAdminSqlTables() {
  return apiFetch<AdminSqlTablesResponse>("/api/v1/admin/sql/tables");
}

export function runAdminSqlQuery(sql: string) {
  return apiSend<AdminSqlQueryResponse>("/api/v1/admin/sql/query", "POST", { sql });
}
