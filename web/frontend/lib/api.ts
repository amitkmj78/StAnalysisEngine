"use client";

import type {
  AdminUser,
  ChatAskResponse,
  ChatProvidersResponse,
  EntryHistory,
  EntryPlan,
  EntryScanRow,
  FundRankResponse,
  FundScoreResponse,
  ManualPositionInput,
  MonthlyPlanResponse,
  PortfolioPosition,
  PortfolioStrategyRow,
  PortfolioSubmitResponse,
  PortfolioSummary,
  PredictionNarrative,
  PredictionSummary,
  SavedPrediction,
  StockRankResponse,
  StockScoreResponse,
  StrategiesSummaryResponse,
  TickerSearchResult,
  Trade,
  TradeCreateInput,
  UniversesResponse,
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

export function getPredictionSummary(ticker: string, period: string) {
  return apiFetch<PredictionSummary>("/api/v1/predict/summary", { ticker, period });
}

export function getPredictionNarrative(ticker: string, period: string, provider?: string) {
  const params: Record<string, string> = { ticker, period };
  if (provider) params.provider = provider;
  return apiFetch<PredictionNarrative>("/api/v1/predict/narrative", params);
}

export function savePrediction(ticker: string, period: string) {
  return apiSend<{ prediction: SavedPrediction }>("/api/v1/predict/save", "POST", { ticker, period });
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

async function apiSend<T>(
  path: string,
  method: "POST" | "DELETE",
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

export function getPortfolioPositions() {
  return apiFetch<{ positions: PortfolioPosition[] }>("/api/v1/portfolio/positions");
}

export function getPortfolioStrategies() {
  return apiFetch<{ strategies: PortfolioStrategyRow[] }>("/api/v1/portfolio/strategies");
}

export function getPortfolioSummary() {
  return apiFetch<{ summary: PortfolioSummary }>("/api/v1/portfolio/summary");
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
