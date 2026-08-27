"use client";

import { useEffect, useState } from "react";

import {
  ApiError,
  deletePortfolioPosition,
  editPortfolioPosition,
  getCurrentPrice,
  getPortfolioInsights,
  getPortfolioPerformance,
  getPortfolioStrategies,
  getPortfolioSummary,
  importPortfolioCsv,
  movePortfolioPosition,
  refreshPortfolio,
  submitManualPositions,
} from "@/lib/api";
import type {
  ManualPositionInput,
  Portfolio,
  PortfolioInsight,
  PortfolioPerformance,
  PortfolioStrategyRow,
  PortfolioSummary,
} from "@/lib/types";
import PlanText from "@/components/PlanText";
import GoalPlan from "@/components/GoalPlan";
import PortfolioDropAlerts from "@/components/PortfolioDropAlerts";
import PortfolioSwitcher from "@/components/PortfolioSwitcher";
import TickerSearchInput from "@/components/TickerSearchInput";
import CurrentPriceBadge from "@/components/CurrentPriceBadge";
import InfoModal, { type ColumnInfo } from "@/components/InfoModal";

const RISK_PROFILES = ["Conservative", "Balanced", "Aggressive"];

type Mode = "manual" | "csv";

const EMPTY_ROW: ManualPositionInput = { name: "", ticker: "", shares: 0, current_price: 0, avg_cost: 0 };

const PERFORMANCE_COLUMN_INFO: Record<string, ColumnInfo> = {
  Ticker: {
    title: "Ticker",
    body: [
      "The position's stock/fund symbol. A percentage badge next to it means this position is concentrated — it makes up a large enough share of your portfolio's total value that it's driving most of the swings.",
    ],
  },
  Signal: {
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
  "Next-Day Forecast": {
    title: "Next-Day Forecast",
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
      "The latest trade price. When the market is in pre-market or after-hours, an extra line shows that session's price and percent change separately from the regular-session price above it.",
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
};

export default function PortfolioPage() {
  const [selectedPortfolioId, setSelectedPortfolioId] = useState<number | null>(null);
  const [allPortfolios, setAllPortfolios] = useState<Portfolio[]>([]);
  const [portfolioReloadSignal, setPortfolioReloadSignal] = useState(0);
  const [mode, setMode] = useState<Mode>("manual");
  const [riskProfile, setRiskProfile] = useState("Balanced");
  const [riskFactor, setRiskFactor] = useState(5);

  const [rows, setRows] = useState<ManualPositionInput[]>([{ ...EMPTY_ROW }]);
  const [file, setFile] = useState<File | null>(null);
  const [priceFetchingRow, setPriceFetchingRow] = useState<number | null>(null);

  const [strategies, setStrategies] = useState<PortfolioStrategyRow[]>([]);
  const [summary, setSummary] = useState<PortfolioSummary | null>(null);
  const [performance, setPerformance] = useState<PortfolioPerformance | null>(null);
  const [performanceError, setPerformanceError] = useState<string | null>(null);
  const [performanceLoading, setPerformanceLoading] = useState(false);
  const [insights, setInsights] = useState<PortfolioInsight[]>([]);
  const [insightsError, setInsightsError] = useState<string | null>(null);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [performanceInfoColumn, setPerformanceInfoColumn] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [watchlistNote, setWatchlistNote] = useState<string | null>(null);

  const [editingTicker, setEditingTicker] = useState<string | null>(null);
  const [editShares, setEditShares] = useState("");
  const [editAvgCost, setEditAvgCost] = useState("");
  const [editSaving, setEditSaving] = useState(false);
  const [editError, setEditError] = useState<string | null>(null);

  const [addTicker, setAddTicker] = useState("");
  const [addShares, setAddShares] = useState("");
  const [addAvgCost, setAddAvgCost] = useState("");
  const [adding, setAdding] = useState(false);
  const [addError, setAddError] = useState<string | null>(null);

  const [deletingTicker, setDeletingTicker] = useState<string | null>(null);
  const [movingTicker, setMovingTicker] = useState<string | null>(null);
  const [moveTargetId, setMoveTargetId] = useState("");
  const [moveSaving, setMoveSaving] = useState(false);
  const [positionActionError, setPositionActionError] = useState<string | null>(null);

  async function refreshPerformance(showLoading: boolean) {
    if (showLoading) setPerformanceLoading(true);
    setPerformanceError(null);
    try {
      setPerformance(await getPortfolioPerformance(30, selectedPortfolioId ?? undefined));
    } catch (err) {
      setPerformanceError(err instanceof ApiError ? err.message : "Could not load 30-day performance.");
    } finally {
      if (showLoading) setPerformanceLoading(false);
    }
  }

  async function refreshInsights() {
    setInsightsLoading(true);
    setInsightsError(null);
    try {
      const res = await getPortfolioInsights(selectedPortfolioId ?? undefined);
      setInsights(res.positions);
    } catch (err) {
      setInsightsError(err instanceof ApiError ? err.message : "Could not load signal/rank data for your holdings.");
    } finally {
      setInsightsLoading(false);
    }
  }

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const [stratRes, summaryRes] = await Promise.all([
        getPortfolioStrategies(selectedPortfolioId ?? undefined),
        getPortfolioSummary(selectedPortfolioId ?? undefined),
      ]);
      setStrategies(stratRes.strategies);
      setSummary(summaryRes.summary);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load portfolio.");
    } finally {
      setLoading(false);
    }

    await refreshPerformance(true);
    await refreshInsights();
  }

  // Waits for PortfolioSwitcher to resolve which portfolio is selected
  // (on mount, and again any time the user switches or creates one)
  // before loading anything portfolio-scoped.
  useEffect(() => {
    if (selectedPortfolioId !== null) {
      refresh();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedPortfolioId]);

  // Live-ish prices without per-position polling: one batched call for the
  // whole portfolio every 10s (well under /performance's 15/min rate limit
  // regardless of how many positions exist), instead of each position
  // polling independently the way CurrentPriceBadge does elsewhere.
  useEffect(() => {
    if (summary === null || summary.total_positions === 0) return;
    const interval = setInterval(() => refreshPerformance(false), 10000);
    return () => clearInterval(interval);
  }, [summary?.total_positions]);

  function updateRow(i: number, patch: Partial<ManualPositionInput>) {
    setRows((prev) => prev.map((r, idx) => (idx === i ? { ...r, ...patch } : r)));
  }

  function addRow() {
    setRows((prev) => [...prev, { ...EMPTY_ROW }]);
  }

  function removeRow(i: number) {
    setRows((prev) => prev.filter((_, idx) => idx !== i));
  }

  async function populateCurrentPrice(i: number, ticker: string) {
    const trimmed = ticker.trim().toUpperCase();
    if (!trimmed) return;
    setPriceFetchingRow(i);
    try {
      const res = await getCurrentPrice(trimmed);
      if (res.price !== null) {
        updateRow(i, { current_price: res.price });
      }
    } catch {
      // Lookup failed (bad ticker, no data) — leave whatever's there so the
      // user can still type a price in by hand, same as before this existed.
    } finally {
      setPriceFetchingRow((cur) => (cur === i ? null : cur));
    }
  }

  async function submitManual(e: React.FormEvent) {
    e.preventDefault();
    const valid = rows.filter((r) => r.ticker.trim() && r.shares > 0);
    if (valid.length === 0) {
      setError("Add at least one position with a ticker and share count.");
      return;
    }
    setSubmitting(true);
    setError(null);
    setWatchlistNote(null);
    try {
      const res = await submitManualPositions(
        valid.map((r) => ({ ...r, ticker: r.ticker.trim().toUpperCase() })),
        riskProfile,
        riskFactor,
        selectedPortfolioId ?? undefined,
      );
      setRows([{ ...EMPTY_ROW }]);
      noteWatchlist(res.watchlist_alerts_created);
      await refresh();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not save positions.");
    } finally {
      setSubmitting(false);
    }
  }

  function noteWatchlist(count: number) {
    if (count > 0) {
      setWatchlistNote(
        `${count} watchlist alert${count === 1 ? "" : "s"} set from your strategies' upside targets and stops.`
      );
    }
  }

  async function handleRefresh() {
    setRefreshing(true);
    setError(null);
    setWatchlistNote(null);
    try {
      const res = await refreshPortfolio(riskProfile, riskFactor, selectedPortfolioId ?? undefined);
      noteWatchlist(res.watchlist_alerts_created);
      await refresh();
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Could not refresh your portfolio against current market prices."
      );
    } finally {
      setRefreshing(false);
    }
  }

  function startEdit(s: PortfolioStrategyRow) {
    setEditingTicker(s.ticker);
    setEditShares(String(s.shares ?? ""));
    setEditAvgCost(String(s.avg_cost ?? ""));
    setEditError(null);
  }

  function cancelEdit() {
    setEditingTicker(null);
    setEditError(null);
  }

  async function saveEdit(ticker: string) {
    const shares = Number(editShares);
    const avgCost = Number(editAvgCost);
    if (!shares || shares <= 0) {
      setEditError("Shares must be a positive number.");
      return;
    }
    if (!avgCost || avgCost <= 0) {
      setEditError("Avg cost must be a positive number.");
      return;
    }
    setEditSaving(true);
    setEditError(null);
    setWatchlistNote(null);
    try {
      const res = await editPortfolioPosition(ticker, shares, avgCost, riskProfile, riskFactor, selectedPortfolioId ?? undefined);
      noteWatchlist(res.watchlist_alerts_created);
      setEditingTicker(null);
      await refresh();
    } catch (err) {
      setEditError(err instanceof ApiError ? err.message : "Could not save this position.");
    } finally {
      setEditSaving(false);
    }
  }

  async function handleDeletePosition(ticker: string) {
    if (!window.confirm(`Delete ${ticker}? This can't be undone.`)) return;
    setDeletingTicker(ticker);
    setPositionActionError(null);
    try {
      await deletePortfolioPosition(ticker, selectedPortfolioId ?? undefined);
      setPortfolioReloadSignal((n) => n + 1);
      await refresh();
    } catch (err) {
      setPositionActionError(err instanceof ApiError ? err.message : `Could not delete ${ticker}.`);
    } finally {
      setDeletingTicker(null);
    }
  }

  function startMove(ticker: string) {
    setMovingTicker(ticker);
    setMoveTargetId("");
    setPositionActionError(null);
  }

  function cancelMove() {
    setMovingTicker(null);
    setPositionActionError(null);
  }

  async function confirmMove(ticker: string) {
    const toId = Number(moveTargetId);
    if (!toId) {
      setPositionActionError("Choose a destination portfolio.");
      return;
    }
    setMoveSaving(true);
    setPositionActionError(null);
    try {
      await movePortfolioPosition(ticker, toId, riskProfile, riskFactor, selectedPortfolioId ?? undefined);
      setMovingTicker(null);
      setPortfolioReloadSignal((n) => n + 1);
      await refresh();
    } catch (err) {
      setPositionActionError(err instanceof ApiError ? err.message : `Could not move ${ticker}.`);
    } finally {
      setMoveSaving(false);
    }
  }

  async function handleAddPosition(e: React.FormEvent) {
    e.preventDefault();
    const ticker = addTicker.trim().toUpperCase();
    const shares = Number(addShares);
    const avgCost = Number(addAvgCost);
    if (!ticker) {
      setAddError("Enter a ticker.");
      return;
    }
    if (!shares || shares <= 0) {
      setAddError("Shares must be a positive number.");
      return;
    }
    if (!avgCost || avgCost <= 0) {
      setAddError("Avg cost must be a positive number.");
      return;
    }
    setAdding(true);
    setAddError(null);
    setWatchlistNote(null);
    try {
      const res = await editPortfolioPosition(ticker, shares, avgCost, riskProfile, riskFactor, selectedPortfolioId ?? undefined);
      noteWatchlist(res.watchlist_alerts_created);
      setAddTicker("");
      setAddShares("");
      setAddAvgCost("");
      await refresh();
    } catch (err) {
      setAddError(err instanceof ApiError ? err.message : "Could not add this position.");
    } finally {
      setAdding(false);
    }
  }

  async function submitCsv(e: React.FormEvent) {
    e.preventDefault();
    if (!file) {
      setError("Choose a Robinhood activity CSV first.");
      return;
    }
    setSubmitting(true);
    setError(null);
    setWatchlistNote(null);
    try {
      const res = await importPortfolioCsv(file, riskProfile, riskFactor, selectedPortfolioId ?? undefined);
      setFile(null);
      noteWatchlist(res.watchlist_alerts_created);
      await refresh();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not process CSV.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">Portfolio Strategies</h1>
      <p className="mt-1 text-sm text-slate-500">
        Import a Robinhood activity CSV or enter positions manually to get short- and long-term plans per holding.
        Each save also sets watchlist alerts by default at the suggested upside target and stop for every position.
      </p>

      <div className="mt-6">
        <PortfolioSwitcher
          selectedPortfolioId={selectedPortfolioId}
          onChange={setSelectedPortfolioId}
          onPortfoliosChange={setAllPortfolios}
          reloadSignal={portfolioReloadSignal}
        />
      </div>

      <PortfolioDropAlerts portfolioId={selectedPortfolioId} />
      <GoalPlan portfolioId={selectedPortfolioId} />

      <div className="mt-6 flex flex-wrap items-end gap-3">
        <Field label="Risk profile">
          <select value={riskProfile} onChange={(e) => setRiskProfile(e.target.value)} className="input">
            {RISK_PROFILES.map((r) => (
              <option key={r} value={r}>{r}</option>
            ))}
          </select>
        </Field>
        <Field label="Risk factor (1-10)">
          <input type="number" min={1} max={10} value={riskFactor} onChange={(e) => setRiskFactor(Number(e.target.value))} className="input w-20" />
        </Field>
      </div>

      <div className="mt-4 flex gap-2">
        <button
          onClick={() => setMode("manual")}
          className={`rounded-md px-3 py-1.5 text-sm font-medium ${mode === "manual" ? "bg-slate-900 text-white" : "border border-slate-300 text-slate-700 hover:bg-slate-100"}`}
        >
          Manual Entry
        </button>
        <button
          onClick={() => setMode("csv")}
          className={`rounded-md px-3 py-1.5 text-sm font-medium ${mode === "csv" ? "bg-slate-900 text-white" : "border border-slate-300 text-slate-700 hover:bg-slate-100"}`}
        >
          Import Robinhood CSV
        </button>
      </div>

      {mode === "manual" ? (
        <form onSubmit={submitManual} className="mt-4 flex flex-col gap-3 rounded-lg border border-slate-200 bg-white p-5">
          {rows.map((row, i) => (
            <div key={i} className="flex flex-wrap items-end gap-2">
              <Field label="Ticker">
                <input
                  value={row.ticker}
                  onChange={(e) => updateRow(i, { ticker: e.target.value })}
                  onBlur={(e) => populateCurrentPrice(i, e.target.value)}
                  className="input w-24 uppercase"
                />
              </Field>
              <Field label="Name">
                <input value={row.name} onChange={(e) => updateRow(i, { name: e.target.value })} className="input w-32" />
              </Field>
              <Field label="Shares">
                <input type="number" step="0.0001" value={row.shares || ""} onChange={(e) => updateRow(i, { shares: Number(e.target.value) })} className="input w-24" />
              </Field>
              <Field label="Avg cost">
                <input type="number" step="0.01" value={row.avg_cost || ""} onChange={(e) => updateRow(i, { avg_cost: Number(e.target.value) })} className="input w-24" />
              </Field>
              <Field label="Current price">
                <input
                  type="number"
                  step="0.01"
                  placeholder={priceFetchingRow === i ? "Fetching…" : undefined}
                  value={row.current_price || ""}
                  onChange={(e) => updateRow(i, { current_price: Number(e.target.value) })}
                  className="input w-24"
                />
              </Field>
              {rows.length > 1 && (
                <button type="button" onClick={() => removeRow(i)} className="text-xs text-red-600 hover:underline">
                  Remove
                </button>
              )}
            </div>
          ))}
          <div className="flex items-center gap-3">
            <button type="button" onClick={addRow} className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-100">
              + Add Position
            </button>
            <button type="submit" disabled={submitting} className="btn-primary">
              {submitting ? "Saving…" : "Save Positions"}
            </button>
          </div>
        </form>
      ) : (
        <form onSubmit={submitCsv} className="mt-4 flex flex-col gap-3 rounded-lg border border-slate-200 bg-white p-5">
          <Field label="Robinhood activity CSV">
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="text-sm text-slate-700"
            />
          </Field>
          <button type="submit" disabled={submitting} className="btn-primary self-start">
            {submitting ? "Processing…" : "Import CSV"}
          </button>
        </form>
      )}

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {watchlistNote && (
        <p className="mt-4 rounded-md bg-emerald-50 px-3 py-2 text-sm text-emerald-700">
          {watchlistNote}{" "}
          <a href="/watchlist" className="underline">
            View watchlist
          </a>
        </p>
      )}

      {summary && (
        <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-3">
          <MetricTile label="Positions" value={String(summary.total_positions)} />
          <MetricTile label="Total Value" value={`$${summary.total_value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
          <MetricTile label="Unrealized PnL" value={`${summary.total_pnl_pct.toFixed(2)}%`} />
        </div>
      )}

      {summary && summary.total_positions > 0 && (
        <div className="mt-6">
          <h2 className="text-lg font-semibold text-slate-900">Value vs. 30 Days Ago &amp; What You Paid</h2>
          <p className="mt-1 text-xs text-slate-500">
            Today&apos;s market price for every holding — against its price {performance?.lookback_days ?? 30} days
            ago, and against your average cost — priced fresh each time, independent of when you last saved or
            refreshed.
          </p>

          {performanceLoading && !performance && <p className="mt-2 text-sm text-slate-500">Loading…</p>}
          {performanceError && (
            <p className="mt-2 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{performanceError}</p>
          )}

          {performance && performance.rows.length > 0 && (
            <>
              <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-4">
                <MetricTile
                  label="Value Now"
                  value={`$${performance.total_value_now.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                />
                <MetricTile
                  label="30D Change"
                  value={`${performance.value_diff >= 0 ? "+" : ""}$${performance.value_diff.toLocaleString(undefined, { maximumFractionDigits: 0 })}${
                    performance.value_diff_pct !== null
                      ? ` (${performance.value_diff_pct >= 0 ? "+" : ""}${performance.value_diff_pct.toFixed(2)}%)`
                      : ""
                  }`}
                  positive={performance.value_diff >= 0}
                />
                <MetricTile
                  label="Total Paid"
                  value={`$${performance.total_cost_basis.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                />
                <MetricTile
                  label="Gain vs. Paid"
                  value={`${performance.total_gain_vs_cost >= 0 ? "+" : ""}$${performance.total_gain_vs_cost.toLocaleString(undefined, { maximumFractionDigits: 0 })}${
                    performance.total_gain_vs_cost_pct !== null
                      ? ` (${performance.total_gain_vs_cost_pct >= 0 ? "+" : ""}${performance.total_gain_vs_cost_pct.toFixed(2)}%)`
                      : ""
                  }`}
                  positive={performance.total_gain_vs_cost >= 0}
                />
              </div>

              <div className="mt-3 max-h-[70vh] overflow-auto rounded-lg border border-slate-200 bg-white">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                      <PerformanceTh label="Ticker" onInfoClick={() => setPerformanceInfoColumn("Ticker")} />
                      <PerformanceTh label="Signal" onInfoClick={() => setPerformanceInfoColumn("Signal")} />
                      <PerformanceTh label="Momentum Rank" onInfoClick={() => setPerformanceInfoColumn("Momentum Rank")} />
                      <PerformanceTh label="Next-Day Forecast" align="right" onInfoClick={() => setPerformanceInfoColumn("Next-Day Forecast")} />
                      <PerformanceTh label="5-Day Forecast" align="right" onInfoClick={() => setPerformanceInfoColumn("5-Day Forecast")} />
                      <PerformanceTh label="10-Day Forecast" align="right" onInfoClick={() => setPerformanceInfoColumn("10-Day Forecast")} />
                      <PerformanceTh label="Shares" align="right" onInfoClick={() => setPerformanceInfoColumn("Shares")} />
                      <PerformanceTh label="Price Now" align="right" onInfoClick={() => setPerformanceInfoColumn("Price Now")} />
                      <PerformanceTh label="Price 30D Ago" align="right" onInfoClick={() => setPerformanceInfoColumn("Price 30D Ago")} />
                      <PerformanceTh label="30D Diff" align="right" onInfoClick={() => setPerformanceInfoColumn("30D Diff")} />
                      <PerformanceTh label="Avg Cost Paid" align="right" onInfoClick={() => setPerformanceInfoColumn("Avg Cost Paid")} />
                      <PerformanceTh label="Gain vs. Paid" align="right" onInfoClick={() => setPerformanceInfoColumn("Gain vs. Paid")} />
                    </tr>
                  </thead>
                  <tbody>
                    {performance.rows.map((r) => {
                      const insight = insights.find((i) => i.ticker === r.ticker) ?? null;
                      return (
                      <tr key={r.ticker} className="border-b border-slate-100 last:border-0">
                        <td className="px-3 py-2 font-medium text-slate-800">
                          {r.ticker}
                          {insight?.concentrated && (
                            <span
                              title="A single position this large drives most of your portfolio's swings."
                              className="ml-1.5 rounded-full bg-amber-50 px-1.5 py-0.5 text-[10px] font-semibold text-amber-700"
                            >
                              {insight.weight_pct?.toFixed(0)}%
                            </span>
                          )}
                        </td>
                        <td className="px-3 py-2">
                          {insight?.signal ? (
                            <span
                              className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
                                insight.signal === "BUY"
                                  ? "bg-emerald-50 text-emerald-700"
                                  : insight.signal === "SELL"
                                  ? "bg-red-50 text-red-700"
                                  : "bg-slate-100 text-slate-600"
                              }`}
                            >
                              {insight.signal}
                            </span>
                          ) : (
                            <span className="text-slate-400">{insightsLoading ? "…" : "—"}</span>
                          )}
                        </td>
                        <td className="px-3 py-2 text-slate-600">
                          {insight?.rank !== null && insight?.rank !== undefined && insight.universe_size
                            ? `#${insight.rank} of ${insight.universe_size}`
                            : insightsLoading
                            ? "…"
                            : "—"}
                        </td>
                        <td className="px-3 py-2 text-right">
                          {insight?.target_price_1d != null && insight?.expected_return_pct_1d != null ? (
                            <span
                              className={`font-medium ${
                                insight.expected_return_pct_1d >= 0 ? "text-emerald-600" : "text-red-600"
                              }`}
                            >
                              ${insight.target_price_1d.toFixed(2)}
                              <span className="ml-1 text-xs">
                                ({insight.expected_return_pct_1d >= 0 ? "+" : ""}
                                {insight.expected_return_pct_1d.toFixed(2)}%)
                              </span>
                            </span>
                          ) : (
                            <span className="text-slate-400">{insightsLoading ? "…" : "—"}</span>
                          )}
                        </td>
                        <td className="px-3 py-2 text-right">
                          {insight?.target_price_5d != null && insight?.expected_return_pct_5d != null ? (
                            <span
                              className={`font-medium ${
                                insight.expected_return_pct_5d >= 0 ? "text-emerald-600" : "text-red-600"
                              }`}
                            >
                              ${insight.target_price_5d.toFixed(2)}
                              <span className="ml-1 text-xs">
                                ({insight.expected_return_pct_5d >= 0 ? "+" : ""}
                                {insight.expected_return_pct_5d.toFixed(2)}%)
                              </span>
                            </span>
                          ) : (
                            <span className="text-slate-400">{insightsLoading ? "…" : "—"}</span>
                          )}
                        </td>
                        <td className="px-3 py-2 text-right">
                          {insight?.target_price != null && insight?.expected_return_pct != null ? (
                            <span
                              className={`font-medium ${
                                insight.expected_return_pct >= 0 ? "text-emerald-600" : "text-red-600"
                              }`}
                            >
                              ${insight.target_price.toFixed(2)}
                              <span className="ml-1 text-xs">
                                ({insight.expected_return_pct >= 0 ? "+" : ""}
                                {insight.expected_return_pct.toFixed(2)}%)
                              </span>
                            </span>
                          ) : (
                            <span className="text-slate-400">{insightsLoading ? "…" : "—"}</span>
                          )}
                        </td>
                        <td className="px-3 py-2 text-right text-slate-600">{r.shares}</td>
                        {r.price_unavailable ? (
                          <td colSpan={4} className="px-3 py-2 text-slate-400">
                            No market data found for this ticker — check it&apos;s a valid, publicly-traded symbol.
                          </td>
                        ) : (
                          <>
                            <td className="px-3 py-2 text-right text-slate-600">
                              ${r.price_now!.toFixed(2)}
                              {r.extended_hours && (
                                <div
                                  className={`text-xs ${
                                    (r.extended_hours.change_pct ?? 0) >= 0 ? "text-emerald-600" : "text-red-600"
                                  }`}
                                >
                                  {r.extended_hours.state === "POST" ? "After hours" : "Pre-market"}: $
                                  {r.extended_hours.price.toFixed(2)}
                                  {r.extended_hours.change_pct !== null && (
                                    <>
                                      {" "}
                                      ({r.extended_hours.change_pct >= 0 ? "+" : ""}
                                      {r.extended_hours.change_pct.toFixed(2)}%)
                                    </>
                                  )}
                                </div>
                              )}
                            </td>
                            <td className="px-3 py-2 text-right text-slate-600">
                              {r.price_30d_ago !== null ? `$${r.price_30d_ago.toFixed(2)}` : "—"}
                            </td>
                            <td
                              className={`px-3 py-2 text-right font-medium ${
                                r.diff === null ? "text-slate-400" : r.diff >= 0 ? "text-emerald-600" : "text-red-600"
                              }`}
                            >
                              {r.diff === null
                                ? "—"
                                : `${r.diff >= 0 ? "+" : ""}${r.diff.toLocaleString(undefined, { maximumFractionDigits: 0 })}${
                                    r.diff_pct !== null ? ` (${r.diff_pct >= 0 ? "+" : ""}${r.diff_pct.toFixed(1)}%)` : ""
                                  }`}
                            </td>
                            <td className="px-3 py-2 text-right text-slate-600">
                              {r.avg_cost !== null ? `$${r.avg_cost.toFixed(2)}` : "—"}
                            </td>
                            <td
                              className={`px-3 py-2 text-right font-medium ${
                                r.gain_vs_cost === null
                                  ? "text-slate-400"
                                  : r.gain_vs_cost >= 0
                                  ? "text-emerald-600"
                                  : "text-red-600"
                              }`}
                            >
                              {r.gain_vs_cost === null
                                ? "—"
                                : `${r.gain_vs_cost >= 0 ? "+" : ""}${r.gain_vs_cost.toLocaleString(undefined, { maximumFractionDigits: 0 })}${
                                    r.gain_vs_cost_pct !== null
                                      ? ` (${r.gain_vs_cost_pct >= 0 ? "+" : ""}${r.gain_vs_cost_pct.toFixed(1)}%)`
                                      : ""
                                  }`}
                            </td>
                          </>
                        )}
                      </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      )}

      <div className="mt-6 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-lg font-semibold text-slate-900">Strategies</h2>
        {strategies.length > 0 && (
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
          >
            {refreshing ? "Refreshing…" : "Refresh with Current Market"}
          </button>
        )}
      </div>
      <p className="mt-1 text-xs text-slate-500">
        Pulls today&apos;s prices for the positions you&apos;ve already saved and recomputes the plans below —
        no need to re-enter or re-upload anything.
      </p>

      <p className="mt-3 text-xs font-medium text-slate-500">
        Add a new position — this only appends this one ticker, it won&apos;t touch anything else you&apos;ve
        saved.
      </p>
      <form onSubmit={handleAddPosition} className="mt-1 flex flex-wrap items-end gap-2 rounded-lg border border-slate-200 bg-white p-4">
        <Field label="Ticker">
          <TickerSearchInput
            value={addTicker}
            onChange={setAddTicker}
            className="input w-32 uppercase"
          />
        </Field>
        <CurrentPriceBadge ticker={addTicker} />
        <Field label="Shares">
          <input
            type="number"
            step="0.0001"
            value={addShares}
            onChange={(e) => setAddShares(e.target.value)}
            className="input w-24"
          />
        </Field>
        <Field label="Avg cost">
          <input
            type="number"
            step="0.01"
            value={addAvgCost}
            onChange={(e) => setAddAvgCost(e.target.value)}
            className="input w-24"
          />
        </Field>
        <button type="submit" disabled={adding} className="btn-primary">
          {adding ? "Adding…" : "Add to Portfolio"}
        </button>
        {addError && <p className="w-full text-xs text-red-600">{addError}</p>}
      </form>

      {loading ? (
        <p className="mt-2 text-sm text-slate-500">Loading…</p>
      ) : strategies.length === 0 ? (
        <p className="mt-2 text-sm text-slate-500">No saved strategies yet.</p>
      ) : (
        <div className="mt-3 grid grid-cols-1 gap-4 xl:grid-cols-2">
          {strategies.map((s) => {
            const pnl = s.unrealized_pnl_pct;
            const pnlPositive = pnl !== null && pnl >= 0;
            const isEditing = editingTicker === s.ticker;
            const extendedHours = performance?.rows.find((r) => r.ticker === s.ticker)?.extended_hours ?? null;
            const insight = insights.find((i) => i.ticker === s.ticker) ?? null;
            return (
              <div key={s.id} className="rounded-lg border border-slate-200 bg-white p-5">
                <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
                  <h3 className="text-base font-semibold text-slate-900">{s.ticker}</h3>
                  <div className="flex items-center gap-2">
                    <span
                      className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
                        pnl === null
                          ? "bg-slate-100 text-slate-500"
                          : pnlPositive
                          ? "bg-emerald-50 text-emerald-700"
                          : "bg-red-50 text-red-700"
                      }`}
                    >
                      {pnl === null ? "—" : `${pnlPositive ? "+" : ""}${pnl.toFixed(2)}%`}
                    </span>
                    {!isEditing && movingTicker !== s.ticker && (
                      <>
                        <button
                          onClick={() => startEdit(s)}
                          className="rounded-md border border-slate-300 px-2 py-0.5 text-xs font-medium text-slate-600 hover:bg-slate-100"
                        >
                          Edit
                        </button>
                        {allPortfolios.length > 1 && (
                          <button
                            onClick={() => startMove(s.ticker)}
                            className="rounded-md border border-slate-300 px-2 py-0.5 text-xs font-medium text-slate-600 hover:bg-slate-100"
                          >
                            Move
                          </button>
                        )}
                        <button
                          onClick={() => handleDeletePosition(s.ticker)}
                          disabled={deletingTicker === s.ticker}
                          className="rounded-md border border-red-200 px-2 py-0.5 text-xs font-medium text-red-600 hover:bg-red-50 disabled:opacity-50"
                        >
                          {deletingTicker === s.ticker ? "Deleting…" : "Delete"}
                        </button>
                      </>
                    )}
                  </div>
                </div>

                {movingTicker === s.ticker && (
                  <div className="mt-2 flex flex-wrap items-end gap-2 rounded-md border border-slate-200 bg-slate-50 p-3">
                    <Field label="Move to">
                      <select
                        value={moveTargetId}
                        onChange={(e) => setMoveTargetId(e.target.value)}
                        className="input"
                      >
                        <option value="">Choose a portfolio…</option>
                        {allPortfolios
                          .filter((p) => p.id !== selectedPortfolioId)
                          .map((p) => (
                            <option key={p.id} value={p.id}>
                              {p.name}
                            </option>
                          ))}
                      </select>
                    </Field>
                    <button onClick={() => confirmMove(s.ticker)} disabled={moveSaving} className="btn-primary">
                      {moveSaving ? "Moving…" : "Confirm Move"}
                    </button>
                    <button
                      onClick={cancelMove}
                      disabled={moveSaving}
                      className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-100"
                    >
                      Cancel
                    </button>
                    {positionActionError && <p className="w-full text-xs text-red-600">{positionActionError}</p>}
                  </div>
                )}

                {isEditing ? (
                  <div className="mt-2 flex flex-wrap items-end gap-2 rounded-md border border-slate-200 bg-slate-50 p-3">
                    <Field label="Shares">
                      <input
                        type="number"
                        step="0.0001"
                        value={editShares}
                        onChange={(e) => setEditShares(e.target.value)}
                        className="input w-24"
                      />
                    </Field>
                    <Field label="Avg cost">
                      <input
                        type="number"
                        step="0.01"
                        value={editAvgCost}
                        onChange={(e) => setEditAvgCost(e.target.value)}
                        className="input w-24"
                      />
                    </Field>
                    <button
                      onClick={() => saveEdit(s.ticker)}
                      disabled={editSaving}
                      className="btn-primary"
                    >
                      {editSaving ? "Saving…" : "Save"}
                    </button>
                    <button
                      onClick={cancelEdit}
                      disabled={editSaving}
                      className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-100"
                    >
                      Cancel
                    </button>
                    {editError && <p className="w-full text-xs text-red-600">{editError}</p>}
                  </div>
                ) : (
                  <p className="mt-1 text-sm text-slate-500">
                    {s.shares} sh @ avg ${s.avg_cost?.toFixed(2)} · now ${s.current_price?.toFixed(2)}
                    {extendedHours && (
                      <span className={extendedHours.change_pct !== null && extendedHours.change_pct >= 0 ? "text-emerald-600" : "text-red-600"}>
                        {" "}
                        · {extendedHours.state === "POST" ? "after hours" : "pre-market"}: $
                        {extendedHours.price.toFixed(2)}
                        {extendedHours.change_pct !== null && (
                          <> ({extendedHours.change_pct >= 0 ? "+" : ""}{extendedHours.change_pct.toFixed(2)}%)</>
                        )}
                      </span>
                    )}
                  </p>
                )}

                {insightsLoading && !insight && (
                  <p className="mt-2 text-xs text-slate-400">Checking live signal &amp; rank…</p>
                )}
                {insight && (
                  <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
                    {insight.signal && (
                      <span
                        title={
                          insight.expected_return_pct !== null
                            ? `Expected ${insight.expected_return_pct >= 0 ? "+" : ""}${insight.expected_return_pct.toFixed(2)}% — same model as /predict`
                            : "Same model as /predict"
                        }
                        className={`rounded-full px-2 py-0.5 font-semibold ${
                          insight.signal === "BUY"
                            ? "bg-emerald-50 text-emerald-700"
                            : insight.signal === "SELL"
                            ? "bg-red-50 text-red-700"
                            : "bg-slate-100 text-slate-600"
                        }`}
                      >
                        {insight.signal}
                      </span>
                    )}
                    {insight.rank !== null && insight.universe_size !== null && (
                      <span className="text-slate-500">
                        Momentum rank #{insight.rank} of {insight.universe_size}
                      </span>
                    )}
                    {insight.concentrated && insight.weight_pct !== null && (
                      <span
                        title="A single position this large drives most of your portfolio's swings — consider whether that's intentional."
                        className="rounded-full bg-amber-50 px-2 py-0.5 font-semibold text-amber-700"
                      >
                        {insight.weight_pct.toFixed(0)}% of portfolio — concentrated
                      </span>
                    )}
                  </div>
                )}

                <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <div className="rounded-md border border-slate-100 bg-slate-50/70 p-3">
                    <PlanText text={withLiveRead(s.short_term_plan, shortTermSignalNote(insight))} />
                  </div>
                  <div className="rounded-md border border-slate-100 bg-slate-50/70 p-3">
                    <PlanText text={withLiveRead(s.long_term_plan, longTermMomentumNote(insight))} />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {insightsError && (
        <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{insightsError}</p>
      )}
      {positionActionError && movingTicker === null && (
        <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{positionActionError}</p>
      )}

      {performanceInfoColumn && PERFORMANCE_COLUMN_INFO[performanceInfoColumn] && (
        <InfoModal
          info={PERFORMANCE_COLUMN_INFO[performanceInfoColumn]}
          onClose={() => setPerformanceInfoColumn(null)}
        />
      )}
    </div>
  );
}

// The stored Short-/Long-Term Plan text (services/portfolio_strategy.py) picks
// its "Stance"/guidance sentence purely from which P&L bucket a position
// falls into — any two tickers at the same P&L% and risk profile get the
// identical sentence, since nothing about the ticker itself (momentum,
// model signal) feeds into it. These append a ticker-specific paragraph
// using data the page has already fetched live (portfolio_insights) —
// no extra request, and the stored plan itself is left untouched.
function shortTermSignalNote(insight: PortfolioInsight | null): string | null {
  if (!insight || !insight.signal) return null;
  const expected =
    insight.expected_return_pct !== null
      ? ` (model expects ${insight.expected_return_pct >= 0 ? "+" : ""}${insight.expected_return_pct.toFixed(1)}% over its forecast horizon)`
      : "";
  if (insight.signal === "BUY") {
    return `The model's current signal is **BUY**${expected} — consistent with the case to keep holding here.`;
  }
  if (insight.signal === "SELL") {
    return `The model's current signal is **SELL**${expected} — this cuts against holding; worth watching closely, or locking in gains if you'd rather not fight the signal.`;
  }
  return `The model's current signal is **HOLD**${expected} — no strong edge either way right now.`;
}

function longTermMomentumNote(insight: PortfolioInsight | null): string | null {
  if (!insight || insight.rank === null || insight.universe_size === null || !insight.universe_size) return null;
  const pct = insight.rank / insight.universe_size;
  let read: string;
  if (pct <= 0.25) read = "near the top of the current ranked universe, suggesting relative strength is still with this name";
  else if (pct <= 0.5) read = "in the upper half of the current ranked universe";
  else if (pct <= 0.75) read = "in the lower half of the current ranked universe";
  else read = "near the bottom of the current ranked universe, worth factoring into how much conviction you have in the long-term thesis";
  return `Momentum rank: **#${insight.rank} of ${insight.universe_size}** — ${read}.`;
}

function withLiveRead(planText: string, note: string | null): string {
  return note ? `${planText}\n\n**Live Read:** ${note}` : planText;
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs font-medium text-slate-500">{label}</label>
      {children}
    </div>
  );
}

function MetricTile({ label, value, positive }: { label: string; value: string; positive?: boolean }) {
  const valueClass =
    positive === undefined ? "text-slate-900" : positive ? "text-emerald-600" : "text-red-600";
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <p className="text-xs text-slate-500">{label}</p>
      <p className={`mt-1 text-lg font-semibold ${valueClass}`}>{value}</p>
    </div>
  );
}

function PerformanceTh({
  label,
  align,
  onInfoClick,
}: {
  label: string;
  align?: "left" | "right";
  onInfoClick: () => void;
}) {
  return (
    <th className={`sticky top-0 z-10 bg-slate-50 px-3 py-2 ${align === "right" ? "text-right" : ""}`}>
      <div className={`flex items-center gap-1 ${align === "right" ? "justify-end" : ""}`}>
        {align === "right" && <ThInfoButton label={label} onClick={onInfoClick} />}
        <span>{label}</span>
        {align !== "right" && <ThInfoButton label={label} onClick={onInfoClick} />}
      </div>
    </th>
  );
}

function ThInfoButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={`What is ${label}?`}
      className="flex h-3.5 w-3.5 items-center justify-center rounded-full border border-slate-300 text-[9px] font-normal normal-case text-slate-400 hover:border-slate-500 hover:text-slate-700"
    >
      i
    </button>
  );
}
