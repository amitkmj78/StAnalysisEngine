"use client";

import { useEffect, useState } from "react";

import {
  ApiError,
  getPortfolioPerformance,
  getPortfolioStrategies,
  getPortfolioSummary,
  importPortfolioCsv,
  refreshPortfolio,
  submitManualPositions,
} from "@/lib/api";
import type { ManualPositionInput, PortfolioPerformance, PortfolioStrategyRow, PortfolioSummary } from "@/lib/types";
import PlanText from "@/components/PlanText";

const RISK_PROFILES = ["Conservative", "Balanced", "Aggressive"];

type Mode = "manual" | "csv";

const EMPTY_ROW: ManualPositionInput = { name: "", ticker: "", shares: 0, current_price: 0, avg_cost: 0 };

export default function PortfolioPage() {
  const [mode, setMode] = useState<Mode>("manual");
  const [riskProfile, setRiskProfile] = useState("Balanced");
  const [riskFactor, setRiskFactor] = useState(5);

  const [rows, setRows] = useState<ManualPositionInput[]>([{ ...EMPTY_ROW }]);
  const [file, setFile] = useState<File | null>(null);

  const [strategies, setStrategies] = useState<PortfolioStrategyRow[]>([]);
  const [summary, setSummary] = useState<PortfolioSummary | null>(null);
  const [performance, setPerformance] = useState<PortfolioPerformance | null>(null);
  const [performanceError, setPerformanceError] = useState<string | null>(null);
  const [performanceLoading, setPerformanceLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [watchlistNote, setWatchlistNote] = useState<string | null>(null);

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const [stratRes, summaryRes] = await Promise.all([getPortfolioStrategies(), getPortfolioSummary()]);
      setStrategies(stratRes.strategies);
      setSummary(summaryRes.summary);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load portfolio.");
    } finally {
      setLoading(false);
    }

    setPerformanceLoading(true);
    setPerformanceError(null);
    try {
      setPerformance(await getPortfolioPerformance(30));
    } catch (err) {
      setPerformanceError(err instanceof ApiError ? err.message : "Could not load 30-day performance.");
    } finally {
      setPerformanceLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  function updateRow(i: number, patch: Partial<ManualPositionInput>) {
    setRows((prev) => prev.map((r, idx) => (idx === i ? { ...r, ...patch } : r)));
  }

  function addRow() {
    setRows((prev) => [...prev, { ...EMPTY_ROW }]);
  }

  function removeRow(i: number) {
    setRows((prev) => prev.filter((_, idx) => idx !== i));
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
      const res = await refreshPortfolio(riskProfile, riskFactor);
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
      const res = await importPortfolioCsv(file, riskProfile, riskFactor);
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
    <div className="mx-auto max-w-5xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">Portfolio Strategies</h1>
      <p className="mt-1 text-sm text-slate-500">
        Import a Robinhood activity CSV or enter positions manually to get short- and long-term plans per holding.
        Each save also sets watchlist alerts by default at the suggested upside target and stop for every position.
      </p>

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
                <input value={row.ticker} onChange={(e) => updateRow(i, { ticker: e.target.value })} className="input w-24 uppercase" />
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
                <input type="number" step="0.01" value={row.current_price || ""} onChange={(e) => updateRow(i, { current_price: Number(e.target.value) })} className="input w-24" />
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

              <div className="mt-3 overflow-x-auto rounded-lg border border-slate-200 bg-white">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                      <th className="px-3 py-2">Ticker</th>
                      <th className="px-3 py-2 text-right">Shares</th>
                      <th className="px-3 py-2 text-right">Price Now</th>
                      <th className="px-3 py-2 text-right">Price 30D Ago</th>
                      <th className="px-3 py-2 text-right">30D Diff</th>
                      <th className="px-3 py-2 text-right">Avg Cost Paid</th>
                      <th className="px-3 py-2 text-right">Gain vs. Paid</th>
                    </tr>
                  </thead>
                  <tbody>
                    {performance.rows.map((r) => (
                      <tr key={r.ticker} className="border-b border-slate-100 last:border-0">
                        <td className="px-3 py-2 font-medium text-slate-800">{r.ticker}</td>
                        <td className="px-3 py-2 text-right text-slate-600">{r.shares}</td>
                        {r.price_unavailable ? (
                          <td colSpan={4} className="px-3 py-2 text-slate-400">
                            No market data found for this ticker — check it&apos;s a valid, publicly-traded symbol.
                          </td>
                        ) : (
                          <>
                            <td className="px-3 py-2 text-right text-slate-600">${r.price_now!.toFixed(2)}</td>
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
                    ))}
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
      {loading ? (
        <p className="mt-2 text-sm text-slate-500">Loading…</p>
      ) : strategies.length === 0 ? (
        <p className="mt-2 text-sm text-slate-500">No saved strategies yet.</p>
      ) : (
        <div className="mt-3 grid grid-cols-1 gap-4 xl:grid-cols-2">
          {strategies.map((s) => {
            const pnl = s.unrealized_pnl_pct;
            const pnlPositive = pnl !== null && pnl >= 0;
            return (
              <div key={s.id} className="rounded-lg border border-slate-200 bg-white p-5">
                <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
                  <h3 className="text-base font-semibold text-slate-900">{s.ticker}</h3>
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
                </div>
                <p className="mt-1 text-sm text-slate-500">
                  {s.shares} sh @ avg ${s.avg_cost?.toFixed(2)} · now ${s.current_price?.toFixed(2)}
                </p>

                <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <div className="rounded-md border border-slate-100 bg-slate-50/70 p-3">
                    <PlanText text={s.short_term_plan} />
                  </div>
                  <div className="rounded-md border border-slate-100 bg-slate-50/70 p-3">
                    <PlanText text={s.long_term_plan} />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
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
