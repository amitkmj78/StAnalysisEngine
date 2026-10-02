"use client";

import { useEffect, useState } from "react";

import EntryChart from "@/components/entry/EntryChart";
import MetricLabel from "@/components/MetricLabel";
import MetricTile from "@/components/MetricTile";
import { ApiError, getEntryPlan, getEntryScan, getEntryUniverses } from "@/lib/api";
import type { EntryHistory, EntryPlan, EntryScanRow } from "@/lib/types";

const SCAN_COLUMNS = ["Ticker", "Signal", "Quant Signal", "Entry Score", "Current Price", "Entry Low", "Entry High", "Stop Loss", "First Target", "RSI"];

const SORTABLE_NUMERIC_COLUMNS = new Set(["Entry Score", "Current Price", "Entry Low", "Entry High", "Stop Loss", "First Target", "RSI"]);

function signalBadgeClass(signal: string): string {
  switch (signal) {
    case "Buy Now":
      return "border-emerald-200 bg-emerald-50 text-emerald-700";
    case "Buy on Pullback":
      return "border-emerald-100 bg-emerald-50 text-emerald-600";
    case "Breakout Entry":
      return "border-blue-200 bg-blue-50 text-blue-700";
    case "Watch for Reversal":
      return "border-amber-200 bg-amber-50 text-amber-700";
    case "Wait for Pullback":
      return "border-amber-100 bg-amber-50 text-amber-600";
    default:
      return "border-slate-200 bg-slate-100 text-slate-600";
  }
}

function SignalBadge({ signal, className = "" }: { signal: string; className?: string }) {
  return (
    <span
      className={`inline-flex items-center whitespace-nowrap rounded-full border px-2.5 py-0.5 text-xs font-medium ${signalBadgeClass(signal)} ${className}`}
    >
      {signal}
    </span>
  );
}

// Matches the BUY/SELL/HOLD palette already established on the
// Quant-vs-Analyst comparison page, for a consistent meaning across
// the app: emerald = buy, red = sell, slate = hold/unknown.
function quantSignalBadgeClass(signal: string): string {
  if (signal === "BUY") return "border-emerald-200 bg-emerald-50 text-emerald-700";
  if (signal === "SELL") return "border-red-200 bg-red-50 text-red-700";
  if (signal === "HOLD") return "border-slate-200 bg-slate-100 text-slate-600";
  return "border-slate-200 bg-slate-100 text-slate-400";
}

function QuantSignalBadge({ signal, className = "" }: { signal: string | null | undefined; className?: string }) {
  if (!signal) {
    return <span className="text-xs text-slate-300">—</span>;
  }
  return (
    <span
      className={`inline-flex items-center whitespace-nowrap rounded-full border px-2.5 py-0.5 text-xs font-medium ${quantSignalBadgeClass(signal)} ${className}`}
    >
      {signal}
    </span>
  );
}

function EntryScoreBar({ score }: { score: number }) {
  // Entry Score is uncapped (a genuinely exceptional setup can exceed
  // 100 — see the Entry Score info panel) — the bar itself still only
  // has 100%-of-its-width to work with, so fill is clamped for display
  // while the number next to it always shows the real value.
  const fillPct = Math.max(0, Math.min(100, score));
  const barColor = score >= 100 ? "bg-emerald-600" : score >= 70 ? "bg-emerald-500" : score >= 40 ? "bg-amber-500" : "bg-slate-400";
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-slate-100">
        <div className={`h-full rounded-full ${barColor}`} style={{ width: `${fillPct}%` }} />
      </div>
      <span className="text-xs tabular-nums text-slate-500">{score}</span>
    </div>
  );
}

export default function EntryPage() {
  const [assetType, setAssetType] = useState<"Fund" | "Stock">("Stock");
  const [mode, setMode] = useState<"scan" | "check">("scan");
  const [universes, setUniverses] = useState<string[]>(["All"]);
  const [universe, setUniverse] = useState("All");
  const [topN, setTopN] = useState(5);
  const [ticker, setTicker] = useState("AAPL");
  const [quantSignalFilter, setQuantSignalFilter] = useState("");
  const [sortColumn, setSortColumn] = useState("Entry Score");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const [scanResults, setScanResults] = useState<EntryScanRow[]>([]);
  const [singlePlan, setSinglePlan] = useState<EntryPlan | null>(null);
  const [singleHistory, setSingleHistory] = useState<EntryHistory | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);

  useEffect(() => {
    getEntryUniverses(assetType)
      .then((res) => {
        setUniverses(res.universes);
        setUniverse(res.universes[0] ?? "All");
      })
      .catch(() => {});
  }, [assetType]);

  async function runSearch(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setHasSearched(true);
    try {
      if (mode === "scan") {
        const res = await getEntryScan(assetType, universe, topN, assetType === "Stock" ? quantSignalFilter || undefined : undefined);
        setScanResults(res.results);
        setSinglePlan(null);
      } else {
        const res = await getEntryPlan(ticker.trim().toUpperCase());
        setSinglePlan(res.plan);
        setSingleHistory(res.history);
        setScanResults([]);
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong.");
      setScanResults([]);
      setSinglePlan(null);
    } finally {
      setLoading(false);
    }
  }

  // The hero always shows the backend's own best-overall pick (Entry
  // Score + Signal Rank) regardless of how the user has the table
  // sorted — "top entry" is a distinct concept from "how I want to
  // browse the list right now".
  const winner = scanResults[0];

  function toggleSort(col: string) {
    if (sortColumn === col) {
      setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    } else {
      setSortColumn(col);
      setSortDir(SORTABLE_NUMERIC_COLUMNS.has(col) ? "desc" : "asc");
    }
  }

  const sortedResults = [...scanResults].sort((a, b) => {
    const av = a[sortColumn];
    const bv = b[sortColumn];
    if (av === null || av === undefined) return 1;
    if (bv === null || bv === undefined) return -1;
    if (typeof av === "number" && typeof bv === "number") {
      return sortDir === "desc" ? bv - av : av - bv;
    }
    return sortDir === "desc" ? String(bv).localeCompare(String(av)) : String(av).localeCompare(String(bv));
  });

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">Entry Signals</h1>
      <p className="mt-1 text-sm text-slate-500">
        Scan for the strongest current entry setups, or check one ticker for a buy-zone view.
      </p>

      <form onSubmit={runSearch} className="mt-6 flex flex-wrap items-end gap-3">
        <Field label="Asset type">
          <select value={assetType} onChange={(e) => setAssetType(e.target.value as "Fund" | "Stock")} className="input">
            <option value="Stock">Stock</option>
            <option value="Fund">Fund</option>
          </select>
        </Field>

        <Field label="Mode">
          <select value={mode} onChange={(e) => setMode(e.target.value as "scan" | "check")} className="input">
            <option value="scan">Scan current best entries</option>
            <option value="check">Check one ticker</option>
          </select>
        </Field>

        {mode === "scan" ? (
          <>
            <Field label="Universe">
              <select value={universe} onChange={(e) => setUniverse(e.target.value)} className="input">
                {universes.map((u) => (
                  <option key={u} value={u}>
                    {u}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Results">
              <input
                type="number"
                min={1}
                max={20}
                value={topN}
                onChange={(e) => setTopN(Number(e.target.value))}
                className="input w-20"
              />
            </Field>
            {assetType === "Stock" && (
              <Field label="Quant Signal">
                <select value={quantSignalFilter} onChange={(e) => setQuantSignalFilter(e.target.value)} className="input">
                  <option value="">Any</option>
                  <option value="BUY">BUY only</option>
                  <option value="HOLD">HOLD only</option>
                  <option value="SELL">SELL only</option>
                </select>
              </Field>
            )}
          </>
        ) : (
          <Field label="Ticker">
            <input value={ticker} onChange={(e) => setTicker(e.target.value)} className="input w-32" />
          </Field>
        )}

        <button type="submit" disabled={loading} className="btn-primary">
          {loading ? "Scanning…" : "Run"}
        </button>
      </form>

      {loading && <p className="mt-4 text-sm text-slate-500">Scanning current setups…</p>}
      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {!loading && hasSearched && !error && scanResults.length === 0 && !singlePlan && (
        <p className="mt-4 text-sm text-slate-500">No results — try another selection or ticker.</p>
      )}

      {mode === "scan" && winner && !loading && (
        <div className="mt-6 flex flex-col gap-6">
          <div className="rounded-lg border border-slate-200 bg-gradient-to-br from-white to-slate-50 p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
                  Top entry right now · {universe}
                </p>
                <h2 className="mt-1 text-2xl font-semibold text-slate-900">{winner.Ticker}</h2>
              </div>
              <div className="flex items-center gap-2">
                <SignalBadge signal={winner.Signal} className="px-3 py-1 text-sm" />
                <MetricLabel term="Entry Signal" />
              </div>
            </div>
            <div className="mt-3 flex items-center gap-3">
              <span className="text-xs font-medium text-slate-500">Entry Score</span>
              <EntryScoreBar score={Number(winner["Entry Score"])} />
              <MetricLabel term="Entry Score" />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricTile label="Entry Score" value={`${winner["Entry Score"]}`} />
            <MetricTile label="Current Price" value={`$${Number(winner["Current Price"]).toFixed(2)}`} />
            <MetricTile label="Entry Low" value={`$${Number(winner["Entry Low"]).toFixed(2)}`} />
            <MetricTile label="Entry High" value={`$${Number(winner["Entry High"]).toFixed(2)}`} />
          </div>

          {/* Table — sm and up. A cramped horizontally-scrolling table is a
              poor fit for a phone screen, so mobile gets its own card list
              below instead of just squeezing this one narrower. */}
          <div className="hidden overflow-x-auto rounded-lg border border-slate-200 bg-white sm:block">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                  {SCAN_COLUMNS.map((col) => (
                    <th key={col} className="px-3 py-2">
                      <button
                        type="button"
                        onClick={() => toggleSort(col)}
                        className="flex items-center gap-1 hover:text-slate-700"
                      >
                        {col}
                        {sortColumn === col && <span className="text-slate-400">{sortDir === "desc" ? "↓" : "↑"}</span>}
                      </button>
                      {/* "Signal"/"Quant Signal" are keyed distinctly here
                          ("Entry Signal"/"Entry Quant Signal") from the
                          Portfolio/Stock Finder pages' own same-named
                          columns, which mean different things. */}
                      <MetricLabel
                        term={col === "Signal" ? "Entry Signal" : col === "Quant Signal" ? "Entry Quant Signal" : col}
                      />
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sortedResults.map((row) => (
                  <tr key={row.Ticker} className="border-b border-slate-100 last:border-0 hover:bg-slate-50">
                    {SCAN_COLUMNS.map((col) => (
                      <td key={col} className="px-3 py-2 text-slate-700">
                        {col === "Signal" ? (
                          <SignalBadge signal={String(row[col])} />
                        ) : col === "Quant Signal" ? (
                          <QuantSignalBadge signal={row[col] as string | null | undefined} />
                        ) : col === "Entry Score" ? (
                          <EntryScoreBar score={Number(row[col])} />
                        ) : (
                          formatCell(row[col])
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Card list — below sm, one card per ticker instead of a
              horizontally-scrolling table. */}
          <div className="flex flex-col gap-3 sm:hidden">
            {sortedResults.map((row) => (
              <div key={row.Ticker} className="rounded-lg border border-slate-200 bg-white p-4">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-semibold text-slate-900">{row.Ticker}</span>
                  <div className="flex items-center gap-1.5">
                    <SignalBadge signal={String(row.Signal)} />
                    <QuantSignalBadge signal={row["Quant Signal"]} />
                  </div>
                </div>
                <div className="mt-2">
                  <EntryScoreBar score={Number(row["Entry Score"])} />
                </div>
                <div className="mt-3 grid grid-cols-2 gap-x-3 gap-y-1 text-xs text-slate-600">
                  <span>
                    Price <span className="tabular-nums text-slate-800">${Number(row["Current Price"]).toFixed(2)}</span>
                  </span>
                  <span>
                    RSI <span className="tabular-nums text-slate-800">{formatCell(row.RSI)}</span>
                  </span>
                  <span>
                    Entry <span className="tabular-nums text-slate-800">${Number(row["Entry Low"]).toFixed(2)}–${Number(row["Entry High"]).toFixed(2)}</span>
                  </span>
                  <span>
                    Stop <span className="tabular-nums text-slate-800">${Number(row["Stop Loss"]).toFixed(2)}</span>
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {mode === "check" && singlePlan && !loading && (
        <div className="mt-6 flex flex-col gap-6">
          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h2 className="text-lg font-semibold text-slate-900">{singlePlan.ticker} Entry Snapshot</h2>
              <div className="flex items-center gap-2">
                <SignalBadge signal={singlePlan.signal} />
                <MetricLabel term="Entry Signal" />
                <span className="text-slate-300">·</span>
                <QuantSignalBadge signal={singlePlan.quant_signal} />
                <MetricLabel term="Entry Quant Signal" />
              </div>
            </div>
            <p className="mt-2 text-sm text-slate-600">{singlePlan.summary}</p>
          </div>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricTile label="Entry Score" value={`${singlePlan.entry_score}`} />
            <MetricTile label="Current Price" value={`$${singlePlan.current_price.toFixed(2)}`} />
            <MetricTile label="Entry Zone" value={`$${singlePlan.ideal_entry_low.toFixed(2)} – $${singlePlan.ideal_entry_high.toFixed(2)}`} />
            <MetricTile label="Breakout" value={`$${singlePlan.breakout_entry.toFixed(2)}`} />
          </div>

          {singleHistory && (
            <div className="rounded-lg border border-slate-200 bg-white p-4">
              <EntryChart plan={singlePlan} history={singleHistory} />
            </div>
          )}

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <div className="rounded-lg border border-slate-200 bg-white p-5">
              <h3 className="font-semibold text-slate-900">Entry Levels</h3>
              <p className="mt-2 text-sm text-slate-600">Buy zone: ${singlePlan.ideal_entry_low.toFixed(2)} – ${singlePlan.ideal_entry_high.toFixed(2)}</p>
              <p className="text-sm text-slate-600">Breakout trigger: ${singlePlan.breakout_entry.toFixed(2)}</p>
              <p className="text-sm text-slate-600">Stop loss: ${singlePlan.stop_loss.toFixed(2)}</p>
              <p className="text-sm text-slate-600">First target: ${singlePlan.first_target.toFixed(2)}</p>
            </div>
            <div className="rounded-lg border border-slate-200 bg-white p-5">
              <h3 className="font-semibold text-slate-900">Trend Read</h3>
              <p className="mt-2 text-sm text-slate-600">RSI: {singlePlan.rsi?.toFixed(2) ?? "N/A"}</p>
              <p className="text-sm text-slate-600">Short-term trend: {singlePlan.trend_up ? "Uptrend" : "Mixed / weak"}</p>
              <p className="text-sm text-slate-600">Long-term trend: {singlePlan.long_term_up ? "Long-term uptrend" : "Not fully supportive"}</p>
              <p className="text-sm text-slate-600">20D support / resistance: ${singlePlan.support_20.toFixed(2)} / ${singlePlan.resistance_20.toFixed(2)}</p>
            </div>
            <div className="rounded-lg border border-slate-200 bg-white p-5">
              <h3 className="flex items-center gap-1.5 font-semibold text-slate-900">
                Quant Forecast
                <MetricLabel term="Entry Quant Signal" />
              </h3>
              {singlePlan.quant_signal ? (
                <>
                  <p className="mt-2 text-sm text-slate-600">
                    10-day signal: <QuantSignalBadge signal={singlePlan.quant_signal} />
                  </p>
                  <p className="text-sm text-slate-600">
                    Expected return: {singlePlan.quant_expected_return_pct !== null ? `${singlePlan.quant_expected_return_pct >= 0 ? "+" : ""}${singlePlan.quant_expected_return_pct.toFixed(2)}%` : "N/A"}
                  </p>
                  <p className="text-sm text-slate-600">
                    Target price: {singlePlan.quant_target_price !== null ? `$${singlePlan.quant_target_price.toFixed(2)}` : "N/A"}
                  </p>
                </>
              ) : (
                <p className="mt-2 text-sm text-slate-400">Forecast unavailable for this ticker right now.</p>
              )}
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

function formatCell(value: string | number | null | undefined) {
  if (value === null || value === undefined) return "N/A";
  if (typeof value === "number") return Number.isInteger(value) ? value : value.toFixed(2);
  return value;
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs font-medium text-slate-500">{label}</label>
      {children}
    </div>
  );
}

