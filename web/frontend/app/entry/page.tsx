"use client";

import { useEffect, useState } from "react";

import EntryChart from "@/components/entry/EntryChart";
import InfoModal, { type ColumnInfo } from "@/components/InfoModal";
import { ApiError, getEntryPlan, getEntryScan, getEntryUniverses } from "@/lib/api";
import type { EntryHistory, EntryPlan, EntryScanRow } from "@/lib/types";

const SCAN_COLUMNS = ["Ticker", "Signal", "Entry Score", "Current Price", "Entry Low", "Entry High", "Stop Loss", "First Target", "RSI"];

const COLUMN_INFO: Record<string, ColumnInfo> = {
  "Entry Score": {
    title: "Entry Score",
    body: [
      "A score built from this ticker's own technical setup right now — not a percentile rank against other tickers like the Screener's Score, so it can be compared across different scans and doesn't shift just because the universe changed. 100 is a strong, well-rounded setup; a genuinely exceptional one (strong on every factor at once) can score above it — there's no artificial ceiling hiding real differences between setups.",
      "Signal strength — up to 90 points: the Signal label (Wait = 0, Wait for Pullback = 1, Watch for Reversal = 2, Breakout Entry = 3, Buy on Pullback = 4, Buy Now = 5) times 18.",
      "RSI closeness to 52 — up to 20 points: full 20 at RSI exactly 52 (strong momentum without being overheated), losing a point per unit away, reaching 0 once RSI is 20+ points from 52 in either direction.",
      "Bullish short-term momentum — +14 if present, except for \"Buy Now\"/\"Breakout Entry\" where it's already required to earn that label (counted once via signal strength, not twice).",
      "Short-term uptrend — +14 if present, except for \"Buy Now\"/\"Wait for Pullback\" where it's already required to earn that label.",
      "Long-term uptrend — +10 if present, except for \"Buy on Pullback\" where it's already required to earn that label.",
      "Proximity to a level — near 20-day support: +12 (except for \"Buy on Pullback\", already required). Otherwise, near a breakout level: +8 (except for \"Breakout Entry\", already required). Only one of these ever applies.",
      "Above-average volume — up to +12: 0 at today's volume equal to its 20-day average, scaling up to the full 12 points once volume is 60%+ above that average.",
      "The exceptions above matter: a stock's signal label already implies certain conditions (e.g. \"Buy Now\" requires an uptrend with momentum), so re-awarding those same points on top would double-count the same evidence. The remaining points only come from genuine extra strength beyond what the label already guarantees — nothing here is capped, so two stocks with the same signal can still show meaningfully different scores.",
    ],
  },
  Signal: {
    title: "Signal — what each label means",
    body: [
      "\"Buy Now\" — short-term uptrend with supportive momentum, and price isn't overextended (RSI below 70). The most straightforward setup.",
      "\"Buy on Pullback\" — the longer-term trend is still intact, and price has pulled back near recent support.",
      "\"Breakout Entry\" — price is pressing against recent resistance with supportive momentum, close to breaking out.",
      "\"Watch for Reversal\" — the stock looks oversold (RSI 35 or below). Washed out, but wait for confirmation before entering.",
      "\"Wait for Pullback\" — the trend is healthy, but price looks stretched (RSI 70+). Healthy trend, risky entry point right now.",
      "\"Wait\" — no clear edge either way; the setup is mixed.",
      "Ranked strongest to weakest for scan ordering: Buy Now → Buy on Pullback → Breakout Entry → Watch for Reversal → Wait for Pullback → Wait.",
    ],
  },
};

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
  const [sortColumn, setSortColumn] = useState("Entry Score");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const [scanResults, setScanResults] = useState<EntryScanRow[]>([]);
  const [singlePlan, setSinglePlan] = useState<EntryPlan | null>(null);
  const [singleHistory, setSingleHistory] = useState<EntryHistory | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [infoColumn, setInfoColumn] = useState<string | null>(null);

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
        const res = await getEntryScan(assetType, universe, topN);
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
                <InfoIcon onClick={() => setInfoColumn("Signal")} />
              </div>
            </div>
            <div className="mt-3 flex items-center gap-3">
              <span className="text-xs font-medium text-slate-500">Entry Score</span>
              <EntryScoreBar score={Number(winner["Entry Score"])} />
              <InfoIcon onClick={() => setInfoColumn("Entry Score")} />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricTile label="Entry Score" value={`${winner["Entry Score"]}`} onInfoClick={() => setInfoColumn("Entry Score")} />
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
                      {COLUMN_INFO[col] && (
                        <InfoIcon title={`What is ${col}?`} onClick={() => setInfoColumn(col)} />
                      )}
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
                  <SignalBadge signal={String(row.Signal)} />
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
                <InfoIcon onClick={() => setInfoColumn("Signal")} />
              </div>
            </div>
            <p className="mt-2 text-sm text-slate-600">{singlePlan.summary}</p>
          </div>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricTile label="Entry Score" value={`${singlePlan.entry_score}`} onInfoClick={() => setInfoColumn("Entry Score")} />
            <MetricTile label="Current Price" value={`$${singlePlan.current_price.toFixed(2)}`} />
            <MetricTile label="Entry Zone" value={`$${singlePlan.ideal_entry_low.toFixed(2)} – $${singlePlan.ideal_entry_high.toFixed(2)}`} />
            <MetricTile label="Breakout" value={`$${singlePlan.breakout_entry.toFixed(2)}`} />
          </div>

          {singleHistory && (
            <div className="rounded-lg border border-slate-200 bg-white p-4">
              <EntryChart plan={singlePlan} history={singleHistory} />
            </div>
          )}

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
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
          </div>
        </div>
      )}

      {infoColumn && COLUMN_INFO[infoColumn] && (
        <InfoModal info={COLUMN_INFO[infoColumn]} onClose={() => setInfoColumn(null)} />
      )}
    </div>
  );
}

function InfoIcon({ onClick, title }: { onClick: () => void; title?: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title ?? "What does this mean?"}
      className="flex h-4 w-4 items-center justify-center rounded-full border border-slate-300 text-[10px] font-normal normal-case text-slate-400 hover:border-slate-500 hover:text-slate-700"
    >
      i
    </button>
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

function MetricTile({
  label,
  value,
  onInfoClick,
}: {
  label: string;
  value: string;
  onInfoClick?: () => void;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <p className="flex items-center gap-1 text-xs text-slate-500">
        {label}
        {onInfoClick && <InfoIcon title={`What is ${label}?`} onClick={onInfoClick} />}
      </p>
      <p className="mt-1 text-lg font-semibold text-slate-900">{value}</p>
    </div>
  );
}
