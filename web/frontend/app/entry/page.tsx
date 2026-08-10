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
      "A 0–100 score built from this ticker's own technical setup right now — not a percentile rank against other tickers like the Screener's Score, so it can be compared across different scans and doesn't shift just because the universe changed.",
      "Signal strength — up to 90 points: the Signal label (Wait = 0, Wait for Pullback = 1, Watch for Reversal = 2, Breakout Entry = 3, Buy on Pullback = 4, Buy Now = 5) times 18.",
      "RSI closeness to 52 — up to 20 points: full 20 at RSI exactly 52 (strong momentum without being overheated), losing a point per unit away, reaching 0 once RSI is 20+ points from 52 in either direction.",
      "Bullish short-term momentum — +14 flat if present.",
      "Short-term uptrend — +14 flat if present.",
      "Long-term uptrend — +10 flat if present.",
      "Proximity to a level — near 20-day support: +12. Otherwise, near a breakout level: +8. Only one of these ever applies.",
      "Above-average volume — up to +12: 0 at today's volume equal to its 20-day average, scaling up to the full 12 points once volume is 60%+ above that average.",
      "These can add up to more than 100 in the best case — the raw total is capped at 100, not rescaled, so hitting the cap just means \"maxed out on setup quality,\" not a mathematical ceiling being approached smoothly.",
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

export default function EntryPage() {
  const [assetType, setAssetType] = useState<"Fund" | "Stock">("Stock");
  const [mode, setMode] = useState<"scan" | "check">("scan");
  const [universes, setUniverses] = useState<string[]>(["All"]);
  const [universe, setUniverse] = useState("All");
  const [topN, setTopN] = useState(5);
  const [ticker, setTicker] = useState("AAPL");

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

  const winner = scanResults[0];

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
          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <h2 className="text-lg font-semibold text-slate-900">Top Entry Right Now: {winner.Ticker}</h2>
            <p className="mt-1 text-sm text-slate-600">
              Signal: <strong>{winner.Signal}</strong> <InfoIcon onClick={() => setInfoColumn("Signal")} /> —
              strongest current {assetType.toLowerCase()} entry setup in {universe}.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricTile label="Entry Score" value={`${winner["Entry Score"]}/100`} onInfoClick={() => setInfoColumn("Entry Score")} />
            <MetricTile label="Current Price" value={`$${Number(winner["Current Price"]).toFixed(2)}`} />
            <MetricTile label="Entry Low" value={`$${Number(winner["Entry Low"]).toFixed(2)}`} />
            <MetricTile label="Entry High" value={`$${Number(winner["Entry High"]).toFixed(2)}`} />
          </div>

          <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                  {SCAN_COLUMNS.map((col) => (
                    <th key={col} className="px-3 py-2">
                      <div className="flex items-center gap-1">
                        {col}
                        {COLUMN_INFO[col] && <InfoIcon title={`What is ${col}?`} onClick={() => setInfoColumn(col)} />}
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {scanResults.map((row) => (
                  <tr key={row.Ticker} className="border-b border-slate-100 last:border-0">
                    {SCAN_COLUMNS.map((col) => (
                      <td key={col} className="px-3 py-2 text-slate-700">
                        {formatCell(row[col])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {mode === "check" && singlePlan && !loading && (
        <div className="mt-6 flex flex-col gap-6">
          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <h2 className="text-lg font-semibold text-slate-900">{singlePlan.ticker} Entry Snapshot</h2>
            <p className="mt-1 text-sm text-slate-600">
              Signal: <strong>{singlePlan.signal}</strong> <InfoIcon onClick={() => setInfoColumn("Signal")} />.{" "}
              {singlePlan.summary}
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricTile label="Entry Score" value={`${singlePlan.entry_score}/100`} onInfoClick={() => setInfoColumn("Entry Score")} />
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
