"use client";

import { useEffect, useState } from "react";

import InfoModal, { type ColumnInfo } from "@/components/InfoModal";
import TickerSearchInput from "@/components/TickerSearchInput";
import { ApiError, getStockRanking, getStockScore, getStockUniverses } from "@/lib/api";
import type { StockRankRow } from "@/lib/types";

const GOALS = ["Short Term", "Long Term"];

const DISPLAY_COLUMNS = [
  "Ticker",
  "Name",
  "Sector",
  "Price",
  "Score",
  "1M Return %",
  "3M Return %",
  "1Y Return %",
  "RSI",
];

const TEXT_COLUMNS = new Set(["Ticker", "Name", "Sector"]);

type SortDirection = "asc" | "desc";

const COLUMN_INFO: Record<string, ColumnInfo> = {
  Score: {
    title: "Score",
    body: [
      "A 0–100 blend of several metrics, each normalized against the other tickers in this result set (the best value in the current list scores highest on that metric, the worst scores lowest) — it's a relative ranking within this run, not an absolute grade. Re-running with a different universe can change a ticker's score even if nothing about the ticker itself changed.",
      "The metrics and their weights depend on the Goal you picked:",
      "\"Short Term\": 3-month return (30%), 1-month return (25%), RSI balance (15%), MACD signal strength (15%), volume strength (10%), 6-month volatility (5%, lower is better).",
      "\"Long Term\": 1-year return (28%), 3-year annualized return (20%), 6-month return (12%), revenue growth (12%), earnings growth (10%), forward P/E (8%, lower is better), 1-year max drawdown (10%, lower is better).",
    ],
  },
  RSI: {
    title: "RSI — Relative Strength Index",
    body: [
      "Measures how fast and how much a stock's price has moved recently, on a 0–100 scale, based on the ratio of average recent gains to average recent losses.",
      "Above 70 is often considered overbought (may be due for a pullback). Below 30 is often considered oversold (may be due for a bounce). Around 50 is neutral momentum.",
      "This app computes it over a standard 14-day window.",
      "It doesn't just reward high RSI: the ranking score prefers RSI near 55 — strong momentum without being overheated — and penalizes distance from 55 in either direction. A stock at RSI 90 scores worse than one at RSI 55, same as a weak stock sitting at RSI 20.",
      "It's only used for the \"Short Term\" goal (15% of that score). \"Long Term\" ranking doesn't use RSI at all — it weights fundamentals and multi-year returns instead.",
    ],
  },
};

export default function StockFinderPage() {
  const [mode, setMode] = useState<"rank" | "score">("rank");
  const [goal, setGoal] = useState("Short Term");
  const [universes, setUniverses] = useState<string[]>(["All"]);
  const [universe, setUniverse] = useState("All");
  const [ticker, setTicker] = useState("AAPL");

  const [results, setResults] = useState<StockRankRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [infoColumn, setInfoColumn] = useState<string | null>(null);

  function handleSort(col: string) {
    if (sortColumn === col) {
      setSortDirection((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortColumn(col);
      setSortDirection(TEXT_COLUMNS.has(col) ? "asc" : "desc");
    }
  }

  useEffect(() => {
    getStockUniverses()
      .then((res) => setUniverses(res.universes))
      .catch(() => {
        // Non-fatal: fall back to "All" already in state.
      });
  }, []);

  async function runSearch(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setHasSearched(true);
    setSortColumn(null);
    try {
      if (mode === "rank") {
        const res = await getStockRanking(goal, universe);
        setResults(res.results);
      } else {
        const res = await getStockScore(goal, ticker.trim().toUpperCase());
        setResults(res.result ? [res.result] : []);
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong.");
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  const winner = results[0];

  const sortedResults = sortColumn
    ? [...results].sort((a, b) => {
        const av = a[sortColumn];
        const bv = b[sortColumn];
        if (av == null && bv == null) return 0;
        if (av == null) return 1;
        if (bv == null) return -1;

        let cmp: number;
        if (typeof av === "number" && typeof bv === "number") {
          cmp = av - bv;
        } else {
          cmp = String(av).localeCompare(String(bv));
        }
        return sortDirection === "asc" ? cmp : -cmp;
      })
    : results;

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">Best Stock Finder</h1>
      <p className="mt-1 text-sm text-slate-500">
        Rank a stock universe by goal, or score one ticker directly.
      </p>

      <form onSubmit={runSearch} className="mt-6 flex flex-wrap items-end gap-3">
        <Field label="Goal">
          <select value={goal} onChange={(e) => setGoal(e.target.value)} className="rounded-md border border-slate-300 px-3 py-2 text-sm">
            {GOALS.map((g) => (
              <option key={g} value={g}>
                {g}
              </option>
            ))}
          </select>
        </Field>

        <Field label="Mode">
          <select value={mode} onChange={(e) => setMode(e.target.value as "rank" | "score")} className="rounded-md border border-slate-300 px-3 py-2 text-sm">
            <option value="rank">Rank a universe</option>
            <option value="score">Score one ticker</option>
          </select>
        </Field>

        {mode === "rank" ? (
          <Field label="Universe">
            <select value={universe} onChange={(e) => setUniverse(e.target.value)} className="rounded-md border border-slate-300 px-3 py-2 text-sm">
              {universes.map((u) => (
                <option key={u} value={u}>
                  {u}
                </option>
              ))}
            </select>
          </Field>
        ) : (
          <Field label="Ticker or company name">
            <TickerSearchInput value={ticker} onChange={setTicker} className="w-56 rounded-md border border-slate-300 px-3 py-2 text-sm" />
          </Field>
        )}

        <button
          type="submit"
          disabled={loading}
          className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
        >
          {loading ? "Scanning…" : "Run"}
        </button>
      </form>

      {loading && (
        <p className="mt-4 text-sm text-slate-500">
          {mode === "rank"
            ? "Scoring every ticker in the universe — first run for a universe can take a while, cached for an hour after."
            : "Scoring this ticker…"}
        </p>
      )}

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {!loading && hasSearched && results.length === 0 && !error && (
        <p className="mt-4 text-sm text-slate-500">No results for that selection.</p>
      )}

      {winner && !loading && (
        <div className="mt-6 flex flex-col gap-6">
          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <h2 className="text-lg font-semibold text-slate-900">
              Top Pick: {winner.Ticker} — {winner.Name}
            </h2>
            <p className="mt-1 text-sm text-slate-600">
              Scored highest for <strong>{goal}</strong>
              {mode === "rank" ? ` in ${universe}` : ""}.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricTile label="Score" value={`${winner.Score}/100`} />
            <MetricTile label="Price" value={`$${Number(winner.Price).toFixed(2)}`} />
            <MetricTile label="Sector" value={String(winner.Sector)} />
            <MetricTile
              label="1Y Return"
              value={winner["1Y Return %"] != null ? `${Number(winner["1Y Return %"]).toFixed(1)}%` : "N/A"}
            />
          </div>

          {results.length > 1 && (
            <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                    {DISPLAY_COLUMNS.map((col) => (
                      <th key={col} className="px-3 py-2">
                        <div className="flex items-center gap-1">
                          <button
                            type="button"
                            onClick={() => handleSort(col)}
                            className="flex items-center gap-1 uppercase tracking-wide text-slate-500 hover:text-slate-900"
                          >
                            {col}
                            <span className="text-[10px] text-slate-400">
                              {sortColumn === col ? (sortDirection === "asc" ? "▲" : "▼") : ""}
                            </span>
                          </button>
                          {COLUMN_INFO[col] && (
                            <button
                              type="button"
                              onClick={() => setInfoColumn(col)}
                              title={`What is ${col}?`}
                              className="flex h-4 w-4 items-center justify-center rounded-full border border-slate-300 text-[10px] font-normal normal-case text-slate-400 hover:border-slate-500 hover:text-slate-700"
                            >
                              i
                            </button>
                          )}
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sortedResults.map((row) => (
                    <tr key={row.Ticker} className="border-b border-slate-100 last:border-0">
                      {DISPLAY_COLUMNS.map((col) => (
                        <td key={col} className="px-3 py-2 text-slate-700">
                          {formatCell(row[col])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {infoColumn && COLUMN_INFO[infoColumn] && (
        <InfoModal info={COLUMN_INFO[infoColumn]} onClose={() => setInfoColumn(null)} />
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

function MetricTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="mt-1 text-lg font-semibold text-slate-900">{value}</p>
    </div>
  );
}
