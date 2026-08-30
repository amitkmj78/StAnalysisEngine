"use client";

import { useEffect, useState } from "react";

import InfoModal, { type ColumnInfo } from "@/components/InfoModal";
import TickerSearchInput from "@/components/TickerSearchInput";
import { ApiError, getFundCategories, getFundGoals, getFundRanking, getFundScore, getFundsByInception } from "@/lib/api";
import type { FundRankRow } from "@/lib/types";

const DISPLAY_COLUMNS = ["Ticker", "Fund", "Category", "Price", "Score", "Expense Ratio %", "1Y Return %", "3Y Annualized %"];
const INCEPTION_DISPLAY_COLUMNS = ["Ticker", "Fund", "Category", "Price", "Years Since Inception", "Since Inception Return %"];
const MIN_YEARS_OPTIONS = [5, 10, 15, 20, 25];

const TEXT_COLUMNS = new Set(["Ticker", "Fund", "Category"]);

type SortDirection = "asc" | "desc";

const COLUMN_INFO: Record<string, ColumnInfo> = {
  Score: {
    title: "Score",
    body: [
      "A 0–100 blend of several metrics, each normalized against the other funds in this result set (the best value in the current list scores highest on that metric, the worst scores lowest) — it's a relative ranking within this run, not an absolute grade. Re-running with a different category can change a fund's score even if nothing about the fund itself changed.",
      "The metrics and their weights depend on the Goal you picked:",
      "\"Balanced Core\": 1-year return (35%), 3-year annualized return (25%), expense ratio (20%, lower is better), 1-year volatility (10%, lower is better), 3-year max drawdown (10%, lower is better).",
      "\"Lowest Cost\": expense ratio (65%, lower is better), 3-year annualized return (20%), 1-year volatility (10%, lower is better), fund assets (5%).",
      "\"Best Growth\": 1-year return (50%), 3-year annualized return (35%), 1-year volatility (10%, lower is better), expense ratio (5%, lower is better).",
      "\"Most Stable\": 1-year volatility (45%, lower is better), 3-year max drawdown (30%, lower is better), expense ratio (15%, lower is better), 3-year annualized return (10%).",
    ],
  },
  "Expense Ratio %": {
    title: "Expense Ratio",
    body: [
      "The fund's annual operating fee, as a percentage of your invested assets — pulled live from Yahoo Finance's fund data for each ticker.",
      "It's deducted automatically from the fund's returns over the year, not billed to you separately, so a higher expense ratio quietly eats into your net return every year you hold it, compounding over time.",
      "Lower is better. It's factored into every Goal's Score, from a minor 5% weight under \"Best Growth\" up to being the single biggest factor (65%) under \"Lowest Cost.\"",
    ],
  },
  "Since Inception Return %": {
    title: "Since Inception Return",
    body: [
      "Real, point-in-time return from the fund's actual inception date to now (inception-date price vs. current price) — not an annualized figure, and not a prediction.",
      "Funds missing a disclosed inception date are excluded from this ranking rather than guessed at.",
    ],
  },
  "Years Since Inception": {
    title: "Years Since Inception",
    body: ["How long the fund has actually existed, based on its disclosed inception date — the minimum you set filters out anything younger than that."],
  },
};

export default function IndexFundPage() {
  const [mode, setMode] = useState<"rank" | "score" | "inception">("rank");
  const [goal, setGoal] = useState("Balanced Core");
  const [goals, setGoals] = useState<string[]>(["Balanced Core"]);
  const [categories, setCategories] = useState<string[]>(["All"]);
  const [category, setCategory] = useState("All");
  const [ticker, setTicker] = useState("VOO");
  const [minYears, setMinYears] = useState(10);

  const [results, setResults] = useState<FundRankRow[]>([]);
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
    getFundGoals().then((res) => setGoals(res.goals)).catch(() => {});
    getFundCategories().then((res) => setCategories(res.categories)).catch(() => {});
  }, []);

  async function runSearch(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setHasSearched(true);
    setSortColumn(null);
    try {
      if (mode === "rank") {
        const res = await getFundRanking(goal, category);
        setResults(res.results);
      } else if (mode === "inception") {
        const res = await getFundsByInception(minYears, category);
        setResults(res.results);
      } else {
        const res = await getFundScore(goal, ticker.trim().toUpperCase());
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
  const activeColumns = mode === "inception" ? INCEPTION_DISPLAY_COLUMNS : DISPLAY_COLUMNS;

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
      <h1 className="text-2xl font-semibold text-slate-900">Fund Screener</h1>
      <p className="mt-1 text-sm text-slate-500">Compare major index ETFs and rank by the goal that matters most to you.</p>

      <form onSubmit={runSearch} className="mt-6 flex flex-wrap items-end gap-3">
        <Field label="Goal">
          <select value={goal} onChange={(e) => setGoal(e.target.value)} className="input">
            {goals.map((g) => (
              <option key={g} value={g}>
                {g}
              </option>
            ))}
          </select>
        </Field>

        <Field label="Mode">
          <select value={mode} onChange={(e) => setMode(e.target.value as "rank" | "score" | "inception")} className="input">
            <option value="rank">Rank a category</option>
            <option value="score">Score one fund</option>
            <option value="inception">Since inception</option>
          </select>
        </Field>

        {mode === "score" ? (
          <Field label="Ticker or fund name">
            <TickerSearchInput value={ticker} onChange={setTicker} className="input w-56" />
          </Field>
        ) : (
          <Field label="Category">
            <select value={category} onChange={(e) => setCategory(e.target.value)} className="input">
              {categories.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </Field>
        )}

        {mode === "inception" && (
          <Field label="Minimum years since inception">
            <select value={minYears} onChange={(e) => setMinYears(Number(e.target.value))} className="input">
              {MIN_YEARS_OPTIONS.map((y) => (
                <option key={y} value={y}>{y}+ years</option>
              ))}
            </select>
          </Field>
        )}

        <button type="submit" disabled={loading} className="btn-primary">
          {loading ? "Scanning…" : "Run"}
        </button>
      </form>

      {loading && <p className="mt-4 text-sm text-slate-500">Pulling fund data — first run for a category can take a moment.</p>}
      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {!loading && hasSearched && results.length === 0 && !error && (
        <p className="mt-4 text-sm text-slate-500">No results for that selection.</p>
      )}

      {winner && !loading && (
        <div className="mt-6 flex flex-col gap-6">
          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <h2 className="text-lg font-semibold text-slate-900">
              Top Pick: {winner.Ticker} — {winner.Fund}
            </h2>
            <p className="mt-1 text-sm text-slate-600">
              {mode === "inception"
                ? `Best since-inception return among ${category === "All" ? "all funds" : category} with ${minYears}+ years of history.`
                : (
                  <>
                    Scored highest for <strong>{goal}</strong>
                    {mode === "rank" ? ` in ${category}` : ""}.
                  </>
                )}
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {mode === "inception" ? (
              <>
                <MetricTile
                  label="Since Inception"
                  value={winner["Since Inception Return %"] != null ? `${Number(winner["Since Inception Return %"]).toFixed(1)}%` : "N/A"}
                  onInfoClick={() => setInfoColumn("Since Inception Return %")}
                />
                <MetricTile
                  label="Years Since Inception"
                  value={winner["Years Since Inception"] != null ? `${Number(winner["Years Since Inception"]).toFixed(1)}y` : "N/A"}
                  onInfoClick={() => setInfoColumn("Years Since Inception")}
                />
                <MetricTile label="Price" value={`$${Number(winner.Price).toFixed(2)}`} />
                <MetricTile
                  label="Expense Ratio"
                  value={winner["Expense Ratio %"] != null ? `${Number(winner["Expense Ratio %"]).toFixed(2)}%` : "N/A"}
                />
              </>
            ) : (
              <>
                <MetricTile label="Score" value={`${winner.Score}/100`} onInfoClick={() => setInfoColumn("Score")} />
                <MetricTile label="Price" value={`$${Number(winner.Price).toFixed(2)}`} />
                <MetricTile
                  label="Expense Ratio"
                  value={winner["Expense Ratio %"] != null ? `${Number(winner["Expense Ratio %"]).toFixed(2)}%` : "N/A"}
                />
                <MetricTile
                  label="1Y Return"
                  value={winner["1Y Return %"] != null ? `${Number(winner["1Y Return %"]).toFixed(1)}%` : "N/A"}
                />
              </>
            )}
          </div>

          {results.length > 1 && (
            <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                    {activeColumns.map((col) => (
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
                      {activeColumns.map((col) => (
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
        {onInfoClick && (
          <button
            type="button"
            onClick={onInfoClick}
            title={`What is ${label}?`}
            className="flex h-4 w-4 items-center justify-center rounded-full border border-slate-300 text-[10px] font-normal normal-case text-slate-400 hover:border-slate-500 hover:text-slate-700"
          >
            i
          </button>
        )}
      </p>
      <p className="mt-1 text-lg font-semibold text-slate-900">{value}</p>
    </div>
  );
}
