"use client";

import { Fragment, useEffect, useMemo, useState } from "react";

import InfoModal, { type ColumnInfo } from "@/components/InfoModal";
import TickerSearchInput from "@/components/TickerSearchInput";
import { ApiError, getFundCategories, getFundGoals, getFundRanking, getFundScore } from "@/lib/api";
import type { FundGoal, FundRankRow } from "@/lib/types";

type SortDirection = "asc" | "desc";

const WINDOW_OPTIONS: { value: string; label: string }[] = [
  { value: "1y", label: "1Y" },
  { value: "3y", label: "3Y" },
  { value: "5y", label: "5Y" },
  { value: "10y", label: "10Y" },
  { value: "max_common", label: "Max common period" },
];

const TEXT_COLUMNS = new Set(["Ticker", "Fund", "Category", "Inception Date"]);

const ALL_COLUMNS: { key: string; label: string; defaultVisible: boolean }[] = [
  { key: "Ticker", label: "Ticker", defaultVisible: true },
  { key: "Fund", label: "Name", defaultVisible: true },
  { key: "Category", label: "Category", defaultVisible: true },
  { key: "Score", label: "Score", defaultVisible: true },
  { key: "Price", label: "Price", defaultVisible: true },
  { key: "Expense Ratio %", label: "Expense Ratio", defaultVisible: true },
  { key: "Tracking Difference %", label: "Tracking Diff.", defaultVisible: false },
  { key: "Assets ($B)", label: "AUM ($B)", defaultVisible: true },
  { key: "Avg Daily Volume", label: "Avg Daily Volume", defaultVisible: false },
  { key: "Bid/Ask Spread %", label: "Bid/Ask Spread (live)", defaultVisible: false },
  { key: "CAGR (Window) %", label: "CAGR (Window)", defaultVisible: true },
  { key: "Max Drawdown (Window) %", label: "Max Drawdown (Window)", defaultVisible: true },
  { key: "Std Dev (Window) %", label: "Std Dev (Window)", defaultVisible: false },
  { key: "Sharpe (Window)", label: "Sharpe (Window)", defaultVisible: true },
  { key: "Sortino (Window)", label: "Sortino (Window)", defaultVisible: false },
  { key: "Distribution Yield %", label: "Distribution Yield", defaultVisible: false },
  { key: "Turnover %", label: "Turnover", defaultVisible: false },
  { key: "Inception Date", label: "Inception Date", defaultVisible: false },
];

const COLUMN_INFO: Record<string, ColumnInfo> = {
  Score: {
    title: "Score",
    body: [
      "A relative rank within this fund's own category: every metric is z-scored against the other funds in the same Category before weighting, so a fund is only ever compared to real peers — a bond fund is never scored against an equity fund's volatility.",
      "It isn't a 0–100 grade. 0 means \"about average for its category\" on the metrics that matter to this Goal; positive means better than its peers, negative means worse — and the further from 0, the bigger the gap. A very small peer group (a handful of nearly-identical funds plus one real outlier) can push a Score well beyond ±100.",
      "The metrics and weights depend on the Goal you picked — see the weights strip above the table for the exact breakdown of whichever Goal is active.",
      "Expand a row (the ▸ on the left) to see the Return/Risk/Cost/Liquidity sub-scores and the raw metric behind each.",
    ],
  },
  "Expense Ratio %": {
    title: "Expense Ratio",
    body: [
      "The fund's annual operating fee, as a percentage of your invested assets — pulled live from Yahoo Finance's fund data for each ticker.",
      "It's deducted automatically from the fund's returns over the year, so a higher expense ratio quietly eats into your net return every year you hold it, compounding over time. Lower is better.",
    ],
  },
  "Tracking Difference %": {
    title: "Tracking Difference",
    body: [
      "This fund's CAGR minus its benchmark index's own CAGR over the selected window — how much the fund gave up (or gained) versus the index it tracks, beyond the stated expense ratio.",
      "Only computed for funds mapped to a benchmark with an unambiguous, free index ticker (the major S&P/Nasdaq/Russell indices). Everything else — Dow-Jones-branded, MSCI/FTSE international, and every bond index — shows N/A rather than a guessed number.",
    ],
  },
  "Assets ($B)": {
    title: "Fund Assets (AUM)",
    body: ["Total net assets under management, in billions — a rough proxy for how liquid and established a fund is."],
  },
  "Avg Daily Volume": {
    title: "Average Daily Volume",
    body: ["Shares traded per day on average (3-month average where available) — higher volume generally means tighter spreads and easier entry/exit at the quoted price."],
  },
  "Bid/Ask Spread %": {
    title: "Bid/Ask Spread (live)",
    body: [
      "A live snapshot of (ask − bid) / midpoint, taken at the time the data was last refreshed — not a historical median, since no historical bid/ask series exists via this data source.",
      "Smaller is better: it's roughly what you give up in one round-trip just from the spread, separate from any commission.",
    ],
  },
  "CAGR (Window) %": {
    title: "CAGR (selected window)",
    body: ["Compound annual growth rate over the currently selected Window, using total return with dividends reinvested — not a simple average of yearly returns."],
  },
  "Max Drawdown (Window) %": {
    title: "Max Drawdown (selected window)",
    body: ["The largest peak-to-trough decline within the selected Window — how much this fund lost from its best point before recovering, expressed as a positive percentage."],
  },
  "Std Dev (Window) %": {
    title: "Std. Dev. (selected window)",
    body: ["Annualized standard deviation of daily returns over the selected Window — a measure of how bumpy the ride was, not of long-run direction."],
  },
  "Sharpe (Window)": {
    title: "Sharpe Ratio (selected window)",
    body: ["Annualized return divided by annualized volatility over the selected Window, assuming a 0% risk-free rate. Higher means more return per unit of risk taken."],
  },
  "Sortino (Window)": {
    title: "Sortino Ratio (selected window)",
    body: ["Like Sharpe, but only penalizes downside volatility (losing days), not all volatility — a fund that's volatile only on the way up scores better here than on Sharpe."],
  },
  "Distribution Yield %": {
    title: "Distribution Yield",
    body: ["The fund's trailing distribution yield, pulled live from Yahoo Finance."],
  },
  "Turnover %": {
    title: "Turnover",
    body: [
      "Annual holdings turnover ratio, as disclosed by the fund. This figure is not reliably available via this data source for most funds — including large, well-known funds like SPY and BND — so it shows N/A far more often than not. Shown anyway rather than hidden, so the gap is visible.",
    ],
  },
};

function windowLabel(w: string) {
  return WINDOW_OPTIONS.find((o) => o.value === w)?.label ?? w;
}

function formatDateLabel(dateStr: string | null) {
  if (!dateStr) return null;
  const d = new Date(`${dateStr}T00:00:00`);
  if (Number.isNaN(d.getTime())) return dateStr;
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

export default function IndexFundPage() {
  const [mode, setMode] = useState<"rank" | "score">("rank");
  const [goal, setGoal] = useState("Balanced Core");
  const [goals, setGoals] = useState<FundGoal[]>([]);
  const [categories, setCategories] = useState<string[]>(["All"]);
  const [category, setCategory] = useState("US Large Blend");
  const [windowValue, setWindowValue] = useState("5y");
  const [ticker, setTicker] = useState("VOO");
  const [debouncedTicker, setDebouncedTicker] = useState("VOO");
  const [customWeights, setCustomWeights] = useState<Record<string, number>>({});

  const [results, setResults] = useState<FundRankRow[]>([]);
  const [windowMeta, setWindowMeta] = useState<{ start: string | null; end: string | null; error: string | null }>({
    start: null,
    end: null,
    error: null,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [infoColumn, setInfoColumn] = useState<string | null>(null);
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
  const [visibleColumns, setVisibleColumns] = useState<Set<string>>(
    new Set(ALL_COLUMNS.filter((c) => c.defaultVisible).map((c) => c.key)),
  );

  function handleSort(col: string) {
    if (sortColumn === col) {
      setSortDirection((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortColumn(col);
      setSortDirection(TEXT_COLUMNS.has(col) ? "asc" : "desc");
    }
  }

  function toggleRow(t: string) {
    setExpandedRows((prev) => {
      const next = new Set(prev);
      if (next.has(t)) next.delete(t);
      else next.add(t);
      return next;
    });
  }

  function toggleColumn(key: string) {
    setVisibleColumns((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  useEffect(() => {
    getFundGoals()
      .then((res) => {
        setGoals(res.goals);
        const custom = res.goals.find((g) => g.name === "Custom");
        if (custom) {
          setCustomWeights((prev) =>
            Object.keys(prev).length > 0 ? prev : Object.fromEntries(custom.weights.map((w) => [w.metric, 1])),
          );
        }
      })
      .catch(() => {});
    getFundCategories().then((res) => setCategories(res.categories)).catch(() => {});
  }, []);

  // Debounce the free-text ticker (TickerSearchInput fires onChange every
  // keystroke) so Score mode doesn't fire a request per character typed.
  useEffect(() => {
    const t = setTimeout(() => setDebouncedTicker(ticker.trim().toUpperCase()), 400);
    return () => clearTimeout(t);
  }, [ticker]);

  const activeGoal = goals.find((g) => g.name === goal);
  const customWeightsTotal = Object.values(customWeights).reduce((sum, v) => sum + (v > 0 ? v : 0), 0);

  const customWeightsKey = goal === "Custom" ? JSON.stringify(customWeights) : "";

  // FS-1: results load automatically, and any control change re-runs the
  // search — no "Run" button. Debounced ticker keeps Score mode from
  // re-fetching per keystroke.
  useEffect(() => {
    if (mode === "score" && !debouncedTicker) return;
    if (goal === "Custom" && customWeightsTotal <= 0) return;

    let cancelled = false;
    setLoading(true);
    setError(null);

    const weights = goal === "Custom" ? customWeights : undefined;
    const request =
      mode === "rank" ? getFundRanking(goal, category, windowValue, weights) : getFundScore(goal, debouncedTicker, windowValue, weights);

    request
      .then((res) => {
        if (cancelled) return;
        setHasSearched(true);
        setSortColumn(null);
        if ("results" in res) {
          setResults(res.results);
        } else {
          setResults(res.result ? [res.result] : []);
        }
        setWindowMeta({ start: res.start, end: res.end, error: res.error });
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof ApiError ? err.message : "Something went wrong.");
        setResults([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, goal, category, windowValue, debouncedTicker, customWeightsKey]);

  const winner = results[0];
  const activeColumns = useMemo(() => ALL_COLUMNS.filter((c) => visibleColumns.has(c.key)), [visibleColumns]);

  function sortRows(rows: FundRankRow[]) {
    if (!sortColumn) return rows;
    return [...rows].sort((a, b) => {
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
    });
  }

  const sortedResults = sortRows(results);

  // FS-3: category="All" groups results by Category instead of one flat
  // ranked list. Grouping happens after sorting so a within-category sort
  // is preserved inside each group.
  const groupedResults = useMemo(() => {
    if (category !== "All") return null;
    const groups = new Map<string, FundRankRow[]>();
    for (const row of sortedResults) {
      const key = row.Category || "Uncategorized";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(row);
    }
    return groups;
  }, [category, sortedResults]);

  const windowStart = formatDateLabel(windowMeta.start);
  const windowEnd = formatDateLabel(windowMeta.end);

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">Fund Screener</h1>
      <p className="mt-1 text-sm text-slate-500">Compare major index ETFs and rank by the goal that matters most to you.</p>

      <div className="mt-6 flex flex-wrap items-end gap-3">
        <Field label="Goal">
          <select value={goal} onChange={(e) => setGoal(e.target.value)} className="input">
            {goals.map((g) => (
              <option key={g.name} value={g.name}>
                {g.name}
              </option>
            ))}
          </select>
        </Field>

        <Field label="Mode">
          <select value={mode} onChange={(e) => setMode(e.target.value as "rank" | "score")} className="input">
            <option value="rank">Rank a category</option>
            <option value="score">Score one fund</option>
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

        <Field label="Window">
          <select value={windowValue} onChange={(e) => setWindowValue(e.target.value)} className="input">
            {WINDOW_OPTIONS.map((w) => (
              <option key={w.value} value={w.value}>
                {w.label}
              </option>
            ))}
          </select>
        </Field>

        {loading && <span className="pb-2 text-xs text-slate-400">Refreshing…</span>}
      </div>

      {windowStart && windowEnd && (
        <p className="mt-2 text-xs text-slate-400">
          Showing {windowLabel(windowValue)}: {windowStart} – {windowEnd}
        </p>
      )}

      {activeGoal && (
        <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-3">
          {goal !== "Custom" ? (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-medium uppercase tracking-wide text-slate-400">Weights</span>
              {activeGoal.weights.map((w) => (
                <span key={w.metric} className="rounded-full border border-slate-200 bg-white px-2 py-0.5 text-xs text-slate-600">
                  {w.label} {Math.round((w.weight ?? 0) * 100)}%{w.lower_is_better ? " (lower is better)" : ""}
                </span>
              ))}
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              <span className="text-xs font-medium uppercase tracking-wide text-slate-400">
                Custom weights ({customWeightsTotal > 0 ? "normalized to 100%" : "set at least one above 0"})
              </span>
              <div className="grid grid-cols-1 gap-x-4 gap-y-2 sm:grid-cols-2">
                {activeGoal.weights.map((w) => {
                  const raw = customWeights[w.metric] ?? 0;
                  const pct = customWeightsTotal > 0 ? Math.round((raw / customWeightsTotal) * 100) : 0;
                  return (
                    <label key={w.metric} className="flex items-center gap-2 text-xs text-slate-600">
                      <span className="w-40 shrink-0">{w.label}</span>
                      <input
                        type="range"
                        min={0}
                        max={100}
                        value={raw}
                        onChange={(e) =>
                          setCustomWeights((prev) => ({ ...prev, [w.metric]: Number(e.target.value) }))
                        }
                        className="flex-1"
                      />
                      <span className="w-10 shrink-0 text-right tabular-nums">{pct}%</span>
                    </label>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {windowMeta.error && <p className="mt-4 rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-700">{windowMeta.error}</p>}
      {!loading && hasSearched && results.length === 0 && !error && !windowMeta.error && (
        <p className="mt-4 text-sm text-slate-500">No results for that selection.</p>
      )}

      {winner && !loading && category !== "All" && (
        <div className="mt-6 flex flex-col gap-6">
          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <h2 className="text-lg font-semibold text-slate-900">
              Top Pick: {winner.Ticker} — {winner.Fund}
            </h2>
            <p className="mt-1 text-sm text-slate-600">
              Scored highest for <strong>{goal}</strong>
              {mode === "rank" ? ` in ${category}` : ""}.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricTile label="Score" value={`${winner.Score >= 0 ? "+" : ""}${winner.Score}`} onInfoClick={() => setInfoColumn("Score")} />
            <MetricTile label="Price" value={`$${Number(winner.Price).toFixed(2)}`} />
            <MetricTile
              label="Expense Ratio"
              value={winner["Expense Ratio %"] != null ? `${Number(winner["Expense Ratio %"]).toFixed(2)}%` : "N/A"}
            />
            <MetricTile
              label="CAGR (Window)"
              value={winner["CAGR (Window) %"] != null ? `${Number(winner["CAGR (Window) %"]).toFixed(1)}%` : "N/A"}
            />
          </div>
        </div>
      )}

      {results.length > 0 && (
        <div className="mt-6 flex flex-col gap-6">
          <details className="rounded-lg border border-slate-200 bg-white p-3 text-sm">
            <summary className="cursor-pointer text-xs font-medium uppercase tracking-wide text-slate-500">Columns</summary>
            <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3">
              {ALL_COLUMNS.map((c) => (
                <label key={c.key} className="flex items-center gap-2 text-xs text-slate-600">
                  <input type="checkbox" checked={visibleColumns.has(c.key)} onChange={() => toggleColumn(c.key)} />
                  {c.label}
                </label>
              ))}
            </div>
          </details>

          {groupedResults
            ? Array.from(groupedResults.entries()).map(([groupName, rows]) => (
                <div key={groupName}>
                  <h3 className="mb-2 text-sm font-semibold text-slate-700">
                    {groupName} <span className="font-normal text-slate-400">({rows.length})</span>
                  </h3>
                  <FundTable
                    rows={rows}
                    columns={activeColumns}
                    sortColumn={sortColumn}
                    sortDirection={sortDirection}
                    onSort={handleSort}
                    onInfoClick={setInfoColumn}
                    expandedRows={expandedRows}
                    onToggleRow={toggleRow}
                  />
                </div>
              ))
            : results.length > 1 && (
                <FundTable
                  rows={sortedResults}
                  columns={activeColumns}
                  sortColumn={sortColumn}
                  sortDirection={sortDirection}
                  onSort={handleSort}
                  onInfoClick={setInfoColumn}
                  expandedRows={expandedRows}
                  onToggleRow={toggleRow}
                />
              )}
        </div>
      )}

      {infoColumn && COLUMN_INFO[infoColumn] && <InfoModal info={COLUMN_INFO[infoColumn]} onClose={() => setInfoColumn(null)} />}
    </div>
  );
}

function FundTable({
  rows,
  columns,
  sortColumn,
  sortDirection,
  onSort,
  onInfoClick,
  expandedRows,
  onToggleRow,
}: {
  rows: FundRankRow[];
  columns: { key: string; label: string }[];
  sortColumn: string | null;
  sortDirection: SortDirection;
  onSort: (col: string) => void;
  onInfoClick: (col: string) => void;
  expandedRows: Set<string>;
  onToggleRow: (ticker: string) => void;
}) {
  return (
    <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
            <th className="w-8 px-2 py-2" />
            {columns.map((col) => (
              <th key={col.key} className="px-3 py-2">
                <div className="flex items-center gap-1">
                  <button
                    type="button"
                    onClick={() => onSort(col.key)}
                    className="flex items-center gap-1 uppercase tracking-wide text-slate-500 hover:text-slate-900"
                  >
                    {col.label}
                    <span className="text-[10px] text-slate-400">
                      {sortColumn === col.key ? (sortDirection === "asc" ? "▲" : "▼") : ""}
                    </span>
                  </button>
                  {COLUMN_INFO[col.key] && (
                    <button
                      type="button"
                      onClick={() => onInfoClick(col.key)}
                      title={`What is ${col.label}?`}
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
          {rows.map((row) => {
            const expanded = expandedRows.has(row.Ticker);
            return (
              <Fragment key={row.Ticker}>
                <tr className="border-b border-slate-100 last:border-0">
                  <td className="px-2 py-2">
                    <button
                      type="button"
                      onClick={() => onToggleRow(row.Ticker)}
                      className="flex h-5 w-5 items-center justify-center rounded text-slate-400 hover:bg-slate-100 hover:text-slate-700"
                      title="Show score breakdown"
                    >
                      {expanded ? "▾" : "▸"}
                    </button>
                  </td>
                  {columns.map((col) => (
                    <td key={col.key} className="px-3 py-2 text-slate-700">
                      {formatCell(row[col.key])}
                    </td>
                  ))}
                </tr>
                {expanded && (
                  <tr className="border-b border-slate-100 bg-slate-50 last:border-0">
                    <td />
                    <td colSpan={columns.length} className="px-3 py-3">
                      <ScoreBreakdown row={row} />
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ScoreBreakdown({ row }: { row: FundRankRow }) {
  const breakdown = row._breakdown;
  if (!breakdown || Object.keys(breakdown).length === 0) {
    return <p className="text-xs text-slate-400">No score breakdown available for this row.</p>;
  }
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {Object.entries(breakdown).map(([bucket, data]) => (
        <div key={bucket} className="rounded-md border border-slate-200 bg-white p-3">
          <p className="flex items-center justify-between text-xs font-semibold text-slate-700">
            {bucket}
            <span className="tabular-nums text-slate-400">{data.sub_score >= 0 ? "+" : ""}{(data.sub_score * 100).toFixed(1)}</span>
          </p>
          <ul className="mt-2 flex flex-col gap-1">
            {data.metrics.map((m) => (
              <li key={m.key} className="flex items-center justify-between text-xs text-slate-500">
                <span>{m.label}</span>
                <span className="tabular-nums text-slate-700">
                  {m.raw_value != null ? `${m.raw_value.toFixed(2)}${m.unit}` : "N/A"}
                </span>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

function formatCell(value: string | number | null | undefined | Record<string, unknown>) {
  if (value === null || value === undefined) return "N/A";
  if (typeof value === "object") return "";
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
