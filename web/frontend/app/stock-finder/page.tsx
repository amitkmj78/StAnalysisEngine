"use client";

import { Fragment, useEffect, useMemo, useState } from "react";
import Link from "next/link";

import InfoModal, { type ColumnInfo } from "@/components/InfoModal";
import TickerSearchInput from "@/components/TickerSearchInput";
import {
  ApiError,
  createWatchlistAlert,
  deleteScreen,
  getScreens,
  getStockRanking,
  getStockScore,
  getStockUniverses,
  saveScreen,
} from "@/lib/api";
import type { AlertConditionType, SavedScreen, ScreenSnapshotRow, StockRankRow } from "@/lib/types";

const GOALS = ["Short Term", "Long Term"];

// Every field _build_stock_row (services/stock_finder_service.py) returns,
// in display order — the column picker offers all of these, not just the
// small default subset shown out of the box.
const ALL_COLUMNS = [
  "Ticker",
  "Name",
  "Sector",
  "Price",
  "Score",
  "Market Cap ($B)",
  "Forward PE",
  "Revenue Growth %",
  "Earnings Growth %",
  "1M Return %",
  "3M Return %",
  "6M Return %",
  "1Y Return %",
  "3Y Annualized %",
  "Return 10D %",
  "Return 30D %",
  "Return 60D %",
  "Return 90D %",
  "RSI",
  "RSI Balance",
  "MACD Strength",
  "Volume Strength %",
  "6M Volatility %",
  "1Y Max Drawdown %",
];

const DEFAULT_COLUMNS = ["Ticker", "Name", "Sector", "Price", "Score", "1M Return %", "3M Return %", "1Y Return %", "RSI"];
const REQUIRED_COLUMN = "Ticker";
const COLUMNS_STORAGE_KEY = "stanalysisengine.stockFinderColumns";

const TEXT_COLUMNS = new Set(["Ticker", "Name", "Sector"]);

type SortDirection = "asc" | "desc";
type SortKey = { column: string; direction: SortDirection };
const MAX_SORT_KEYS = 3;

interface FilterState {
  marketCapMin: string;
  marketCapMax: string;
  forwardPeMin: string;
  forwardPeMax: string;
  volumeStrengthMin: string;
  sectors: string[];
}

const EMPTY_FILTERS: FilterState = {
  marketCapMin: "",
  marketCapMax: "",
  forwardPeMin: "",
  forwardPeMax: "",
  volumeStrengthMin: "",
  sectors: [],
};

function filtersActive(f: FilterState): boolean {
  return (
    f.marketCapMin !== "" ||
    f.marketCapMax !== "" ||
    f.forwardPeMin !== "" ||
    f.forwardPeMax !== "" ||
    f.volumeStrengthMin !== "" ||
    f.sectors.length > 0
  );
}

const COLUMN_INFO: Record<string, ColumnInfo> = {
  Score: {
    title: "Score",
    body: [
      "A 0–100 blend of several metrics, each normalized against the other tickers in this result set (the best value in the current list scores highest on that metric, the worst scores lowest) — it's a relative ranking within this run, not an absolute grade. Re-running with a different universe can change a ticker's score even if nothing about the ticker itself changed.",
      "The metrics and their weights depend on the Goal you picked:",
      "\"Short Term\": 3-month return (30%), 1-month return (25%), RSI balance (15%), MACD signal strength (15%), volume strength (10%), 6-month volatility (5%, lower is better).",
      "\"Long Term\": 1-year return (28%), 3-year annualized return (20%), 6-month return (12%), revenue growth (12%), earnings growth (10%), forward P/E (8%, lower is better), 1-year max drawdown (10%, lower is better).",
      "Filters below narrow which rows are shown, but never change how Score is computed — scores stay comparable across different filter selections since they're calculated before filtering.",
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
  const [sortKeys, setSortKeys] = useState<SortKey[]>([]);
  const [infoColumn, setInfoColumn] = useState<string | null>(null);

  const [filters, setFilters] = useState<FilterState>(EMPTY_FILTERS);
  const [showFilters, setShowFilters] = useState(false);

  const [visibleColumns, setVisibleColumns] = useState<string[]>(DEFAULT_COLUMNS);
  const [showColumnPicker, setShowColumnPicker] = useState(false);

  const [screens, setScreens] = useState<SavedScreen[]>([]);
  const [screensLoading, setScreensLoading] = useState(false);
  const [screenName, setScreenName] = useState("");
  const [savingScreen, setSavingScreen] = useState(false);
  const [saveScreenMessage, setSaveScreenMessage] = useState<string | null>(null);
  const [deletingScreenId, setDeletingScreenId] = useState<number | null>(null);
  const [compareScreenId, setCompareScreenId] = useState<number | null>(null);

  const [watchlistTicker, setWatchlistTicker] = useState<string | null>(null);
  const [watchlistCondition, setWatchlistCondition] = useState<AlertConditionType>("price_above");
  const [watchlistThreshold, setWatchlistThreshold] = useState("");
  const [watchlistSaving, setWatchlistSaving] = useState(false);
  const [watchlistMessage, setWatchlistMessage] = useState<string | null>(null);

  useEffect(() => {
    getStockUniverses()
      .then((res) => setUniverses(res.universes))
      .catch(() => {
        // Non-fatal: fall back to "All" already in state.
      });

    const stored = localStorage.getItem(COLUMNS_STORAGE_KEY);
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed) && parsed.every((c) => typeof c === "string")) {
          setVisibleColumns(parsed.includes(REQUIRED_COLUMN) ? parsed : [REQUIRED_COLUMN, ...parsed]);
        }
      } catch {
        // Malformed value — keep the default.
      }
    }

    loadScreens();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function loadScreens() {
    setScreensLoading(true);
    try {
      const res = await getScreens();
      setScreens(res.screens);
    } catch {
      // Non-fatal — saved screens are supplementary.
    } finally {
      setScreensLoading(false);
    }
  }

  async function fetchResults(forGoal: string, forMode: "rank" | "score", forUniverse: string, forTicker: string) {
    setLoading(true);
    setError(null);
    setHasSearched(true);
    setSortKeys([]);
    try {
      if (forMode === "rank") {
        const res = await getStockRanking(forGoal, forUniverse);
        setResults(res.results);
      } else {
        const res = await getStockScore(forGoal, forTicker.trim().toUpperCase());
        setResults(res.result ? [res.result] : []);
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong.");
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  async function runSearch(e: React.FormEvent) {
    e.preventDefault();
    setCompareScreenId(null);
    setFilters(EMPTY_FILTERS);
    await fetchResults(goal, mode, universe, ticker);
  }

  function toggleColumn(col: string) {
    if (col === REQUIRED_COLUMN) return;
    setVisibleColumns((prev) => {
      const next = prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col];
      localStorage.setItem(COLUMNS_STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  }

  function toggleSector(sector: string) {
    setFilters((prev) => ({
      ...prev,
      sectors: prev.sectors.includes(sector) ? prev.sectors.filter((s) => s !== sector) : [...prev.sectors, sector],
    }));
  }

  function handleSort(col: string, additive: boolean) {
    setSortKeys((prev) => {
      if (!additive) {
        if (prev.length === 1 && prev[0].column === col) {
          return [{ column: col, direction: prev[0].direction === "asc" ? "desc" : "asc" }];
        }
        return [{ column: col, direction: TEXT_COLUMNS.has(col) ? "asc" : "desc" }];
      }
      const idx = prev.findIndex((k) => k.column === col);
      if (idx === -1) {
        if (prev.length >= MAX_SORT_KEYS) return prev;
        return [...prev, { column: col, direction: TEXT_COLUMNS.has(col) ? "asc" : "desc" }];
      }
      const next = [...prev];
      next[idx] = { column: col, direction: next[idx].direction === "asc" ? "desc" : "asc" };
      return next;
    });
  }

  const winner = results[0];

  const availableSectors = useMemo(
    () => Array.from(new Set(results.map((r) => String(r.Sector ?? "Unknown")))).sort(),
    [results],
  );

  const filteredResults = useMemo(() => {
    if (!filtersActive(filters)) return results;
    return results.filter((row) => {
      const cap = row["Market Cap ($B)"] as number | null;
      if (filters.marketCapMin !== "" && (cap == null || cap < Number(filters.marketCapMin))) return false;
      if (filters.marketCapMax !== "" && (cap == null || cap > Number(filters.marketCapMax))) return false;
      const pe = row["Forward PE"] as number | null;
      if (filters.forwardPeMin !== "" && (pe == null || pe < Number(filters.forwardPeMin))) return false;
      if (filters.forwardPeMax !== "" && (pe == null || pe > Number(filters.forwardPeMax))) return false;
      const vol = row["Volume Strength %"] as number | null;
      if (filters.volumeStrengthMin !== "" && (vol == null || vol < Number(filters.volumeStrengthMin))) return false;
      if (filters.sectors.length > 0 && !filters.sectors.includes(String(row.Sector ?? "Unknown"))) return false;
      return true;
    });
  }, [results, filters]);

  const sortedResults = useMemo(() => {
    if (sortKeys.length === 0) return filteredResults;
    return [...filteredResults].sort((a, b) => {
      for (const { column, direction } of sortKeys) {
        const av = a[column];
        const bv = b[column];
        if (av == null && bv == null) continue;
        if (av == null) return 1;
        if (bv == null) return -1;
        let cmp: number;
        if (typeof av === "number" && typeof bv === "number") {
          cmp = av - bv;
        } else {
          cmp = String(av).localeCompare(String(bv));
        }
        cmp = direction === "asc" ? cmp : -cmp;
        if (cmp !== 0) return cmp;
      }
      return 0;
    });
  }, [filteredResults, sortKeys]);

  async function handleSaveScreen() {
    const name = screenName.trim();
    if (!name) {
      setSaveScreenMessage("Enter a name for this screen.");
      return;
    }
    setSavingScreen(true);
    setSaveScreenMessage(null);
    try {
      const snapshot: ScreenSnapshotRow[] = sortedResults.slice(0, 10).map((r) => ({
        Ticker: String(r.Ticker),
        Score: Number(r.Score),
        Price: Number(r.Price),
      }));
      await saveScreen({
        name,
        goal,
        universe,
        filters: { ...filters },
        visible_columns: visibleColumns,
        sort_keys: sortKeys,
        snapshot_top10: snapshot,
      });
      setScreenName("");
      setSaveScreenMessage("Screen saved.");
      await loadScreens();
    } catch (err) {
      setSaveScreenMessage(err instanceof ApiError ? err.message : "Could not save this screen.");
    } finally {
      setSavingScreen(false);
    }
  }

  async function handleLoadScreen(screen: SavedScreen) {
    setMode("rank");
    setGoal(screen.goal);
    setUniverse(screen.universe);
    const f = screen.filters as Partial<Record<keyof FilterState, unknown>>;
    setFilters({
      marketCapMin: typeof f.marketCapMin === "string" ? f.marketCapMin : "",
      marketCapMax: typeof f.marketCapMax === "string" ? f.marketCapMax : "",
      forwardPeMin: typeof f.forwardPeMin === "string" ? f.forwardPeMin : "",
      forwardPeMax: typeof f.forwardPeMax === "string" ? f.forwardPeMax : "",
      volumeStrengthMin: typeof f.volumeStrengthMin === "string" ? f.volumeStrengthMin : "",
      sectors: Array.isArray(f.sectors) ? (f.sectors as string[]) : [],
    });
    if (screen.visible_columns.length > 0) setVisibleColumns(screen.visible_columns);
    setSortKeys(screen.sort_keys);
    setCompareScreenId(screen.id);
    await fetchResults(screen.goal, "rank", screen.universe, ticker);
  }

  async function handleDeleteScreen(id: number) {
    setDeletingScreenId(id);
    try {
      await deleteScreen(id);
      setScreens((prev) => prev.filter((s) => s.id !== id));
      if (compareScreenId === id) setCompareScreenId(null);
    } catch {
      // Non-fatal — leave the row in place if delete failed.
    } finally {
      setDeletingScreenId(null);
    }
  }

  async function handleAddToWatchlist(t: string) {
    const threshold = Number(watchlistThreshold);
    if (!watchlistThreshold || Number.isNaN(threshold) || threshold <= 0) {
      setWatchlistMessage("Enter a positive threshold price.");
      return;
    }
    setWatchlistSaving(true);
    setWatchlistMessage(null);
    try {
      await createWatchlistAlert(t, watchlistCondition, threshold);
      setWatchlistMessage(`Added ${t} to your watchlist.`);
      setWatchlistThreshold("");
    } catch (err) {
      setWatchlistMessage(err instanceof ApiError ? err.message : "Could not add to watchlist.");
    } finally {
      setWatchlistSaving(false);
    }
  }

  const compareScreen = compareScreenId != null ? screens.find((s) => s.id === compareScreenId) : undefined;
  const comparisonRows = useMemo(() => {
    if (!compareScreen) return [];
    const before = new Map(compareScreen.snapshot_top10.map((r, i) => [r.Ticker, { rank: i + 1, score: r.Score }]));
    const currentTop10 = sortedResults.slice(0, 10);
    const after = new Map(currentTop10.map((r, i) => [String(r.Ticker), { rank: i + 1, score: Number(r.Score) }]));
    const allTickers = Array.from(new Set([...before.keys(), ...after.keys()]));
    return allTickers
      .map((t) => ({ ticker: t, before: before.get(t), after: after.get(t) }))
      .sort((a, b) => (a.after?.rank ?? 99) - (b.after?.rank ?? 99));
  }, [compareScreen, sortedResults]);

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">Stock Screener</h1>
      <p className="mt-1 text-sm text-slate-500">
        Rank a stock universe by goal, or score one ticker directly. Filter, customize columns, and save screens
        to reuse later.
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
            <MetricTile label="Score" value={`${winner.Score}/100`} onInfoClick={() => setInfoColumn("Score")} />
            <MetricTile label="Price" value={`$${Number(winner.Price).toFixed(2)}`} />
            <MetricTile label="Sector" value={String(winner.Sector)} />
            <MetricTile
              label="1Y Return"
              value={winner["1Y Return %"] != null ? `${Number(winner["1Y Return %"]).toFixed(1)}%` : "N/A"}
            />
          </div>

          {mode === "rank" && results.length > 1 && (
            <div className="rounded-lg border border-slate-200 bg-white p-5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <h3 className="font-semibold text-slate-900">Saved Screens</h3>
                <div className="flex flex-wrap items-end gap-2">
                  <input
                    value={screenName}
                    onChange={(e) => setScreenName(e.target.value)}
                    placeholder="Screen name"
                    className="rounded-md border border-slate-300 px-3 py-1.5 text-sm"
                  />
                  <button
                    onClick={handleSaveScreen}
                    disabled={savingScreen}
                    className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                  >
                    {savingScreen ? "Saving…" : "Save this screen"}
                  </button>
                </div>
              </div>
              {saveScreenMessage && <p className="mt-2 text-xs text-slate-500">{saveScreenMessage}</p>}

              {screensLoading ? (
                <p className="mt-3 text-sm text-slate-500">Loading saved screens…</p>
              ) : screens.length > 0 ? (
                <div className="mt-3 flex flex-col gap-2">
                  {screens.map((s) => (
                    <div key={s.id} className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-slate-200 px-3 py-2 text-sm">
                      <span className="text-slate-700">
                        <strong>{s.name}</strong> &middot; {s.goal} &middot; {s.universe} &middot;{" "}
                        {new Date(s.saved_at).toLocaleDateString()}
                      </span>
                      <span className="flex items-center gap-2">
                        <button
                          onClick={() => handleLoadScreen(s)}
                          className="rounded-md border border-slate-300 px-2 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
                        >
                          Load &amp; Compare
                        </button>
                        <button
                          onClick={() => handleDeleteScreen(s.id)}
                          disabled={deletingScreenId === s.id}
                          className="rounded-md border border-red-200 px-2 py-1 text-xs font-medium text-red-700 hover:bg-red-50 disabled:opacity-50"
                        >
                          {deletingScreenId === s.id ? "…" : "Delete"}
                        </button>
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="mt-3 text-sm text-slate-500">No saved screens yet.</p>
              )}

              {compareScreen && (
                <div className="mt-4 border-t border-slate-200 pt-4">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                    Top 10 vs. &quot;{compareScreen.name}&quot; (saved {new Date(compareScreen.saved_at).toLocaleDateString()})
                  </p>
                  <div className="mt-2 overflow-x-auto">
                    <table className="min-w-full text-sm">
                      <thead>
                        <tr className="border-b border-slate-200 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                          <th className="px-2 py-1.5">Ticker</th>
                          <th className="px-2 py-1.5">Then</th>
                          <th className="px-2 py-1.5">Now</th>
                          <th className="px-2 py-1.5">Change</th>
                        </tr>
                      </thead>
                      <tbody>
                        {comparisonRows.map((row) => (
                          <tr key={row.ticker} className="border-b border-slate-100 last:border-0">
                            <td className="px-2 py-1.5 font-medium text-slate-900">{row.ticker}</td>
                            <td className="px-2 py-1.5 text-slate-700">
                              {row.before ? `#${row.before.rank} (${row.before.score.toFixed(1)})` : "—"}
                            </td>
                            <td className="px-2 py-1.5 text-slate-700">
                              {row.after ? `#${row.after.rank} (${row.after.score.toFixed(1)})` : "—"}
                            </td>
                            <td className="px-2 py-1.5">
                              {!row.before ? (
                                <span className="text-emerald-600">New entrant</span>
                              ) : !row.after ? (
                                <span className="text-red-600">Dropped out of top 10</span>
                              ) : row.before.rank === row.after.rank ? (
                                <span className="text-slate-400">Same rank</span>
                              ) : row.before.rank > row.after.rank ? (
                                <span className="text-emerald-600">Up {row.before.rank - row.after.rank}</span>
                              ) : (
                                <span className="text-red-600">Down {row.after.rank - row.before.rank}</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {mode === "rank" && results.length > 1 && (
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={() => setShowFilters((v) => !v)}
                className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
              >
                {showFilters ? "Hide Filters" : "Filters"}
                {filtersActive(filters) ? ` (active)` : ""}
              </button>
              <button
                type="button"
                onClick={() => setShowColumnPicker((v) => !v)}
                className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
              >
                Columns
              </button>
              <span className="text-xs text-slate-500">
                Showing {sortedResults.length} of {results.length} tickers
                {sortKeys.length > 0 &&
                  ` · sorted by ${sortKeys.map((k) => `${k.column} (${k.direction})`).join(", ")}`}
              </span>
            </div>
          )}

          {showFilters && mode === "rank" && (
            <div className="rounded-lg border border-slate-200 bg-white p-4">
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                <RangeFilter
                  label="Market Cap ($B)"
                  min={filters.marketCapMin}
                  max={filters.marketCapMax}
                  onMinChange={(v) => setFilters((prev) => ({ ...prev, marketCapMin: v }))}
                  onMaxChange={(v) => setFilters((prev) => ({ ...prev, marketCapMax: v }))}
                />
                <RangeFilter
                  label="Forward P/E"
                  min={filters.forwardPeMin}
                  max={filters.forwardPeMax}
                  onMinChange={(v) => setFilters((prev) => ({ ...prev, forwardPeMin: v }))}
                  onMaxChange={(v) => setFilters((prev) => ({ ...prev, forwardPeMax: v }))}
                />
                <div className="flex flex-col gap-1">
                  <label className="text-xs font-medium text-slate-500">Volume Strength % (min)</label>
                  <input
                    type="number"
                    value={filters.volumeStrengthMin}
                    onChange={(e) => setFilters((prev) => ({ ...prev, volumeStrengthMin: e.target.value }))}
                    className="rounded-md border border-slate-300 px-2 py-1.5 text-sm"
                    placeholder="e.g. 0"
                  />
                </div>
              </div>
              <div className="mt-3">
                <label className="text-xs font-medium text-slate-500">Sector</label>
                <div className="mt-1 flex flex-wrap gap-2">
                  {availableSectors.map((s) => (
                    <button
                      key={s}
                      type="button"
                      onClick={() => toggleSector(s)}
                      className={`rounded-full border px-2.5 py-1 text-xs font-medium ${
                        filters.sectors.includes(s)
                          ? "border-slate-900 bg-slate-900 text-white"
                          : "border-slate-300 text-slate-600 hover:bg-slate-50"
                      }`}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
              {filtersActive(filters) && (
                <button
                  type="button"
                  onClick={() => setFilters(EMPTY_FILTERS)}
                  className="mt-3 text-xs font-medium text-slate-500 underline hover:text-slate-800"
                >
                  Clear all filters
                </button>
              )}
            </div>
          )}

          {showColumnPicker && mode === "rank" && (
            <div className="rounded-lg border border-slate-200 bg-white p-4">
              <p className="text-xs font-medium text-slate-500">Choose visible columns</p>
              <div className="mt-2 flex flex-wrap gap-x-4 gap-y-2">
                {ALL_COLUMNS.map((col) => (
                  <label key={col} className="flex items-center gap-1.5 text-sm text-slate-700">
                    <input
                      type="checkbox"
                      checked={visibleColumns.includes(col)}
                      disabled={col === REQUIRED_COLUMN}
                      onChange={() => toggleColumn(col)}
                    />
                    {col}
                  </label>
                ))}
              </div>
            </div>
          )}

          {results.length > 1 && (
            <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                    {visibleColumns.map((col) => {
                      const keyIndex = sortKeys.findIndex((k) => k.column === col);
                      return (
                        <th key={col} className="px-3 py-2">
                          <div className="flex items-center gap-1">
                            <button
                              type="button"
                              onClick={(e) => handleSort(col, e.shiftKey)}
                              title="Click to sort; shift-click to add as a secondary sort key"
                              className="flex items-center gap-1 uppercase tracking-wide text-slate-500 hover:text-slate-900"
                            >
                              {col}
                              <span className="text-[10px] text-slate-400">
                                {keyIndex !== -1
                                  ? `${sortKeys[keyIndex].direction === "asc" ? "▲" : "▼"}${
                                      sortKeys.length > 1 ? keyIndex + 1 : ""
                                    }`
                                  : ""}
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
                      );
                    })}
                    <th className="px-3 py-2"></th>
                  </tr>
                </thead>
                <tbody>
                  {sortedResults.map((row) => {
                    const t = String(row.Ticker);
                    return (
                      <Fragment key={t}>
                        <tr className="border-b border-slate-100 last:border-0">
                          {visibleColumns.map((col) => (
                            <td key={col} className="px-3 py-2 text-slate-700">
                              {formatCell(row[col])}
                            </td>
                          ))}
                          <td className="px-3 py-2 text-right">
                            <div className="flex items-center justify-end gap-2">
                              <button
                                type="button"
                                onClick={() => {
                                  setWatchlistTicker(watchlistTicker === t ? null : t);
                                  setWatchlistMessage(null);
                                  setWatchlistThreshold("");
                                }}
                                className="rounded-md border border-slate-300 px-2 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
                              >
                                Watchlist
                              </button>
                              <Link
                                href={`/predict?ticker=${encodeURIComponent(t)}`}
                                className="rounded-md border border-slate-300 px-2 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
                              >
                                Forecast
                              </Link>
                            </div>
                          </td>
                        </tr>
                        {watchlistTicker === t && (
                          <tr className="border-b border-slate-100 bg-slate-50 last:border-0">
                            <td colSpan={visibleColumns.length + 1} className="px-3 py-2">
                              <div className="flex flex-wrap items-center gap-2">
                                <span className="text-xs font-medium text-slate-500">Alert me when {t} is</span>
                                <select
                                  value={watchlistCondition}
                                  onChange={(e) => setWatchlistCondition(e.target.value as AlertConditionType)}
                                  className="rounded-md border border-slate-300 px-2 py-1 text-xs"
                                >
                                  <option value="price_above">above</option>
                                  <option value="price_below">below</option>
                                </select>
                                <input
                                  type="number"
                                  value={watchlistThreshold}
                                  onChange={(e) => setWatchlistThreshold(e.target.value)}
                                  placeholder="Price"
                                  className="w-24 rounded-md border border-slate-300 px-2 py-1 text-xs"
                                />
                                <button
                                  type="button"
                                  onClick={() => handleAddToWatchlist(t)}
                                  disabled={watchlistSaving}
                                  className="rounded-md bg-slate-900 px-2.5 py-1 text-xs font-medium text-white hover:bg-slate-800 disabled:opacity-50"
                                >
                                  {watchlistSaving ? "Adding…" : "Add"}
                                </button>
                                {watchlistMessage && <span className="text-xs text-slate-500">{watchlistMessage}</span>}
                              </div>
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    );
                  })}
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

function RangeFilter({
  label,
  min,
  max,
  onMinChange,
  onMaxChange,
}: {
  label: string;
  min: string;
  max: string;
  onMinChange: (v: string) => void;
  onMaxChange: (v: string) => void;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs font-medium text-slate-500">{label}</label>
      <div className="flex items-center gap-2">
        <input
          type="number"
          value={min}
          onChange={(e) => onMinChange(e.target.value)}
          placeholder="Min"
          className="w-full rounded-md border border-slate-300 px-2 py-1.5 text-sm"
        />
        <span className="text-slate-400">–</span>
        <input
          type="number"
          value={max}
          onChange={(e) => onMaxChange(e.target.value)}
          placeholder="Max"
          className="w-full rounded-md border border-slate-300 px-2 py-1.5 text-sm"
        />
      </div>
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
