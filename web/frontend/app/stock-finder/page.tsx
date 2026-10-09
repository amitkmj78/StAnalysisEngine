"use client";

import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";

import MetricLabel from "@/components/MetricLabel";
import Sparkline from "@/components/portfolio/Sparkline";
import TickerSearchInput from "@/components/TickerSearchInput";
import {
  ApiError,
  createWatchlistAlert,
  deleteScreen,
  getAnalystRating,
  getDefaultColumns,
  getPredictionSummary,
  getPresetScreens,
  getSavedScreenAlerts,
  getScreens,
  getStockRanking,
  getStockScore,
  getStockUniverses,
  saveScreen,
  setDefaultColumns,
} from "@/lib/api";
import type {
  AlertConditionType,
  AnalystRatingSummary,
  PresetScreen,
  SavedScreen,
  SavedScreenAlert,
  ScreenSnapshotRow,
  SignalOut,
  StockRankRow,
} from "@/lib/types";

// "Ledger" direction (warm paper, Fraunces for headings/numbers, IBM Plex
// for body/UI/tabular data) is now the whole app's shared look -- see
// app/layout.tsx + app/globals.css's --pf-* tokens, loaded globally there.
const PF = {
  page: "bg-[var(--pf-bg)]",
  ink: "text-[var(--pf-text)]",
  muted: "text-slate-500",
  line: "border-[var(--pf-border)]",
  card: "rounded-xl border border-[var(--pf-border)] bg-white",
  surface2: "bg-slate-100",
  good: "text-[var(--pf-up)]",
  bad: "text-[var(--pf-down)]",
  warnBg: "bg-[var(--pf-warn-soft)]",
  warnText: "text-[var(--pf-warn)]",
  btn: "rounded-md border border-[var(--pf-border)] bg-white px-3 py-1.5 text-sm font-medium text-slate-900 hover:border-[var(--pf-accent)] hover:text-[var(--pf-accent)]",
  btnPrimary: "rounded-md bg-[var(--pf-accent)] px-3 py-1.5 text-sm font-semibold text-white hover:opacity-90",
  input: "rounded-md border border-[var(--pf-border)] bg-white px-3 py-2 text-sm text-slate-900",
  chip: "inline-flex items-center gap-1 rounded-full border border-[var(--pf-border)] bg-white px-2.5 py-1 text-xs text-slate-900",
};

function goodBad(v: number | null | undefined): string {
  if (v === null || v === undefined) return PF.muted;
  return v >= 0 ? PF.good : PF.bad;
}

const GOALS = ["Short Term", "Long Term"];

// Mirrors exactly the weights already documented in the Score column's own
// tooltip below -- shown inline (rather than only on click) so the active
// Goal's weighting is visible without hunting for the info icon.
const GOAL_WEIGHTS_DISPLAY: Record<string, { label: string; pct: number; lowerIsBetter?: boolean }[]> = {
  "Short Term": [
    { label: "3-Month Return", pct: 30 },
    { label: "1-Month Return", pct: 25 },
    { label: "RSI Balance", pct: 15 },
    { label: "MACD Signal Strength", pct: 15 },
    { label: "Volume Strength", pct: 10 },
    { label: "6-Month Volatility", pct: 5, lowerIsBetter: true },
  ],
  "Long Term": [
    { label: "1-Year Return", pct: 28 },
    { label: "3-Year Annualized Return", pct: 20 },
    { label: "6-Month Return", pct: 12 },
    { label: "Revenue Growth", pct: 12 },
    { label: "Earnings Growth", pct: 10 },
    { label: "Forward P/E", pct: 8, lowerIsBetter: true },
    { label: "1-Year Max Drawdown", pct: 10, lowerIsBetter: true },
  ],
};

// Every field _build_stock_row (services/stock_finder_service.py) returns,
// in display order — the column picker offers all of these, not just the
// small default subset shown out of the box.
const ALL_COLUMNS = [
  "Ticker",
  "Name",
  "Sector",
  "Price",
  "Score",
  "Quant Signal",
  "Analyst Rating",
  "Market Cap ($B)",
  "Forward PE",
  "Dividend Yield %",
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
  "Close vs SMA50 %",
  "Close vs SMA200 %",
  "Dist. 52W High %",
  "Dist. 52W Low %",
  "New 52W High",
  "New 52W Low",
  "SMA20 Crossed Above SMA50",
  "SMA20 Crossed Below SMA50",
  "Spark 90D",
  "Short-Term Score",
  "Short-Term Signal",
  "Long-Term Score",
  "Long-Term Signal",
  "Score As Of",
  "Owned",
  "Watchlisted",
];

// Data-driven columns come straight from the API row; "Quant Signal" and
// "Analyst Rating" are the exceptions — both fetched lazily per row on
// click (see quantSignals/analystRatings state, and their dedicated <td>
// branches in the table body below) since each is its own yfinance/model
// call per ticker, not something to eagerly run across a 500-ticker
// universe.
const DEFAULT_COLUMNS = ["Ticker", "Name", "Sector", "Price", "Score", "Quant Signal", "Analyst Rating", "1M Return %", "3M Return %", "1Y Return %", "RSI", "Spark 90D"];
const REQUIRED_COLUMN = "Ticker";
const COLUMNS_STORAGE_KEY = "stanalysisengine.stockFinderColumns";

const TEXT_COLUMNS = new Set(["Ticker", "Name", "Sector", "Short-Term Signal", "Long-Term Signal", "Owned", "Watchlisted"]);

type SortDirection = "asc" | "desc";
type SortKey = { column: string; direction: SortDirection };
const MAX_SORT_KEYS = 3;

// "Any" / "Only" / "Exclude" -- a plain boolean can't express "only show
// tickers I don't own", so Owned/Watchlisted get their own tri-state type
// instead of overloading `boolean | null`.
type TriState = "any" | "only" | "exclude";

interface FilterState {
  marketCapMin: string;
  marketCapMax: string;
  forwardPeMin: string;
  forwardPeMax: string;
  volumeStrengthMin: string;
  sectors: string[];
  dividendYieldMin: string;
  volatilityMax: string;
  momentumMin: string;
  momentumMax: string;
  earningsGrowthMin: string;
  earningsGrowthMax: string;
  shortScoreMin: string;
  shortScoreMax: string;
  longScoreMin: string;
  longScoreMax: string;
  shortSignal: string[];
  longSignal: string[];
  owned: TriState;
  watchlisted: TriState;
  // ALX-5: technical filters alongside the score filters above.
  closeVsSma50Min: string;
  closeVsSma50Max: string;
  closeVsSma200Min: string;
  closeVsSma200Max: string;
  dist52wHighMin: string;
  dist52wHighMax: string;
  dist52wLowMin: string;
  dist52wLowMax: string;
  new52wHighOnly: boolean;
  new52wLowOnly: boolean;
  crossedAbove50Only: boolean;
  crossedBelow50Only: boolean;
}

const EMPTY_FILTERS: FilterState = {
  marketCapMin: "",
  marketCapMax: "",
  forwardPeMin: "",
  forwardPeMax: "",
  volumeStrengthMin: "",
  sectors: [],
  dividendYieldMin: "",
  volatilityMax: "",
  momentumMin: "",
  momentumMax: "",
  earningsGrowthMin: "",
  earningsGrowthMax: "",
  shortScoreMin: "",
  shortScoreMax: "",
  longScoreMin: "",
  longScoreMax: "",
  shortSignal: [],
  longSignal: [],
  owned: "any",
  watchlisted: "any",
  closeVsSma50Min: "",
  closeVsSma50Max: "",
  closeVsSma200Min: "",
  closeVsSma200Max: "",
  dist52wHighMin: "",
  dist52wHighMax: "",
  dist52wLowMin: "",
  dist52wLowMax: "",
  new52wHighOnly: false,
  new52wLowOnly: false,
  crossedAbove50Only: false,
  crossedBelow50Only: false,
};

function filtersActive(f: FilterState): boolean {
  return (
    f.marketCapMin !== "" ||
    f.marketCapMax !== "" ||
    f.forwardPeMin !== "" ||
    f.forwardPeMax !== "" ||
    f.volumeStrengthMin !== "" ||
    f.sectors.length > 0 ||
    f.dividendYieldMin !== "" ||
    f.volatilityMax !== "" ||
    f.momentumMin !== "" ||
    f.momentumMax !== "" ||
    f.earningsGrowthMin !== "" ||
    f.earningsGrowthMax !== "" ||
    f.shortScoreMin !== "" ||
    f.shortScoreMax !== "" ||
    f.longScoreMin !== "" ||
    f.longScoreMax !== "" ||
    f.shortSignal.length > 0 ||
    f.longSignal.length > 0 ||
    f.owned !== "any" ||
    f.watchlisted !== "any" ||
    f.closeVsSma50Min !== "" ||
    f.closeVsSma50Max !== "" ||
    f.closeVsSma200Min !== "" ||
    f.closeVsSma200Max !== "" ||
    f.dist52wHighMin !== "" ||
    f.dist52wHighMax !== "" ||
    f.dist52wLowMin !== "" ||
    f.dist52wLowMax !== "" ||
    f.new52wHighOnly ||
    f.new52wLowOnly ||
    f.crossedAbove50Only ||
    f.crossedBelow50Only
  );
}

const SIGNAL_VALUES = ["Buy", "Hold", "Trim"];

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

  const [filters, setFilters] = useState<FilterState>(EMPTY_FILTERS);
  const [showFilters, setShowFilters] = useState(false);

  const [visibleColumns, setVisibleColumns] = useState<string[]>(DEFAULT_COLUMNS);
  const [showColumnPicker, setShowColumnPicker] = useState(false);
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
  const tableWrapRef = useRef<HTMLDivElement | null>(null);
  const [tableOverflowing, setTableOverflowing] = useState(false);

  const [presets, setPresets] = useState<PresetScreen[]>([]);

  const [screens, setScreens] = useState<SavedScreen[]>([]);
  const [screensLoading, setScreensLoading] = useState(false);
  const [screenAlerts, setScreenAlerts] = useState<Record<number, SavedScreenAlert>>({});
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

  const [quantSignals, setQuantSignals] = useState<
    Record<string, { status: "loading" } | { status: "error" } | { status: "ok"; signal: SignalOut }>
  >({});
  const [analystRatings, setAnalystRatings] = useState<
    Record<string, { status: "loading" } | { status: "error" } | { status: "ok"; rating: AnalystRatingSummary }>
  >({});

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
    // ALX-6: the server-side default (if the user has one saved) wins
    // over the localStorage value above once it arrives -- this is what
    // makes the column choice follow the user across devices instead of
    // being per-browser only. Signed-out visitors get a 401 here, which
    // is expected and non-fatal: they just keep whatever localStorage/
    // DEFAULT_COLUMNS already set.
    getDefaultColumns()
      .then((res) => {
        if (res.columns && res.columns.length > 0) {
          setVisibleColumns(res.columns.includes(REQUIRED_COLUMN) ? res.columns : [REQUIRED_COLUMN, ...res.columns]);
        }
      })
      .catch(() => {
        // Non-fatal -- signed out, or no server default saved yet.
      });

    loadScreens();
    getPresetScreens()
      .then((res) => setPresets(res.presets))
      .catch(() => {
        // Non-fatal -- presets are supplementary.
      });
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
    try {
      const res = await getSavedScreenAlerts();
      setScreenAlerts(Object.fromEntries(res.alerts.map((a) => [a.screen_id, a])));
    } catch {
      // Non-fatal -- the enter/leave badge is supplementary (SCN-3 may
      // also just be disabled server-side, which returns an empty list).
    }
  }

  async function fetchResults(forGoal: string, forMode: "rank" | "score", forUniverse: string, forTicker: string) {
    setLoading(true);
    setError(null);
    setHasSearched(true);
    setSortKeys([]);
    setQuantSignals({});
    setAnalystRatings({});
    setExpandedRows(new Set());
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
      // ALX-6: best-effort server-side save too, so this choice follows
      // the user to another device -- a signed-out visitor's 401 here is
      // expected and silently ignored, same as the initial load above.
      setDefaultColumns(next).catch(() => undefined);
      return next;
    });
  }

  function toggleSector(sector: string) {
    setFilters((prev) => ({
      ...prev,
      sectors: prev.sectors.includes(sector) ? prev.sectors.filter((s) => s !== sector) : [...prev.sectors, sector],
    }));
  }

  function toggleShortSignal(signal: string) {
    setFilters((prev) => ({
      ...prev,
      shortSignal: prev.shortSignal.includes(signal) ? prev.shortSignal.filter((s) => s !== signal) : [...prev.shortSignal, signal],
    }));
  }

  function toggleLongSignal(signal: string) {
    setFilters((prev) => ({
      ...prev,
      longSignal: prev.longSignal.includes(signal) ? prev.longSignal.filter((s) => s !== signal) : [...prev.longSignal, signal],
    }));
  }

  function toggleRow(t: string) {
    setExpandedRows((prev) => {
      const next = new Set(prev);
      if (next.has(t)) next.delete(t);
      else next.add(t);
      return next;
    });
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

      const divYield = row["Dividend Yield %"] as number | null;
      if (filters.dividendYieldMin !== "" && (divYield == null || divYield < Number(filters.dividendYieldMin))) return false;

      const volatility = row["6M Volatility %"] as number | null;
      if (filters.volatilityMax !== "" && (volatility == null || volatility > Number(filters.volatilityMax))) return false;

      // Momentum: 3-Month Return, the same metric this app already treats
      // as the dominant momentum signal (the largest single weight in the
      // Short Term goal -- see the 3M Return % column's own tooltip).
      const momentum = row["3M Return %"] as number | null;
      if (filters.momentumMin !== "" && (momentum == null || momentum < Number(filters.momentumMin))) return false;
      if (filters.momentumMax !== "" && (momentum == null || momentum > Number(filters.momentumMax))) return false;

      const earningsGrowth = row["Earnings Growth %"] as number | null;
      if (filters.earningsGrowthMin !== "" && (earningsGrowth == null || earningsGrowth < Number(filters.earningsGrowthMin))) return false;
      if (filters.earningsGrowthMax !== "" && (earningsGrowth == null || earningsGrowth > Number(filters.earningsGrowthMax))) return false;

      const shortScore = row["Short-Term Score"] as number | null;
      if (filters.shortScoreMin !== "" && (shortScore == null || shortScore < Number(filters.shortScoreMin))) return false;
      if (filters.shortScoreMax !== "" && (shortScore == null || shortScore > Number(filters.shortScoreMax))) return false;

      const longScore = row["Long-Term Score"] as number | null;
      if (filters.longScoreMin !== "" && (longScore == null || longScore < Number(filters.longScoreMin))) return false;
      if (filters.longScoreMax !== "" && (longScore == null || longScore > Number(filters.longScoreMax))) return false;

      if (filters.shortSignal.length > 0 && !filters.shortSignal.includes(String(row["Short-Term Signal"] ?? ""))) return false;
      if (filters.longSignal.length > 0 && !filters.longSignal.includes(String(row["Long-Term Signal"] ?? ""))) return false;

      const owned = row.Owned as boolean | undefined;
      if (filters.owned === "only" && !owned) return false;
      if (filters.owned === "exclude" && owned) return false;
      const watchlisted = row.Watchlisted as boolean | undefined;
      if (filters.watchlisted === "only" && !watchlisted) return false;
      if (filters.watchlisted === "exclude" && watchlisted) return false;

      // ALX-5: technical filters -- must mirror services/stock_finder_
      // service.py::apply_filters exactly, same maintenance note that
      // function's docstring gives, or a saved-screen alert would
      // silently diverge from what this page shows.
      const closeVsSma50 = row["Close vs SMA50 %"] as number | null;
      if (filters.closeVsSma50Min !== "" && (closeVsSma50 == null || closeVsSma50 < Number(filters.closeVsSma50Min))) return false;
      if (filters.closeVsSma50Max !== "" && (closeVsSma50 == null || closeVsSma50 > Number(filters.closeVsSma50Max))) return false;
      const closeVsSma200 = row["Close vs SMA200 %"] as number | null;
      if (filters.closeVsSma200Min !== "" && (closeVsSma200 == null || closeVsSma200 < Number(filters.closeVsSma200Min))) return false;
      if (filters.closeVsSma200Max !== "" && (closeVsSma200 == null || closeVsSma200 > Number(filters.closeVsSma200Max))) return false;
      const dist52wHigh = row["Dist. 52W High %"] as number | null;
      if (filters.dist52wHighMin !== "" && (dist52wHigh == null || dist52wHigh < Number(filters.dist52wHighMin))) return false;
      if (filters.dist52wHighMax !== "" && (dist52wHigh == null || dist52wHigh > Number(filters.dist52wHighMax))) return false;
      const dist52wLow = row["Dist. 52W Low %"] as number | null;
      if (filters.dist52wLowMin !== "" && (dist52wLow == null || dist52wLow < Number(filters.dist52wLowMin))) return false;
      if (filters.dist52wLowMax !== "" && (dist52wLow == null || dist52wLow > Number(filters.dist52wLowMax))) return false;
      if (filters.new52wHighOnly && !row["New 52W High"]) return false;
      if (filters.new52wLowOnly && !row["New 52W Low"]) return false;
      if (filters.crossedAbove50Only && !row["SMA20 Crossed Above SMA50"]) return false;
      if (filters.crossedBelow50Only && !row["SMA20 Crossed Below SMA50"]) return false;

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

  // NFR-3: Short-Term/Long-Term Score come from the nightly job (SCR-1..4),
  // not this request -- a single shared date covers the normal case (the
  // job runs for the whole universe at once); if rows actually disagree
  // (e.g. a partial night), say the range instead of picking one silently.
  const scoreAsOfSummary = useMemo(() => {
    const dates = Array.from(
      new Set(results.map((r) => r["Score As Of"] as string | null).filter((d): d is string => !!d))
    ).sort();
    if (dates.length === 0) return null;
    if (dates.length === 1) return dates[0];
    return `${dates[0]} to ${dates[dates.length - 1]} (varies by ticker)`;
  }, [results]);

  // Selecting more columns than fit the viewport makes the table wider than
  // its wrapper with no visible cue that the rest is one scroll away
  // (native scrollbars are overlay/hover-only in most browsers) — this
  // tracks real overflow so the toolbar can say so explicitly instead of
  // just looking cut off.
  useEffect(() => {
    const el = tableWrapRef.current;
    if (!el) {
      setTableOverflowing(false);
      return;
    }
    const check = () => setTableOverflowing(el.scrollWidth > el.clientWidth + 1);
    check();
    const observer = new ResizeObserver(check);
    observer.observe(el);
    return () => observer.disconnect();
  }, [visibleColumns, sortedResults]);

  // A plain scroll wheel has no horizontal axis at all on most mice (only
  // trackpads/tilt-wheels do), so the wide table's horizontal overflow was
  // effectively unreachable for anyone without one -- these buttons work
  // regardless of input device.
  function scrollTable(deltaX: number) {
    tableWrapRef.current?.scrollBy({ left: deltaX, behavior: "smooth" });
  }

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

  // Defensive round-trip of a persisted filters blob (a saved screen or a
  // hardcoded preset) into FilterState -- any key that's missing, wrongly
  // typed, or from a since-removed filter falls back to EMPTY_FILTERS'
  // value instead of corrupting state.
  function parseFilters(f: Partial<Record<keyof FilterState, unknown>>): FilterState {
    return {
      marketCapMin: typeof f.marketCapMin === "string" ? f.marketCapMin : "",
      marketCapMax: typeof f.marketCapMax === "string" ? f.marketCapMax : "",
      forwardPeMin: typeof f.forwardPeMin === "string" ? f.forwardPeMin : "",
      forwardPeMax: typeof f.forwardPeMax === "string" ? f.forwardPeMax : "",
      volumeStrengthMin: typeof f.volumeStrengthMin === "string" ? f.volumeStrengthMin : "",
      sectors: Array.isArray(f.sectors) ? (f.sectors as string[]) : [],
      dividendYieldMin: typeof f.dividendYieldMin === "string" ? f.dividendYieldMin : "",
      volatilityMax: typeof f.volatilityMax === "string" ? f.volatilityMax : "",
      momentumMin: typeof f.momentumMin === "string" ? f.momentumMin : "",
      momentumMax: typeof f.momentumMax === "string" ? f.momentumMax : "",
      earningsGrowthMin: typeof f.earningsGrowthMin === "string" ? f.earningsGrowthMin : "",
      earningsGrowthMax: typeof f.earningsGrowthMax === "string" ? f.earningsGrowthMax : "",
      shortScoreMin: typeof f.shortScoreMin === "string" ? f.shortScoreMin : "",
      shortScoreMax: typeof f.shortScoreMax === "string" ? f.shortScoreMax : "",
      longScoreMin: typeof f.longScoreMin === "string" ? f.longScoreMin : "",
      longScoreMax: typeof f.longScoreMax === "string" ? f.longScoreMax : "",
      shortSignal: Array.isArray(f.shortSignal) ? (f.shortSignal as string[]) : [],
      longSignal: Array.isArray(f.longSignal) ? (f.longSignal as string[]) : [],
      owned: f.owned === "only" || f.owned === "exclude" ? f.owned : "any",
      watchlisted: f.watchlisted === "only" || f.watchlisted === "exclude" ? f.watchlisted : "any",
      closeVsSma50Min: typeof f.closeVsSma50Min === "string" ? f.closeVsSma50Min : "",
      closeVsSma50Max: typeof f.closeVsSma50Max === "string" ? f.closeVsSma50Max : "",
      closeVsSma200Min: typeof f.closeVsSma200Min === "string" ? f.closeVsSma200Min : "",
      closeVsSma200Max: typeof f.closeVsSma200Max === "string" ? f.closeVsSma200Max : "",
      dist52wHighMin: typeof f.dist52wHighMin === "string" ? f.dist52wHighMin : "",
      dist52wHighMax: typeof f.dist52wHighMax === "string" ? f.dist52wHighMax : "",
      dist52wLowMin: typeof f.dist52wLowMin === "string" ? f.dist52wLowMin : "",
      dist52wLowMax: typeof f.dist52wLowMax === "string" ? f.dist52wLowMax : "",
      new52wHighOnly: f.new52wHighOnly === true,
      new52wLowOnly: f.new52wLowOnly === true,
      crossedAbove50Only: f.crossedAbove50Only === true,
      crossedBelow50Only: f.crossedBelow50Only === true,
    };
  }

  async function handleLoadScreen(screen: SavedScreen) {
    setMode("rank");
    setGoal(screen.goal);
    setUniverse(screen.universe);
    setFilters(parseFilters(screen.filters as Partial<Record<keyof FilterState, unknown>>));
    if (screen.visible_columns.length > 0) setVisibleColumns(screen.visible_columns);
    setSortKeys(screen.sort_keys);
    setCompareScreenId(screen.id);
    await fetchResults(screen.goal, "rank", screen.universe, ticker);
  }

  async function handleLoadPreset(preset: PresetScreen) {
    setMode("rank");
    setGoal(preset.goal);
    setUniverse(preset.universe);
    setFilters(parseFilters(preset.filters as Partial<Record<keyof FilterState, unknown>>));
    setCompareScreenId(null);
    await fetchResults(preset.goal, "rank", preset.universe, ticker);
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

  async function loadQuantSignal(t: string) {
    setQuantSignals((prev) => ({ ...prev, [t]: { status: "loading" } }));
    try {
      const res = await getPredictionSummary(t, "1y", 10);
      if (res.signal) {
        setQuantSignals((prev) => ({ ...prev, [t]: { status: "ok", signal: res.signal! } }));
      } else {
        setQuantSignals((prev) => ({ ...prev, [t]: { status: "error" } }));
      }
    } catch {
      setQuantSignals((prev) => ({ ...prev, [t]: { status: "error" } }));
    }
  }

  async function loadAnalystRating(t: string) {
    setAnalystRatings((prev) => ({ ...prev, [t]: { status: "loading" } }));
    try {
      const rating = await getAnalystRating(t);
      setAnalystRatings((prev) => ({ ...prev, [t]: { status: "ok", rating } }));
    } catch {
      setAnalystRatings((prev) => ({ ...prev, [t]: { status: "error" } }));
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

  const activeWeights = GOAL_WEIGHTS_DISPLAY[goal];
  // "Detail" columns for the row-expand panel: whichever of ALL_COLUMNS the
  // user hasn't already chosen to show inline via the column picker, minus
  // the two lazy-loaded ones (they get their own dedicated table cell, not
  // a duplicate in the expand panel).
  const detailColumns = useMemo(
    () =>
      ALL_COLUMNS.filter(
        (c) => !visibleColumns.includes(c) && c !== "Quant Signal" && c !== "Analyst Rating" && c !== "Spark 90D",
      ),
    [visibleColumns],
  );

  return (
    <div className={`${PF.page} ${PF.ink}`}>
      <div className="mx-auto max-w-6xl px-4 py-8">
        <h1 className="font-display text-3xl font-semibold" style={{ fontFamily: "var(--font-pf-display)" }}>
          Stock Screener
        </h1>
        <p className={`mt-1 max-w-2xl text-sm ${PF.muted}`}>
          Rank a stock universe by goal, or score one ticker directly. Filter, customize columns, and save screens
          to reuse later.{" "}
          <Link href="/guides/signals" className="text-[var(--pf-accent)] hover:underline">
            Learn more about reading signals →
          </Link>
        </p>

        <form onSubmit={runSearch} className={`mt-6 flex flex-wrap items-end gap-3 ${PF.card} p-4`}>
          <Field label="Goal">
            <select value={goal} onChange={(e) => setGoal(e.target.value)} className={PF.input}>
              {GOALS.map((g) => (
                <option key={g} value={g}>
                  {g}
                </option>
              ))}
            </select>
          </Field>

          <Field label="Mode">
            <select value={mode} onChange={(e) => setMode(e.target.value as "rank" | "score")} className={PF.input}>
              <option value="rank">Rank a universe</option>
              <option value="score">Score one ticker</option>
            </select>
          </Field>

          {mode === "rank" ? (
            <Field label="Universe">
              <select value={universe} onChange={(e) => setUniverse(e.target.value)} className={PF.input}>
                {universes.map((u) => (
                  <option key={u} value={u}>
                    {u}
                  </option>
                ))}
              </select>
            </Field>
          ) : (
            <Field label="Ticker or company name">
              <TickerSearchInput value={ticker} onChange={setTicker} className={`${PF.input} w-56`} />
            </Field>
          )}

          <button type="submit" disabled={loading} className={`${PF.btnPrimary} disabled:opacity-50`}>
            {loading ? "Scanning…" : "Run"}
          </button>
        </form>

        {mode === "rank" && (
          <div className={`mt-3 flex flex-wrap items-center gap-2 rounded-xl ${PF.surface2} px-4 py-3`}>
            <span className={`text-xs font-medium uppercase tracking-wide ${PF.muted}`}>Weights &middot; {goal}</span>
            {activeWeights.map((w) => (
              <span key={w.label} className={PF.chip}>
                {w.label} <strong>{w.pct}%</strong>
                {w.lowerIsBetter && <span className={PF.muted}>(lower is better)</span>}
              </span>
            ))}
          </div>
        )}

        {mode === "rank" && presets.length > 0 && (
          <div className={`mt-3 ${PF.card} p-4`}>
            <p className={`text-xs font-semibold uppercase tracking-wide ${PF.muted}`}>Preset screens</p>
            <div className="mt-2 grid grid-cols-1 gap-3 sm:grid-cols-3">
              {presets.map((p) => (
                <div key={p.key} className={`rounded-md border ${PF.line} p-3`}>
                  <button
                    type="button"
                    onClick={() => handleLoadPreset(p)}
                    className={`text-sm font-semibold underline-offset-2 hover:underline ${PF.ink}`}
                  >
                    {p.name}
                  </button>
                  <p className={`mt-1 text-xs ${PF.muted}`}>{p.rules}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {loading && (
          <p className={`mt-4 text-sm ${PF.muted}`}>
            {mode === "rank"
              ? universe === "US - S&P 500" || universe === "All"
                ? "Scoring roughly 500 tickers — this first run can take several minutes, cached for an hour after."
                : "Scoring every ticker in the universe — first run for a universe can take a while, cached for an hour after."
              : "Scoring this ticker…"}
          </p>
        )}

        {error && <p className="mt-4 rounded-md bg-[var(--pf-down-soft)] px-3 py-2 text-sm text-[var(--pf-down)]">{error}</p>}

        {!loading && hasSearched && results.length === 0 && !error && (
          <p className={`mt-4 text-sm ${PF.muted}`}>No results for that selection.</p>
        )}

        {winner && !loading && (
          <div className="mt-6 flex flex-col gap-6">
            <div className={`${PF.card} p-5`}>
              <h2 className="text-xl font-semibold" style={{ fontFamily: "var(--font-pf-display)" }}>
                Top Pick: {winner.Ticker} — {winner.Name}
              </h2>
              <p className={`mt-1 text-sm ${PF.muted}`}>
                Scored highest for <strong className={PF.ink}>{goal}</strong>
                {mode === "rank" ? ` in ${universe}` : ""}
                {mode === "rank" && `, out of ${results.length} ticker${results.length === 1 ? "" : "s"} screened.`}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <MetricTile label="Score" value={`${winner.Score}/100`} />
              <MetricTile label="Price" value={`$${Number(winner.Price).toFixed(2)}`} />
              <MetricTile label="Sector" value={String(winner.Sector)} />
              <MetricTile
                label="1Y Return"
                value={winner["1Y Return %"] != null ? `${Number(winner["1Y Return %"]).toFixed(1)}%` : "N/A"}
                tone={goodBad(winner["1Y Return %"] as number | null)}
              />
            </div>

            {mode === "rank" && results.length > 1 && (
              <div className={`${PF.card} p-5`}>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <h3 className="text-lg font-semibold" style={{ fontFamily: "var(--font-pf-display)" }}>
                    Saved Screens
                  </h3>
                  <div className="flex flex-wrap items-end gap-2">
                    <input
                      value={screenName}
                      onChange={(e) => setScreenName(e.target.value)}
                      placeholder="Screen name"
                      className={`${PF.input} py-1.5`}
                    />
                    <button onClick={handleSaveScreen} disabled={savingScreen} className={`${PF.btn} disabled:opacity-50`}>
                      {savingScreen ? "Saving…" : "Save this screen"}
                    </button>
                  </div>
                </div>
                {saveScreenMessage && <p className={`mt-2 text-xs ${PF.muted}`}>{saveScreenMessage}</p>}

                {screensLoading ? (
                  <p className={`mt-3 text-sm ${PF.muted}`}>Loading saved screens…</p>
                ) : screens.length > 0 ? (
                  <div className="mt-3 flex flex-col gap-2">
                    {screens.map((s) => (
                      <div key={s.id} className={`flex flex-wrap items-center justify-between gap-2 rounded-md border ${PF.line} px-3 py-2 text-sm`}>
                        <span className={PF.ink}>
                          <strong>{s.name}</strong> &middot; {s.goal} &middot; {s.universe} &middot;{" "}
                          {new Date(s.saved_at).toLocaleDateString()}
                          {screenAlerts[s.id] && (screenAlerts[s.id].entered.length > 0 || screenAlerts[s.id].left_tickers.length > 0) && (
                            <span
                              title={`Entered: ${screenAlerts[s.id].entered.join(", ") || "none"} · Left: ${screenAlerts[s.id].left_tickers.join(", ") || "none"}`}
                              className={`ml-2 rounded-full px-2 py-0.5 text-[11px] font-medium ${PF.warnBg} ${PF.warnText}`}
                            >
                              +{screenAlerts[s.id].entered.length} entered, {screenAlerts[s.id].left_tickers.length} dropped since last check
                            </span>
                          )}
                        </span>
                        <span className="flex items-center gap-2">
                          <button onClick={() => handleLoadScreen(s)} className={`${PF.btn} px-2 py-1 text-xs`}>
                            Load &amp; Compare
                          </button>
                          <button
                            onClick={() => handleDeleteScreen(s.id)}
                            disabled={deletingScreenId === s.id}
                            className={`rounded-md border border-red-200 px-2 py-1 text-xs font-medium text-red-700 hover:bg-red-50 disabled:opacity-50`}
                          >
                            {deletingScreenId === s.id ? "…" : "Delete"}
                          </button>
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className={`mt-3 text-sm ${PF.muted}`}>No saved screens yet.</p>
                )}

                {compareScreen && (
                  <div className={`mt-4 border-t ${PF.line} pt-4`}>
                    <p className={`text-xs font-semibold uppercase tracking-wide ${PF.muted}`}>
                      Top 10 vs. &quot;{compareScreen.name}&quot; (saved {new Date(compareScreen.saved_at).toLocaleDateString()})
                    </p>
                    <div className="mt-2 overflow-x-auto">
                      <table className="min-w-full text-sm">
                        <thead>
                          <tr className={`border-b ${PF.line} text-left text-xs font-medium uppercase tracking-wide ${PF.muted}`}>
                            <th className="px-2 py-1.5">Ticker</th>
                            <th className="px-2 py-1.5">Then</th>
                            <th className="px-2 py-1.5">Now</th>
                            <th className="px-2 py-1.5">Change</th>
                          </tr>
                        </thead>
                        <tbody>
                          {comparisonRows.map((row) => (
                            <tr key={row.ticker} className={`border-b ${PF.line} last:border-0`}>
                              <td className="px-2 py-1.5 font-medium" style={{ fontFamily: "var(--font-pf-mono)" }}>
                                {row.ticker}
                              </td>
                              <td className="px-2 py-1.5">
                                {row.before ? `#${row.before.rank} (${row.before.score.toFixed(1)})` : "—"}
                              </td>
                              <td className="px-2 py-1.5">
                                {row.after ? `#${row.after.rank} (${row.after.score.toFixed(1)})` : "—"}
                              </td>
                              <td className="px-2 py-1.5">
                                {!row.before ? (
                                  <span className={PF.good}>New entrant</span>
                                ) : !row.after ? (
                                  <span className={PF.bad}>Dropped out of top 10</span>
                                ) : row.before.rank === row.after.rank ? (
                                  <span className={PF.muted}>Same rank</span>
                                ) : row.before.rank > row.after.rank ? (
                                  <span className={PF.good}>Up {row.before.rank - row.after.rank}</span>
                                ) : (
                                  <span className={PF.bad}>Down {row.after.rank - row.before.rank}</span>
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
                  className={`${PF.btn} ${showFilters ? "border-[var(--pf-accent)] text-[var(--pf-accent)]" : ""}`}
                >
                  {showFilters ? "Hide Filters" : "Filters"}
                  {filtersActive(filters) ? ` (active)` : ""}
                </button>
                <button
                  type="button"
                  onClick={() => setShowColumnPicker((v) => !v)}
                  className={`${PF.btn} ${showColumnPicker ? "border-[var(--pf-accent)] text-[var(--pf-accent)]" : ""}`}
                >
                  Columns
                </button>
                <span className={`text-xs ${PF.muted}`}>
                  Showing {sortedResults.length} of {results.length} tickers
                  {sortKeys.length > 0 &&
                    ` · sorted by ${sortKeys.map((k) => `${k.column} (${k.direction})`).join(", ")}`}
                </span>
                {scoreAsOfSummary && (
                  // NFR-3: Short/Long-Term Score come from the nightly job, not
                  // this request -- they can be a day or more stale if a
                  // night's run failed, so this must say when they're from.
                  <span className={`text-xs ${PF.muted}`} title="Short-Term/Long-Term Score and Signal columns">
                    Scores as of {scoreAsOfSummary}
                  </span>
                )}
                {tableOverflowing && (
                  <span className={`flex items-center gap-1.5 text-xs font-medium ${PF.warnText} ${PF.warnBg} rounded-full py-0.5 pl-2.5 pr-1`}>
                    {visibleColumns.length} columns selected — table scrolls sideways
                    <button
                      type="button"
                      onClick={() => scrollTable(-400)}
                      title="Scroll table left"
                      className="flex h-5 w-5 items-center justify-center rounded-full border border-current"
                    >
                      ←
                    </button>
                    <button
                      type="button"
                      onClick={() => scrollTable(400)}
                      title="Scroll table right"
                      className="flex h-5 w-5 items-center justify-center rounded-full border border-current"
                    >
                      →
                    </button>
                  </span>
                )}
              </div>
            )}

            {showFilters && mode === "rank" && (
              <div className={`${PF.card} p-4`}>
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
                    <label className={`text-xs font-medium ${PF.muted}`}>Volume Strength % (min)</label>
                    <input
                      type="number"
                      value={filters.volumeStrengthMin}
                      onChange={(e) => setFilters((prev) => ({ ...prev, volumeStrengthMin: e.target.value }))}
                      className={PF.input}
                      placeholder="e.g. 0"
                    />
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className={`text-xs font-medium ${PF.muted}`}>Dividend Yield % (min)</label>
                    <input
                      type="number"
                      value={filters.dividendYieldMin}
                      onChange={(e) => setFilters((prev) => ({ ...prev, dividendYieldMin: e.target.value }))}
                      className={PF.input}
                      placeholder="e.g. 2"
                    />
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className={`text-xs font-medium ${PF.muted}`}>6M Volatility % (max)</label>
                    <input
                      type="number"
                      value={filters.volatilityMax}
                      onChange={(e) => setFilters((prev) => ({ ...prev, volatilityMax: e.target.value }))}
                      className={PF.input}
                      placeholder="e.g. 20"
                    />
                  </div>
                  <RangeFilter
                    label="Momentum (3M Return %)"
                    min={filters.momentumMin}
                    max={filters.momentumMax}
                    onMinChange={(v) => setFilters((prev) => ({ ...prev, momentumMin: v }))}
                    onMaxChange={(v) => setFilters((prev) => ({ ...prev, momentumMax: v }))}
                  />
                  <RangeFilter
                    label="Earnings Growth %"
                    min={filters.earningsGrowthMin}
                    max={filters.earningsGrowthMax}
                    onMinChange={(v) => setFilters((prev) => ({ ...prev, earningsGrowthMin: v }))}
                    onMaxChange={(v) => setFilters((prev) => ({ ...prev, earningsGrowthMax: v }))}
                  />
                  <RangeFilter
                    label="Short-Term Score"
                    min={filters.shortScoreMin}
                    max={filters.shortScoreMax}
                    onMinChange={(v) => setFilters((prev) => ({ ...prev, shortScoreMin: v }))}
                    onMaxChange={(v) => setFilters((prev) => ({ ...prev, shortScoreMax: v }))}
                  />
                  <RangeFilter
                    label="Long-Term Score"
                    min={filters.longScoreMin}
                    max={filters.longScoreMax}
                    onMinChange={(v) => setFilters((prev) => ({ ...prev, longScoreMin: v }))}
                    onMaxChange={(v) => setFilters((prev) => ({ ...prev, longScoreMax: v }))}
                  />
                </div>
                <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <div>
                    <label className={`text-xs font-medium ${PF.muted}`}>Short-Term Signal</label>
                    <div className="mt-1 flex flex-wrap gap-2">
                      {SIGNAL_VALUES.map((sig) => (
                        <button
                          key={sig}
                          type="button"
                          onClick={() => toggleShortSignal(sig)}
                          className={`rounded-full border px-2.5 py-1 text-xs font-medium ${
                            filters.shortSignal.includes(sig)
                              ? "border-[var(--pf-accent)] bg-[var(--pf-accent)] text-white"
                              : `${PF.line} ${PF.muted} hover:bg-slate-100`
                          }`}
                        >
                          {sig}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div>
                    <label className={`text-xs font-medium ${PF.muted}`}>Long-Term Signal</label>
                    <div className="mt-1 flex flex-wrap gap-2">
                      {SIGNAL_VALUES.map((sig) => (
                        <button
                          key={sig}
                          type="button"
                          onClick={() => toggleLongSignal(sig)}
                          className={`rounded-full border px-2.5 py-1 text-xs font-medium ${
                            filters.longSignal.includes(sig)
                              ? "border-[var(--pf-accent)] bg-[var(--pf-accent)] text-white"
                              : `${PF.line} ${PF.muted} hover:bg-slate-100`
                          }`}
                        >
                          {sig}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
                <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <div>
                    <label className={`text-xs font-medium ${PF.muted}`}>Owned</label>
                    <div className="mt-1 flex gap-2">
                      {(["any", "only", "exclude"] as TriState[]).map((v) => (
                        <button
                          key={v}
                          type="button"
                          onClick={() => setFilters((prev) => ({ ...prev, owned: v }))}
                          className={`rounded-full border px-2.5 py-1 text-xs font-medium capitalize ${
                            filters.owned === v ? "border-[var(--pf-accent)] bg-[var(--pf-accent)] text-white" : `${PF.line} ${PF.muted} hover:bg-slate-100`
                          }`}
                        >
                          {v === "only" ? "Owned only" : v === "exclude" ? "Not owned" : "Any"}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div>
                    <label className={`text-xs font-medium ${PF.muted}`}>Watchlisted</label>
                    <div className="mt-1 flex gap-2">
                      {(["any", "only", "exclude"] as TriState[]).map((v) => (
                        <button
                          key={v}
                          type="button"
                          onClick={() => setFilters((prev) => ({ ...prev, watchlisted: v }))}
                          className={`rounded-full border px-2.5 py-1 text-xs font-medium capitalize ${
                            filters.watchlisted === v ? "border-[var(--pf-accent)] bg-[var(--pf-accent)] text-white" : `${PF.line} ${PF.muted} hover:bg-slate-100`
                          }`}
                        >
                          {v === "only" ? "Watchlisted only" : v === "exclude" ? "Not watchlisted" : "Any"}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
                <div className="mt-3">
                  <label className={`text-xs font-medium ${PF.muted}`}>Sector</label>
                  <div className="mt-1 flex flex-wrap gap-2">
                    {availableSectors.map((s) => (
                      <button
                        key={s}
                        type="button"
                        onClick={() => toggleSector(s)}
                        className={`rounded-full border px-2.5 py-1 text-xs font-medium ${
                          filters.sectors.includes(s)
                            ? "border-[var(--pf-accent)] bg-[var(--pf-accent)] text-white"
                            : `${PF.line} ${PF.muted} hover:bg-slate-100`
                        }`}
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="mt-3">
                  {/* A group heading, not a label for one control -- a <span>, not <label>, unlike
                      this file's older group headings (Sector, Short-Term Signal) which reuse
                      <label> for the same purpose; not fixed here, see NFR-7's tracker note. */}
                  <span className={`text-xs font-medium ${PF.muted}`}>Technical (ALX-5)</span>
                  <div className="mt-1 grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <RangeFilter
                      label="Close vs 50-day SMA %"
                      min={filters.closeVsSma50Min}
                      max={filters.closeVsSma50Max}
                      onMinChange={(v) => setFilters((prev) => ({ ...prev, closeVsSma50Min: v }))}
                      onMaxChange={(v) => setFilters((prev) => ({ ...prev, closeVsSma50Max: v }))}
                    />
                    <RangeFilter
                      label="Close vs 200-day SMA %"
                      min={filters.closeVsSma200Min}
                      max={filters.closeVsSma200Max}
                      onMinChange={(v) => setFilters((prev) => ({ ...prev, closeVsSma200Min: v }))}
                      onMaxChange={(v) => setFilters((prev) => ({ ...prev, closeVsSma200Max: v }))}
                    />
                    <RangeFilter
                      label="Distance from 52-week high %"
                      min={filters.dist52wHighMin}
                      max={filters.dist52wHighMax}
                      onMinChange={(v) => setFilters((prev) => ({ ...prev, dist52wHighMin: v }))}
                      onMaxChange={(v) => setFilters((prev) => ({ ...prev, dist52wHighMax: v }))}
                    />
                    <RangeFilter
                      label="Distance from 52-week low %"
                      min={filters.dist52wLowMin}
                      max={filters.dist52wLowMax}
                      onMinChange={(v) => setFilters((prev) => ({ ...prev, dist52wLowMin: v }))}
                      onMaxChange={(v) => setFilters((prev) => ({ ...prev, dist52wLowMax: v }))}
                    />
                  </div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {([
                      ["new52wHighOnly", "New 52W high"],
                      ["new52wLowOnly", "New 52W low"],
                      ["crossedAbove50Only", "SMA20 crossed above SMA50"],
                      ["crossedBelow50Only", "SMA20 crossed below SMA50"],
                    ] as const).map(([key, label]) => (
                      <button
                        key={key}
                        type="button"
                        onClick={() => setFilters((prev) => ({ ...prev, [key]: !prev[key] }))}
                        className={`rounded-full border px-2.5 py-1 text-xs font-medium ${
                          filters[key] ? "border-[var(--pf-accent)] bg-[var(--pf-accent)] text-white" : `${PF.line} ${PF.muted} hover:bg-slate-100`
                        }`}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                </div>
                {filtersActive(filters) && (
                  <button
                    type="button"
                    onClick={() => setFilters(EMPTY_FILTERS)}
                    className={`mt-3 text-xs font-medium ${PF.muted} underline hover:text-slate-900`}
                  >
                    Clear all filters
                  </button>
                )}
              </div>
            )}

            {showColumnPicker && mode === "rank" && (
              <div className={`${PF.card} p-4`}>
                <p className={`text-xs font-medium ${PF.muted}`}>Choose visible columns</p>
                <div className="mt-2 flex flex-wrap gap-x-4 gap-y-2">
                  {ALL_COLUMNS.map((col) => (
                    <label key={col} className="flex items-center gap-1.5 text-sm">
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
              <div ref={tableWrapRef} className={`max-h-[70vh] overflow-auto rounded-xl border ${PF.line} bg-white`}>
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className={`border-b ${PF.line} ${PF.surface2} text-left text-[11px] font-medium uppercase tracking-wide ${PF.muted}`}>
                      <th className={`sticky left-0 top-0 z-20 ${PF.surface2} w-8 px-2 py-2`} />
                      {visibleColumns.map((col) => {
                        const keyIndex = sortKeys.findIndex((k) => k.column === col);
                        const pinned = col === "Ticker";
                        return (
                          <th
                            key={col}
                            className={`sticky top-0 ${PF.surface2} px-3 py-2 ${pinned ? "left-8 z-20" : "z-10"}`}
                          >
                            <div className="flex items-center gap-1">
                              {col === "Spark 90D" ? (
                                <span className="uppercase tracking-wide">{col}</span>
                              ) : (
                                <button
                                  type="button"
                                  onClick={(e) => handleSort(col, e.shiftKey)}
                                  title="Click to sort; shift-click to add as a secondary sort key"
                                  className={`flex items-center gap-1 uppercase tracking-wide ${PF.muted} hover:text-slate-900`}
                                >
                                  {col}
                                  <span className="text-[10px]">
                                    {keyIndex !== -1
                                      ? `${sortKeys[keyIndex].direction === "asc" ? "▲" : "▼"}${
                                          sortKeys.length > 1 ? keyIndex + 1 : ""
                                        }`
                                      : ""}
                                  </span>
                                </button>
                              )}
                              <MetricLabel term={col} />
                            </div>
                          </th>
                        );
                      })}
                      <th className={`sticky top-0 z-10 ${PF.surface2} px-3 py-2`}></th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedResults.map((row) => {
                      const t = String(row.Ticker);
                      const quantSignal = quantSignals[t];
                      const analystRating = analystRatings[t];
                      const expanded = expandedRows.has(t);
                      return (
                        <Fragment key={t}>
                          <tr className={`border-b ${PF.line} last:border-0 hover:bg-slate-50`}>
                            <td className={`sticky left-0 z-[5] bg-white px-2 py-2`}>
                              <button
                                type="button"
                                onClick={() => toggleRow(t)}
                                className={`flex h-5 w-5 items-center justify-center rounded ${PF.muted} hover:bg-slate-100 hover:text-slate-900`}
                                title="Show more stats"
                              >
                                {expanded ? "▾" : "▸"}
                              </button>
                            </td>
                            {visibleColumns.map((col) => {
                              if (col === "Quant Signal") {
                                return (
                                  <td key={col} className="px-3 py-2">
                                    {!quantSignal ? (
                                      <button type="button" onClick={() => loadQuantSignal(t)} className={`${PF.btn} px-2 py-0.5 text-xs`}>
                                        Load
                                      </button>
                                    ) : quantSignal.status === "loading" ? (
                                      <span className={`text-xs ${PF.muted}`}>…</span>
                                    ) : quantSignal.status === "error" ? (
                                      <span className={`text-xs ${PF.muted}`}>—</span>
                                    ) : (
                                      <span className="flex items-center gap-1.5">
                                        <span
                                          className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
                                            quantSignal.signal.signal === "BUY"
                                              ? "bg-[var(--pf-accent-soft)] text-[var(--pf-up)]"
                                              : quantSignal.signal.signal === "SELL"
                                              ? "bg-[var(--pf-down-soft)] text-[var(--pf-down)]"
                                              : `${PF.surface2} ${PF.muted}`
                                          }`}
                                        >
                                          {quantSignal.signal.signal}
                                        </span>
                                        <span className={`text-xs ${PF.muted}`} style={{ fontFamily: "var(--font-pf-mono)" }}>
                                          {quantSignal.signal.expected_return_pct >= 0 ? "+" : ""}
                                          {quantSignal.signal.expected_return_pct.toFixed(2)}%
                                        </span>
                                        {quantSignal.signal.signal_flip_count !== null && (
                                          <span
                                            title={`Signal has flipped over its trailing ${quantSignal.signal.signal_days_captured}-day history`}
                                            className={`rounded-full px-1.5 py-0.5 text-[10px] font-semibold ${
                                              quantSignal.signal.signal_unstable
                                                ? `${PF.warnBg} ${PF.warnText}`
                                                : `${PF.surface2} ${PF.muted}`
                                            }`}
                                          >
                                            {quantSignal.signal.signal_flip_count} flip{quantSignal.signal.signal_flip_count === 1 ? "" : "s"}
                                          </span>
                                        )}
                                      </span>
                                    )}
                                  </td>
                                );
                              }
                              if (col === "Analyst Rating") {
                                return (
                                  <td key={col} className="px-3 py-2">
                                    {!analystRating ? (
                                      <button type="button" onClick={() => loadAnalystRating(t)} className={`${PF.btn} px-2 py-0.5 text-xs`}>
                                        Load
                                      </button>
                                    ) : analystRating.status === "loading" ? (
                                      <span className={`text-xs ${PF.muted}`}>…</span>
                                    ) : analystRating.status === "error" ? (
                                      <span className={`text-xs ${PF.muted}`}>No coverage</span>
                                    ) : (
                                      <span className="flex flex-col gap-0.5">
                                        <span className="flex items-center gap-1.5">
                                          <span
                                            className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
                                              /buy/i.test(analystRating.rating.consensus)
                                                ? "bg-[var(--pf-accent-soft)] text-[var(--pf-up)]"
                                                : /sell|underperform/i.test(analystRating.rating.consensus)
                                                ? "bg-[var(--pf-down-soft)] text-[var(--pf-down)]"
                                                : `${PF.surface2} ${PF.muted}`
                                            }`}
                                          >
                                            {analystRating.rating.consensus}
                                          </span>
                                          {analystRating.rating.analyst_count !== null && (
                                            <span className={`text-xs ${PF.muted}`}>({analystRating.rating.analyst_count})</span>
                                          )}
                                        </span>
                                        {analystRating.rating.target_mean !== null && (
                                          <span className={`text-xs ${PF.muted}`} style={{ fontFamily: "var(--font-pf-mono)" }}>
                                            Target ${analystRating.rating.target_mean.toFixed(2)}
                                          </span>
                                        )}
                                      </span>
                                    )}
                                  </td>
                                );
                              }
                              if (col === "Spark 90D") {
                                const values = (row["Spark 90D"] as number[] | undefined) ?? [];
                                return (
                                  <td key={col} className="px-3 py-2">
                                    <Sparkline
                                      values={values}
                                      color={goodBad(row["1M Return %"] as number | null) === PF.good ? "#047857" : "#b91c1c"}
                                    />
                                  </td>
                                );
                              }
                              const isNumeric = !TEXT_COLUMNS.has(col);
                              const pinned = col === "Ticker";
                              return (
                                <td
                                  key={col}
                                  className={`px-3 py-2 ${isNumeric ? "text-right" : ""} ${
                                    pinned ? "sticky left-8 z-[5] bg-white font-medium" : ""
                                  }`}
                                  style={isNumeric || pinned ? { fontFamily: "var(--font-pf-mono)" } : undefined}
                                >
                                  {formatCell(row[col])}
                                </td>
                              );
                            })}
                            <td className="px-3 py-2 text-right">
                              <div className="flex items-center justify-end gap-2">
                                <button
                                  type="button"
                                  onClick={() => {
                                    setWatchlistTicker(watchlistTicker === t ? null : t);
                                    setWatchlistMessage(null);
                                    setWatchlistThreshold("");
                                  }}
                                  className={`${PF.btn} px-2 py-1 text-xs`}
                                >
                                  Watchlist
                                </button>
                                <Link href={`/predict?ticker=${encodeURIComponent(t)}`} className={`${PF.btn} px-2 py-1 text-xs`}>
                                  Forecast
                                </Link>
                                <Link href={`/stock/${encodeURIComponent(t)}`} className={`${PF.btn} px-2 py-1 text-xs`}>
                                  Score
                                </Link>
                              </div>
                            </td>
                          </tr>
                          {expanded && (
                            <tr className={`border-b ${PF.line} ${PF.surface2} last:border-0`}>
                              <td />
                              <td colSpan={visibleColumns.length + 1} className="px-4 py-3">
                                {detailColumns.length > 0 ? (
                                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                                    {detailColumns.map((col) => (
                                      <div key={col}>
                                        <p className={`text-[10.5px] font-medium uppercase tracking-wide ${PF.muted}`}>{col}</p>
                                        <p className="mt-0.5 text-sm" style={{ fontFamily: "var(--font-pf-mono)" }}>
                                          {formatCell(row[col])}
                                        </p>
                                      </div>
                                    ))}
                                  </div>
                                ) : (
                                  <p className={`text-xs ${PF.muted}`}>Every available column is already shown — use Columns to hide some and see them here instead.</p>
                                )}
                              </td>
                            </tr>
                          )}
                          {watchlistTicker === t && (
                            <tr className={`border-b ${PF.line} ${PF.surface2} last:border-0`}>
                              <td />
                              <td colSpan={visibleColumns.length + 1} className="px-3 py-2">
                                <div className="flex flex-wrap items-center gap-2">
                                  <span className={`text-xs font-medium ${PF.muted}`}>Alert me when {t} is</span>
                                  <select
                                    value={watchlistCondition}
                                    onChange={(e) => setWatchlistCondition(e.target.value as AlertConditionType)}
                                    className={`${PF.input} px-2 py-1 text-xs`}
                                  >
                                    <option value="price_above">above</option>
                                    <option value="price_below">below</option>
                                  </select>
                                  <input
                                    type="number"
                                    value={watchlistThreshold}
                                    onChange={(e) => setWatchlistThreshold(e.target.value)}
                                    placeholder="Price"
                                    className={`${PF.input} w-24 px-2 py-1 text-xs`}
                                  />
                                  <button
                                    type="button"
                                    onClick={() => handleAddToWatchlist(t)}
                                    disabled={watchlistSaving}
                                    className={`${PF.btnPrimary} px-2.5 py-1 text-xs disabled:opacity-50`}
                                  >
                                    {watchlistSaving ? "Adding…" : "Add"}
                                  </button>
                                  {watchlistMessage && <span className={`text-xs ${PF.muted}`}>{watchlistMessage}</span>}
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

      </div>
    </div>
  );
}

function formatCell(value: string | number | boolean | number[] | null | undefined) {
  if (value === null || value === undefined) return "N/A";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (Array.isArray(value)) return "—";
  if (typeof value === "number") return Number.isInteger(value) ? value : value.toFixed(2);
  return value;
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <label className={`text-xs font-medium ${PF.muted}`}>{label}</label>
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
      <label className={`text-xs font-medium ${PF.muted}`}>{label}</label>
      <div className="flex items-center gap-2">
        <input type="number" value={min} onChange={(e) => onMinChange(e.target.value)} placeholder="Min" className={`${PF.input} w-full py-1.5`} />
        <span className={PF.muted}>–</span>
        <input type="number" value={max} onChange={(e) => onMaxChange(e.target.value)} placeholder="Max" className={`${PF.input} w-full py-1.5`} />
      </div>
    </div>
  );
}

function MetricTile({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: string;
}) {
  return (
    <div className={`${PF.card} p-3`}>
      <p className={`flex items-center gap-1 text-xs ${PF.muted}`}>
        <MetricLabel>{label}</MetricLabel>
      </p>
      <p className={`mt-1 text-lg font-semibold ${tone ?? ""}`} style={{ fontFamily: "var(--font-pf-display)" }}>
        {value}
      </p>
    </div>
  );
}
