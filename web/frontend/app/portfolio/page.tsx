"use client";

import { useEffect, useRef, useState } from "react";
import { Fraunces, IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";

import {
  ApiError,
  deletePortfolioPosition,
  editPortfolioPosition,
  getCurrentUser,
  getPortfolioInsights,
  getPortfolioPerformance,
  getPortfolioSentiment,
  getPortfolioStrategies,
  getPortfolioSummary,
  movePortfolioPosition,
  refreshPortfolio,
  setPortfolioMargin,
} from "@/lib/api";
import { isAdmin } from "@/lib/admin";
import type {
  Portfolio,
  PortfolioInsight,
  PortfolioPerformance,
  PortfolioStrategyRow,
  PortfolioSummary,
  TickerSentiment,
} from "@/lib/types";
import Link from "next/link";
import PlanText from "@/components/PlanText";
import GoalPlan from "@/components/GoalPlan";
import PortfolioSwitcher from "@/components/PortfolioSwitcher";
import TickerSearchInput from "@/components/TickerSearchInput";
import CurrentPriceBadge from "@/components/CurrentPriceBadge";
import BenchmarkComparisonCard from "@/components/portfolio/BenchmarkComparisonCard";
import GainVsPaidChart from "@/components/portfolio/GainVsPaidChart";
import MarketNewsTicker from "@/components/MarketNewsTicker";
import PortfolioReviewCard from "@/components/portfolio/PortfolioReviewCard";
import InfoModal, { type ColumnInfo } from "@/components/InfoModal";

// Scoped to this page only -- the rest of the site keeps its existing
// Geist font (see app/layout.tsx) and slate palette. "Ledger" direction
// from the published redesign concepts: warm paper, Fraunces for
// numbers/headings, IBM Plex for body/UI text and tabular data.
const fraunces = Fraunces({ subsets: ["latin"], weight: ["500", "600", "700"], variable: "--font-pf-display" });
const plexSans = IBM_Plex_Sans({ subsets: ["latin"], weight: ["400", "500", "600", "700"], variable: "--font-pf-sans" });
const plexMono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-pf-mono" });

const DISPLAY_FONT = { fontFamily: "var(--font-pf-display)" };
const MONO_FONT = { fontFamily: "var(--font-pf-mono)" };

// One small set of reusable class strings instead of the hex literal
// repeated at every call site -- this is the page's whole "Ledger"
// palette (see the published mockup): warm paper background, forest-
// green accent, and good/bad kept close to (but distinct from) that
// accent hue.
const PF = {
  page: "bg-[#f4f1ea]",
  ink: "text-[#1f2420]",
  muted: "text-[#857d6e]",
  line: "border-[#ddd8cd]",
  card: "rounded-xl border border-[#ddd8cd] bg-white",
  good: "text-[#2f6b4f]",
  bad: "text-[#a23b34]",
  btn: "rounded-md border border-[#ddd8cd] bg-white px-3 py-1.5 text-sm font-medium text-[#1f2420] hover:border-[#2f5d50] hover:text-[#2f5d50]",
  btnPrimary: "rounded-md bg-[#2f5d50] px-3 py-1.5 text-sm font-semibold text-[#f4f1ea] hover:bg-[#274e43]",
};

function goodBad(v: number | null | undefined): string {
  if (v === null || v === undefined) return PF.muted;
  return v >= 0 ? PF.good : PF.bad;
}

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
      "The latest trade price used for this row's value/gain figures. When the market is in pre-market or after-hours and a quote is available, this is that session's price, not the regular session's stale close — a badge marks it, and the regular-session price is shown underneath for reference.",
    ],
  },
  "Market Value": {
    title: "Market Value",
    body: [
      "What this position is worth right now: Shares × Price Now. Summed across every holding, this is the same number shown in the Total Value figure above.",
    ],
  },
  Today: {
    title: "Today",
    body: [
      "Today's dollar and percent gain/loss versus yesterday's regular-session close: (Price Now − Previous Close) × Shares — the standard \"day P&L\" figure most brokerages show.",
      "While the market is in pre-market or after-hours, this uses that session's price, so it reflects the after-hours move too, not just the regular session's.",
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

const LIVE_READ_INFO: ColumnInfo = {
  title: "Signal, Sentiment & Live Read",
  body: [
    "Signal (BUY/HOLD/SELL badge): the app's own quant model's call, based on a forecast over the next several trading days — same model used on /predict. Most positions land on HOLD by design; BUY/SELL require an expected return of at least +5%/-5% after the raw forecast is deliberately damped toward zero.",
    "Momentum Rank: where this ticker currently sits versus the rest of the tracked universe on trailing return — a real ranking, not a prediction.",
    "Sentiment (in Live Read): today's real news/earnings sentiment (Bullish/Neutral/Bearish), scored from actual search results — a current reading, not a 5-day/10-day forecast. \"Agrees\"/\"conflicts\" describes whether it lines up with the quant Signal above; when it says the two conflict, that's exactly the kind of tension worth digging into before acting.",
    "\"Stance\" is a general risk-management principle for any position at that P&L level and risk profile. \"Live Read\" is what's specific to this ticker right now (Signal, Sentiment, Momentum Rank) — read both together, not the Stance sentence alone.",
    "None of this is investment advice — it describes what the app's own signals currently show, not a recommendation.",
  ],
};

export default function PortfolioPage() {
  const [selectedPortfolioId, setSelectedPortfolioId] = useState<number | null>(null);
  // Guards against out-of-order responses: switching portfolios quickly
  // fires a new refresh() before a slower, now-stale one (e.g. under
  // Yahoo rate-limiting) has resolved. Without checking this ref before
  // applying a response, the stale request's data can land *after* the
  // fresh one and silently overwrite it with the wrong portfolio's
  // numbers — confirmed happening in practice, not just theoretical.
  const latestPortfolioIdRef = useRef<number | null>(null);
  useEffect(() => {
    latestPortfolioIdRef.current = selectedPortfolioId;
  }, [selectedPortfolioId]);
  const [allPortfolios, setAllPortfolios] = useState<Portfolio[]>([]);
  const [portfolioReloadSignal, setPortfolioReloadSignal] = useState(0);
  const [isAdminUser, setIsAdminUser] = useState(false);
  const [showGoalPlan, setShowGoalPlan] = useState(false);
  const [riskProfile] = useState("Balanced");
  const [riskFactor] = useState(5);

  const [marginInput, setMarginInput] = useState("");
  const [marginSaving, setMarginSaving] = useState(false);
  const [marginSaved, setMarginSaved] = useState(false);
  const [marginError, setMarginError] = useState<string | null>(null);
  const currentPortfolio = allPortfolios.find((p) => p.id === selectedPortfolioId) ?? null;

  useEffect(() => {
    setMarginInput(currentPortfolio ? String(currentPortfolio.margin_balance) : "");
    setMarginSaved(false);
    setMarginError(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentPortfolio?.id, currentPortfolio?.margin_balance]);

  async function saveMargin() {
    if (selectedPortfolioId === null) return;
    const value = Number(marginInput);
    if (!Number.isFinite(value) || value < 0) {
      setMarginError("Enter a non-negative number.");
      return;
    }
    setMarginSaving(true);
    setMarginError(null);
    setMarginSaved(false);
    try {
      await setPortfolioMargin(selectedPortfolioId, value);
      setAllPortfolios((prev) => prev.map((p) => (p.id === selectedPortfolioId ? { ...p, margin_balance: value } : p)));
      await refreshPerformance(false);
      setMarginSaved(true);
    } catch (err) {
      setMarginError(err instanceof ApiError ? err.message : "Could not save margin balance.");
    } finally {
      setMarginSaving(false);
    }
  }

  const [strategies, setStrategies] = useState<PortfolioStrategyRow[]>([]);
  const [summary, setSummary] = useState<PortfolioSummary | null>(null);
  const [performance, setPerformance] = useState<PortfolioPerformance | null>(null);
  const [performanceError, setPerformanceError] = useState<string | null>(null);
  const [performanceLoading, setPerformanceLoading] = useState(false);
  const [insights, setInsights] = useState<PortfolioInsight[]>([]);
  const [insightsError, setInsightsError] = useState<string | null>(null);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [sentiment, setSentiment] = useState<Record<string, TickerSentiment>>({});
  const [performanceInfoColumn, setPerformanceInfoColumn] = useState<string | null>(null);
  const [showLiveReadInfo, setShowLiveReadInfo] = useState(false);
  // Collapsed by default — with 15+ positions, every row's full Short-/
  // Long-Term Plan text (each with its own bullets, Stance, and Live
  // Read) made this page a very long scroll of mostly-repeated structure.
  // The header/price/badges row alone is enough to scan a whole
  // portfolio; the full narrative is one click away per position.
  const [expandedTickers, setExpandedTickers] = useState<Set<string>>(new Set());

  function toggleExpanded(ticker: string) {
    setExpandedTickers((prev) => {
      const next = new Set(prev);
      if (next.has(ticker)) next.delete(ticker);
      else next.add(ticker);
      return next;
    });
  }
  const [loading, setLoading] = useState(true);
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
    const requestedId = selectedPortfolioId;
    if (showLoading) setPerformanceLoading(true);
    setPerformanceError(null);
    try {
      const res = await getPortfolioPerformance(30, requestedId ?? undefined);
      if (latestPortfolioIdRef.current !== requestedId) return; // superseded by a newer portfolio switch
      setPerformance(res);
    } catch (err) {
      if (latestPortfolioIdRef.current !== requestedId) return;
      setPerformanceError(err instanceof ApiError ? err.message : "Could not load 30-day performance.");
    } finally {
      if (showLoading && latestPortfolioIdRef.current === requestedId) setPerformanceLoading(false);
    }
  }

  async function refreshInsights() {
    const requestedId = selectedPortfolioId;
    setInsightsLoading(true);
    setInsightsError(null);
    try {
      const res = await getPortfolioInsights(requestedId ?? undefined);
      if (latestPortfolioIdRef.current !== requestedId) return;
      setInsights(res.positions);
    } catch (err) {
      if (latestPortfolioIdRef.current !== requestedId) return;
      setInsightsError(err instanceof ApiError ? err.message : "Could not load signal/rank data for your holdings.");
    } finally {
      if (latestPortfolioIdRef.current === requestedId) setInsightsLoading(false);
    }
  }

  async function refresh() {
    const requestedId = selectedPortfolioId;
    setLoading(true);
    setError(null);
    try {
      const [stratRes, summaryRes] = await Promise.all([
        getPortfolioStrategies(requestedId ?? undefined),
        getPortfolioSummary(requestedId ?? undefined),
      ]);
      if (latestPortfolioIdRef.current !== requestedId) return; // a newer switch has already taken over
      setStrategies(stratRes.strategies);
      setSummary(summaryRes.summary);
    } catch (err) {
      if (latestPortfolioIdRef.current === requestedId) {
        setError(err instanceof ApiError ? err.message : "Could not load portfolio.");
      }
    } finally {
      if (latestPortfolioIdRef.current === requestedId) setLoading(false);
    }

    if (latestPortfolioIdRef.current !== requestedId) return;
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

  // Separate from refresh() deliberately, same rationale as /portfolio/
  // compare: sentiment is shared/cached by ticker across every user, but
  // the first request each day for an uncached ticker still runs a real
  // LLM call, so it shouldn't block the rest of this page's (fast) render.
  useEffect(() => {
    if (selectedPortfolioId === null) return;
    let cancelled = false;
    getPortfolioSentiment(selectedPortfolioId)
      .then((res) => {
        if (!cancelled) setSentiment(res.sentiment);
      })
      .catch(() => {
        if (!cancelled) setSentiment({});
      });
    return () => {
      cancelled = true;
    };
  }, [selectedPortfolioId]);

  useEffect(() => {
    getCurrentUser()
      .then((u) => setIsAdminUser(isAdmin(u.email)))
      .catch(() => {
        // Silent — worst case the polling effect below just stays off.
      });
  }, []);

  // Live-ish prices without per-position polling: one batched call for the
  // whole portfolio every 10s (well under /performance's 15/min rate limit
  // regardless of how many positions exist), instead of each position
  // polling independently the way CurrentPriceBadge does elsewhere.
  //
  // Admin-only: every open tab polling every 10s adds up to real,
  // continuous yfinance load across all of a user's positions — for the
  // admin account that's an accepted cost of "live-ish" data, but it's
  // not worth every user's browser tab contributing to the same
  // Yahoo-rate-limit pressure the rest of the app is already fighting.
  // Non-admin users still get fresh prices on every explicit refresh
  // (switching portfolios, editing a position, etc.), just not this
  // background tick.
  //
  // /performance can take well over 10s under live Yahoo rate-limiting
  // (observed up to ~50s) — a plain setInterval doesn't wait for the
  // previous call to resolve, so it would otherwise stack overlapping
  // in-flight requests every tick during exactly the conditions where
  // that's most harmful, further loading down the already-throttled
  // yfinance backend. inFlight skips a tick instead of stacking one.
  useEffect(() => {
    if (!isAdminUser || summary === null || summary.total_positions === 0) return;
    let inFlight = false;
    const tick = () => {
      // Backgrounded/minimized tab: skip this tick rather than burning a
      // yfinance-backed call on data nobody's looking at. A returning tab
      // fires document.visibilitychange below and catches up immediately
      // instead of waiting out the rest of this interval.
      if (document.hidden || inFlight) return;
      inFlight = true;
      refreshPerformance(false).finally(() => {
        inFlight = false;
      });
    };
    const interval = setInterval(tick, 10000);
    const onVisibilityChange = () => {
      if (!document.hidden) tick();
    };
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      clearInterval(interval);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
    // selectedPortfolioId must be a dependency: refreshPerformance closes
    // over it, and without it here, switching to a portfolio with the same
    // total_positions count as the previous one wouldn't change any
    // dependency React can see — the old interval (and its stale
    // portfolio_id) would keep running instead of being torn down and
    // recreated against the newly-selected portfolio.
  }, [isAdminUser, summary?.total_positions, selectedPortfolioId]);

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

  const totalValue = performance?.total_value_now ?? summary?.total_value ?? null;

  return (
    <div className={`${fraunces.variable} ${plexSans.variable} ${plexMono.variable} ${PF.page} ${PF.ink}`} style={{ fontFamily: "var(--font-pf-sans)" }}>
      <div className="mx-auto max-w-7xl px-4 py-8">
        <MarketNewsTicker />

        {/* ---------- Toolbar: portfolio switcher + entry points ---------- */}
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-b border-[#ddd8cd] pb-5">
          <PortfolioSwitcher
            selectedPortfolioId={selectedPortfolioId}
            onChange={setSelectedPortfolioId}
            onPortfoliosChange={setAllPortfolios}
            reloadSignal={portfolioReloadSignal}
          />
          <div className="flex flex-wrap gap-2">
            <Link href="/portfolio/add" className={PF.btn}>
              + Add Positions
            </Link>
            <Link href="/portfolio/add?mode=plaid" className={PF.btn}>
              Connect Brokerage
            </Link>
            <Link href="/portfolio/build-index" className={PF.btn}>
              + Build Diversified Index
            </Link>
            <Link href="/portfolio/compare" className={PF.btn}>
              Compare vs. Best Fund
            </Link>
          </div>
        </div>

        {/* ---------- Goal plan + margin (utility row) ---------- */}
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <input
            type="checkbox"
            id="show-goal-plan"
            checked={showGoalPlan}
            onChange={(e) => setShowGoalPlan(e.target.checked)}
            className="h-3.5 w-3.5 rounded border-[#ddd8cd]"
          />
          <label htmlFor="show-goal-plan" className="text-sm font-medium text-[#1f2420]">
            Goal-Based Investing Plan
          </label>
        </div>

        {showGoalPlan && <GoalPlan portfolioId={selectedPortfolioId} />}

        {selectedPortfolioId !== null && (
          <div className="mt-3 flex flex-wrap items-end gap-2">
            <Field label="Margin balance ($ borrowed from broker)">
              <input
                type="number"
                min={0}
                step="0.01"
                value={marginInput}
                onChange={(e) => setMarginInput(e.target.value)}
                className="w-44 rounded-md border border-[#ddd8cd] bg-white px-3 py-1.5 text-sm text-[#1f2420]"
                style={MONO_FONT}
              />
            </Field>
            <button type="button" onClick={saveMargin} disabled={marginSaving} className={`${PF.btn} disabled:opacity-50`}>
              {marginSaving ? "Saving…" : "Save"}
            </button>
            {marginSaved && <span className={`text-xs font-medium ${PF.good}`}>Saved</span>}
            {marginError && <span className={`text-xs font-medium ${PF.bad}`}>{marginError}</span>}
          </div>
        )}

        {error && <p className={`mt-4 rounded-md border border-[#e4c9c5] bg-[#fbeceb] px-3 py-2 text-sm ${PF.bad}`}>{error}</p>}
        {watchlistNote && (
          <p className={`mt-4 rounded-md border border-[#cfe0d8] bg-[#ecf3ef] px-3 py-2 text-sm ${PF.good}`}>
            {watchlistNote}{" "}
            <a href="/watchlist" className="underline">
              View watchlist
            </a>
          </p>
        )}

        {/* ---------- Hero: total value + sub-line + chip row ---------- */}
        {summary && (
          <div className="mt-6 grid grid-cols-1 gap-8 border-b border-[#ddd8cd] pb-7 lg:grid-cols-[1.1fr_1fr]">
            <div>
              <p className="font-mono text-[11px] uppercase tracking-wider text-[#857d6e]" style={MONO_FONT}>
                Total value · {currentPortfolio?.name ?? "Portfolio"}
              </p>
              <p className="mt-1 text-5xl font-semibold leading-none" style={DISPLAY_FONT}>
                {totalValue !== null ? `$${totalValue.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : "—"}
              </p>
              {performance && (
                <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-sm text-[#514c43]">
                  {performance.total_day_gain !== null && (
                    <span>
                      Today{" "}
                      <b className={goodBad(performance.total_day_gain)} style={MONO_FONT}>
                        {performance.total_day_gain >= 0 ? "+" : ""}$
                        {performance.total_day_gain.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        {performance.total_day_gain_pct !== null &&
                          ` (${performance.total_day_gain_pct >= 0 ? "+" : ""}${performance.total_day_gain_pct.toFixed(2)}%)`}
                      </b>
                    </span>
                  )}
                  <span>
                    30 days{" "}
                    <b className={goodBad(performance.value_diff)} style={MONO_FONT}>
                      {performance.value_diff >= 0 ? "+" : ""}$
                      {performance.value_diff.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                      {performance.value_diff_pct !== null &&
                        ` (${performance.value_diff_pct >= 0 ? "+" : ""}${performance.value_diff_pct.toFixed(2)}%)`}
                    </b>
                  </span>
                  <span>
                    Since cost{" "}
                    <b className={goodBad(performance.total_gain_vs_cost)} style={MONO_FONT}>
                      {performance.total_gain_vs_cost >= 0 ? "+" : ""}$
                      {performance.total_gain_vs_cost.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                      {performance.total_gain_vs_cost_pct !== null &&
                        ` (${performance.total_gain_vs_cost_pct >= 0 ? "+" : ""}${performance.total_gain_vs_cost_pct.toFixed(2)}%)`}
                    </b>
                  </span>
                </div>
              )}
            </div>
            <div className="flex flex-wrap content-start gap-3">
              <Chip label="Positions" value={String(summary.total_positions)} />
              <Chip label="Unrealized PnL" value={`${summary.total_pnl_pct.toFixed(2)}%`} tone={summary.total_pnl_pct} />
              {performance && <Chip label="Total Paid" value={`$${performance.total_cost_basis.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />}
              {performance && performance.margin_balance > 0 && (
                <Chip
                  label="Net Equity"
                  value={`$${performance.net_equity.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                  tone={performance.net_equity}
                />
              )}
            </div>
          </div>
        )}

        {summary && summary.total_positions > 0 && <PortfolioReviewCard portfolioId={selectedPortfolioId} />}

        {summary && summary.total_positions > 0 && (
          <div className="mt-2">
            {performanceLoading && !performance && <p className="mt-4 text-sm text-[#857d6e]">Loading…</p>}
            {performanceError && (
              <p className={`mt-4 rounded-md border border-[#e4c9c5] bg-[#fbeceb] px-3 py-2 text-sm ${PF.bad}`}>{performanceError}</p>
            )}

            {performance && performance.rows.length > 0 && (
              <>
                <BenchmarkComparisonCard portfolioId={selectedPortfolioId} />

                {performance.rows.some((r) => r.used_extended_hours) && (
                  <p className="mt-2 text-xs text-[#857d6e]">
                    Includes after-hours/pre-market prices for{" "}
                    {performance.rows.filter((r) => r.used_extended_hours).length} holding
                    {performance.rows.filter((r) => r.used_extended_hours).length === 1 ? "" : "s"} — see the Price
                    column for which.
                  </p>
                )}

                <GainVsPaidChart rows={performance.rows} />
              </>
            )}
          </div>
        )}

        {/* ---------- Holdings: one table, one row per position, expand for detail ---------- */}
        <div className="mt-9 flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-xl font-semibold" style={DISPLAY_FONT}>
            Holdings
          </h2>
          <div className="flex flex-wrap items-center gap-2">
            {strategies.length > 0 && (
              <button onClick={() => setExpandedTickers(new Set(strategies.map((s) => s.ticker)))} className={PF.btn}>
                Expand All
              </button>
            )}
            {expandedTickers.size > 0 && (
              <button onClick={() => setExpandedTickers(new Set())} className={PF.btn}>
                Collapse All
              </button>
            )}
            {strategies.length > 0 && (
              <button onClick={handleRefresh} disabled={refreshing} className={`${PF.btn} disabled:opacity-50`}>
                {refreshing ? "Refreshing…" : "Refresh with Current Market"}
              </button>
            )}
          </div>
        </div>
        <p className="mt-1 text-xs text-[#857d6e]">
          Today&apos;s price for every holding, priced fresh each load. Click a row for forecasts, the 30-day/cost
          comparison, and its short-/long-term plan.
        </p>

        <p className="mt-4 text-xs font-medium text-[#857d6e]">
          Add a new position — this only appends this one ticker, it won&apos;t touch anything else you&apos;ve
          saved.
        </p>
        <form onSubmit={handleAddPosition} className={`mt-1 flex flex-wrap items-end gap-2 ${PF.card} p-4`}>
          <Field label="Ticker">
            <TickerSearchInput value={addTicker} onChange={setAddTicker} className="input w-32 uppercase" />
          </Field>
          <CurrentPriceBadge ticker={addTicker} />
          <Field label="Shares">
            <input
              type="number"
              step="0.0001"
              value={addShares}
              onChange={(e) => setAddShares(e.target.value)}
              className="w-24 rounded-md border border-[#ddd8cd] bg-white px-3 py-1.5 text-sm text-[#1f2420]"
            />
          </Field>
          <Field label="Avg cost">
            <input
              type="number"
              step="0.01"
              value={addAvgCost}
              onChange={(e) => setAddAvgCost(e.target.value)}
              className="w-24 rounded-md border border-[#ddd8cd] bg-white px-3 py-1.5 text-sm text-[#1f2420]"
            />
          </Field>
          <button type="submit" disabled={adding} className={`${PF.btnPrimary} disabled:opacity-50`}>
            {adding ? "Adding…" : "Add to Portfolio"}
          </button>
          {addError && <p className={`w-full text-xs ${PF.bad}`}>{addError}</p>}
        </form>

        {loading ? (
          <p className="mt-3 text-sm text-[#857d6e]">Loading…</p>
        ) : strategies.length === 0 ? (
          <p className="mt-3 text-sm text-[#857d6e]">No saved strategies yet.</p>
        ) : (
          <div className={`mt-3 overflow-hidden ${PF.card}`}>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="border-b border-[#ddd8cd] bg-[#efebe3] text-left text-[10.5px] font-semibold uppercase tracking-wide text-[#857d6e]">
                    <th className="px-4 py-3">Ticker</th>
                    <th className="px-4 py-3">Signal</th>
                    <th className="px-4 py-3 text-right">Shares</th>
                    <th className="px-4 py-3 text-right">Price</th>
                    <th className="px-4 py-3 text-right">Value</th>
                    <th className="px-4 py-3 text-right">Today</th>
                    <th className="px-4 py-3 text-right">Since Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {strategies.map((s) => {
                    const perfRow = performance?.rows.find((r) => r.ticker === s.ticker) ?? null;
                    const insight = insights.find((i) => i.ticker === s.ticker) ?? null;
                    const tickerSentiment = sentiment[s.ticker] ?? null;
                    const isExpanded = expandedTickers.has(s.ticker);
                    const isEditing = editingTicker === s.ticker;
                    const isMoving = movingTicker === s.ticker;

                    const livePrice = perfRow?.price_now ?? s.current_price;
                    const pnlPct = perfRow?.gain_vs_cost_pct ?? s.unrealized_pnl_pct;
                    const gainVsCostDollar = perfRow?.gain_vs_cost ?? null;

                    return (
                      <FragmentRow key={s.id}>
                        <tr
                          className="cursor-pointer border-b border-[#ede9df] last:border-0 hover:bg-[#faf8f3]"
                          onClick={() => toggleExpanded(s.ticker)}
                        >
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-2">
                              <Chevron open={isExpanded} />
                              <span className="font-semibold">{s.ticker}</span>
                              {insight?.concentrated && (
                                <span
                                  title="A single position this large drives most of your portfolio's swings."
                                  className="rounded-full bg-[#f4e3c9] px-1.5 py-0.5 text-[10px] font-bold text-[#8a6417]"
                                >
                                  {insight.weight_pct?.toFixed(0)}%
                                </span>
                              )}
                            </div>
                            {perfRow?.acquired_at && (
                              <div className="mt-0.5 pl-[22px] text-[11px] text-[#a39b8b]">
                                Held since {fmtAcquiredAt(perfRow.acquired_at)}
                              </div>
                            )}
                          </td>
                          <td className="px-4 py-3">
                            {insight?.signal ? (
                              <span
                                className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
                                  insight.signal === "BUY"
                                    ? "bg-[#e3ede8] text-[#2f6b4f]"
                                    : insight.signal === "SELL"
                                    ? "bg-[#f6e5e3] text-[#a23b34]"
                                    : "bg-[#efece4] text-[#6b6459]"
                                }`}
                              >
                                {insight.signal}
                              </span>
                            ) : (
                              <span className="text-[#a39b8b]">{insightsLoading ? "…" : "—"}</span>
                            )}
                          </td>
                          <td className="px-4 py-3 text-right" style={MONO_FONT}>
                            {s.shares?.toFixed(2) ?? "—"}
                          </td>
                          <td className="px-4 py-3 text-right" style={MONO_FONT}>
                            {livePrice !== null && livePrice !== undefined ? `$${livePrice.toFixed(2)}` : "—"}
                            {perfRow?.used_extended_hours && perfRow.extended_hours && (
                              <div
                                className={`mt-0.5 text-[10px] font-semibold ${
                                  (perfRow.extended_hours.change_pct ?? 0) >= 0 ? PF.good : PF.bad
                                }`}
                              >
                                {perfRow.extended_hours.state === "POST" ? "after hours" : "pre-market"}
                              </div>
                            )}
                          </td>
                          <td className="px-4 py-3 text-right font-medium" style={MONO_FONT}>
                            {perfRow && perfRow.value_now !== null
                              ? `$${perfRow.value_now.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
                              : "—"}
                          </td>
                          <td className={`px-4 py-3 text-right font-medium ${goodBad(perfRow?.day_gain ?? null)}`} style={MONO_FONT}>
                            {perfRow?.day_gain != null
                              ? `${perfRow.day_gain >= 0 ? "+" : ""}${perfRow.day_gain.toLocaleString(undefined, { maximumFractionDigits: 0 })}${
                                  perfRow.day_gain_pct !== null ? ` (${perfRow.day_gain_pct >= 0 ? "+" : ""}${perfRow.day_gain_pct.toFixed(1)}%)` : ""
                                }`
                              : "—"}
                          </td>
                          <td className={`px-4 py-3 text-right font-medium ${goodBad(pnlPct)}`} style={MONO_FONT}>
                            {gainVsCostDollar !== null
                              ? `${gainVsCostDollar >= 0 ? "+" : ""}${gainVsCostDollar.toLocaleString(undefined, { maximumFractionDigits: 0 })}${
                                  pnlPct !== null ? ` (${pnlPct >= 0 ? "+" : ""}${pnlPct.toFixed(1)}%)` : ""
                                }`
                              : pnlPct !== null
                              ? `${pnlPct >= 0 ? "+" : ""}${pnlPct.toFixed(2)}%`
                              : "—"}
                          </td>
                        </tr>

                        {isExpanded && (
                          <tr className="border-b border-[#ede9df] bg-[#faf8f3] last:border-0">
                            <td colSpan={7} className="px-4 py-5 pl-11">
                              {perfRow?.price_unavailable ? (
                                <p className="text-sm text-[#a39b8b]">
                                  No market data found for this ticker — check it&apos;s a valid, publicly-traded
                                  symbol.
                                </p>
                              ) : (
                                <div className="grid grid-cols-2 gap-5 sm:grid-cols-4">
                                  <DetailStat label="Momentum Rank">
                                    {insight?.rank != null && insight.universe_size
                                      ? `#${insight.rank} of ${insight.universe_size}`
                                      : insightsLoading
                                      ? "…"
                                      : "—"}
                                  </DetailStat>
                                  <DetailStat label="1-Day Forecast" tone={insight?.expected_return_pct_1d ?? null}>
                                    {insight?.target_price_1d != null && insight?.expected_return_pct_1d != null
                                      ? `$${insight.target_price_1d.toFixed(2)} (${insight.expected_return_pct_1d >= 0 ? "+" : ""}${insight.expected_return_pct_1d.toFixed(2)}%)`
                                      : "—"}
                                  </DetailStat>
                                  <DetailStat label="5-Day Forecast" tone={insight?.expected_return_pct_5d ?? null}>
                                    {insight?.target_price_5d != null && insight?.expected_return_pct_5d != null
                                      ? `$${insight.target_price_5d.toFixed(2)} (${insight.expected_return_pct_5d >= 0 ? "+" : ""}${insight.expected_return_pct_5d.toFixed(2)}%)`
                                      : "—"}
                                  </DetailStat>
                                  <DetailStat label="10-Day Forecast" tone={insight?.expected_return_pct ?? null}>
                                    {insight?.target_price != null && insight?.expected_return_pct != null
                                      ? `$${insight.target_price.toFixed(2)} (${insight.expected_return_pct >= 0 ? "+" : ""}${insight.expected_return_pct.toFixed(2)}%)`
                                      : "—"}
                                  </DetailStat>
                                  <DetailStat label="Price 30D Ago">
                                    {perfRow?.price_30d_ago != null ? `$${perfRow.price_30d_ago.toFixed(2)}` : "—"}
                                  </DetailStat>
                                  <DetailStat label="30D Diff" tone={perfRow?.diff ?? null}>
                                    {perfRow?.diff != null
                                      ? `${perfRow.diff >= 0 ? "+" : ""}${perfRow.diff.toLocaleString(undefined, { maximumFractionDigits: 0 })}${
                                          perfRow.diff_pct !== null ? ` (${perfRow.diff_pct >= 0 ? "+" : ""}${perfRow.diff_pct.toFixed(1)}%)` : ""
                                        }`
                                      : "—"}
                                  </DetailStat>
                                  <DetailStat label="Avg Cost Paid">
                                    {s.avg_cost !== null && s.avg_cost !== undefined ? `$${s.avg_cost.toFixed(2)}` : "—"}
                                  </DetailStat>
                                  <DetailStat label="Look up">
                                    <div className="flex flex-col gap-0.5">
                                      <Link
                                        href={`/predict?ticker=${s.ticker}&from=portfolio`}
                                        className="text-[#2f5d50] hover:underline"
                                        onClick={(e) => e.stopPropagation()}
                                      >
                                        Forecast
                                      </Link>
                                      <Link
                                        href={`/signal-comparison?ticker=${s.ticker}&from=portfolio`}
                                        className="text-[#2f5d50] hover:underline"
                                        onClick={(e) => e.stopPropagation()}
                                      >
                                        Quant vs Analyst
                                      </Link>
                                    </div>
                                  </DetailStat>
                                </div>
                              )}

                              {insight && (
                                <div className="mt-4 flex flex-wrap items-center gap-2 text-xs">
                                  {insight.concentrated && insight.weight_pct !== null && (
                                    <span
                                      title="A single position this large drives most of your portfolio's swings — consider whether that's intentional."
                                      className="rounded-full bg-[#f4e3c9] px-2 py-0.5 font-semibold text-[#8a6417]"
                                    >
                                      {insight.weight_pct.toFixed(0)}% of portfolio — concentrated
                                    </span>
                                  )}
                                  <button
                                    type="button"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setShowLiveReadInfo(true);
                                    }}
                                    title="What do Signal, Sentiment, and Live Read mean?"
                                    aria-label="What do Signal, Sentiment, and Live Read mean?"
                                    className="flex h-4 w-4 items-center justify-center rounded-full border border-[#ddd8cd] text-[10px] font-normal text-[#a39b8b] hover:border-[#857d6e] hover:text-[#1f2420]"
                                  >
                                    i
                                  </button>
                                </div>
                              )}

                              <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
                                <div className="rounded-md border border-[#ede9df] bg-white p-3">
                                  <PlanText text={withLiveRead(s.short_term_plan, shortTermSignalNote(insight, tickerSentiment))} />
                                </div>
                                <div className="rounded-md border border-[#ede9df] bg-white p-3">
                                  <PlanText text={withLiveRead(s.long_term_plan, longTermMomentumNote(insight))} />
                                </div>
                              </div>

                              <div className="mt-4 border-t border-[#ede9df] pt-4" onClick={(e) => e.stopPropagation()}>
                                {isMoving ? (
                                  <div className="flex flex-wrap items-end gap-2">
                                    <Field label="Move to">
                                      <select
                                        value={moveTargetId}
                                        onChange={(e) => setMoveTargetId(e.target.value)}
                                        className="rounded-md border border-[#ddd8cd] bg-white px-3 py-1.5 text-sm"
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
                                    <button onClick={() => confirmMove(s.ticker)} disabled={moveSaving} className={`${PF.btnPrimary} disabled:opacity-50`}>
                                      {moveSaving ? "Moving…" : "Confirm Move"}
                                    </button>
                                    <button onClick={cancelMove} disabled={moveSaving} className={PF.btn}>
                                      Cancel
                                    </button>
                                    {positionActionError && <p className={`w-full text-xs ${PF.bad}`}>{positionActionError}</p>}
                                  </div>
                                ) : isEditing ? (
                                  <div className="flex flex-wrap items-end gap-2">
                                    <Field label="Shares">
                                      <input
                                        type="number"
                                        step="0.0001"
                                        value={editShares}
                                        onChange={(e) => setEditShares(e.target.value)}
                                        className="w-24 rounded-md border border-[#ddd8cd] bg-white px-3 py-1.5 text-sm"
                                      />
                                    </Field>
                                    <Field label="Avg cost">
                                      <input
                                        type="number"
                                        step="0.01"
                                        value={editAvgCost}
                                        onChange={(e) => setEditAvgCost(e.target.value)}
                                        className="w-24 rounded-md border border-[#ddd8cd] bg-white px-3 py-1.5 text-sm"
                                      />
                                    </Field>
                                    <button onClick={() => saveEdit(s.ticker)} disabled={editSaving} className={`${PF.btnPrimary} disabled:opacity-50`}>
                                      {editSaving ? "Saving…" : "Save"}
                                    </button>
                                    <button onClick={cancelEdit} disabled={editSaving} className={PF.btn}>
                                      Cancel
                                    </button>
                                    {editError && <p className={`w-full text-xs ${PF.bad}`}>{editError}</p>}
                                  </div>
                                ) : (
                                  <div className="flex flex-wrap gap-2">
                                    <button onClick={() => startEdit(s)} className={PF.btn}>
                                      Edit
                                    </button>
                                    {allPortfolios.length > 1 && (
                                      <button onClick={() => startMove(s.ticker)} className={PF.btn}>
                                        Move
                                      </button>
                                    )}
                                    <button
                                      onClick={() => handleDeletePosition(s.ticker)}
                                      disabled={deletingTicker === s.ticker}
                                      className="rounded-md border border-[#e4c9c5] px-3 py-1.5 text-sm font-medium text-[#a23b34] hover:bg-[#fbeceb] disabled:opacity-50"
                                    >
                                      {deletingTicker === s.ticker ? "Deleting…" : "Delete"}
                                    </button>
                                  </div>
                                )}
                              </div>
                            </td>
                          </tr>
                        )}
                      </FragmentRow>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {insightsError && (
          <p className={`mt-3 rounded-md border border-[#e4c9c5] bg-[#fbeceb] px-3 py-2 text-sm ${PF.bad}`}>{insightsError}</p>
        )}
        {positionActionError && movingTicker === null && (
          <p className={`mt-3 rounded-md border border-[#e4c9c5] bg-[#fbeceb] px-3 py-2 text-sm ${PF.bad}`}>{positionActionError}</p>
        )}

        {performanceInfoColumn && PERFORMANCE_COLUMN_INFO[performanceInfoColumn] && (
          <InfoModal
            info={PERFORMANCE_COLUMN_INFO[performanceInfoColumn]}
            onClose={() => setPerformanceInfoColumn(null)}
          />
        )}

        {showLiveReadInfo && <InfoModal info={LIVE_READ_INFO} onClose={() => setShowLiveReadInfo(false)} />}
      </div>
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
function shortTermSignalNote(insight: PortfolioInsight | null, sentiment: TickerSentiment | null): string | null {
  if (!insight || !insight.signal) return null;
  const expected =
    insight.expected_return_pct !== null
      ? ` (model expects ${insight.expected_return_pct >= 0 ? "+" : ""}${insight.expected_return_pct.toFixed(1)}% over its forecast horizon)`
      : "";

  let signalLine: string;
  if (insight.signal === "BUY") {
    signalLine = `The model's current signal is **BUY**${expected} — consistent with the case to keep holding here.`;
  } else if (insight.signal === "SELL") {
    signalLine = `The model's current signal is **SELL**${expected} — this cuts against holding; worth watching closely, or locking in gains if you'd rather not fight the signal.`;
  } else {
    signalLine = `The model's current signal is **HOLD**${expected} — no strong edge either way right now.`;
  }

  // Signal alone clusters heavily on HOLD (BUY/SELL need a >=5%/<=-5%
  // expected return), which made same-signal cards read as near-identical
  // even with the % attached. Sentiment — scored per ticker from real
  // news/earnings text — varies more, so it's appended as a second,
  // genuinely differentiating layer rather than replacing the signal note.
  if (!sentiment || !sentiment.label) return signalLine;

  const agrees =
    (sentiment.label === "Bullish" && insight.signal === "BUY") ||
    (sentiment.label === "Bearish" && insight.signal === "SELL");
  const conflicts =
    (sentiment.label === "Bullish" && insight.signal === "SELL") ||
    (sentiment.label === "Bearish" && insight.signal === "BUY");

  let sentimentClause: string;
  if (agrees) {
    sentimentClause = `Today's news/earnings sentiment reads **${sentiment.label}** too — the two line up.`;
  } else if (conflicts) {
    sentimentClause = `Today's news/earnings sentiment reads **${sentiment.label}**, which cuts against the model's own call — a real tension worth digging into.`;
  } else {
    sentimentClause = `Today's news/earnings sentiment reads **${sentiment.label}**.`;
  }
  if (sentiment.reasoning) {
    sentimentClause += ` (${sentiment.reasoning})`;
  }

  return `${signalLine} ${sentimentClause}`;
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

function fmtAcquiredAt(isoDate: string): string {
  // acquired_at is a date-only value ("YYYY-MM-DD") -- new Date(isoDate)
  // would parse it as UTC midnight, which can display as the previous
  // day in any timezone behind UTC. Building a local-time Date from the
  // parsed components avoids that off-by-one.
  const [year, month, day] = isoDate.split("-").map(Number);
  if (!year || !month || !day) return isoDate;
  return new Date(year, month - 1, day).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs font-medium text-[#857d6e]">{label}</label>
      {children}
    </div>
  );
}

function Chip({ label, value, tone }: { label: string; value: string; tone?: number | null }) {
  const valueClass = tone === undefined ? "" : goodBad(tone);
  return (
    <div className="min-w-[128px] rounded-lg border border-[#ddd8cd] bg-white px-4 py-3">
      <p className="font-mono text-[10.5px] uppercase tracking-wide text-[#857d6e]" style={MONO_FONT}>
        {label}
      </p>
      <p className={`mt-0.5 text-lg font-semibold ${valueClass}`} style={MONO_FONT}>
        {value}
      </p>
    </div>
  );
}

function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2.5}
      className={`h-3.5 w-3.5 flex-none text-[#857d6e] transition-transform ${open ? "rotate-90" : ""}`}
    >
      <path d="M9 5l7 7-7 7" />
    </svg>
  );
}

function DetailStat({ label, tone, children }: { label: string; tone?: number | null; children: React.ReactNode }) {
  const valueClass = tone === undefined ? "" : goodBad(tone);
  return (
    <div>
      <p className="font-mono text-[10px] uppercase tracking-wide text-[#857d6e]" style={MONO_FONT}>
        {label}
      </p>
      <p className={`mt-0.5 text-sm font-semibold ${valueClass}`} style={MONO_FONT}>
        {children}
      </p>
    </div>
  );
}

// A holdings row is really two <tr>s (the row itself, plus an optional
// detail row) that must stay adjacent siblings inside <tbody> -- a
// wrapping element would break table semantics. React.Fragment does this
// without one, but needs a key when used in a list; this thin wrapper
// keeps the call sites above readable.
function FragmentRow({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
