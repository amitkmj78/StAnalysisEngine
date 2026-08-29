"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";

import {
  ApiError,
  getFundGoals,
  getFundRanking,
  getFundReturnSince,
  getPortfolio1yForecast,
  getPortfolioInsights,
  getPortfolioPerformance,
  getPortfolioSummary,
} from "@/lib/api";
import type {
  FundRankRow,
  FundReturnSince,
  Portfolio,
  PortfolioInsight,
  PortfolioPerformance,
  PortfolioSummary,
} from "@/lib/types";
import PortfolioSwitcher from "@/components/PortfolioSwitcher";
import InfoModal, { type ColumnInfo } from "@/components/InfoModal";

const GOAL_INFO: ColumnInfo = {
  title: "What do these goals mean?",
  body: [
    "Each goal describes what the fund's Score weighs — not a rule about what kind of fund can win. A single-sector fund can still top \"Balanced Core\" if it scores well on that blend, even though the fund itself isn't diversified.",
    "Balanced Core: a well-rounded blend — 1Y return 35%, 3Y annualized 25%, expense ratio 20%, 1Y volatility 10%, 3Y max drawdown 10%.",
    "Lowest Cost: minimizing fees above almost everything else — expense ratio 65%, 3Y annualized 20%, 1Y volatility 10%, fund assets 5%.",
    "Best Growth: chasing the highest returns — 1Y return 50%, 3Y annualized 35%, 1Y volatility 10%, expense ratio 5%.",
    "Most Stable: minimizing swings and drawdowns — 1Y volatility 45%, 3Y max drawdown 30%, expense ratio 15%, 3Y annualized 10%.",
  ],
};

function fmtPct(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
}

function pctClass(v: number | null | undefined): string {
  if (v === null || v === undefined) return "text-slate-500";
  return v >= 0 ? "text-emerald-600" : "text-red-600";
}

function daysSince(isoDate: string): number {
  return Math.max(0, Math.round((Date.now() - new Date(isoDate).getTime()) / (1000 * 60 * 60 * 24)));
}

function fmtDuration(days: number): string {
  if (days < 30) return `${days} day${days === 1 ? "" : "s"}`;
  if (days < 365) return `${Math.round(days / 30)} month${Math.round(days / 30) === 1 ? "" : "s"}`;
  const years = Math.floor(days / 365);
  const months = Math.round((days % 365) / 30);
  return months > 0 ? `${years}y ${months}mo` : `${years} year${years === 1 ? "" : "s"}`;
}

const SIGNAL_BADGE_CLASS: Record<string, string> = {
  BUY: "bg-emerald-50 text-emerald-700",
  SELL: "bg-red-50 text-red-700",
  HOLD: "bg-slate-100 text-slate-600",
};

const MATCHED_WINDOWS = [
  { label: "30 Days", days: 30 },
  { label: "6 Months", days: 182 },
  { label: "1 Year", days: 365 },
];

interface MatchedWindow {
  label: string;
  days: number;
  portfolioPct: number | null;
  fundPct: number | null;
}

interface PositionReturns {
  d30: number | null;
  m6: number | null;
  y1: number | null;
}

function windowKey(days: number): keyof PositionReturns {
  if (days <= 30) return "d30";
  if (days <= 182) return "m6";
  return "y1";
}

export default function ComparePage() {
  const [selectedPortfolioId, setSelectedPortfolioId] = useState<number | null>(null);
  const [allPortfolios, setAllPortfolios] = useState<Portfolio[]>([]);
  const [goal, setGoal] = useState("Balanced Core");
  const [goals, setGoals] = useState<string[]>(["Balanced Core"]);

  const [summary, setSummary] = useState<PortfolioSummary | null>(null);
  const [performance, setPerformance] = useState<PortfolioPerformance | null>(null);
  const [insights, setInsights] = useState<PortfolioInsight[]>([]);
  const [topFunds, setTopFunds] = useState<FundRankRow[]>([]);

  const [fundSince, setFundSince] = useState<FundReturnSince | null>(null);
  const [fundSinceError, setFundSinceError] = useState<string | null>(null);

  const [inceptionReturn, setInceptionReturn] = useState<FundReturnSince | null>(null);
  const [matchedWindows, setMatchedWindows] = useState<MatchedWindow[]>([]);
  const [matchedWindowsLoading, setMatchedWindowsLoading] = useState(false);
  const [positionReturns, setPositionReturns] = useState<Record<string, PositionReturns>>({});
  const [forecasts1y, setForecasts1y] = useState<
    Record<string, { status: "loading" } | { status: "error" } | { status: "ok"; pct: number | null }>
  >({});

  async function loadForecast1y(ticker: string) {
    setForecasts1y((prev) => ({ ...prev, [ticker]: { status: "loading" } }));
    try {
      const res = await getPortfolio1yForecast(ticker);
      setForecasts1y((prev) => ({ ...prev, [ticker]: { status: "ok", pct: res.expected_return_pct } }));
    } catch {
      setForecasts1y((prev) => ({ ...prev, [ticker]: { status: "error" } }));
    }
  }

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showGoalInfo, setShowGoalInfo] = useState(false);

  useEffect(() => {
    getFundGoals()
      .then((res) => setGoals(res.goals))
      .catch(() => {
        // Non-fatal: fall back to "Balanced Core" already in state.
      });
  }, []);

  useEffect(() => {
    if (selectedPortfolioId === null) return;
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedPortfolioId, goal]);

  async function load() {
    setLoading(true);
    setError(null);
    setFundSince(null);
    setFundSinceError(null);
    setForecasts1y({});
    try {
      const [summaryRes, performanceRes, insightsRes, fundRes] = await Promise.all([
        getPortfolioSummary(selectedPortfolioId ?? undefined),
        getPortfolioPerformance(30, selectedPortfolioId ?? undefined),
        getPortfolioInsights(selectedPortfolioId ?? undefined),
        getFundRanking(goal, "All"),
      ]);
      setSummary(summaryRes.summary);
      setPerformance(performanceRes);
      setInsights(insightsRes.positions);
      setTopFunds(fundRes.results.slice(0, 5));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load this comparison.");
    } finally {
      setLoading(false);
    }
  }

  const selectedPortfolio = useMemo(
    () => allPortfolios.find((p) => p.id === selectedPortfolioId) ?? null,
    [allPortfolios, selectedPortfolioId],
  );
  const daysHeld = useMemo(() => {
    if (!selectedPortfolio) return null;
    const created = new Date(selectedPortfolio.created_at);
    return Math.max(1, Math.round((Date.now() - created.getTime()) / (1000 * 60 * 60 * 24)));
  }, [selectedPortfolio]);

  const topFund = topFunds[0] ?? null;

  // Once we know how long this portfolio has actually existed, fetch the
  // top fund's real point-in-time return over that identical window —
  // "what if you'd put this money there instead, starting the same day" —
  // rather than the fixed 30d/1Y/3Y windows the ranking table shows.
  useEffect(() => {
    if (!topFund || !selectedPortfolio) return;
    const sinceDate = selectedPortfolio.created_at.slice(0, 10);
    let cancelled = false;
    setFundSinceError(null);
    getFundReturnSince(topFund.Ticker, sinceDate)
      .then((res) => {
        if (!cancelled) setFundSince(res);
      })
      .catch((err) => {
        if (!cancelled) setFundSinceError(err instanceof ApiError ? err.message : "Could not load the fund's return over this period.");
      });
    return () => {
      cancelled = true;
    };
  }, [topFund, selectedPortfolio]);

  // The fund's total return from its actual inception date to now — not
  // duration-matched to anything, just "how has this fund done over its
  // whole real life."
  useEffect(() => {
    const inceptionDate = topFund?.["Inception Date"] as string | null | undefined;
    if (!topFund || !inceptionDate) {
      setInceptionReturn(null);
      return;
    }
    let cancelled = false;
    getFundReturnSince(topFund.Ticker, inceptionDate)
      .then((res) => {
        if (!cancelled) setInceptionReturn(res);
      })
      .catch(() => {
        if (!cancelled) setInceptionReturn(null);
      });
    return () => {
      cancelled = true;
    };
  }, [topFund]);

  // Your portfolio's real trailing performance next to the top fund's real
  // point-in-time return, over identical 30-day/6-month/1-year windows —
  // fixed, matched horizons, unlike "Since You Started" above which tracks
  // however long this specific portfolio has actually existed.
  useEffect(() => {
    if (!topFund) {
      setMatchedWindows([]);
      setPositionReturns({});
      return;
    }
    let cancelled = false;
    setMatchedWindowsLoading(true);
    setMatchedWindows(MATCHED_WINDOWS.map((w) => ({ ...w, portfolioPct: null, fundPct: null })));

    Promise.all(
      MATCHED_WINDOWS.map(async (w) => {
        const sinceDate = new Date(Date.now() - w.days * 24 * 60 * 60 * 1000).toISOString().slice(0, 10);
        const [perf, fundRet] = await Promise.all([
          getPortfolioPerformance(w.days, selectedPortfolioId ?? undefined).catch(() => null),
          getFundReturnSince(topFund.Ticker, sinceDate).catch(() => null),
        ]);
        return {
          ...w,
          portfolioPct: perf?.value_diff_pct ?? null,
          fundPct: fundRet?.return_pct ?? null,
          rows: perf?.rows ?? [],
        };
      }),
    ).then((results) => {
      if (cancelled) return;
      setMatchedWindows(results.map(({ rows, ...w }) => w));

      // Same per-position rows the portfolio-level 30d/6mo/1yr figures above
      // are built from — each position's own real trailing return at each
      // window, not a single "expected return" number from the prediction
      // model (that stays in the Signal column, which already has its own
      // ~10-day horizon).
      const byTicker: Record<string, PositionReturns> = {};
      for (const w of results) {
        const key = windowKey(w.days);
        for (const row of w.rows) {
          byTicker[row.ticker] = byTicker[row.ticker] ?? { d30: null, m6: null, y1: null };
          byTicker[row.ticker][key] = row.diff_pct;
        }
      }
      setPositionReturns(byTicker);
    }).finally(() => {
      if (!cancelled) setMatchedWindowsLoading(false);
    });

    return () => {
      cancelled = true;
    };
  }, [topFund, selectedPortfolioId]);

  const signalCounts = useMemo(() => {
    const c = { BUY: 0, SELL: 0, HOLD: 0 };
    for (const p of insights) {
      if (p.signal === "BUY" || p.signal === "SELL" || p.signal === "HOLD") c[p.signal]++;
    }
    return c;
  }, [insights]);

  const concentrated = useMemo(() => insights.filter((p) => p.concentrated), [insights]);

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">Portfolio vs. Best Fund</h1>
          <p className="mt-1 text-sm text-slate-500">
            Your portfolio&apos;s real performance and current signals, next to the Fund Screener&apos;s
            top-ranked picks for the goal you choose — two different things, not a claim that one should
            replace the other.
          </p>
        </div>
        <Link href="/portfolio" className="text-sm font-medium text-slate-600 hover:underline">
          ← Back to Portfolio
        </Link>
      </div>

      <div className="mt-6 flex flex-wrap items-end gap-3">
        <PortfolioSwitcher
          selectedPortfolioId={selectedPortfolioId}
          onChange={setSelectedPortfolioId}
          onPortfoliosChange={setAllPortfolios}
        />
        <Field label="Compare against goal">
          <div className="flex items-center gap-1.5">
            <select value={goal} onChange={(e) => setGoal(e.target.value)} className="input">
              {goals.map((g) => (
                <option key={g} value={g}>{g}</option>
              ))}
            </select>
            <button
              type="button"
              onClick={() => setShowGoalInfo(true)}
              title="What do these goals mean?"
              className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-slate-300 text-xs font-normal text-slate-400 hover:border-slate-500 hover:text-slate-700"
            >
              i
            </button>
          </div>
        </Field>
      </div>

      {showGoalInfo && <InfoModal info={GOAL_INFO} onClose={() => setShowGoalInfo(false)} />}

      {loading && <p className="mt-6 text-sm text-slate-500">Loading…</p>}
      {error && <p className="mt-6 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {!loading && summary && performance && (
        <>
          <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="rounded-lg border border-slate-200 bg-white p-5">
              <h2 className="text-sm font-semibold text-slate-900">Your Portfolio</h2>
              <p className="mt-1 text-xs text-slate-500">
                {summary.total_positions} position{summary.total_positions === 1 ? "" : "s"} · $
                {summary.total_value.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                {daysHeld !== null && ` · held ${fmtDuration(daysHeld)}`}
              </p>
              <div className="mt-3 grid grid-cols-2 gap-3">
                <Metric label="30-Day Change" value={fmtPct(performance.value_diff_pct)} valueClass={pctClass(performance.value_diff_pct)} />
                <Metric label="Gain vs. Cost" value={fmtPct(performance.total_gain_vs_cost_pct)} valueClass={pctClass(performance.total_gain_vs_cost_pct)} />
              </div>
            </div>

            <div className="rounded-lg border border-slate-200 bg-white p-5">
              {topFund ? (
                <>
                  <h2 className="text-sm font-semibold text-slate-900">
                    Top Fund for &quot;{goal}&quot;: {topFund.Ticker}
                  </h2>
                  <p className="mt-1 text-xs text-slate-500">{topFund.Fund} · Score {topFund.Score}/100</p>
                  <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3">
                    <Metric label="1Y Return" value={fmtPct(topFund["1Y Return %"] as number)} valueClass={pctClass(topFund["1Y Return %"] as number)} />
                    <Metric label="3Y Annualized" value={fmtPct(topFund["3Y Annualized %"] as number)} valueClass={pctClass(topFund["3Y Annualized %"] as number)} />
                    <Metric label="Expense Ratio" value={topFund["Expense Ratio %"] != null ? `${(topFund["Expense Ratio %"] as number).toFixed(2)}%` : "—"} />
                    <Metric
                      label="Time Since Inception"
                      value={topFund["Inception Date"] ? fmtDuration(daysSince(topFund["Inception Date"] as string)) : "unknown"}
                    />
                    <Metric
                      label="% Since Inception"
                      value={topFund["Inception Date"] ? (inceptionReturn ? fmtPct(inceptionReturn.return_pct) : "…") : "unknown"}
                      valueClass={inceptionReturn ? pctClass(inceptionReturn.return_pct) : undefined}
                    />
                  </div>
                </>
              ) : (
                <p className="text-sm text-slate-500">No fund data for this goal yet.</p>
              )}
            </div>
          </div>

          {topFund && daysHeld !== null && selectedPortfolio && (
            <div className="mt-4 rounded-lg border border-slate-200 bg-white p-5">
              <h2 className="text-sm font-semibold text-slate-900">
                Since You Started ({fmtDuration(daysHeld)})
              </h2>
              <p className="mt-1 text-xs text-slate-500">
                Your portfolio&apos;s gain vs. cost isn&apos;t tied to one exact start date — different
                positions were added at different times. This is the real point-in-time comparison instead:
                what {topFund.Ticker} actually returned over the exact same window your portfolio has existed.
              </p>
              {fundSinceError && <p className="mt-2 text-xs text-red-600">{fundSinceError}</p>}
              <div className="mt-3 grid grid-cols-2 gap-3">
                <Metric
                  label="Your Gain vs. Cost"
                  value={fmtPct(performance.total_gain_vs_cost_pct)}
                  valueClass={pctClass(performance.total_gain_vs_cost_pct)}
                />
                <Metric
                  label={`${topFund.Ticker} Since ${new Date(selectedPortfolio.created_at).toLocaleDateString()}`}
                  value={fundSince ? fmtPct(fundSince.return_pct) : "…"}
                  valueClass={fundSince ? pctClass(fundSince.return_pct) : undefined}
                />
              </div>
            </div>
          )}

          {topFund && matchedWindows.length > 0 && (
            <div className="mt-4 rounded-lg border border-slate-200 bg-white p-5">
              <h2 className="text-sm font-semibold text-slate-900">30 Days / 6 Months / 1 Year</h2>
              <p className="mt-1 text-xs text-slate-500">
                Your portfolio&apos;s real trailing performance next to what {topFund.Ticker} actually returned
                over the same fixed windows — regardless of how long you&apos;ve personally held this portfolio.
              </p>
              <div className="mt-3 overflow-x-auto rounded-md border border-slate-200">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                      <th className="px-3 py-2">Window</th>
                      <th className="px-3 py-2 text-right">Your Portfolio</th>
                      <th className="px-3 py-2 text-right">{topFund.Ticker}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {matchedWindows.map((w) => (
                      <tr key={w.label} className="border-b border-slate-100 last:border-0">
                        <td className="px-3 py-2 font-medium text-slate-800">{w.label}</td>
                        <td className={`px-3 py-2 text-right font-medium ${pctClass(w.portfolioPct)}`}>
                          {matchedWindowsLoading && w.portfolioPct === null ? "…" : fmtPct(w.portfolioPct)}
                        </td>
                        <td className={`px-3 py-2 text-right font-medium ${pctClass(w.fundPct)}`}>
                          {matchedWindowsLoading && w.fundPct === null ? "…" : fmtPct(w.fundPct)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {topFunds.length > 0 && (
            <div className="mt-4 rounded-lg border border-slate-200 bg-white p-5">
              <h2 className="text-sm font-semibold text-slate-900">Top 5 Funds for &quot;{goal}&quot;</h2>
              <div className="mt-3 overflow-x-auto rounded-md border border-slate-200">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                      <th className="px-3 py-2">Ticker</th>
                      <th className="px-3 py-2">Fund</th>
                      <th className="px-3 py-2 text-right">Score</th>
                      <th className="px-3 py-2 text-right">1Y Return</th>
                      <th className="px-3 py-2 text-right">3Y Annualized</th>
                      <th className="px-3 py-2 text-right">Expense Ratio</th>
                      <th className="px-3 py-2 text-right">Since Inception</th>
                    </tr>
                  </thead>
                  <tbody>
                    {topFunds.map((f, i) => (
                      <tr key={f.Ticker} className="border-b border-slate-100 last:border-0">
                        <td className="px-3 py-2 font-medium text-slate-800">
                          {i === 0 && <span className="mr-1.5 text-amber-500">#1</span>}
                          {f.Ticker}
                        </td>
                        <td className="px-3 py-2 text-slate-600">{f.Fund}</td>
                        <td className="px-3 py-2 text-right text-slate-600">{f.Score}</td>
                        <td className={`px-3 py-2 text-right font-medium ${pctClass(f["1Y Return %"] as number)}`}>
                          {fmtPct(f["1Y Return %"] as number)}
                        </td>
                        <td className={`px-3 py-2 text-right font-medium ${pctClass(f["3Y Annualized %"] as number)}`}>
                          {fmtPct(f["3Y Annualized %"] as number)}
                        </td>
                        <td className="px-3 py-2 text-right text-slate-600">
                          {f["Expense Ratio %"] != null ? `${(f["Expense Ratio %"] as number).toFixed(2)}%` : "—"}
                        </td>
                        <td className="px-3 py-2 text-right text-slate-600">
                          {f["Inception Date"] ? fmtDuration(daysSince(f["Inception Date"] as string)) : "unknown"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="mt-6 rounded-lg border border-slate-200 bg-white p-5">
            <h2 className="text-sm font-semibold text-slate-900">Current Signals Across Your Positions</h2>
            <div className="mt-3 flex flex-wrap gap-2">
              <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${SIGNAL_BADGE_CLASS.BUY}`}>{signalCounts.BUY} BUY</span>
              <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${SIGNAL_BADGE_CLASS.HOLD}`}>{signalCounts.HOLD} HOLD</span>
              <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${SIGNAL_BADGE_CLASS.SELL}`}>{signalCounts.SELL} SELL</span>
            </div>
            <p className="mt-2 text-xs text-slate-500">
              10D/30D Forecast are the prediction model's expected return at those horizons — both within the
              5-to-60-day range the model is actually backtested for. 30D/6M/1Y Trailing are each position's
              real historical return over that window instead — not predicted.
            </p>
            <p className="mt-1 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-700">
              1Y Forecast is the same model pushed to a 252-day recursive forecast — far beyond the ~5-60 day
              range it&apos;s actually backtested for, and compounding forecast error at every step. Shown
              because it was asked for, not because it&apos;s reliable — treat 1Y Trailing as the trustworthy
              number for that horizon.
            </p>

            {concentrated.length > 0 && (
              <p className="mt-3 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-700">
                {concentrated.map((p) => `${p.ticker} (${p.weight_pct?.toFixed(0)}%)`).join(", ")}{" "}
                {concentrated.length === 1 ? "makes up" : "make up"} a large enough share of this portfolio to
                drive most of its swings — worth a deliberate decision, not an accident.
              </p>
            )}

            <div className="mt-4 max-h-[24rem] overflow-auto rounded-md border border-slate-200">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="sticky top-0 border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                    <th className="px-3 py-2">Ticker</th>
                    <th className="px-3 py-2 text-right">Weight</th>
                    <th className="px-3 py-2">Signal</th>
                    <th className="px-3 py-2 text-right">10D Forecast</th>
                    <th className="px-3 py-2 text-right">30D Forecast</th>
                    <th className="px-3 py-2 text-right" title="Unvalidated — far beyond the model's backtested range">
                      1Y Forecast <span className="text-amber-500">*</span>
                    </th>
                    <th className="px-3 py-2 text-right">30D Trailing</th>
                    <th className="px-3 py-2 text-right">6M Trailing</th>
                    <th className="px-3 py-2 text-right">1Y Trailing</th>
                  </tr>
                </thead>
                <tbody>
                  {[...insights]
                    .sort((a, b) => (b.weight_pct ?? 0) - (a.weight_pct ?? 0))
                    .map((p) => {
                      const ret = positionReturns[p.ticker];
                      return (
                      <tr key={p.ticker} className="border-b border-slate-100 last:border-0">
                        <td className="px-3 py-2 font-medium text-slate-800">
                          {p.ticker}
                          {p.concentrated && (
                            <span
                              title="A single position this large drives most of your portfolio's swings."
                              className="ml-1.5 rounded-full bg-amber-50 px-1.5 py-0.5 text-[10px] font-semibold text-amber-700"
                            >
                              concentrated
                            </span>
                          )}
                        </td>
                        <td className="px-3 py-2 text-right text-slate-600">
                          {p.weight_pct !== null ? `${p.weight_pct.toFixed(1)}%` : "—"}
                        </td>
                        <td className="px-3 py-2">
                          {p.signal ? (
                            <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${SIGNAL_BADGE_CLASS[p.signal]}`}>
                              {p.signal}
                            </span>
                          ) : (
                            <span className="text-slate-400">—</span>
                          )}
                        </td>
                        <td className={`px-3 py-2 text-right font-medium ${pctClass(p.expected_return_pct)}`}>
                          {fmtPct(p.expected_return_pct)}
                        </td>
                        <td className={`px-3 py-2 text-right font-medium ${pctClass(p.expected_return_pct_30d)}`}>
                          {fmtPct(p.expected_return_pct_30d)}
                        </td>
                        <td className="px-3 py-2 text-right">
                          {(() => {
                            const f1y = forecasts1y[p.ticker];
                            if (!f1y) {
                              return (
                                <button
                                  type="button"
                                  onClick={() => loadForecast1y(p.ticker)}
                                  className="rounded-md border border-slate-300 px-2 py-0.5 text-xs font-medium text-slate-600 hover:bg-slate-50"
                                >
                                  Load
                                </button>
                              );
                            }
                            if (f1y.status === "loading") return <span className="text-xs text-slate-400">…</span>;
                            if (f1y.status === "error") {
                              return (
                                <button
                                  type="button"
                                  onClick={() => loadForecast1y(p.ticker)}
                                  className="text-xs font-medium text-red-600 hover:underline"
                                >
                                  Failed — retry
                                </button>
                              );
                            }
                            return <span className={`font-medium ${pctClass(f1y.pct)}`}>{fmtPct(f1y.pct)}</span>;
                          })()}
                        </td>
                        <td className={`px-3 py-2 text-right font-medium ${pctClass(ret?.d30)}`}>
                          {matchedWindowsLoading && !ret ? "…" : fmtPct(ret?.d30)}
                        </td>
                        <td className={`px-3 py-2 text-right font-medium ${pctClass(ret?.m6)}`}>
                          {matchedWindowsLoading && !ret ? "…" : fmtPct(ret?.m6)}
                        </td>
                        <td className={`px-3 py-2 text-right font-medium ${pctClass(ret?.y1)}`}>
                          {matchedWindowsLoading && !ret ? "…" : fmtPct(ret?.y1)}
                        </td>
                      </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
          </div>
        </>
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

function Metric({ label, value, valueClass }: { label: string; value: string; valueClass?: string }) {
  return (
    <div>
      <p className="text-xs text-slate-500">{label}</p>
      <p className={`mt-0.5 text-lg font-semibold ${valueClass ?? "text-slate-900"}`}>{value}</p>
    </div>
  );
}
