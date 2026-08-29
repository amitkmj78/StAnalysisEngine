"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";

import {
  ApiError,
  getFundGoals,
  getFundRanking,
  getFundReturnSince,
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

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
          <select value={goal} onChange={(e) => setGoal(e.target.value)} className="input">
            {goals.map((g) => (
              <option key={g} value={g}>{g}</option>
            ))}
          </select>
        </Field>
      </div>

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
                  <div className="mt-3 grid grid-cols-2 gap-3">
                    <Metric label="1Y Return" value={fmtPct(topFund["1Y Return %"] as number)} valueClass={pctClass(topFund["1Y Return %"] as number)} />
                    <Metric label="3Y Annualized" value={fmtPct(topFund["3Y Annualized %"] as number)} valueClass={pctClass(topFund["3Y Annualized %"] as number)} />
                    <Metric label="Expense Ratio" value={topFund["Expense Ratio %"] != null ? `${(topFund["Expense Ratio %"] as number).toFixed(2)}%` : "—"} />
                    <Metric
                      label="Time Since Inception"
                      value={topFund["Inception Date"] ? fmtDuration(daysSince(topFund["Inception Date"] as string)) : "unknown"}
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
                    <th className="px-3 py-2 text-right">Expected Return</th>
                  </tr>
                </thead>
                <tbody>
                  {[...insights]
                    .sort((a, b) => (b.weight_pct ?? 0) - (a.weight_pct ?? 0))
                    .map((p) => (
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
                      </tr>
                    ))}
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
