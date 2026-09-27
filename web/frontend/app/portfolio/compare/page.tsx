"use client";

import { useEffect, useState } from "react";

import { ApiError, getFundGoals, getPortfolioCompare, getPredictionSummary } from "@/lib/api";
import type { CompareHolding, CompareSignal, CompareTopStock, CompareWindowCode, ForecastOut, FundGoal, PortfolioCompareResponse } from "@/lib/types";
import PortfolioSwitcher from "@/components/PortfolioSwitcher";
import CompareGrowthChart, { COMPARE_COLORS } from "@/components/portfolio/CompareGrowthChart";
import Sparkline from "@/components/portfolio/Sparkline";
import StockForecastPanel from "@/components/portfolio/StockForecastPanel";
import { useUrlState } from "@/lib/useUrlState";

const WINDOWS: { code: CompareWindowCode; label: string }[] = [
  { code: "10D", label: "10D" },
  { code: "30D", label: "30D" },
  { code: "60D", label: "60D" },
  { code: "90D", label: "90D" },
  { code: "1Y", label: "1Y" },
];

const DEFAULT_GOAL = "Balanced Core";
const DEFAULT_WINDOW = "90D";

function fmtPct(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  return `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`;
}

function pctColor(v: number | null | undefined): string {
  if (v === null || v === undefined) return "#6b7280";
  return v >= 0 ? COMPARE_COLORS.gain : COMPARE_COLORS.loss;
}

function fmtDate(iso: string | null): string {
  if (!iso) return "";
  return new Date(iso + "T00:00:00").toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function isStale(asOf: string | null): boolean {
  if (!asOf) return false;
  const days = (Date.now() - new Date(asOf + "T00:00:00").getTime()) / 86400000;
  return days > 7; // ~5 trading days
}

function SignalPill({ signal }: { signal: CompareSignal | null | undefined }) {
  if (!signal || !signal.action) {
    return <span className="text-xs text-slate-400">No current signal</span>;
  }
  const color =
    signal.action === "buy" ? COMPARE_COLORS.buy : signal.action === "trim" ? COMPARE_COLORS.trim : COMPARE_COLORS.hold;
  const label = signal.action === "buy" ? "Buy" : signal.action === "trim" ? "Trim" : "Hold";
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-semibold text-white"
      style={{ backgroundColor: color }}
      title={`Confidence: ${signal.label}${isStale(signal.as_of) ? " (stale)" : ""}`}
    >
      {label}
      {isStale(signal.as_of) && <span className="opacity-80">·stale</span>}
    </span>
  );
}

function SkeletonBlock({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded-md bg-slate-200 ${className}`} />;
}

export default function PortfolioComparePage() {
  const [{ p: portfolioParam, goal, window: windowCode }, setUrlState] = useUrlState({
    p: "",
    goal: DEFAULT_GOAL,
    window: DEFAULT_WINDOW,
  });

  const [selectedPortfolioId, setSelectedPortfolioId] = useState<number | null>(
    portfolioParam ? Number(portfolioParam) : null,
  );
  const [goals, setGoals] = useState<FundGoal[]>([]);

  const [data, setData] = useState<PortfolioCompareResponse | null>(null);
  const [loading, setLoading] = useState(true); // first load only
  const [refetching, setRefetching] = useState(false); // goal/window switch
  const [error, setError] = useState<string | null>(null);

  // On-demand only, never fetched for all 10 stocks automatically -- a
  // real multi-day forecast trains a model per ticker per horizon (the
  // same expensive path the Predict page uses), so this only runs for
  // whichever one stock the user actually expands.
  const [expandedForecastTicker, setExpandedForecastTicker] = useState<string | null>(null);
  const [forecastByTicker, setForecastByTicker] = useState<Record<string, ForecastOut | null>>({});
  const [forecastLoading, setForecastLoading] = useState<string | null>(null);
  const [forecastError, setForecastError] = useState<string | null>(null);

  useEffect(() => {
    getFundGoals()
      .then((res) => setGoals(res.goals.filter((g) => g.name !== "Custom")))
      .catch(() => {});
  }, []);

  async function toggleForecast(ticker: string) {
    if (expandedForecastTicker === ticker) {
      setExpandedForecastTicker(null);
      return;
    }
    setExpandedForecastTicker(ticker);
    setForecastError(null);
    if (forecastByTicker[ticker] !== undefined) return; // already fetched
    setForecastLoading(ticker);
    try {
      const res = await getPredictionSummary(ticker, "1y", 10);
      setForecastByTicker((prev) => ({ ...prev, [ticker]: res.forecast }));
    } catch (err) {
      setForecastError(err instanceof ApiError ? err.message : "Could not load a forecast for this stock.");
    } finally {
      setForecastLoading((cur) => (cur === ticker ? null : cur));
    }
  }

  async function load(isSwitch: boolean) {
    if (selectedPortfolioId === null) return;
    if (isSwitch) setRefetching(true);
    else setLoading(true);
    setError(null);
    try {
      const res = await getPortfolioCompare(goal, windowCode, selectedPortfolioId);
      setData(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't load the comparison.");
    } finally {
      setLoading(false);
      setRefetching(false);
    }
  }

  useEffect(() => {
    load(data !== null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedPortfolioId, goal, windowCode]);

  function handlePortfolioChange(id: number) {
    setSelectedPortfolioId(id);
    setUrlState({ p: String(id) });
  }

  const dimmed = refetching ? "opacity-50 transition-opacity" : "transition-opacity";

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">Portfolio vs. Top Picks</h1>
          <p className="mt-1 text-sm text-slate-500">
            How your portfolio has actually done against the S&amp;P 500 and the top-ranked fund for a goal,
            over a window you pick.
          </p>
        </div>
      </div>

      <div className="mt-6">
        <PortfolioSwitcher
          selectedPortfolioId={selectedPortfolioId}
          onChange={handlePortfolioChange}
          initialPreferId={portfolioParam ? Number(portfolioParam) : undefined}
        />
      </div>

      {/* Goal picker */}
      <div className="mt-6">
        <p className="text-xs font-medium text-slate-500">Goal</p>
        <div className="mt-2 grid grid-cols-2 gap-2 sm:flex sm:flex-wrap">
          {goals.map((g) => (
            <button
              key={g.name}
              type="button"
              onClick={() => setUrlState({ goal: g.name })}
              className={`min-h-[44px] rounded-lg border px-3 py-2 text-left text-sm ${
                goal === g.name ? "border-slate-900 bg-slate-900 text-white" : "border-slate-200 bg-white text-slate-700 hover:border-slate-400"
              }`}
            >
              <div className="font-medium">{g.name}</div>
              {g.description && (
                <div className={`mt-0.5 text-xs ${goal === g.name ? "text-slate-200" : "text-slate-500"}`}>
                  {g.description}
                </div>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Window picker */}
      <div className="mt-4">
        <p className="text-xs font-medium text-slate-500">Window</p>
        <div className="mt-2 grid grid-cols-5 gap-2 sm:flex">
          {WINDOWS.map((w) => (
            <button
              key={w.code}
              type="button"
              onClick={() => setUrlState({ window: w.code })}
              className={`min-h-[44px] rounded-lg border px-4 py-2 text-sm font-medium ${
                windowCode === w.code ? "border-slate-900 bg-slate-900 text-white" : "border-slate-200 bg-white text-slate-700 hover:border-slate-400"
              }`}
            >
              {w.label}
            </button>
          ))}
          {refetching && <span className="ml-2 self-center text-xs text-slate-400">Updating…</span>}
        </div>
      </div>

      {/* Loading skeleton (first load only) */}
      {loading && !data && (
        <div className="mt-8 flex flex-col gap-4">
          <SkeletonBlock className="h-6 w-2/3" />
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <SkeletonBlock className="h-24" />
            <SkeletonBlock className="h-24" />
            <SkeletonBlock className="h-24" />
          </div>
          <SkeletonBlock className="h-72" />
          <SkeletonBlock className="h-64" />
        </div>
      )}

      {error && (
        <div className="mt-8 rounded-lg border border-red-200 bg-red-50 p-5 text-center">
          <p className="text-sm text-red-700">Couldn&apos;t load the comparison. Try again.</p>
          <button
            type="button"
            onClick={() => load(false)}
            className="mt-3 rounded-md bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700"
          >
            Retry
          </button>
        </div>
      )}

      {!loading && !error && data && (
        <div className={`mt-8 flex flex-col gap-6 ${dimmed}`}>
          <p className="text-xs text-slate-500">Prices as of {fmtDate(data.as_of)} close</p>

          <p className="text-lg text-slate-800">{data.headline}</p>

          {data.portfolio.holdings_count === 0 ? (
            <div className="rounded-lg border border-slate-200 bg-white p-5 text-sm text-slate-600">
              Add holdings to compare.{" "}
              <a href="/portfolio" className="underline">
                Go to Portfolio
              </a>
            </div>
          ) : (
            <>
              {/* Summary cards */}
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                <SummaryCard
                  title="Your portfolio"
                  color={COMPARE_COLORS.portfolio}
                  returnPct={data.portfolio.return_pct}
                  volatilityPct={data.portfolio.volatility_pct}
                  extraLabel="Max drawdown"
                  extraValue={fmtPct(data.portfolio.max_drawdown_pct)}
                />
                <SummaryCard
                  title={`S&P 500 (${data.benchmark.ticker})`}
                  color={COMPARE_COLORS.benchmark}
                  returnPct={data.benchmark.return_pct}
                  volatilityPct={data.benchmark.volatility_pct}
                  extraLabel="Expense ratio"
                  extraValue={data.benchmark.expense_ratio_pct !== null ? `${data.benchmark.expense_ratio_pct.toFixed(2)}%` : "—"}
                />
                {data.top_funds[0] ? (
                  <SummaryCard
                    title={`Top ${data.goal.label} pick: ${data.top_funds[0].ticker}`}
                    color={COMPARE_COLORS.topPick}
                    returnPct={data.top_funds[0].return_pct}
                    volatilityPct={data.top_funds[0].volatility_pct}
                    extraLabel="Expense ratio"
                    extraValue={data.top_funds[0].expense_ratio_pct !== null ? `${data.top_funds[0].expense_ratio_pct.toFixed(2)}%` : "—"}
                  />
                ) : (
                  <div className="rounded-lg border border-slate-200 bg-white p-4 text-sm text-slate-500">
                    No fund data for this goal yet.
                  </div>
                )}
              </div>

              {/* Growth chart */}
              <div className="rounded-lg border border-slate-200 bg-white p-4">
                <CompareGrowthChart
                  portfolioSeries={data.portfolio.series}
                  benchmarkSeries={data.benchmark.series}
                  topFundSeries={data.top_funds[0]?.series ?? null}
                  topFundTicker={data.top_funds[0]?.ticker ?? null}
                />
              </div>

              {/* Gap drivers */}
              {data.gap_drivers.length > 0 && (
                <div className="rounded-lg border border-slate-200 bg-white p-4">
                  <h3 className="font-semibold text-slate-900">Where the gap comes from</h3>
                  <div className="mt-3 flex flex-col gap-2">
                    {data.gap_drivers.map((d) => (
                      <div key={d.ticker} className="flex items-center justify-between text-sm">
                        <span className="font-medium text-slate-700">
                          {d.kind === "lead" ? "Leading: " : "Dragging: "}
                          {d.ticker}
                        </span>
                        <span style={{ color: pctColor(d.contribution_pts) }} className="font-mono font-semibold">
                          {d.contribution_pts >= 0 ? "+" : ""}
                          {d.contribution_pts.toFixed(2)} pts
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Holdings table */}
              <div>
                <h3 className="font-semibold text-slate-900">Holdings</h3>
                <div className="mt-2 overflow-x-auto rounded-lg border border-slate-200 bg-white">
                  <table className="min-w-full text-sm">
                    <thead>
                      <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                        <th className="px-3 py-2">Ticker</th>
                        <th className="px-3 py-2">Signal</th>
                        <th className="px-3 py-2 text-right">Weight</th>
                        <th className="px-3 py-2 text-right">Return</th>
                        <th className="px-3 py-2 text-right">Contribution</th>
                        <th className="px-3 py-2">Trend</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...data.holdings]
                        .sort((a, b) => b.contribution_pts - a.contribution_pts)
                        .map((h: CompareHolding) => (
                          <tr key={h.ticker} className="border-b border-slate-100 last:border-0">
                            <td className="px-3 py-2 font-medium text-slate-800">
                              {h.ticker}
                              {h.since && <div className="text-[11px] font-normal text-slate-400">Since {fmtDate(h.since)}</div>}
                            </td>
                            <td className="px-3 py-2">
                              <SignalPill signal={h.signal} />
                            </td>
                            <td className="px-3 py-2 text-right text-slate-600">{h.weight_pct.toFixed(1)}%</td>
                            <td className="px-3 py-2 text-right font-medium" style={{ color: pctColor(h.return_pct) }}>
                              {fmtPct(h.return_pct)}
                            </td>
                            <td className="px-3 py-2 text-right font-medium" style={{ color: pctColor(h.contribution_pts) }}>
                              {h.contribution_pts >= 0 ? "+" : ""}
                              {h.contribution_pts.toFixed(2)} pts
                            </td>
                            <td className="px-3 py-2">
                              <Sparkline values={h.spark} color={pctColor(h.return_pct)} />
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}

          {/* Top funds */}
          <div>
            <h3 className="font-semibold text-slate-900">Top-Ranked Funds — {data.goal.label}</h3>
            <div className="mt-2 overflow-x-auto rounded-lg border border-slate-200 bg-white">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                    <th className="px-3 py-2">#</th>
                    <th className="px-3 py-2">Fund</th>
                    <th className="px-3 py-2">Why it ranks here</th>
                    <th className="px-3 py-2 text-right">Return</th>
                    <th className="px-3 py-2 text-right">Expense Ratio</th>
                  </tr>
                </thead>
                <tbody>
                  {data.top_funds.map((f) => (
                    <tr key={f.ticker} className="border-b border-slate-100 last:border-0">
                      <td className="px-3 py-2 text-slate-500">{f.rank}</td>
                      <td className="px-3 py-2">
                        <span className="font-medium text-slate-800">{f.ticker}</span>{" "}
                        <span className="text-slate-500">{f.name}</span>
                      </td>
                      <td className="px-3 py-2 text-slate-600">{f.reason}</td>
                      <td className="px-3 py-2 text-right font-medium" style={{ color: pctColor(f.return_pct) }}>
                        {fmtPct(f.return_pct)}
                      </td>
                      <td className="px-3 py-2 text-right text-slate-600">
                        {f.expense_ratio_pct !== null ? `${f.expense_ratio_pct.toFixed(2)}%` : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Top stocks */}
          <div>
            <h3 className="font-semibold text-slate-900">Best-Performing Stocks — {windowCode}</h3>
            <div className="mt-2 grid grid-cols-2 gap-3 sm:grid-cols-5">
              {data.top_stocks.map((s: CompareTopStock) => {
                const expanded = expandedForecastTicker === s.ticker;
                return (
                  <div
                    key={s.ticker}
                    className={`rounded-lg border border-slate-200 bg-white p-3 ${expanded ? "col-span-2 sm:col-span-5" : ""}`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-slate-800">{s.ticker}</span>
                      {s.owned && (
                        <span className="rounded-full bg-slate-900 px-1.5 py-0.5 text-[10px] font-semibold text-white">
                          You own
                        </span>
                      )}
                    </div>
                    <p className="truncate text-xs text-slate-500">{s.name}</p>
                    <p className="text-xs text-slate-400">{s.sector}</p>
                    <p className="mt-1 font-medium" style={{ color: pctColor(s.return_pct) }}>
                      {fmtPct(s.return_pct)}
                    </p>
                    <div className="mt-2 border-t border-slate-100 pt-2">
                      <SignalPill signal={s.signal} />
                      {s.expected_return_pct !== null && (
                        <p className="mt-1 text-xs text-slate-500">
                          Forecast: <span style={{ color: pctColor(s.expected_return_pct) }}>{fmtPct(s.expected_return_pct)}</span>
                          {s.target_price !== null && ` (target $${s.target_price.toFixed(2)})`}
                        </p>
                      )}
                      <button
                        type="button"
                        onClick={() => toggleForecast(s.ticker)}
                        className="mt-1 text-xs font-medium text-blue-700 hover:underline"
                      >
                        {expanded ? "Hide 10-day forecast" : "Show 10-day forecast"}
                      </button>
                    </div>
                    {expanded && (
                      <div className="mt-2 border-t border-slate-100 pt-2">
                        {forecastLoading === s.ticker && <p className="text-xs text-slate-400">Loading forecast…</p>}
                        {forecastError && forecastLoading !== s.ticker && (
                          <p className="text-xs text-red-600">{forecastError}</p>
                        )}
                        {forecastLoading !== s.ticker && forecastByTicker[s.ticker] && (
                          <StockForecastPanel ticker={s.ticker} forecast={forecastByTicker[s.ticker]!} />
                        )}
                        {forecastLoading !== s.ticker && forecastByTicker[s.ticker] === null && (
                          <p className="text-xs text-slate-400">Not enough price history to forecast this stock.</p>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function SummaryCard({
  title,
  color,
  returnPct,
  volatilityPct,
  extraLabel,
  extraValue,
}: {
  title: string;
  color: string;
  returnPct: number | null;
  volatilityPct: number | null;
  extraLabel: string;
  extraValue: string;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex items-center gap-2">
        <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
        <span className="text-sm font-medium text-slate-700">{title}</span>
      </div>
      <p className="mt-2 text-2xl font-semibold" style={{ color: pctColor(returnPct) }}>
        {fmtPct(returnPct)}
      </p>
      <div className="mt-2 flex justify-between text-xs text-slate-500">
        <span>Volatility {volatilityPct !== null ? `${volatilityPct.toFixed(1)}%` : "—"}</span>
        <span>
          {extraLabel} {extraValue}
        </span>
      </div>
    </div>
  );
}
