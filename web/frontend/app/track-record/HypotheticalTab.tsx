"use client";

import { useEffect, useState } from "react";

import { ApiError, getMomentumBacktest, getPredictAlgoComparison } from "@/lib/api";
import type { MomentumBacktestResponse, PredictAlgoComparisonResponse, PublishedSignalsResponse } from "@/lib/types";
import { RecordTile } from "./page";

const COMPARE_HORIZONS = [1, 5, 10, 30];

function fmtPct(v: number | null): string {
  if (v === null) return "—";
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
}

export default function HypotheticalTab({ data }: { data: PublishedSignalsResponse }) {
  const [backtest, setBacktest] = useState<MomentumBacktestResponse | null>(null);
  const [backtestLoading, setBacktestLoading] = useState(true);
  const [backtestError, setBacktestError] = useState<string | null>(null);

  const [compareHorizon, setCompareHorizon] = useState(30);
  const [comparison, setComparison] = useState<PredictAlgoComparisonResponse | null>(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [comparisonError, setComparisonError] = useState<string | null>(null);

  useEffect(() => {
    setBacktestLoading(true);
    setBacktestError(null);
    // top_n capped at 10 by the backtest endpoint itself (GET /momentum/backtest's
    // own le=10 validation) -- narrower than the live record's top 25, called
    // out in the panel copy below rather than silently claiming exact parity.
    getMomentumBacktest("Stock", data.universe_id, data.lookback_days, 10, 3, 30)
      .then(setBacktest)
      .catch((err) => setBacktestError(err instanceof ApiError ? err.message : "Failed to load the backtest."))
      .finally(() => setBacktestLoading(false));
  }, [data.universe_id, data.lookback_days]);

  async function loadComparison(horizon: number) {
    setCompareHorizon(horizon);
    setComparisonLoading(true);
    setComparisonError(null);
    try {
      setComparison(await getPredictAlgoComparison(horizon));
    } catch (err) {
      setComparisonError(err instanceof ApiError ? err.message : "Failed to load the comparison.");
    } finally {
      setComparisonLoading(false);
    }
  }

  return (
    <>
      <div className="mt-6 rounded-md border border-indigo-200 bg-indigo-50 px-3 py-2 text-sm font-medium text-indigo-800">
        Hypothetical — everything on this tab is a simulation or a live secondary opinion, not part of the
        audited live record on the Live tab. Past and hypothetical performance do not indicate future results.
      </div>

      <div className="mt-6 rounded-lg border border-indigo-200 bg-indigo-50/40 p-5">
        <h2 className="font-semibold text-slate-900">Hypothetical Backtest</h2>
        <p className="mt-2 text-sm leading-relaxed text-slate-600">
          A walk-forward simulation of the same ranking rule shown on the Live tab, holding the top{" "}
          {backtest?.top_n ?? 10}{" "}
          picks (narrower than the Live tab&apos;s top 25 — the backtest endpoint caps at 10) over{" "}
          {backtest?.years ?? 3} years of history, rebalanced every {backtest?.horizon_days ?? 30}{" "}
          trading days, after estimated trading costs. This is retroactive — it is NOT the live, out-of-sample
          record above, and it can look better or worse than live results ever will.
        </p>

        {backtestLoading && <p className="mt-3 text-sm text-slate-500">Loading…</p>}
        {backtestError && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{backtestError}</p>}

        {backtest && !backtestLoading && (
          <>
            <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <RecordTile label="Cumulative Return" value={fmtPct(backtest.strategy_cumulative_return_pct)} />
              <RecordTile label="Benchmark Return" value={fmtPct(backtest.benchmark_cumulative_return_pct)} />
              <RecordTile label="CAGR" value={fmtPct(backtest.cagr_pct)} />
              <RecordTile label="Hit Rate" value={backtest.hit_rate_pct !== null ? `${backtest.hit_rate_pct.toFixed(1)}%` : "—"} />
              <RecordTile label="Volatility" value={fmtPct(backtest.volatility_pct)} />
              <RecordTile label="Sharpe" value={backtest.sharpe_ratio !== null ? backtest.sharpe_ratio.toFixed(2) : "—"} />
              <RecordTile label="Max Drawdown" value={fmtPct(backtest.max_drawdown_pct)} />
              <RecordTile label="Avg Turnover" value={backtest.avg_turnover_pct !== null ? `${backtest.avg_turnover_pct.toFixed(0)}%` : "—"} />
            </div>
            <p className="mt-3 text-xs text-slate-500">
              {backtest.num_periods} rebalance periods · costs assumed: {backtest.slippage_bps}bps slippage,{" "}
              {backtest.commission_bps}bps commission, {backtest.borrow_cost_bps_annual}bps/yr borrow · risk-free
              rate {(backtest.risk_free_rate_annual * 100).toFixed(1)}%/yr.
            </p>
          </>
        )}
      </div>

      {data.signals.length > 0 && (
        <div className="mt-6 rounded-lg border border-indigo-200 bg-indigo-50/40 p-5">
          <h2 className="font-semibold text-slate-900">Live Secondary Opinion: Predict-Page Algorithm</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-600">
            This app also has a separate, trained forecasting model (used on the Price Prediction page) — a
            different algorithm from the simple momentum rule above, not a validation of it, and not a
            backtest either: this is what that model currently says, live, about today&apos;s published picks.
            Only ever shown against the latest publication, since re-running today&apos;s model against an
            older publish date would unfairly give it information it couldn&apos;t have had at the time.
          </p>
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <div className="flex flex-col gap-1">
              <label className="text-xs font-medium text-slate-500">Forecast Horizon</label>
              <div className="flex gap-1 rounded-md border border-indigo-300 bg-white p-1">
                {COMPARE_HORIZONS.map((h) => (
                  <button
                    key={h}
                    type="button"
                    onClick={() => setCompareHorizon(h)}
                    className={`rounded px-3 py-1 text-sm font-medium ${
                      compareHorizon === h ? "bg-indigo-600 text-white" : "text-slate-600 hover:bg-indigo-50"
                    }`}
                  >
                    {h}d
                  </button>
                ))}
              </div>
            </div>
            <button
              type="button"
              onClick={() => loadComparison(compareHorizon)}
              disabled={comparisonLoading}
              className="self-end rounded-md border border-indigo-300 bg-white px-3 py-1.5 text-sm font-medium text-indigo-700 hover:bg-indigo-50 disabled:opacity-50"
            >
              {comparisonLoading ? "Comparing…" : comparison ? "Refresh" : "Compare"}
            </button>
          </div>

          {comparisonError && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{comparisonError}</p>}

          {comparison && !comparisonLoading && (
            <div className="mt-4 overflow-x-auto rounded-xl border border-slate-200 bg-white">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                    <th className="px-3 py-2">Ticker</th>
                    <th className="px-3 py-2 text-right">Momentum Return</th>
                    <th className="px-3 py-2">Predict-Algo Signal</th>
                    <th className="px-3 py-2 text-right">Predict-Algo Expected Return</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.comparisons.map((c) => (
                    <tr key={c.ticker} className="border-b border-slate-100 last:border-0">
                      <td className="px-3 py-2 font-medium text-slate-800">{c.ticker}</td>
                      <td className={`px-3 py-2 text-right ${c.trailing_return_pct >= 0 ? "text-emerald-600" : "text-red-600"}`}>
                        {c.trailing_return_pct >= 0 ? "+" : ""}
                        {c.trailing_return_pct.toFixed(2)}%
                      </td>
                      <td className="px-3 py-2">
                        {c.predict_signal ? (
                          <span
                            className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
                              c.predict_signal === "BUY"
                                ? "bg-emerald-50 text-emerald-700"
                                : c.predict_signal === "SELL"
                                ? "bg-red-50 text-red-700"
                                : "bg-slate-100 text-slate-600"
                            }`}
                          >
                            {c.predict_signal}
                          </span>
                        ) : (
                          <span className="text-slate-400">—</span>
                        )}
                      </td>
                      <td className="px-3 py-2 text-right text-slate-600">
                        {c.predict_expected_return_pct !== null
                          ? `${c.predict_expected_return_pct >= 0 ? "+" : ""}${c.predict_expected_return_pct.toFixed(2)}%`
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="border-t border-slate-100 px-3 py-2 text-xs text-slate-500">
                Predict-algo forecast: {comparison.predict_days_ahead} trading day
                {comparison.predict_days_ahead === 1 ? "" : "s"} ahead, using {comparison.predict_period} of
                history. The Momentum Return column stays fixed at the published {data.lookback_days}-day
                trailing window regardless of the horizon chosen here — pick a shorter horizon to ask
                &quot;does the algorithm agree over the near term?&quot; rather than over the full window the
                picks were ranked on.
              </p>
            </div>
          )}
        </div>
      )}
    </>
  );
}
