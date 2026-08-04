"use client";

import { useEffect, useState } from "react";

import { ApiError, getPredictAlgoComparison, getPublishedSignals, getSignalOutcomes } from "@/lib/api";
import type { PredictAlgoComparisonResponse, PublishedSignalsResponse, SignalOutcomesResponse } from "@/lib/types";

const COMPARE_HORIZONS = [1, 5, 10, 30];

export default function TrackRecordPage() {
  const [data, setData] = useState<PublishedSignalsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [outcomes, setOutcomes] = useState<SignalOutcomesResponse | null>(null);
  const [outcomesError, setOutcomesError] = useState<string | null>(null);

  const [compareHorizon, setCompareHorizon] = useState(30);
  const [comparison, setComparison] = useState<PredictAlgoComparisonResponse | null>(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [comparisonError, setComparisonError] = useState<string | null>(null);

  useEffect(() => {
    getPublishedSignals()
      .then(setData)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Failed to load the track record."))
      .finally(() => setLoading(false));

    getSignalOutcomes()
      .then(setOutcomes)
      .catch((err) => setOutcomesError(err instanceof ApiError ? err.message : "Failed to load evaluated outcomes."));
  }, []);

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
    <div className="mx-auto max-w-2xl px-4 py-10">
      <h1 className="text-2xl font-semibold text-slate-900">Live Track Record</h1>
      <p className="mt-2 text-sm leading-relaxed text-slate-600">
        A daily, timestamped, append-only record of one ranking rule&apos;s picks — published before each
        day&apos;s outcome is known, and never edited after the fact. This is impersonal research: the same
        content for every reader, describing what the model ranked and why, not a recommendation to buy or
        sell anything.
      </p>

      {loading && <p className="mt-6 text-sm text-slate-500">Loading…</p>}
      {error && <p className="mt-6 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {data && !loading && (
        <>
          <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-3">
            <RecordTile label="Record Started" value={data.record_start_date ?? "Not yet started"} />
            <RecordTile label="Days Published" value={String(data.days_published)} />
            <RecordTile label="Latest Publication" value={data.target_date ?? "—"} />
          </div>

          {data.signals.length === 0 ? (
            <div className="mt-6 rounded-lg border border-slate-200 bg-white p-5 text-sm text-slate-500">
              No signals have been published yet. This page will show the current picks and the full history
              as soon as publication begins — nothing is backfilled or reconstructed after the fact.
            </div>
          ) : (
            <div className="mt-6 overflow-x-auto rounded-lg border border-slate-200 bg-white">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                    <th className="px-3 py-2">Rank</th>
                    <th className="px-3 py-2">Ticker</th>
                    <th className="px-3 py-2 text-right">{data.lookback_days}-Day Trailing Return</th>
                  </tr>
                </thead>
                <tbody>
                  {data.signals.map((s) => (
                    <tr key={s.id} className="border-b border-slate-100 last:border-0">
                      <td className="px-3 py-2 text-slate-400">{s.rank}</td>
                      <td className="px-3 py-2 font-medium text-slate-800">{s.ticker}</td>
                      <td
                        className={`px-3 py-2 text-right font-medium ${
                          s.trailing_return_pct >= 0 ? "text-emerald-600" : "text-red-600"
                        }`}
                      >
                        {s.trailing_return_pct >= 0 ? "+" : ""}
                        {s.trailing_return_pct.toFixed(2)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="border-t border-slate-100 px-3 py-2 text-xs text-slate-500">
                Published {data.target_date} · universe &quot;{data.universe_id}&quot; · model version{" "}
                <code className="rounded bg-slate-100 px-1 py-0.5 font-mono text-[11px]">
                  {data.signals[0]?.model_version_hash.slice(0, 12)}
                </code>
              </p>
            </div>
          )}

          <div className="mt-8 rounded-lg border border-emerald-200 bg-emerald-50/40 p-5">
            <h2 className="font-semibold text-slate-900">Live Performance to Date</h2>
            <p className="mt-2 text-sm leading-relaxed text-slate-600">
              Real, out-of-sample results only — never a simulation, never blended with any backtest. Each
              published pick is scored once its full holding window has actually elapsed: entry priced at
              publication, exit priced {outcomes?.horizon_days ?? 30} trading days later, compared against
              equally owning the whole universe over that identical stretch.
            </p>

            {outcomesError && (
              <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{outcomesError}</p>
            )}

            {outcomes && !outcomesError && (
              <>
                {outcomes.num_evaluated_dates === 0 ? (
                  <p className="mt-3 rounded-md border border-slate-200 bg-white px-3 py-3 text-sm text-slate-500">
                    No published picks have completed their {outcomes.horizon_days}-trading-day holding window
                    yet — nothing is evaluated or estimated before it&apos;s actually knowable. Check back once
                    the earliest publication is that far out.
                  </p>
                ) : (
                  <>
                    <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-5">
                      <RecordTile label="Dates Evaluated" value={String(outcomes.num_evaluated_dates)} />
                      <RecordTile label="Picks Evaluated" value={String(outcomes.num_evaluated_picks)} />
                      <RecordTile
                        label="Hit Rate"
                        value={outcomes.hit_rate_pct !== null ? `${outcomes.hit_rate_pct.toFixed(1)}%` : "—"}
                      />
                      <RecordTile
                        label="Information Coeff."
                        value={outcomes.information_coefficient !== null ? outcomes.information_coefficient.toFixed(3) : "—"}
                      />
                      <RecordTile
                        label="Quintile Spread"
                        value={outcomes.quintile_spread_pct !== null ? `${outcomes.quintile_spread_pct >= 0 ? "+" : ""}${outcomes.quintile_spread_pct.toFixed(2)}%` : "—"}
                      />
                    </div>
                    <p className="mt-3 text-xs text-slate-500">
                      <strong>Hit Rate</strong> is the share of individual picks that beat the equal-weight
                      universe. <strong>Information Coefficient</strong> is the average correlation between a
                      pick&apos;s rank and its realized return — positive means better-ranked picks really did do
                      better. <strong>Quintile Spread</strong> is the best-ranked fifth&apos;s average return
                      minus the worst-ranked fifth&apos;s, averaged across evaluated dates.
                    </p>

                    <details className="mt-3 text-xs text-slate-600">
                      <summary className="cursor-pointer font-medium text-slate-700">
                        Show all {outcomes.outcomes.length} evaluated picks
                      </summary>
                      <div className="mt-2 max-h-80 overflow-y-auto overflow-x-auto rounded-md border border-slate-200 bg-white">
                        <table className="min-w-full text-xs">
                          <thead>
                            <tr className="border-b border-slate-200 bg-slate-50 text-left uppercase tracking-wide text-slate-500">
                              <th className="px-2 py-1.5">Date</th>
                              <th className="px-2 py-1.5">Ticker</th>
                              <th className="px-2 py-1.5 text-right">Rank</th>
                              <th className="px-2 py-1.5 text-right">Realized</th>
                              <th className="px-2 py-1.5 text-right">Benchmark</th>
                              <th className="px-2 py-1.5 text-center">Beat?</th>
                            </tr>
                          </thead>
                          <tbody>
                            {outcomes.outcomes.map((o) => (
                              <tr key={`${o.target_date}-${o.ticker}`} className="border-b border-slate-100 last:border-0">
                                <td className="px-2 py-1.5 text-slate-600">{o.target_date}</td>
                                <td className="px-2 py-1.5 font-medium text-slate-800">{o.ticker}</td>
                                <td className="px-2 py-1.5 text-right text-slate-500">{o.rank}</td>
                                <td className={`px-2 py-1.5 text-right ${o.realized_return_pct >= 0 ? "text-emerald-600" : "text-red-600"}`}>
                                  {o.realized_return_pct >= 0 ? "+" : ""}
                                  {o.realized_return_pct.toFixed(2)}%
                                </td>
                                <td className="px-2 py-1.5 text-right text-slate-600">
                                  {o.benchmark_return_pct >= 0 ? "+" : ""}
                                  {o.benchmark_return_pct.toFixed(2)}%
                                </td>
                                <td className="px-2 py-1.5 text-center">{o.beat_benchmark ? "✓" : ""}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </details>
                  </>
                )}
              </>
            )}
          </div>

          {data.signals.length > 0 && (
            <div className="mt-8 rounded-lg border border-indigo-200 bg-indigo-50/40 p-5">
              <h2 className="font-semibold text-slate-900">Compare Against the Predict-Page Algorithm</h2>
              <p className="mt-2 text-sm leading-relaxed text-slate-600">
                This app also has a separate, trained forecasting model (used on the Price Prediction page) —
                a different algorithm from the simple momentum rule above, not a validation of it. This shows
                what that model currently says about today&apos;s published picks, for comparison. Only ever
                shown against the latest publication, since re-running today&apos;s model against an older
                publish date would unfairly give it information it couldn&apos;t have had at the time.
              </p>
              <div className="mt-3 flex flex-wrap items-center gap-3">
                <div className="flex flex-col gap-1">
                  <label className="text-xs font-medium text-slate-500">Forecast Horizon</label>
                  <div className="flex gap-1 rounded-md border border-indigo-300 bg-white p-1">
                    {COMPARE_HORIZONS.map((h) => (
                      <button
                        key={h}
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
                  onClick={() => loadComparison(compareHorizon)}
                  disabled={comparisonLoading}
                  className="rounded-md border border-indigo-300 bg-white px-3 py-1.5 text-sm font-medium text-indigo-700 hover:bg-indigo-50 disabled:opacity-50 self-end"
                >
                  {comparisonLoading ? "Comparing…" : comparison ? "Refresh" : "Compare"}
                </button>
              </div>

              {comparisonError && (
                <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{comparisonError}</p>
              )}

              {comparison && !comparisonLoading && (
                <div className="mt-4 overflow-x-auto rounded-lg border border-slate-200 bg-white">
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
                          <td
                            className={`px-3 py-2 text-right ${
                              c.trailing_return_pct >= 0 ? "text-emerald-600" : "text-red-600"
                            }`}
                          >
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
                    history. The Momentum Return column stays fixed at the published{" "}
                    {data.lookback_days}-day trailing window regardless of the horizon chosen here — pick a
                    shorter horizon to ask &quot;does the algorithm agree over the near term?&quot; rather than
                    over the full window the picks were ranked on.
                  </p>
                </div>
              )}
            </div>
          )}

          <div className="mt-8 rounded-lg border border-slate-200 bg-white p-5">
            <h2 className="font-semibold text-slate-900">Methodology</h2>
            <p className="mt-2 text-sm leading-relaxed text-slate-600">
              Once a day, after market close, the rule ranks a fixed universe of large, liquid US stocks by
              trailing price return over the stated lookback window, and publishes the top{" "}
              {data.signals.length || 25}. It uses only price data available at the time of publication — no
              future information, no fundamentals, no subjective judgment. The same rule, applied consistently,
              so any past publication can be checked against what actually happened next.
            </p>
            <p className="mt-3 text-sm leading-relaxed text-slate-600">
              This is not personalized to any reader, does not consider anyone&apos;s holdings or goals, and is
              not investment advice. Past performance does not indicate future results.
            </p>
          </div>
        </>
      )}
    </div>
  );
}

function RecordTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="mt-1 text-lg font-semibold text-slate-900">{value}</p>
    </div>
  );
}
