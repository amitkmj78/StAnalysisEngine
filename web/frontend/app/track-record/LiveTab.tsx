"use client";

import { useEffect, useState } from "react";

import { ApiError, getSignalOutcomes, getSignalsCsvExportUrl, getTrackRecord } from "@/lib/api";
import type { PublishedSignalsResponse, SignalOutcomesResponse, TrackRecordResponse } from "@/lib/types";
import PlotlyChart from "@/components/PlotlyChart";
import { RecordTile } from "./page";

const HORIZONS = [10, 30, 60, 90];

function fmtPct(v: number | null): string {
  if (v === null) return "—";
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
}

export default function LiveTab({ data }: { data: PublishedSignalsResponse }) {
  const [horizon, setHorizon] = useState(30);
  const [outcomes, setOutcomes] = useState<SignalOutcomesResponse | null>(null);
  const [outcomesError, setOutcomesError] = useState<string | null>(null);
  const [trackRecord, setTrackRecord] = useState<TrackRecordResponse | null>(null);
  const [trackRecordError, setTrackRecordError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    setOutcomesError(null);
    setTrackRecordError(null);
    Promise.all([
      getSignalOutcomes(horizon).catch((err) => {
        setOutcomesError(err instanceof ApiError ? err.message : "Failed to load evaluated outcomes.");
        return null;
      }),
      getTrackRecord(horizon).catch((err) => {
        setTrackRecordError(err instanceof ApiError ? err.message : "Failed to load the enhanced track record.");
        return null;
      }),
    ])
      .then(([o, tr]) => {
        setOutcomes(o);
        setTrackRecord(tr);
      })
      .finally(() => setLoading(false));
  }, [horizon]);

  const modelVersions = trackRecord ? Object.keys(trackRecord.metrics_by_model_version) : [];
  const signalGroups = trackRecord ? Object.keys(trackRecord.metrics_by_signal) : [];

  return (
    <>
      {data.is_lagged && (
        <div className="mt-6 flex items-center justify-between gap-3 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
          <span>Showing delayed data (free tier).</span>
        </div>
      )}

      <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <RecordTile label="Record Started" value={data.record_start_date ?? "Not yet started"} />
        <RecordTile label="Days Published" value={String(data.days_published)} />
        <RecordTile label="Latest Publication" value={data.target_date ?? "—"} />
      </div>

      {data.tier === "paid" && (
        <div className="mt-3">
          <a
            href={getSignalsCsvExportUrl(data.universe_id, data.lookback_days)}
            className="inline-block rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-100"
          >
            Export CSV
          </a>
        </div>
      )}

      {data.signals.length === 0 ? (
        <div className="mt-6 rounded-lg border border-slate-200 bg-white p-5 text-sm text-slate-500">
          No signals have been published yet. This page will show the current picks and the full history as
          soon as publication begins — nothing is backfilled or reconstructed after the fact.
        </div>
      ) : (
        <div className="mt-6 overflow-x-auto rounded-lg border border-slate-200 bg-white">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                <th className="px-3 py-2">Rank</th>
                <th className="px-3 py-2">Ticker</th>
                <th className="px-3 py-2 text-right">{data.lookback_days}-Day Trailing Return</th>
                <th className="px-3 py-2 text-center">Data Source</th>
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
                  <td className="px-3 py-2 text-center">
                    <span
                      title={
                        s.data_source === "pit"
                          ? "Computed from the point-in-time store — independently reconstructible from recorded history."
                          : "PIT store didn't have enough history for this ticker yet — computed from a live price fetch at publication time."
                      }
                      className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
                        s.data_source === "pit" ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"
                      }`}
                    >
                      {s.data_source === "pit" ? "PIT" : "live"}
                    </span>
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
            {" · "}
            {data.signals.filter((s) => s.data_source === "pit").length}/{data.signals.length} picks sourced from
            the point-in-time store
            {data.signals.every((s) => s.data_source === "pit")
              ? " (fully reconstructible)"
              : " — the rest fell back to a live price fetch because the PIT store didn't have enough history for that ticker yet"}
            .
          </p>
        </div>
      )}

      <div className="mt-8 rounded-lg border border-emerald-200 bg-emerald-50/40 p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="font-semibold text-slate-900">Live Performance to Date</h2>
          <div className="flex gap-1 rounded-md border border-emerald-300 bg-white p-1">
            {HORIZONS.map((h) => (
              <button
                key={h}
                type="button"
                onClick={() => setHorizon(h)}
                className={`rounded px-2.5 py-1 text-xs font-medium ${
                  horizon === h ? "bg-emerald-600 text-white" : "text-slate-600 hover:bg-emerald-50"
                }`}
              >
                {h}d
              </button>
            ))}
          </div>
        </div>
        <p className="mt-2 text-sm leading-relaxed text-slate-600">
          Real, out-of-sample results only — never a simulation, never blended with any backtest. Each published
          pick is scored once its full holding window has actually elapsed: entry priced at publication, exit
          priced {horizon} trading days later, compared against equally owning the whole universe over that
          identical stretch.
        </p>

        {outcomesError && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{outcomesError}</p>}
        {trackRecordError && (
          <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{trackRecordError}</p>
        )}
        {loading && <p className="mt-3 text-sm text-slate-500">Loading…</p>}

        {outcomes && trackRecord && !loading && (
          <>
            {outcomes.num_evaluated_dates === 0 ? (
              <p className="mt-3 rounded-md border border-slate-200 bg-white px-3 py-3 text-sm text-slate-500">
                No published picks have completed their {horizon}-trading-day holding window yet — nothing is
                evaluated or estimated before it&apos;s actually knowable. Check back once the earliest
                publication is that far out.
              </p>
            ) : (
              <>
                <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <RecordTile label="Dates Evaluated" value={String(outcomes.num_evaluated_dates)} />
                  <RecordTile label="Picks Evaluated" value={String(outcomes.num_evaluated_picks)} />
                  <RecordTile label="Hit Rate" value={outcomes.hit_rate_pct !== null ? `${outcomes.hit_rate_pct.toFixed(1)}%` : "—"} />
                  <RecordTile label="Avg Return" value={fmtPct(outcomes.avg_return_pct)} />
                  <RecordTile label="Avg Excess vs SPY" value={fmtPct(trackRecord.avg_excess_vs_spy_pct)} />
                  <RecordTile
                    label="Information Coeff."
                    value={outcomes.information_coefficient !== null ? outcomes.information_coefficient.toFixed(3) : "—"}
                  />
                  <RecordTile
                    label="Quintile Spread"
                    value={
                      outcomes.quintile_spread_pct !== null
                        ? `${outcomes.quintile_spread_pct >= 0 ? "+" : ""}${outcomes.quintile_spread_pct.toFixed(2)}%`
                        : "—"
                    }
                  />
                </div>
                <p className="mt-3 text-xs text-slate-500">
                  <strong>Hit Rate</strong> is the share of individual picks that beat the equal-weight universe.{" "}
                  <strong>Avg Excess vs SPY</strong> compares each pick&apos;s return to SPY&apos;s own return
                  over the identical window (a separate comparison from the equal-weight-universe benchmark used
                  elsewhere here). <strong>Information Coefficient</strong> is the average correlation between a
                  pick&apos;s rank and its realized return — positive means better-ranked picks really did do
                  better. <strong>Quintile Spread</strong> is the best-ranked fifth&apos;s average return minus
                  the worst-ranked fifth&apos;s, averaged across evaluated dates.
                </p>

                {modelVersions.length > 1 && (
                  <div className="mt-4">
                    <h3 className="text-sm font-semibold text-slate-800">By Model Version</h3>
                    <p className="mt-1 text-xs text-slate-500">
                      A model-version change starts a new record — older versions stay visible, never merged
                      away.
                    </p>
                    <div className="mt-2 overflow-x-auto rounded-md border border-slate-200 bg-white">
                      <table className="min-w-full text-xs">
                        <thead>
                          <tr className="border-b border-slate-200 bg-slate-50 text-left uppercase tracking-wide text-slate-500">
                            <th className="px-2 py-1.5">Model Version</th>
                            <th className="px-2 py-1.5 text-right">Picks</th>
                            <th className="px-2 py-1.5 text-right">Hit Rate</th>
                            <th className="px-2 py-1.5 text-right">Avg Return</th>
                          </tr>
                        </thead>
                        <tbody>
                          {modelVersions.map((v) => {
                            const m = trackRecord.metrics_by_model_version[v];
                            return (
                              <tr key={v} className="border-b border-slate-100 last:border-0">
                                <td className="px-2 py-1.5 font-mono">{v.slice(0, 12)}</td>
                                <td className="px-2 py-1.5 text-right">{m.num_evaluated_picks}</td>
                                <td className="px-2 py-1.5 text-right">{m.hit_rate_pct !== null ? `${m.hit_rate_pct.toFixed(1)}%` : "—"}</td>
                                <td className="px-2 py-1.5 text-right">{fmtPct(m.avg_return_pct)}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {signalGroups.length > 0 && (
                  <div className="mt-4">
                    <h3 className="text-sm font-semibold text-slate-800">By Signal</h3>
                    <p className="mt-1 text-xs text-slate-500">{trackRecord.signal_note}</p>
                    <div className="mt-2 overflow-x-auto rounded-md border border-slate-200 bg-white">
                      <table className="min-w-full text-xs">
                        <thead>
                          <tr className="border-b border-slate-200 bg-slate-50 text-left uppercase tracking-wide text-slate-500">
                            <th className="px-2 py-1.5">Signal</th>
                            <th className="px-2 py-1.5 text-right">Picks</th>
                            <th className="px-2 py-1.5 text-right">Hit Rate</th>
                            <th className="px-2 py-1.5 text-right">Avg Return</th>
                          </tr>
                        </thead>
                        <tbody>
                          {signalGroups.map((s) => {
                            const m = trackRecord.metrics_by_signal[s];
                            return (
                              <tr key={s} className="border-b border-slate-100 last:border-0">
                                <td className="px-2 py-1.5">{s}</td>
                                <td className="px-2 py-1.5 text-right">{m.num_evaluated_picks}</td>
                                <td className="px-2 py-1.5 text-right">{m.hit_rate_pct !== null ? `${m.hit_rate_pct.toFixed(1)}%` : "—"}</td>
                                <td className="px-2 py-1.5 text-right">{fmtPct(m.avg_return_pct)}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                <div className="mt-4">
                  <h3 className="text-sm font-semibold text-slate-800">Calibration</h3>
                  <p className="mt-1 text-xs text-slate-500">
                    Stated confidence vs. actual hit rate, per bucket. Confidence here is a stability proxy
                    derived from this ranking&apos;s own publication history — how consistently a ticker has
                    stayed in (or out of) the published top-N over the trailing 30 days — not a stated
                    probability the ranking rule outputs directly (it&apos;s a pure rank, not a probabilistic
                    forecast).
                  </p>
                  <div className="mt-2 overflow-x-auto rounded-md border border-slate-200 bg-white">
                    <table className="min-w-full text-xs">
                      <thead>
                        <tr className="border-b border-slate-200 bg-slate-50 text-left uppercase tracking-wide text-slate-500">
                          <th className="px-2 py-1.5">Confidence Bucket</th>
                          <th className="px-2 py-1.5 text-right">Hit Rate</th>
                          <th className="px-2 py-1.5 text-right">Sample Size</th>
                        </tr>
                      </thead>
                      <tbody>
                        {trackRecord.calibration.map((b) => (
                          <tr key={b.bucket_label} className="border-b border-slate-100 last:border-0">
                            <td className="px-2 py-1.5">{b.bucket_label}</td>
                            <td className="px-2 py-1.5 text-right">{b.hit_rate_pct !== null ? `${b.hit_rate_pct.toFixed(1)}%` : "not enough data yet"}</td>
                            <td className="px-2 py-1.5 text-right">{b.sample_size}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="mt-4">
                  <h3 className="text-sm font-semibold text-slate-800">Worst Misses</h3>
                  <p className="mt-1 text-xs text-slate-500">{trackRecord.trim_note}</p>
                  {trackRecord.worst_misses.length === 0 ? (
                    <p className="mt-2 text-xs text-slate-400">No evaluated picks yet.</p>
                  ) : (
                    <div className="mt-2 overflow-x-auto rounded-md border border-slate-200 bg-white">
                      <table className="min-w-full text-xs">
                        <thead>
                          <tr className="border-b border-slate-200 bg-slate-50 text-left uppercase tracking-wide text-slate-500">
                            <th className="px-2 py-1.5">Date</th>
                            <th className="px-2 py-1.5">Ticker</th>
                            <th className="px-2 py-1.5 text-right">Realized</th>
                          </tr>
                        </thead>
                        <tbody>
                          {trackRecord.worst_misses.map((m) => (
                            <tr key={`${m.target_date}-${m.ticker}`} className="border-b border-slate-100 last:border-0">
                              <td className="px-2 py-1.5 text-slate-600">{m.target_date}</td>
                              <td className="px-2 py-1.5 font-medium text-slate-800">{m.ticker}</td>
                              <td className="px-2 py-1.5 text-right text-red-600">{fmtPct(m.realized_return_pct)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>

                {trackRecord.model_portfolio_series.length >= 2 && (
                  <div className="mt-4">
                    <h3 className="text-sm font-semibold text-slate-800">Model Portfolio vs. SPY</h3>
                    <p className="mt-1 text-xs text-slate-500">
                      Equal-weight Buys, rebalanced at each non-overlapping {horizon}-day period (publication is
                      daily, but chaining every day would double-count overlapping windows — see the methodology
                      note below). Assumes zero trading costs. Thin with only a few weeks of history; deepens
                      over time.
                    </p>
                    <PlotlyChart
                      data={[
                        {
                          x: trackRecord.model_portfolio_series.map((p) => p[0]),
                          y: trackRecord.model_portfolio_series.map((p) => p[1]),
                          type: "scatter",
                          mode: "lines",
                          name: "Model portfolio",
                          line: { color: "#13795B", width: 2.5 },
                        },
                        {
                          x: trackRecord.spy_portfolio_series.map((p) => p[0]),
                          y: trackRecord.spy_portfolio_series.map((p) => p[1]),
                          type: "scatter",
                          mode: "lines",
                          name: "SPY",
                          line: { color: "#7C8794", width: 2, dash: "dash" },
                        },
                      ]}
                      layout={{
                        font: { color: "#6b7280", family: "Arial, Helvetica, sans-serif", size: 11 },
                        xaxis: { gridcolor: "#e5e7eb", linecolor: "#e5e7eb" },
                        yaxis: { gridcolor: "#e5e7eb", linecolor: "#e5e7eb", tickprefix: "$" },
                        paper_bgcolor: "#ffffff",
                        plot_bgcolor: "#ffffff",
                        height: 280,
                        margin: { t: 16, r: 16, b: 32, l: 56 },
                        autosize: true,
                        legend: { orientation: "h", y: -0.2 },
                      }}
                      style={{ width: "100%" }}
                      useResizeHandler
                      config={{ displayModeBar: false }}
                    />
                  </div>
                )}

                <details className="mt-4 text-xs text-slate-600">
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

      <div className="mt-8 rounded-lg border border-slate-200 bg-white p-5">
        <h2 className="font-semibold text-slate-900">Methodology</h2>
        <p className="mt-2 text-sm leading-relaxed text-slate-600">
          Once a day, after market close, the rule ranks a fixed universe of large, liquid US stocks by trailing
          price return over the stated lookback window, and publishes the top {data.signals.length || 25}. It
          uses only price data available at the time of publication — no future information, no fundamentals,
          no subjective judgment. The same rule, applied consistently, so any past publication can be checked
          against what actually happened next.
        </p>
        <p className="mt-3 text-sm leading-relaxed text-slate-600">
          This is not personalized to any reader, does not consider anyone&apos;s holdings or goals, and is not
          investment advice. Past performance does not indicate future results.
        </p>
      </div>
    </>
  );
}
