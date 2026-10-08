"use client";

import { useEffect, useState } from "react";

import { ApiError, getSignalsCsvExportUrl, getTrackRecord } from "@/lib/api";
import type { PublishedSignalsResponse, TrackRecordResponse } from "@/lib/types";
import PlotlyChart from "@/components/PlotlyChart";
import { RecordTile } from "./page";

const HORIZONS = [10, 30, 60, 90];

function fmtPct(v: number | null): string {
  if (v === null) return "—";
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
}

export default function LiveTab({ data }: { data: PublishedSignalsResponse }) {
  const [horizon, setHorizon] = useState(30);
  // REG-2: empty string means "all regimes" (no filter sent to the API).
  const [regime, setRegime] = useState("");
  const [trackRecord, setTrackRecord] = useState<TrackRecordResponse | null>(null);
  const [trackRecordError, setTrackRecordError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    setTrackRecordError(null);
    getTrackRecord(horizon, regime || undefined)
      .then(setTrackRecord)
      .catch((err) => {
        setTrackRecordError(err instanceof ApiError ? err.message : "Failed to load the enhanced track record.");
        setTrackRecord(null);
      })
      .finally(() => setLoading(false));
  }, [horizon, regime]);

  const modelVersions = trackRecord ? Object.keys(trackRecord.metrics_by_model_version) : [];
  const signalGroups = trackRecord ? Object.keys(trackRecord.metrics_by_signal) : [];
  // metrics_by_regime is always the full, unfiltered breakdown (see
  // signals.py::get_track_record) regardless of the active `regime`
  // filter, so this list of options never collapses to one entry once a
  // filter is chosen.
  const regimeOptions = trackRecord ? Object.keys(trackRecord.metrics_by_regime) : [];

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
        <div className="mt-6 rounded-xl border border-slate-200 bg-white p-5 text-sm text-slate-500">
          No signals have been published yet. This page will show the current picks and the full history as
          soon as publication begins — nothing is backfilled or reconstructed after the fact.
        </div>
      ) : (
        <div className="mt-6 overflow-x-auto rounded-xl border border-slate-200 bg-white">
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
          <div className="flex flex-wrap items-center gap-2">
            {regimeOptions.length > 0 && (
              <label className="flex items-center gap-1.5 text-xs text-slate-600">
                Regime
                <select
                  value={regime}
                  onChange={(e) => setRegime(e.target.value)}
                  className="rounded-md border border-slate-300 bg-white px-2 py-1 text-xs"
                >
                  <option value="">All regimes</option>
                  {regimeOptions.map((r) => (
                    <option key={r} value={r}>
                      {r}
                    </option>
                  ))}
                </select>
              </label>
            )}
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
        </div>
        <p className="mt-2 text-sm leading-relaxed text-slate-600">
          Real, out-of-sample results only — never a simulation, never blended with any backtest.{" "}
          <strong>This scores the actual Buy/Trim signal shown on each stock&apos;s own page</strong> (not the
          trailing-return picks list above, which is a separate momentum rule — see the methodology note at the
          bottom), for every stock in the universe, not just a top-N subset. Each signal is scored once its full
          holding window has actually elapsed: entry priced the day it was issued, exit priced {horizon} trading
          days later, compared against SPY over that identical stretch.
        </p>

        {trackRecordError && (
          <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{trackRecordError}</p>
        )}
        {loading && <p className="mt-3 text-sm text-slate-500">Loading…</p>}

        {trackRecord && !loading && (
          <>
            {trackRecord.metrics.num_evaluated_dates === 0 ? (
              <p className="mt-3 rounded-md border border-slate-200 bg-white px-3 py-3 text-sm text-slate-500">
                No Buy/Trim signal has completed its {horizon}-trading-day holding window yet — nothing is
                evaluated or estimated before it&apos;s actually knowable. Check back once the earliest captured
                signal is that far out.
              </p>
            ) : (
              <>
                <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <RecordTile label="Dates Evaluated" value={String(trackRecord.metrics.num_evaluated_dates)} />
                  <RecordTile label="Signals Evaluated" value={String(trackRecord.metrics.num_evaluated_picks)} />
                  <RecordTile
                    label="Hit Rate"
                    value={trackRecord.metrics.hit_rate_pct !== null ? `${trackRecord.metrics.hit_rate_pct.toFixed(1)}%` : "—"}
                  />
                  <RecordTile label="Avg Return" value={fmtPct(trackRecord.metrics.avg_return_pct)} />
                  <RecordTile label="Avg Excess vs SPY" value={fmtPct(trackRecord.avg_excess_vs_spy_pct)} />
                </div>
                <p className="mt-3 text-xs text-slate-500">
                  <strong>Hit Rate</strong> is the share of signals that beat SPY&apos;s own return over the
                  identical window (a Buy beats it by rising more, a Trim by falling more/rising less).{" "}
                  <strong>Avg Excess vs SPY</strong> is the average of that same gap in percentage points. There
                  is no Information Coefficient or Quintile Spread here — those measure a RANKING&apos;s quality
                  (did rank 1 really beat rank 500), and every signal here is independent, not ranked against the
                  others.
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

                {regimeOptions.length > 0 && (
                  <div className="mt-4">
                    <h3 className="text-sm font-semibold text-slate-800">By Regime</h3>
                    <p className="mt-1 text-xs text-slate-500">
                      REG-2: hit rate by market regime on the day each signal was issued. &quot;unknown&quot;
                      covers dates before a regime reading existed for that day.{" "}
                      {regime ? `Filtered to ${regime} above.` : "Use the Regime selector above to filter."}{" "}
                      The regime label itself is shown for information only — it has not passed its own
                      validation gate; see the{" "}
                      <a href="/methodology" className="underline hover:text-slate-700">
                        methodology page
                      </a>
                      .
                    </p>
                    <div className="mt-2 overflow-x-auto rounded-md border border-slate-200 bg-white">
                      <table className="min-w-full text-xs">
                        <thead>
                          <tr className="border-b border-slate-200 bg-slate-50 text-left uppercase tracking-wide text-slate-500">
                            <th className="px-2 py-1.5">Regime</th>
                            <th className="px-2 py-1.5 text-right">Picks</th>
                            <th className="px-2 py-1.5 text-right">Hit Rate</th>
                            <th className="px-2 py-1.5 text-right">Avg Return</th>
                          </tr>
                        </thead>
                        <tbody>
                          {regimeOptions.map((r) => {
                            const m = trackRecord.metrics_by_regime[r];
                            return (
                              <tr key={r} className="border-b border-slate-100 last:border-0">
                                <td className="px-2 py-1.5">{r}</td>
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
                    Stated confidence vs. actual hit rate, grouped into coarse ranges. Confidence here is the
                    real value captured when each signal was issued (services.portfolio_compare_service.
                    derive_confidence — a stability-based score, not yet itself calibrated; see the Confidence
                    Calibration check below, which tests exactly that).
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
                  <h3 className="text-sm font-semibold text-slate-800">Confidence Calibration (fit vs. holdout)</h3>
                  <p className="mt-1 text-xs text-slate-500">
                    FND-4: is the stated confidence score actually a calibrated probability of a hit? The
                    earlier {trackRecord.confidence_calibration.fit_set_size} signals (by date) fit an empirical
                    hit rate per exact score; the later {trackRecord.confidence_calibration.holdout_set_size}
                    {" "}validate it independently — &quot;Agrees&quot; means the two are within{" "}
                    {trackRecord.confidence_calibration.agreement_threshold_points} points of each other, shown
                    only once both halves have at least {trackRecord.confidence_calibration.min_samples_per_bucket}
                    {" "}signals for that score. This does not change the confidence number shown elsewhere on the
                    site today — it is a check of that number, not yet a replacement for it.
                  </p>
                  {Object.keys(trackRecord.confidence_calibration.buckets).length === 0 ? (
                    <p className="mt-2 text-xs text-slate-400">Not enough history yet to check any score.</p>
                  ) : (
                    <div className="mt-2 overflow-x-auto rounded-md border border-slate-200 bg-white">
                      <table className="min-w-full text-xs">
                        <thead>
                          <tr className="border-b border-slate-200 bg-slate-50 text-left uppercase tracking-wide text-slate-500">
                            <th className="px-2 py-1.5">Stated Score</th>
                            <th className="px-2 py-1.5 text-right">Fit Hit Rate (n)</th>
                            <th className="px-2 py-1.5 text-right">Holdout Hit Rate (n)</th>
                            <th className="px-2 py-1.5 text-center">Agrees?</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(trackRecord.confidence_calibration.buckets)
                            .sort((a, b) => Number(b[0]) - Number(a[0]))
                            .map(([score, b]) => (
                              <tr key={score} className="border-b border-slate-100 last:border-0">
                                <td className="px-2 py-1.5">{score}</td>
                                <td className="px-2 py-1.5 text-right">
                                  {b.fit_hit_rate_pct !== null ? `${b.fit_hit_rate_pct.toFixed(1)}% (n=${b.fit_n})` : "insufficient data"}
                                </td>
                                <td className="px-2 py-1.5 text-right">
                                  {b.holdout_hit_rate_pct !== null
                                    ? `${b.holdout_hit_rate_pct.toFixed(1)}% (n=${b.holdout_n})`
                                    : `insufficient data (n=${b.holdout_n})`}
                                </td>
                                <td className="px-2 py-1.5 text-center">
                                  {b.agrees_within_5_points === null ? "—" : b.agrees_within_5_points ? "✓" : "✗"}
                                </td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  )}
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
                            <th className="px-2 py-1.5">Signal</th>
                            <th className="px-2 py-1.5 text-right">Realized</th>
                          </tr>
                        </thead>
                        <tbody>
                          {trackRecord.worst_misses.map((m) => (
                            <tr key={`${m.target_date}-${m.ticker}`} className="border-b border-slate-100 last:border-0">
                              <td className="px-2 py-1.5 text-slate-600">{m.target_date}</td>
                              <td className="px-2 py-1.5 font-medium text-slate-800">{m.ticker}</td>
                              <td className="px-2 py-1.5 text-slate-600">{m.signal ?? "Buy"}</td>
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
                      note below). After assumed trading costs: {trackRecord.model_portfolio_cost_bps_one_way} bps
                      one-way ({trackRecord.model_portfolio_cost_bps_one_way * 2} bps round-trip per rebalance,
                      selling the outgoing cohort and buying the incoming one) — a reasoned retail slippage/spread
                      estimate for liquid S&amp;P 500 names, not backtested or empirically derived. SPY&apos;s own
                      curve isn&apos;t charged this, since a real buy-and-hold benchmark isn&apos;t repeatedly
                      traded the way this portfolio is. Thin with only a few weeks of history; deepens over time.
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

              </>
            )}
          </>
        )}
      </div>

      <div className="mt-8 rounded-xl border border-slate-200 bg-white p-5">
        <h2 className="font-semibold text-slate-900">Methodology</h2>
        <p className="mt-2 text-sm leading-relaxed text-slate-600">
          <strong>Today&apos;s Picks (table above):</strong> once a day, after market close, a fixed rule ranks a
          fixed universe of large, liquid US stocks by trailing price return over the stated lookback window, and
          publishes the top {data.signals.length || 25}. It uses only price data available at the time of
          publication — no future information, no fundamentals, no subjective judgment. The same rule, applied
          consistently, so any past publication can be checked against what actually happened next.
        </p>
        <p className="mt-3 text-sm leading-relaxed text-slate-600">
          <strong>Live Performance to Date (section above):</strong> a separate, different measurement — every
          stock&apos;s own real short-term Buy/Hold/Trim signal from its stock page (the same two-score system
          described on the{" "}
          <a href="/methodology" className="underline hover:text-slate-700">
            methodology page
          </a>
          ), scored against its own realized move once the holding window elapses. This is the app&apos;s actual
          published signal, not the trailing-return picks list above — the two are intentionally different
          systems shown on the same page.
        </p>
        <p className="mt-3 text-sm leading-relaxed text-slate-600">
          Neither is personalized to any reader, considers anyone&apos;s holdings or goals, or is investment
          advice. Past performance does not indicate future results.
        </p>
      </div>
    </>
  );
}
