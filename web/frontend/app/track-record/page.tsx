"use client";

import { useEffect, useState } from "react";

import { ApiError, getPublishedSignals } from "@/lib/api";
import type { PublishedSignalsResponse } from "@/lib/types";

export default function TrackRecordPage() {
  const [data, setData] = useState<PublishedSignalsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getPublishedSignals()
      .then(setData)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Failed to load the track record."))
      .finally(() => setLoading(false));
  }, []);

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

          <div className="mt-8 rounded-lg border border-slate-200 bg-white p-5">
            <h2 className="font-semibold text-slate-900">Methodology</h2>
            <p className="mt-2 text-sm leading-relaxed text-slate-600">
              Once a day, after market close, the rule ranks a fixed universe of large, liquid US stocks by
              trailing price return over the stated lookback window, and publishes the top 5. It uses only
              price data available at the time of publication — no future information, no fundamentals, no
              subjective judgment. The same rule, applied consistently, so any past publication can be checked
              against what actually happened next.
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
