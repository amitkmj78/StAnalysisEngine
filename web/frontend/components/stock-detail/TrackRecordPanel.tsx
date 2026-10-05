"use client";

import { useEffect, useState } from "react";
import MetricLabel from "@/components/MetricLabel";
import { getStockTrackRecord } from "@/lib/api";
import type { StockTrackRecord } from "@/lib/types";

// DIF-2: how this stock's own short-term signals have done, against SPY over the same dates.
// Shows no figures until enough signals have finished their test, and says so instead.

const INFO = {
  title: "Track record",
  body: [
    "Each past Buy or Trim signal for this stock, checked after its 10-session test period.",
    "A Buy is a hit if the stock rose over that period. A Trim is a hit if it did not. Hold signals are not counted.",
    "The excess return is the stock's move minus SPY's move over the same dates. Past results, not a forecast.",
  ],
};

function pct(v: number | null | undefined) {
  return v === null || v === undefined ? "–" : `${v > 0 ? "+" : ""}${v.toFixed(1)}%`;
}

export default function TrackRecordPanel({ ticker }: { ticker: string }) {
  const [data, setData] = useState<StockTrackRecord | null>(null);

  useEffect(() => {
    let cancelled = false;
    getStockTrackRecord(ticker)
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch(() => {
        if (!cancelled) setData(null);
      });
    return () => {
      cancelled = true;
    };
  }, [ticker]);

  if (!data) return null;

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5">
      <h3 className="flex items-center gap-1.5 text-sm font-semibold text-slate-900">
        {ticker} track record
        <MetricLabel info={INFO} />
      </h3>
      {!data.enough_data ? (
        <p className="mt-2 text-sm text-slate-600">
          {data.message ?? "Not enough data yet."}
        </p>
      ) : (
        <dl className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
          <div>
            <dt className="text-xs text-slate-500">Hit rate</dt>
            <dd className="font-mono text-lg font-semibold text-slate-900">{data.hit_rate_pct?.toFixed(0)}%</dd>
            <dd className="text-xs text-slate-400">of {data.signals_evaluated} signals</dd>
          </div>
          <div>
            <dt className="text-xs text-slate-500">Average vs SPY</dt>
            <dd className="font-mono text-lg font-semibold text-slate-900">{pct(data.avg_excess_vs_spy_pct)}</dd>
            <dd className="text-xs text-slate-400">per signal, {data.horizon_sessions} sessions</dd>
          </div>
          <div>
            <dt className="text-xs text-slate-500">Worst miss</dt>
            <dd className="font-mono text-lg font-semibold text-slate-900">
              {data.worst_miss ? pct(data.worst_miss.realized_return_pct) : "–"}
            </dd>
            <dd className="text-xs text-slate-400">
              {data.worst_miss ? `${data.worst_miss.signal} on ${data.worst_miss.as_of_date}` : "No misses"}
            </dd>
          </div>
        </dl>
      )}
    </section>
  );
}
