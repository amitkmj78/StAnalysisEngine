"use client";

import { useEffect, useState } from "react";
import { getSimilarSetups } from "@/lib/api";
import type { SimilarSetupsResponse } from "@/lib/types";

// DIF-7: what the price did after past days that looked like today's (similar stored
// short-term score and the same regime). A record, not a forecast; under 30 cases the
// caveat says the sample is too small to show a pattern.

function pct(v: number | null | undefined) {
  return v === null || v === undefined ? "n/a" : `${v > 0 ? "+" : ""}${v.toFixed(2)}%`;
}

export default function SimilarSetupsCard({ ticker }: { ticker: string }) {
  const [data, setData] = useState<SimilarSetupsResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    getSimilarSetups(ticker)
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
    <div className="rounded-xl border border-slate-200 bg-white p-5">
      <h3 className="text-sm font-semibold text-slate-900">Similar past setups</h3>
      {!data.available ? (
        <p className="mt-2 text-xs text-slate-400">{data.reason}</p>
      ) : (
        <>
          <p className="mt-1 text-xs text-slate-500">
            Days with a short-term score within {data.score_band_points} points of {data.current_score?.toFixed(1)} and the
            same regime ({data.current_regime ?? "no regime label"}). Measured {data.horizon_sessions} sessions after each day.
          </p>
          <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 text-sm sm:grid-cols-4">
            <div>
              <dt className="text-xs text-slate-400">Cases</dt>
              <dd className="font-mono font-semibold text-slate-800">{data.n}</dd>
            </div>
            <div>
              <dt className="text-xs text-slate-400">Median move</dt>
              <dd className="font-mono font-semibold text-slate-800">{pct(data.median_return_pct)}</dd>
            </div>
            <div>
              <dt className="text-xs text-slate-400">Lowest</dt>
              <dd className="font-mono text-slate-700">{pct(data.min_return_pct)}</dd>
            </div>
            <div>
              <dt className="text-xs text-slate-400">Highest</dt>
              <dd className="font-mono text-slate-700">{pct(data.max_return_pct)}</dd>
            </div>
          </dl>
          {data.caveat && <p className="mt-3 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">{data.caveat}</p>}
          <p className="mt-2 text-xs text-slate-400">{data.note}</p>
        </>
      )}
    </div>
  );
}
