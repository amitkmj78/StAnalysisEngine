"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { ApiError, getEtf } from "@/lib/api";
import type { EtfDetail } from "@/lib/types";

function plain(v: number | null | undefined, digits = 2) {
  return v === null || v === undefined ? "–" : v.toFixed(digits);
}

export default function EtfDetailPage() {
  const params = useParams<{ ticker: string }>();
  const ticker = params.ticker?.toUpperCase() ?? "";

  const [data, setData] = useState<EtfDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!ticker) return;
    getEtf(ticker)
      .then(setData)
      .catch((err) => setError(err instanceof ApiError ? err.message : "This ETF could not be loaded."));
  }, [ticker]);

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h1 className="font-display text-2xl font-semibold text-slate-900 dark:text-slate-50">{ticker}</h1>
        <Link href="/stock-finder/etf" className="text-sm font-medium text-slate-700 dark:text-slate-200 hover:underline">
          ← Back to ETF Screener
        </Link>
      </div>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {!data && !error && <p className="mt-6 text-sm text-slate-500 dark:text-slate-400">Loading…</p>}

      {data && (
        <div className="mt-6 flex flex-col gap-4">
          <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5">
            <h2 className="font-semibold text-slate-900 dark:text-slate-50">{data.name}</h2>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              {data.category} · benchmarked to {data.benchmark}
            </p>
            <dl className="mt-3 grid grid-cols-2 gap-3 text-sm sm:grid-cols-3">
              <div>
                <dt className="text-xs text-slate-500 dark:text-slate-400">Weighted score</dt>
                <dd className="font-mono">{plain(data.weighted_score, 1)}</dd>
              </div>
              <div>
                <dt className="text-xs text-slate-500 dark:text-slate-400">Coverage</dt>
                <dd className="font-mono text-xs">
                  {data.coverage_pct.toFixed(0)}% ({data.covered_holdings}/{data.total_holdings} holdings)
                </dd>
              </div>
              <div>
                <dt className="text-xs text-slate-500 dark:text-slate-400">Expense ratio</dt>
                <dd className="font-mono">{plain(data.expense_ratio_pct)}%</dd>
              </div>
              <div>
                <dt className="text-xs text-slate-500 dark:text-slate-400">Tracking error</dt>
                <dd className="font-mono">
                  {data.tracking_error_vs_benchmark_pct !== null
                    ? `${plain(data.tracking_error_vs_benchmark_pct)}% vs ${data.tracking_error_benchmark_ticker}`
                    : "N/A — no confidently-mapped benchmark"}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-slate-500 dark:text-slate-400">Your portfolio overlap</dt>
                <dd className="font-mono">
                  {data.portfolio_overlap.overlap_pct > 0
                    ? `${data.portfolio_overlap.overlap_pct}% (${data.portfolio_overlap.overlapping_tickers.join(", ")})`
                    : "None"}
                </dd>
              </div>
            </dl>
            <p className="mt-3 text-xs text-slate-400 dark:text-slate-500">
              Weighted score and overlap are both based on this fund&apos;s top ~10 disclosed holdings only — no free
              source gives a full holdings list. Tracking error is computed against {data.tracking_error_benchmark_ticker ?? "a benchmark"},
              not the fund&apos;s own underlying index, when that index has no clean free data source.
            </p>
          </div>

          <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5">
            <h2 className="font-semibold text-slate-900 dark:text-slate-50">Top Holdings</h2>
            {data.holdings.length === 0 ? (
              <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">No holdings data available for this fund.</p>
            ) : (
              <ul className="mt-2 flex flex-col gap-1 text-sm">
                {data.holdings.map((h) => (
                  <li key={h.symbol} className="flex justify-between font-mono">
                    <span>
                      {h.symbol} <span className="text-slate-500 dark:text-slate-400">{h.name}</span>
                    </span>
                    <span>{(h.weight * 100).toFixed(2)}%</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
