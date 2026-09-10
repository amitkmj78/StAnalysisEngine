"use client";

import { useEffect, useState } from "react";

import { ApiError, getPortfolioBenchmarkComparison } from "@/lib/api";
import type { PortfolioBenchmarkComparison } from "@/lib/types";

function fmtPct(pct: number | null): string {
  if (pct === null) return "—";
  return `${pct >= 0 ? "+" : ""}${pct.toFixed(1)}%`;
}

export default function BenchmarkComparisonCard({ portfolioId }: { portfolioId: number | null }) {
  const [data, setData] = useState<PortfolioBenchmarkComparison | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getPortfolioBenchmarkComparison(portfolioId ?? undefined)
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof ApiError ? err.message : "Could not load the S&P 500 comparison.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [portfolioId]);

  if (loading) return null;
  if (error) return <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>;
  if (!data || data.gap_pct === null) return null;

  const positive = data.gap_pct >= 0;

  return (
    <div
      className={`mt-3 rounded-lg border p-4 ${
        data.underperforming ? "border-amber-200 bg-amber-50" : "border-slate-200 bg-white"
      }`}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-4 text-sm">
          <span className="font-semibold text-slate-900">vs. S&amp;P 500 ({data.benchmark_ticker})</span>
          <span className="text-slate-600">
            Portfolio <strong>{fmtPct(data.portfolio_return_pct)}</strong>
          </span>
          <span className="text-slate-600">
            {data.benchmark_ticker} <strong>{fmtPct(data.benchmark_return_pct)}</strong>
          </span>
        </div>
        <span
          className={`rounded-full px-2.5 py-1 text-xs font-semibold ${
            positive ? "bg-emerald-50 text-emerald-700" : "bg-red-50 text-red-700"
          }`}
        >
          {positive ? "Ahead by" : "Trailing by"} {Math.abs(data.gap_pct).toFixed(1)} pts
        </span>
      </div>
      {data.underperforming && data.suggestion && (
        <p className="mt-2 text-sm text-amber-800">{data.suggestion}</p>
      )}
      <p className="mt-2 text-xs text-slate-400">
        Since this portfolio was created — an approximation, not a date-matched return for each individual
        position.
      </p>
    </div>
  );
}
