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
  if (error) return <p className="mt-3 rounded-md border border-[#e4c9c5] bg-[#fbeceb] px-3 py-2 text-sm text-[#a23b34]">{error}</p>;
  // Renders as long as there's at least one real stat to show -- the
  // since-inception comparison (gap_pct) needs the portfolio's creation
  // date priced, but benchmark_today_pct only needs today's quotes and
  // shouldn't disappear just because the other one failed.
  if (!data || (data.gap_pct === null && data.benchmark_today_pct === null)) return null;

  const positive = data.gap_pct !== null && data.gap_pct >= 0;

  return (
    <div
      className={`mt-3 rounded-xl border p-4 ${
        data.underperforming ? "border-[#e3cf9c] bg-[#faf3e2]" : "border-[#ddd8cd] bg-white"
      }`}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-4 text-sm">
          <span className="font-semibold text-[#1f2420]">vs. S&amp;P 500 ({data.benchmark_ticker})</span>
          <span className="text-[#514c43]">
            {data.benchmark_ticker} today{" "}
            <strong
              className={
                data.benchmark_today_pct === null
                  ? undefined
                  : data.benchmark_today_pct >= 0
                  ? "text-[#2f6b4f]"
                  : "text-[#a23b34]"
              }
            >
              {fmtPct(data.benchmark_today_pct)}
            </strong>
          </span>
          <span className="text-[#514c43]">
            Portfolio <strong>{fmtPct(data.portfolio_return_pct)}</strong>
          </span>
          <span className="text-[#514c43]">
            {data.benchmark_ticker} <strong>{fmtPct(data.benchmark_return_pct)}</strong>
          </span>
        </div>
        {data.gap_pct !== null && (
          <span
            className={`rounded-full px-2.5 py-1 text-xs font-semibold ${
              positive ? "bg-[#e3ede8] text-[#2f6b4f]" : "bg-[#f6e5e3] text-[#a23b34]"
            }`}
          >
            {positive ? "Ahead by" : "Trailing by"} {Math.abs(data.gap_pct).toFixed(1)} pts
          </span>
        )}
      </div>
      {data.underperforming && data.suggestion && (
        <p className="mt-2 text-sm text-[#8a6417]">{data.suggestion}</p>
      )}
      <p className="mt-2 text-xs text-[#a39b8b]">
        {data.benchmark_ticker} today is a real day-over-day move. Portfolio vs. {data.benchmark_ticker} is since
        this portfolio was created — an approximation, not a date-matched return for each individual position.
      </p>
    </div>
  );
}
