"use client";

import { useEffect, useState } from "react";

import { getPaperTradingPerformance } from "@/lib/api";
import type { PaperTradingPerformance } from "@/lib/types";

function pct(v: number | null | undefined, digits = 1) {
  return v === null || v === undefined ? "–" : `${v > 0 ? "+" : ""}${v.toFixed(digits)}%`;
}
function plain(v: number | null | undefined, digits = 2) {
  return v === null || v === undefined ? "–" : v.toFixed(digits);
}

// BRK-5/PPR-1: the paper account's own real equity curve vs SPY and the
// app's model portfolio -- labelled "Live paper" the same way STB-6's
// strategy track record is, since this is a real linked account's actual
// fills, not a replay.
export default function PaperTrackRecord({ portfolioId }: { portfolioId: number }) {
  const [data, setData] = useState<PaperTradingPerformance | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getPaperTradingPerformance(portfolioId)
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch(() => {
        if (!cancelled) setData(null);
      })
      .finally(() => {
        if (!cancelled) setLoaded(true);
      });
    return () => {
      cancelled = true;
    };
  }, [portfolioId]);

  if (!loaded) return null;
  if (!data || !data.available) return null;

  const hasAccountData = (data.account?.days_of_data ?? 0) >= 2;

  return (
    <div className="mt-4 rounded-md border border-slate-200 dark:border-slate-700 p-3">
      <div className="flex items-center gap-2">
        <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-50">Track record</h3>
        <span className="rounded-full bg-[var(--pf-accent-soft)] px-2 py-0.5 text-xs font-medium text-[var(--pf-accent)]">
          Live paper
        </span>
      </div>
      {hasAccountData ? (
        <dl className="mt-2 grid grid-cols-2 gap-2 text-sm sm:grid-cols-3">
          <div>
            <dt className="text-xs text-slate-500 dark:text-slate-400">Return</dt>
            <dd className="font-mono">{pct(data.account?.return_pct)}</dd>
          </div>
          <div>
            <dt className="text-xs text-slate-500 dark:text-slate-400">Max drawdown</dt>
            <dd className="font-mono">{pct(data.account?.max_drawdown_pct)}</dd>
          </div>
          <div>
            <dt className="text-xs text-slate-500 dark:text-slate-400">Sharpe</dt>
            <dd className="font-mono">{plain(data.account?.sharpe)}</dd>
          </div>
          <div>
            <dt className="text-xs text-slate-500 dark:text-slate-400">vs SPY</dt>
            <dd className="font-mono">
              {data.spy_return_pct !== null && data.spy_return_pct !== undefined && data.account?.return_pct !== null && data.account?.return_pct !== undefined
                ? pct(data.account.return_pct - data.spy_return_pct)
                : "–"}
            </dd>
          </div>
          <div>
            <dt className="text-xs text-slate-500 dark:text-slate-400">vs model portfolio</dt>
            <dd className="font-mono">
              {data.model_portfolio?.available && data.account?.return_pct !== null && data.account?.return_pct !== undefined && data.model_portfolio.total_return_pct !== null && data.model_portfolio.total_return_pct !== undefined
                ? pct(data.account.return_pct - data.model_portfolio.total_return_pct)
                : "–"}
            </dd>
          </div>
          <div>
            <dt className="text-xs text-slate-500 dark:text-slate-400">Days of data</dt>
            <dd className="font-mono">{data.account?.days_of_data}</dd>
          </div>
        </dl>
      ) : (
        <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
          Not enough days of real equity history yet{data.account ? ` (${data.account.days_of_data} so far)` : ""} — check
          back after a few more trading days.
        </p>
      )}
      <p className="mt-2 text-xs text-slate-400 dark:text-slate-500">
        This account&apos;s own real fills and balance over time — not a backtest or a forecast.
      </p>
    </div>
  );
}
