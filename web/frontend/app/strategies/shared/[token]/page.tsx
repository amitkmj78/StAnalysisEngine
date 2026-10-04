"use client";

import Link from "next/link";
import { use, useEffect, useState } from "react";
import { ApiError, getSharedStrategy } from "@/lib/api";
import type { StrategySharedResponse } from "@/lib/types";

// Read-only view of a shared strategy run. Shows the saved result only; no account details.

function pct(v: number | null | undefined, digits = 1) {
  return v === null || v === undefined ? "–" : `${v > 0 ? "+" : ""}${v.toFixed(digits)}%`;
}
function plain(v: number | null | undefined, digits = 2) {
  return v === null || v === undefined ? "–" : v.toFixed(digits);
}

export default function SharedStrategyPage({ params }: { params: Promise<{ token: string }> }) {
  const { token } = use(params);
  const [data, setData] = useState<StrategySharedResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getSharedStrategy(token)
      .then(setData)
      .catch((err) => setError(err instanceof ApiError ? err.message : "This link is not available."));
  }, [token]);

  if (error) return <div className="mx-auto max-w-3xl px-4 py-10 text-sm text-red-700">{error}</div>;
  if (!data) return <div className="mx-auto max-w-3xl px-4 py-10 text-sm text-slate-500">Loading…</div>;

  const r = data.result;
  const tickers = (data.definition.tickers as string[] | undefined) ?? r.tickers;
  const verdict = r.verdict.beats_benchmark_after_costs && (r.verdict.sharpe_vs_basket ?? 0) > 0
    ? "The rules beat holding the same stocks after costs, and on risk-adjusted return too."
    : r.verdict.beats_benchmark_after_costs
      ? "The rules beat the benchmark on return after costs, but not after adjusting for risk."
      : "The rules did not beat the benchmark after costs.";

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <p className="text-xs uppercase tracking-wide text-slate-400">Shared strategy · read-only</p>
      <h1 className="mt-1 text-2xl font-semibold text-slate-900">{data.name}</h1>
      <p className="mt-1 text-sm text-slate-500">
        {r.period.start} to {r.period.end} · {tickers.join(", ")} · {r.trades} trades
      </p>

      <section className="mt-6 rounded-lg border border-slate-200 bg-white p-5">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Did the rules add value?</p>
        <p className="mt-2 text-lg font-medium text-slate-900">{verdict}</p>
        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
          <div className="rounded-md border border-slate-200 p-3">
            <p className="text-xs text-slate-500">Strategy CAGR</p>
            <p className="mt-1 font-mono text-lg">{pct(r.strategy.cagr_pct)}</p>
          </div>
          <div className="rounded-md border border-slate-200 p-3">
            <p className="text-xs text-slate-500">Same stocks, held</p>
            <p className="mt-1 font-mono text-lg">{pct(r.basket.cagr_pct)}</p>
          </div>
          <div className="rounded-md border border-slate-200 p-3">
            <p className="text-xs text-slate-500">Sharpe: strategy vs SPY</p>
            <p className="mt-1 font-mono text-lg">{plain(r.strategy.sharpe)} vs {plain(r.benchmark_spy.sharpe)}</p>
          </div>
        </div>
        <ul className="mt-4 flex flex-col gap-2">
          {r.checks.map((c) => (
            <li key={c.label} className="flex items-start gap-2 text-xs">
              <span className="shrink-0 rounded px-1.5 py-0.5 font-semibold text-slate-700 ring-1 ring-slate-300">{c.status.toUpperCase()}</span>
              <span>
                <span className="font-medium text-slate-800">{c.label}.</span> <span className="text-slate-600">{c.detail}</span>
              </span>
            </li>
          ))}
        </ul>
      </section>

      <p className="mt-4 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
        Backtest of past prices with the rules and costs shown. Not a forecast, not a recommendation, and not an order: nothing is placed.
      </p>
      <p className="mt-6 text-xs text-slate-500">
        Want to build your own? <Link href="/strategies/builder" className="font-medium text-slate-700 hover:underline">Open the strategy builder</Link>.
      </p>
    </div>
  );
}
