"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import MetricLabel from "@/components/MetricLabel";
import { STRATEGY_INFO } from "@/components/strategies/strategyInfo";
import { ApiError, getStrategyScan, startStrategyScan } from "@/lib/api";
import type { StrategyScanGroup, StrategyScanResult } from "@/lib/types";

// Scan the starting templates on a random S&P 500 sample. The output is two short lists of
// candidates to study, one for short-term and one for long-term holds. Nothing here is a
// recommendation, and nothing places an order.

function pct(v: number | null | undefined) {
  return v === null || v === undefined ? "–" : `${v > 0 ? "+" : ""}${v.toFixed(1)}%`;
}

function plain(v: number | null | undefined, digits = 2) {
  return v === null || v === undefined ? "–" : v.toFixed(digits);
}

function GroupList({
  title,
  group,
  benchmark,
}: {
  title: string;
  group: StrategyScanGroup;
  benchmark?: { name: string; oos_cagr_pct: number | null; oos_max_drawdown_pct: number | null; full_cagr_pct: number | null; full_max_drawdown_pct: number | null };
}) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5">
      <h2 className="text-sm font-semibold text-slate-900">{title}</h2>
      <p className="mt-1 text-xs text-slate-500">
        Ranked by {group.ranked_by.toLowerCase()}. Minimum: {group.minimum.toLowerCase()}.
      </p>
      {benchmark && (
        <p className="mt-1 text-xs text-slate-500">
          S&amp;P 500 over the full five years: {pct(benchmark.full_cagr_pct)} a year, worst fall {pct(benchmark.full_max_drawdown_pct)}.
        </p>
      )}
      {group.candidates.length === 0 ? (
        <p className="mt-3 rounded-md bg-slate-50 px-3 py-2 text-sm text-slate-600">{group.message}</p>
      ) : (
        <ol className="mt-3 flex flex-col gap-3">
          {group.candidates.map((c) => (
            <li key={c.key} className="rounded-md border border-slate-200 p-3">
              <p className="font-medium text-slate-900">
                {c.rank}. {c.name}
              </p>
              <p className="mt-1 text-xs text-slate-600">{c.profile}</p>
              {c.warnings.map((w) => (
                <p key={w} className="mt-1 text-xs text-amber-700">
                  {w}
                </p>
              ))}
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

export default function StrategyScanPage() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const [done, setDone] = useState(0);
  const [total, setTotal] = useState(0);
  const [result, setResult] = useState<StrategyScanResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [seedInput, setSeedInput] = useState("");
  // SCAN-2: off by default. On tests today's index members back in time, which flatters every result.
  const [currentOnly, setCurrentOnly] = useState(false);

  useEffect(() => {
    if (!jobId) return;
    const timer = setInterval(async () => {
      try {
        const s = await getStrategyScan(jobId);
        setDone(s.done);
        setTotal(s.total);
        if (s.status === "done") {
          setResult(s.result);
          setStatus("done");
          clearInterval(timer);
        } else if (s.status === "error") {
          setError(s.error ?? "The scan failed.");
          setStatus("error");
          clearInterval(timer);
        }
      } catch (err) {
        setError(err instanceof ApiError ? err.message : "Lost contact with the scan.");
        setStatus("error");
        clearInterval(timer);
      }
    }, 2000);
    return () => clearInterval(timer);
  }, [jobId]);

  async function handleStart() {
    setError(null);
    setResult(null);
    setDone(0);
    setStatus("running");
    try {
      const seed = seedInput.trim() ? Number(seedInput) : undefined;
      const res = await startStrategyScan(seed, currentOnly);
      setJobId(res.job_id);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "The scan could not be started.");
      setStatus("error");
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h1 className="font-display flex items-center gap-2 text-2xl font-semibold text-slate-900">
          Scan for candidates
          <MetricLabel info={STRATEGY_INFO.scan} />
        </h1>
        <Link href="/strategies" className="text-sm font-medium text-slate-700 hover:underline">
          ← Strategies
        </Link>
      </div>
      <p className="mt-1 text-sm text-slate-500">
        Tests the starting templates on a random sample of 20 S&amp;P 500 stocks over the last five years. Candidates to study, not recommendations.
      </p>

      <div className="mt-6 flex flex-wrap items-end gap-3 rounded-xl border border-slate-200 bg-white p-5">
        <label className="flex flex-col gap-1 text-xs text-slate-600">
          Sample seed (optional)
          <input
            value={seedInput}
            onChange={(e) => setSeedInput(e.target.value.replace(/[^0-9]/g, ""))}
            placeholder="random"
            className="input w-32 py-1 text-sm"
          />
        </label>
        <label className="flex items-center gap-2 pb-1 text-xs text-slate-600">
          <input type="checkbox" checked={currentOnly} onChange={(e) => setCurrentOnly(e.target.checked)} />
          Current members only (biased)
        </label>
        <button
          type="button"
          onClick={handleStart}
          disabled={status === "running"}
          className="rounded-md bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800 disabled:opacity-50"
        >
          {status === "running" ? "Scanning…" : "Start scan"}
        </button>
        {status === "running" && (
          <p className="text-sm text-slate-600">Testing template {Math.min(done + 1, total)} of {total}. This takes a minute or two.</p>
        )}
      </div>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {result && (
        <div className="mt-6 flex flex-col gap-5">
          <p className="text-xs text-slate-500">
            Seed {result.seed} · {result.sample_size} stocks · {result.period.start} to {result.period.end} · later{" "}
            {result.out_of_sample.years.toFixed(1)} years measured · {result.variants_tried} templates tried
          </p>
          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Stocks in this random sample</p>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {result.tickers.map((t) => (
                <span key={t} className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-700">
                  {t}
                </span>
              ))}
            </div>
          </div>
          {result.universe && (
            <div className="rounded-md bg-slate-50 px-3 py-2 text-xs text-slate-600">
              <p>
                {result.universe.basis === "point_in_time"
                  ? <>Universe: point-in-time S&amp;P 500 as of {result.universe.as_of}.</>
                  : <>Universe: <strong>current members only (biased)</strong>. Stocks that left the index are not in the sample, which flatters the results.</>}
                {" "}{result.universe.members_at_start} members on that date.
              </p>
              {result.universe.basis === "point_in_time" && result.universe.left_index_in_window.length > 0 && (
                <p className="mt-1">Left the index since then ({result.universe.left_index_in_window.length}): {result.universe.left_index_in_window.join(", ")}.</p>
              )}
              {result.universe.skipped_no_prices && result.universe.skipped_no_prices.length > 0 && (
                <p className="mt-1">Skipped, no price history: {result.universe.skipped_no_prices.join(", ")}.</p>
              )}
            </div>
          )}
          <p className="rounded-md bg-slate-50 px-3 py-2 text-sm text-slate-700">
            Holding these {result.sample_size} stocks, equal weight: <span className="font-mono font-semibold">{pct(result.holding_oos_return_pct)}</span> out-of-sample ({result.out_of_sample.start} to {result.out_of_sample.end}).
          </p>
          {!result.candidates.some((c) => c.passes_short_test) && (
            <p className="rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-800">
              <strong>None of these templates beat holding the same stocks on the later dates.</strong> Treat the lists below as a record of how the templates did on this sample, not as ideas to use.
            </p>
          )}

          <GroupList title="Short-term candidates" group={result.groups.short_term} />
          <GroupList title="Long-term candidates" group={result.groups.long_term} benchmark={result.benchmark} />

          <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white p-5">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">All templates</p>
            <table className="w-full min-w-[52rem] text-right text-sm">
              <thead className="text-xs text-slate-400">
                <tr>
                  <th className="py-1 text-left font-medium">Template</th>
                  <th className="font-medium">Holding</th>
                  <th className="font-medium">Avg hold (days)</th>
                  <th className="font-medium">Trades</th>
                  <th className="font-medium">
                    <span className="inline-flex items-center gap-1">
                      Later dates, after costs
                      <MetricLabel info={STRATEGY_INFO["in-sample"]} />
                    </span>
                  </th>
                  <th className="font-medium">Holding return</th>
                  <th className="font-medium">Difference (pts)</th>
                  <th className="font-medium">Sharpe</th>
                  <th className="font-medium">Max DD</th>
                  <th className="font-medium">Calmar</th>
                  <th className="font-medium">Time in market</th>
                  <th className="font-medium">Return per 100% invested</th>
                  <th className="font-medium">
                    <span className="inline-flex items-center gap-1">
                      Chance Sharpe above zero
                      <MetricLabel info={STRATEGY_INFO["deflated sharpe"]} />
                    </span>
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono text-xs">
                <tr className="bg-slate-50 font-semibold">
                  <td className="py-2 text-left font-sans text-sm text-slate-900">{result.holding.label}</td>
                  <td className="font-sans">Holding</td>
                  <td>–</td>
                  <td>–</td>
                  <td>{pct(result.holding.total_return_pct)}</td>
                  <td>{pct(result.holding.total_return_pct)}</td>
                  <td>–</td>
                  <td>{plain(result.holding.sharpe)}</td>
                  <td>{pct(result.holding.max_drawdown_pct)}</td>
                  <td>{plain(result.holding.calmar)}</td>
                  <td>{result.holding.time_in_market_pct}%</td>
                  <td>{pct(result.holding.exposure_adjusted_cagr_pct)}</td>
                  <td>–</td>
                </tr>
                {result.candidates.map((c) => (
                  <tr key={c.key}>
                    <td className="py-2 text-left font-sans text-sm font-medium text-slate-800">{c.name}</td>
                    <td className="font-sans capitalize">{c.holding_class === "none" ? "–" : c.holding_class}</td>
                    <td>{c.avg_hold_days ?? "–"}</td>
                    <td>{c.trades_oos}</td>
                    <td>{pct(c.oos_return_after_costs_pct)}</td>
                    <td>{pct(c.oos_holding_return_pct)}</td>
                    <td>{c.oos_difference_pts > 0 ? "+" : ""}{c.oos_difference_pts.toFixed(1)}</td>
                    <td>{plain(c.oos_sharpe)}</td>
                    <td>{pct(c.oos_max_drawdown_pct)}</td>
                    <td>{plain(c.oos_calmar)}</td>
                    <td>{c.time_in_market_pct != null ? `${c.time_in_market_pct}%` : "–"}</td>
                    <td>{pct(c.exposure_adjusted_cagr_pct)}</td>
                    <td>{c.deflated_probability != null ? `${Math.round(c.deflated_probability * 100)}%` : "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">{result.note}</p>
          <p className="text-xs text-slate-500">
            Want to test one of these yourself? Open it in the{" "}
            <Link href="/strategies/builder" className="font-medium text-slate-700 hover:underline">
              builder
            </Link>{" "}
            and pick your own stocks.
          </p>
        </div>
      )}
    </div>
  );
}
