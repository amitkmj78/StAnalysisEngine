"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import MetricLabel from "@/components/MetricLabel";
import { STRATEGY_INFO } from "@/components/strategies/strategyInfo";
import { ApiError, getStrategyScan, startStrategyScan } from "@/lib/api";
import type { StrategyScanResult } from "@/lib/types";

// Scan the starting templates on a random S&P 500 sample. The output is a ranked list of
// candidates to study. Nothing here is a recommendation, and nothing places an order.

function pct(v: number | null | undefined) {
  return v === null || v === undefined ? "–" : `${v > 0 ? "+" : ""}${v.toFixed(1)}%`;
}

export default function StrategyScanPage() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const [done, setDone] = useState(0);
  const [total, setTotal] = useState(0);
  const [result, setResult] = useState<StrategyScanResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [seedInput, setSeedInput] = useState("");

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
      const res = await startStrategyScan(seed);
      setJobId(res.job_id);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "The scan could not be started.");
      setStatus("error");
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h1 className="flex items-center gap-2 text-2xl font-semibold text-slate-900">
          Scan for candidates
          <MetricLabel info={STRATEGY_INFO.scan} />
        </h1>
        <Link href="/strategies" className="text-sm font-medium text-slate-700 hover:underline">← Strategies</Link>
      </div>
      <p className="mt-1 text-sm text-slate-500">
        Tests the starting templates on a random sample of 20 S&amp;P 500 stocks over the last five years. Candidates to study, not recommendations.
      </p>

      <div className="mt-6 flex flex-wrap items-end gap-3 rounded-lg border border-slate-200 bg-white p-5">
        <label className="flex flex-col gap-1 text-xs text-slate-600">
          Sample seed (optional)
          <input value={seedInput} onChange={(e) => setSeedInput(e.target.value.replace(/[^0-9]/g, ""))} placeholder="random" className="input w-32 py-1 text-sm" />
        </label>
        <button type="button" onClick={handleStart} disabled={status === "running"} className="rounded-md bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800 disabled:opacity-50">
          {status === "running" ? "Scanning…" : "Start scan"}
        </button>
        {status === "running" && (
          <p className="text-sm text-slate-600">
            Testing template {Math.min(done + 1, total)} of {total}. This takes a minute or two.
          </p>
        )}
      </div>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {result && (
        <div className="mt-6 flex flex-col gap-4">
          <p className="text-xs text-slate-500">
            Seed {result.seed} · {result.sample_size} stocks · {result.period.start} to {result.period.end} · {result.variants_tried} templates tried
          </p>
          <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white p-5">
            <table className="w-full min-w-[44rem] text-right text-sm">
              <thead className="text-xs text-slate-400">
                <tr>
                  <th className="py-1 text-left font-medium">Rank</th>
                  <th className="py-1 text-left font-medium">Template</th>
                  <th className="font-medium">
                    <span className="inline-flex items-center gap-1">Later dates, extra return<MetricLabel info={STRATEGY_INFO["in-sample"]} /></span>
                  </th>
                  <th className="font-medium">Earlier dates, extra return</th>
                  <th className="font-medium">
                    <span className="inline-flex items-center gap-1">Could this be luck?<MetricLabel info={STRATEGY_INFO["deflated sharpe"]} /></span>
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono text-xs">
                {result.candidates.map((c) => (
                  <tr key={c.key}>
                    <td className="py-2 text-left font-sans text-sm text-slate-700">{c.rank}</td>
                    <td className="py-2 text-left font-sans text-sm font-medium text-slate-800">
                      {c.name}
                      {c.warnings.map((w) => (
                        <span key={w} className="mt-0.5 block font-sans text-xs font-normal text-amber-700">{w}</span>
                      ))}
                    </td>
                    <td>{pct(c.oos_excess_return_pct)}</td>
                    <td>{pct(c.in_sample_excess_return_pct)}</td>
                    <td>{c.deflated_probability != null ? `${Math.round(c.deflated_probability * 100)}%` : "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">{result.note}</p>
          <p className="text-xs text-slate-500">
            Want to test one of these yourself? Open it in the <Link href="/strategies/builder" className="font-medium text-slate-700 hover:underline">builder</Link> and pick your own stocks.
          </p>
        </div>
      )}
    </div>
  );
}
