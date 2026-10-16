"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { ApiError, listEtfs } from "@/lib/api";
import type { EtfSummary } from "@/lib/types";

function plain(v: number | null | undefined, digits = 2) {
  return v === null || v === undefined ? "–" : v.toFixed(digits);
}

type SortKey = "ticker" | "weighted_score" | "expense_ratio_pct" | "coverage_pct";

export default function EtfScreenerPage() {
  const [etfs, setEtfs] = useState<EtfSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("weighted_score");
  const [sortDesc, setSortDesc] = useState(true);

  useEffect(() => {
    listEtfs()
      .then((res) => setEtfs(res.etfs))
      .catch((err) => setError(err instanceof ApiError ? err.message : "ETFs could not be loaded."));
  }, []);

  function toggleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDesc((d) => !d);
    } else {
      setSortKey(key);
      setSortDesc(true);
    }
  }

  const sorted = etfs
    ? [...etfs].sort((a, b) => {
        const av = a[sortKey];
        const bv = b[sortKey];
        if (av === null || av === undefined) return 1;
        if (bv === null || bv === undefined) return -1;
        const cmp = typeof av === "string" ? av.localeCompare(String(bv)) : Number(av) - Number(bv);
        return sortDesc ? -cmp : cmp;
      })
    : null;

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="font-display text-2xl font-semibold text-slate-900 dark:text-slate-50">ETF Screener</h1>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Holdings-weighted scores, expense ratio and tracking error across {etfs?.length ?? "…"} curated ETFs.
            Weighted scores are built from each fund&apos;s top ~10 disclosed holdings only — coverage % shows how
            much of the fund that sample actually represents.
          </p>
        </div>
        <Link href="/stock-finder" className="text-sm font-medium text-slate-700 dark:text-slate-200 hover:underline">
          ← Back to Stock Finder
        </Link>
      </div>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <div className="mt-6 overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900">
        {!sorted && !error && <p className="p-5 text-sm text-slate-500 dark:text-slate-400">Loading…</p>}
        {sorted && (
          <table className="w-full min-w-[40rem] text-sm">
            <thead className="border-b border-slate-200 dark:border-slate-700 text-xs text-slate-400 dark:text-slate-500">
              <tr>
                <th className="cursor-pointer px-4 py-2 text-left font-medium" onClick={() => toggleSort("ticker")}>
                  ETF
                </th>
                <th className="px-4 py-2 text-left font-medium">Category</th>
                <th className="cursor-pointer px-4 py-2 text-right font-medium" onClick={() => toggleSort("weighted_score")}>
                  Weighted Score
                </th>
                <th className="cursor-pointer px-4 py-2 text-right font-medium" onClick={() => toggleSort("coverage_pct")}>
                  Coverage
                </th>
                <th className="cursor-pointer px-4 py-2 text-right font-medium" onClick={() => toggleSort("expense_ratio_pct")}>
                  Expense Ratio
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
              {sorted.map((e) => (
                <tr key={e.ticker} className="hover:bg-slate-50 dark:hover:bg-slate-800/50">
                  <td className="px-4 py-2">
                    <Link href={`/stock-finder/etf/${e.ticker}`} className="font-medium text-[var(--pf-accent)] hover:underline">
                      {e.ticker}
                    </Link>
                    <span className="ml-2 text-xs text-slate-500 dark:text-slate-400">{e.name}</span>
                  </td>
                  <td className="px-4 py-2 text-xs text-slate-500 dark:text-slate-400">{e.category ?? "–"}</td>
                  <td className="px-4 py-2 text-right font-mono">{plain(e.weighted_score, 1)}</td>
                  <td className="px-4 py-2 text-right font-mono text-xs text-slate-500 dark:text-slate-400">
                    {e.coverage_pct.toFixed(0)}% ({e.covered_holdings}/{e.total_holdings})
                  </td>
                  <td className="px-4 py-2 text-right font-mono">{plain(e.expense_ratio_pct)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
