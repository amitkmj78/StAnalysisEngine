"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { ApiError, getStrategyLeaderboard, listPublishedStrategies } from "@/lib/api";
import type { PublishedStrategySummary, StrategyLeaderboardEntry } from "@/lib/types";

function fmtPct(v: number | null): string {
  if (v === null) return "—";
  return `${v > 0 ? "+" : ""}${v.toFixed(2)}%`;
}

function Leaderboard() {
  const [rows, setRows] = useState<StrategyLeaderboardEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getStrategyLeaderboard()
      .then((res) => setRows(res.leaderboard))
      .catch((err) => setError(err instanceof ApiError ? err.message : "The leaderboard could not be loaded."));
  }, []);

  if (error) return <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>;
  if (rows === null) return <p className="p-5 text-sm text-slate-500">Loading…</p>;
  if (rows.length === 0) return <p className="p-5 text-sm text-slate-500">Nothing published yet.</p>;

  return (
    <div className="mt-6 overflow-x-auto rounded-xl border border-slate-200 bg-white">
      <table className="w-full text-sm">
        <thead className="border-b border-slate-200 bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
          <tr>
            <th className="px-4 py-2">Strategy</th>
            <th className="px-4 py-2">Risk-adj. excess vs SPY</th>
            <th className="px-4 py-2">Forward return</th>
            <th className="px-4 py-2">Excess vs SPY</th>
            <th className="px-4 py-2">Trades</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {rows.map((row) => (
            <tr key={row.id} className={row.eligible ? "" : "text-slate-400"}>
              <td className="px-4 py-2">
                <Link href={`/strategies/published/${row.id}`} className="font-medium text-indigo-600 hover:underline">
                  {row.name}
                </Link>
                <span className="block text-xs text-slate-400">by {row.author_display_name ?? "a member"}</span>
              </td>
              <td className="px-4 py-2 font-mono tabular-nums">
                {row.eligible ? row.risk_adjusted_excess_return?.toFixed(2) ?? "—" : row.reason}
              </td>
              <td className="px-4 py-2 font-mono tabular-nums">{row.eligible ? fmtPct(row.cumulative_return_pct) : "—"}</td>
              <td className="px-4 py-2 font-mono tabular-nums">{row.eligible ? fmtPct(row.excess_return_pct) : "—"}</td>
              <td className="px-4 py-2 font-mono tabular-nums">{row.trades}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Directory() {
  const [rows, setRows] = useState<PublishedStrategySummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listPublishedStrategies()
      .then((res) => setRows(res.published))
      .catch((err) => setError(err instanceof ApiError ? err.message : "Published strategies could not be loaded."));
  }, []);

  return (
    <div className="mt-6 rounded-xl border border-slate-200 bg-white">
      {error && <p className="p-5 text-sm text-red-700">{error}</p>}
      {rows === null && !error && <p className="p-5 text-sm text-slate-500">Loading…</p>}
      {rows?.length === 0 && <p className="p-5 text-sm text-slate-500">Nothing published yet.</p>}
      <ul className="divide-y divide-slate-100">
        {rows?.map((row) => (
          <li key={row.id}>
            <Link href={`/strategies/published/${row.id}`} className="flex items-center justify-between gap-3 p-4 hover:bg-slate-50">
              <span>
                <span className="font-medium text-slate-900">{row.name}</span>
                <span className="ml-2 text-xs text-slate-400">v{row.version}</span>
                <span className="block text-xs text-slate-500">
                  by {row.author_display_name ?? "a member"} · published {new Date(row.published_at).toLocaleDateString()}
                  {row.rules_visibility === "summary_only" && " · rules private"}
                </span>
              </span>
              <span className="text-sm text-indigo-600">View →</span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function PublishedStrategiesPage() {
  const [tab, setTab] = useState<"directory" | "leaderboard">("directory");

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h1 className="font-display text-2xl font-semibold text-slate-900">Published strategies</h1>
        <Link href="/strategies/saved" className="text-sm font-medium text-slate-700 hover:underline">
          ← Your saved strategies
        </Link>
      </div>
      <p className="mt-1 text-sm text-slate-500">
        Published by other members, with their rules (or a plain-language summary when the author keeps rules
        private) and an ongoing forward paper track record from the day each was published. Backtests are past
        prices only; nothing here places an order, real or paper.
      </p>

      <div className="mt-5 flex gap-1 border-b border-slate-200">
        {(["directory", "leaderboard"] as const).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            aria-pressed={tab === t}
            className={`px-3 py-2 text-sm font-medium ${
              tab === t ? "border-b-2 border-indigo-600 text-indigo-600" : "text-slate-500 hover:text-slate-700"
            }`}
          >
            {t === "directory" ? "Directory" : "Leaderboard"}
          </button>
        ))}
      </div>

      {tab === "directory" ? (
        <Directory />
      ) : (
        <>
          <p className="mt-4 text-xs text-slate-500">
            Ranked by forward risk-adjusted excess return vs SPY (the information ratio of each strategy&apos;s own
            daily returns minus SPY&apos;s, over the same window). Strategies need at least 3 months since
            publishing and 30 closed trades to rank; others show &quot;not enough data yet&quot;.
          </p>
          <Leaderboard />
        </>
      )}
    </div>
  );
}
