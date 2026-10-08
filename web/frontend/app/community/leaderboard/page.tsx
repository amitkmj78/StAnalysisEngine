"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { ApiError, getCommunityLeaderboard } from "@/lib/api";
import type { CommunityLeaderboardEntry } from "@/lib/types";

function fmtPct(v: number | null): string {
  if (v === null) return "—";
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
}

export default function CommunityLeaderboardPage() {
  const [board, setBoard] = useState<CommunityLeaderboardEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getCommunityLeaderboard()
      .then((res) => setBoard(res.leaderboard))
      .catch((err) => setError(err instanceof ApiError ? err.message : "Failed to load the leaderboard."));
  }, []);

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <div className="flex items-center justify-between">
        <h1 className="font-display text-2xl font-semibold text-slate-900">Community Leaderboard</h1>
        <Link href="/community" className="text-sm font-medium text-slate-600 hover:underline">
          Idea Feed
        </Link>
      </div>
      <p className="mt-1 text-sm text-slate-500">
        Ranked by risk-adjusted excess return vs SPY, not raw return alone — an author with too few scored ideas
        still shows their return/sample size, just no rank score, and always sorts last. The app&apos;s own model
        appears here as its own author, scored by the same rules.
      </p>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {board === null && !error && <p className="mt-6 text-sm text-slate-500">Loading…</p>}

      {board !== null && (
        <div className="mt-6 overflow-x-auto rounded-xl border border-slate-200 bg-white">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                <th className="px-3 py-2">Rank</th>
                <th className="px-3 py-2">Author</th>
                <th className="px-3 py-2 text-right">Risk-Adj. Score</th>
                <th className="px-3 py-2 text-right">Avg Excess vs SPY</th>
                <th className="px-3 py-2 text-right">Volatility</th>
                <th className="px-3 py-2 text-right">Hit Rate</th>
                <th className="px-3 py-2 text-right">Ideas (n)</th>
              </tr>
            </thead>
            <tbody>
              {board.map((entry, i) => (
                <tr key={entry.author_user_id ?? "model"} className="border-b border-slate-100 last:border-0">
                  <td className="px-3 py-2 text-slate-400">{i + 1}</td>
                  <td className="px-3 py-2 font-medium text-slate-800">
                    <Link href={`/community/authors/${entry.author_user_id ?? "model"}`} className="hover:underline">
                      {entry.display_name}
                    </Link>
                    {entry.is_model && <span className="ml-1.5 rounded-full bg-indigo-50 px-2 py-0.5 text-[10px] font-semibold text-indigo-700">MODEL</span>}
                  </td>
                  <td className="px-3 py-2 text-right">
                    {entry.score !== null ? entry.score.toFixed(2) : <span className="text-slate-400">not enough data yet</span>}
                  </td>
                  <td className={`px-3 py-2 text-right ${entry.avg_excess_vs_spy_pct >= 0 ? "text-emerald-600" : "text-red-600"}`}>
                    {fmtPct(entry.avg_excess_vs_spy_pct)}
                  </td>
                  <td className="px-3 py-2 text-right text-slate-600">{entry.volatility_pct.toFixed(2)}%</td>
                  <td className="px-3 py-2 text-right text-slate-600">{entry.hit_rate_pct.toFixed(1)}%</td>
                  <td className="px-3 py-2 text-right text-slate-500">{entry.num_ideas}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {board.length === 0 && <p className="px-3 py-6 text-center text-sm text-slate-500">No scored ideas yet.</p>}
        </div>
      )}
    </div>
  );
}
