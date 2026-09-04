"use client";

import { useEffect, useState } from "react";

import { getPortfolioPerformance } from "@/lib/api";
import type { PortfolioPerformanceRow } from "@/lib/types";

// Same key PortfolioSwitcher.tsx persists the user's last-selected
// portfolio under — read directly here instead of requiring this widget
// to sit inside a page that renders <PortfolioSwitcher>. Falls back to
// the user's default portfolio (server-side, when portfolio_id is
// omitted) if nothing's stored yet.
const STORAGE_KEY = "stanalysisengine.selectedPortfolioId";

export default function PortfolioMoversWidget() {
  const [winner, setWinner] = useState<PortfolioPerformanceRow | null>(null);
  const [loser, setLoser] = useState<PortfolioPerformanceRow | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const stored = Number(localStorage.getItem(STORAGE_KEY));
    const portfolioId = Number.isFinite(stored) && stored > 0 ? stored : undefined;

    getPortfolioPerformance(30, portfolioId)
      .then((res) => {
        if (cancelled) return;
        // day_gain_pct is null for positions with no previous-close on
        // file (see services/portfolio_performance_service.py) — those
        // can't be ranked as a winner/loser for today.
        const ranked = res.rows.filter((r) => r.day_gain_pct !== null);
        if (ranked.length === 0) {
          setLoaded(true);
          return;
        }
        const sorted = [...ranked].sort((a, b) => (b.day_gain_pct ?? 0) - (a.day_gain_pct ?? 0));
        setWinner(sorted[0]);
        setLoser(sorted[sorted.length - 1]);
        setLoaded(true);
      })
      .catch(() => setLoaded(true));

    return () => {
      cancelled = true;
    };
  }, []);

  if (!loaded || (!winner && !loser)) return null;

  // A single-position portfolio has one ticker as both "winner" and
  // "loser" of the same value — showing it twice would just be noise.
  const sameTicker = winner && loser && winner.ticker === loser.ticker;

  return (
    <div className="mb-4 flex flex-wrap gap-3 text-sm">
      {winner && (
        <div className="flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-emerald-700">
            Today&apos;s Winner
          </span>
          <span className="font-semibold text-emerald-800">{winner.ticker}</span>
          {winner.day_gain_pct !== null && (
            <span className="font-medium text-emerald-700">
              {winner.day_gain_pct >= 0 ? "+" : ""}
              {winner.day_gain_pct.toFixed(2)}%
            </span>
          )}
        </div>
      )}
      {loser && !sameTicker && (
        <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-red-700">Today&apos;s Loser</span>
          <span className="font-semibold text-red-800">{loser.ticker}</span>
          {loser.day_gain_pct !== null && (
            <span className="font-medium text-red-700">{loser.day_gain_pct.toFixed(2)}%</span>
          )}
        </div>
      )}
    </div>
  );
}
