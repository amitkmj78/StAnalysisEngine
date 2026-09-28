"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";

import { ApiError, getTwoScore, getTwoScoreHistory, getTwoScoreWeeklyChange } from "@/lib/api";
import type { TwoScoreHistoryResponse, TwoScoreResponse, TwoScoreWeeklyChangeResponse } from "@/lib/types";
import Sparkline from "@/components/portfolio/Sparkline";

function signalBadgeClass(signal: string): string {
  if (signal === "Buy") return "bg-emerald-50 text-emerald-700";
  if (signal === "Trim") return "bg-red-50 text-red-700";
  return "bg-slate-100 text-slate-600";
}

function confidenceClass(label: string): string {
  if (label === "high") return "text-emerald-700";
  if (label === "medium") return "text-amber-700";
  if (label === "low") return "text-red-700";
  return "text-slate-400";
}

function ScoreCard({
  title,
  horizon,
  score,
  signal,
  confidence,
  sectorPercentile,
  sectorKey,
  trend,
}: {
  title: string;
  horizon: string;
  score: number | null;
  signal: string;
  confidence: { score: number | null; label: string };
  sectorPercentile: number | null;
  sectorKey: string;
  trend?: { weekly_series: [string, number][]; flagged: boolean; change_pts: number | null };
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-slate-900">{title}</h2>
          <p className="text-xs text-slate-500">{horizon}</p>
        </div>
        <span className={`rounded-full px-3 py-1 text-sm font-semibold ${signalBadgeClass(signal)}`}>{signal}</span>
      </div>
      <div className="mt-3 flex items-end gap-2">
        <span className="text-4xl font-semibold text-slate-900">{score !== null ? score.toFixed(0) : "—"}</span>
        <span className="pb-1 text-sm text-slate-400">/ 100</span>
      </div>
      <p className="mt-1 text-xs text-slate-500">
        {sectorPercentile !== null
          ? `Top ${Math.max(1, Math.round(100 - sectorPercentile))}% of ${sectorKey}`
          : `No sector rank yet — ${sectorKey}`}
      </p>
      <p className={`mt-1 text-xs font-medium ${confidenceClass(confidence.label)}`}>
        Confidence: {confidence.label}
        {confidence.score !== null ? ` (${confidence.score.toFixed(0)})` : ""}
      </p>
      {trend && trend.weekly_series.length >= 2 && (
        <div className="mt-3 flex items-center gap-2">
          <Sparkline values={trend.weekly_series.map(([, v]) => v)} color="#1F4FD1" />
          {trend.flagged && trend.change_pts !== null && (
            <span className={`text-xs font-semibold ${trend.change_pts >= 0 ? "text-emerald-700" : "text-red-700"}`}>
              {trend.change_pts >= 0 ? "+" : ""}
              {trend.change_pts.toFixed(0)} pts this week
            </span>
          )}
        </div>
      )}
    </div>
  );
}

export default function StockScorePage() {
  const params = useParams<{ ticker: string }>();
  const router = useRouter();
  const ticker = (params.ticker as string)?.toUpperCase() ?? "";

  const [jumpTicker, setJumpTicker] = useState("");
  const [data, setData] = useState<TwoScoreResponse | null>(null);
  const [history, setHistory] = useState<TwoScoreHistoryResponse | null>(null);
  const [weeklyChange, setWeeklyChange] = useState<TwoScoreWeeklyChangeResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!ticker) return;
    setLoading(true);
    setError(null);
    setData(null);
    Promise.all([getTwoScore(ticker), getTwoScoreHistory(ticker).catch(() => null), getTwoScoreWeeklyChange(ticker).catch(() => null)])
      .then(([score, hist, weekly]) => {
        setData(score);
        setHistory(hist);
        setWeeklyChange(weekly);
      })
      .catch((err) => {
        setError(
          err instanceof ApiError && err.status === 404
            ? `No score on record yet for ${ticker}. Scores are computed nightly after market close — check back after the next trading day.`
            : err instanceof ApiError
            ? err.message
            : "Could not load this stock's score.",
        );
      })
      .finally(() => setLoading(false));
  }, [ticker]);

  function handleJump(e: React.FormEvent) {
    e.preventDefault();
    const t = jumpTicker.trim().toUpperCase();
    if (t) router.push(`/stock/${t}`);
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Stock Score</p>
          <h1 className="text-2xl font-semibold text-slate-900">{ticker}</h1>
          {data && (
            <p className="text-xs text-slate-500">
              {data.sector_key} · as of {data.as_of_date}
            </p>
          )}
        </div>
        <form onSubmit={handleJump} className="flex gap-2">
          <input
            value={jumpTicker}
            onChange={(e) => setJumpTicker(e.target.value.toUpperCase())}
            placeholder="Jump to ticker…"
            className="w-32 rounded-md border border-slate-300 px-3 py-1.5 text-sm uppercase"
          />
          <button
            type="submit"
            className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-100"
          >
            Go
          </button>
        </form>
      </div>

      {loading && <p className="mt-6 text-sm text-slate-500">Loading…</p>}
      {error && <p className="mt-6 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {data && (
        <>
          <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2">
            <ScoreCard
              title="Short-Term Score"
              horizon="10-90 day horizon"
              score={data.short_score}
              signal={data.short_signal}
              confidence={data.short_confidence}
              sectorPercentile={data.short_sector_percentile}
              sectorKey={data.sector_key}
              trend={history?.short_term}
            />
            <ScoreCard
              title="Long-Term Score"
              horizon="1-3 year horizon"
              score={data.long_score}
              signal={data.long_signal}
              confidence={data.long_confidence}
              sectorPercentile={data.long_sector_percentile}
              sectorKey={data.sector_key}
              trend={history?.long_term}
            />
          </div>

          <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="rounded-lg border border-slate-200 bg-white p-5">
              <h3 className="text-sm font-semibold text-slate-900">Top Drivers</h3>
              {data.explanations.drivers.length === 0 ? (
                <p className="mt-2 text-xs text-slate-400">No positive drivers today.</p>
              ) : (
                <ul className="mt-2 flex flex-col gap-1 text-sm text-emerald-700">
                  {data.explanations.drivers.map((d) => (
                    <li key={d.factor}>
                      +{d.contribution.toFixed(1)} pts — {d.factor}
                    </li>
                  ))}
                </ul>
              )}
            </div>
            <div className="rounded-lg border border-slate-200 bg-white p-5">
              <h3 className="text-sm font-semibold text-slate-900">Top Drags</h3>
              {data.explanations.drags.length === 0 ? (
                <p className="mt-2 text-xs text-slate-400">No negative drags today.</p>
              ) : (
                <ul className="mt-2 flex flex-col gap-1 text-sm text-red-700">
                  {data.explanations.drags.map((d) => (
                    <li key={d.factor}>
                      {d.contribution.toFixed(1)} pts — {d.factor}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>

          {weeklyChange?.change && (
            <div className="mt-4 rounded-md bg-blue-50 px-3 py-2 text-sm text-blue-800">
              Since {weeklyChange.compared_to}: {weeklyChange.change.factor} moved the most (
              {weeklyChange.change.delta_contribution >= 0 ? "+" : ""}
              {weeklyChange.change.delta_contribution.toFixed(1)} pts contribution).
            </div>
          )}

          <div className="mt-6 rounded-lg border border-slate-200 bg-white p-5">
            <h3 className="text-sm font-semibold text-slate-900">Why these scores</h3>
            <ul className="mt-2 flex flex-col gap-2 text-sm text-slate-700">
              <li>{data.sentences.momentum}</li>
              <li>{data.sentences.reversal}</li>
              <li>{data.sentences.value}</li>
              <li>{data.sentences.growth}</li>
              <li>{data.sentences.low_vol}</li>
            </ul>
            <p className="mt-3 text-xs text-slate-400">
              Rules-based composite scores from real, already-captured data — not a trained prediction model.
              Earnings-revisions/surprise and Quality factors aren&apos;t included yet (no data source for them exists).
            </p>
          </div>
        </>
      )}
    </div>
  );
}
