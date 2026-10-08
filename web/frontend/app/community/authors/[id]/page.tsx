"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { ApiError, followAuthor, getCommunityAuthorProfile, unfollowAuthor } from "@/lib/api";
import type { CommunityAuthorProfile } from "@/lib/types";

function fmtPct(v: number | null): string {
  if (v === null) return "—";
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
}

export default function CommunityAuthorProfilePage() {
  const params = useParams<{ id: string }>();
  const authorId = params.id;

  const [profile, setProfile] = useState<CommunityAuthorProfile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [following, setFollowing] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getCommunityAuthorProfile(authorId)
      .then(setProfile)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Failed to load this profile."));
  }, [authorId]);

  async function handleFollowToggle() {
    setBusy(true);
    try {
      if (following) {
        await unfollowAuthor(authorId);
        setFollowing(false);
      } else {
        await followAuthor(authorId);
        setFollowing(true);
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not update your follow status.");
    } finally {
      setBusy(false);
    }
  }

  if (error) return <div className="mx-auto max-w-3xl px-4 py-8 text-sm text-red-700">{error}</div>;
  if (!profile) return <div className="mx-auto max-w-3xl px-4 py-8 text-sm text-slate-500">Loading…</div>;

  const isModel = authorId === "model";

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <div className="flex items-center justify-between">
        <h1 className="font-display text-2xl font-semibold text-slate-900">
          {profile.display_name}
          {isModel && <span className="ml-2 rounded-full bg-indigo-50 px-2 py-0.5 text-xs font-semibold text-indigo-700">MODEL</span>}
        </h1>
        <div className="flex gap-3">
          {!isModel && (
            <button
              onClick={handleFollowToggle}
              disabled={busy}
              className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
            >
              {following ? "Unfollow" : "Follow"}
            </button>
          )}
          <Link href="/community/leaderboard" className="text-sm font-medium text-slate-600 hover:underline">
            Leaderboard
          </Link>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div className="rounded-md border border-slate-200 bg-white p-3">
          <p className="text-xs text-slate-500">Ideas (scored / total)</p>
          <p className="text-lg font-semibold text-slate-900">{profile.num_ideas_scored} / {profile.num_ideas_total}</p>
        </div>
        <div className="rounded-md border border-slate-200 bg-white p-3">
          <p className="text-xs text-slate-500">Hit Rate</p>
          <p className="text-lg font-semibold text-slate-900">{profile.hit_rate_pct !== null ? `${profile.hit_rate_pct.toFixed(1)}%` : "—"}</p>
        </div>
        <div className="rounded-md border border-slate-200 bg-white p-3">
          <p className="text-xs text-slate-500">Avg Excess vs SPY</p>
          <p className="text-lg font-semibold text-slate-900">{fmtPct(profile.avg_excess_vs_spy_pct)}</p>
        </div>
        <div className="rounded-md border border-slate-200 bg-white p-3">
          <p className="text-xs text-slate-500">Risk-Adj. Score</p>
          <p className="text-lg font-semibold text-slate-900">{profile.score !== null ? profile.score.toFixed(2) : "not enough data yet"}</p>
        </div>
      </div>

      {profile.worst_idea && (
        <div className="mt-4 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-800">
          Worst idea: {profile.worst_idea.direction} {profile.worst_idea.ticker} — {fmtPct(profile.worst_idea.realized_return_pct)}
        </div>
      )}

      <h2 className="mt-8 font-semibold text-slate-900">All ideas</h2>
      <div className="mt-2 flex flex-col gap-2">
        {profile.ideas.map((idea) => (
          <div key={idea.id} className="rounded-md border border-slate-200 bg-white p-3 text-sm">
            <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${idea.direction === "LONG" ? "bg-emerald-50 text-emerald-700" : "bg-red-50 text-red-700"}`}>
              {idea.direction}
            </span>{" "}
            <Link href={`/stock/${idea.ticker}`} className="font-medium text-slate-800 hover:underline">
              {idea.ticker}
            </Link>{" "}
            <span className="text-xs text-slate-400">{new Date(idea.created_at).toLocaleDateString()}</span>
            {idea.scored_at ? (
              <span className={`ml-2 text-xs ${idea.outcome === "hit" ? "text-emerald-700" : "text-red-700"}`}>
                {idea.outcome} ({fmtPct(idea.realized_return_pct)})
              </span>
            ) : (
              <span className="ml-2 text-xs text-slate-400">not yet scored</span>
            )}
          </div>
        ))}
        {profile.ideas.length === 0 && <p className="text-sm text-slate-500">No ideas published yet.</p>}
      </div>
    </div>
  );
}
