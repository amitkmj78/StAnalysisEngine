"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import CommunityDiscussionPanel from "@/components/stock-detail/CommunityDiscussionPanel";
import {
  ApiError, followPublishedStrategy, forkPublishedStrategy, getPublishedStrategy, getStrategyForwardRecord,
  listPublishedStrategyVersions, unfollowPublishedStrategy,
} from "@/lib/api";
import type { PublishedStrategyDetail, PublishedStrategyVersion, StrategyForwardSnapshot } from "@/lib/types";

function pct(v: number | null | undefined, digits = 1) {
  return v === null || v === undefined ? "–" : `${v > 0 ? "+" : ""}${v.toFixed(digits)}%`;
}

export default function PublishedStrategyPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const id = Number(params.id);

  const [detail, setDetail] = useState<PublishedStrategyDetail | null>(null);
  const [versions, setVersions] = useState<PublishedStrategyVersion[] | null>(null);
  const [forward, setForward] = useState<StrategyForwardSnapshot[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [forking, setForking] = useState(false);
  const [following, setFollowing] = useState(false);
  const [followBusy, setFollowBusy] = useState(false);

  useEffect(() => {
    getPublishedStrategy(id)
      .then((res) => {
        setDetail(res);
        setFollowing(res.is_following);
      })
      .catch((err) => setError(err instanceof ApiError ? err.message : "This published strategy could not be loaded."));
    listPublishedStrategyVersions(id).then((res) => setVersions(res.versions)).catch(() => setVersions([]));
    getStrategyForwardRecord(id).then((res) => setForward(res.snapshots)).catch(() => setForward([]));
  }, [id]);

  async function handleFork() {
    setForking(true);
    setError(null);
    try {
      // No existing flow anywhere in this app loads a saved strategy's
      // rules back into the builder's editor by id -- that's a separate,
      // pre-existing gap, not something STS-2 is scoped to also build.
      // The fork lands in /strategies/saved (its own private workspace
      // copy, with a link back to this exact version) same as any other
      // saved run.
      await forkPublishedStrategy(id);
      router.push("/strategies/saved");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "This strategy could not be forked.");
      setForking(false);
    }
  }

  async function handleFollow() {
    setFollowBusy(true);
    try {
      if (following) {
        await unfollowPublishedStrategy(id);
        setFollowing(false);
      } else {
        await followPublishedStrategy(id);
        setFollowing(true);
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not update follow status.");
    } finally {
      setFollowBusy(false);
    }
  }

  if (error) return <div className="mx-auto max-w-3xl px-4 py-8 text-sm text-red-700">{error}</div>;
  if (!detail) return <div className="mx-auto max-w-3xl px-4 py-8 text-sm text-slate-500">Loading…</div>;

  const latestForward = forward && forward.length > 0 ? forward[forward.length - 1] : null;

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <Link href="/strategies/published" className="text-sm font-medium text-slate-600 hover:underline">
        ← Published strategies
      </Link>
      <div className="mt-2 flex flex-wrap items-baseline justify-between gap-2">
        <h1 className="font-display text-2xl font-semibold text-slate-900">
          {detail.name} <span className="text-base font-normal text-slate-400">v{detail.version}</span>
        </h1>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleFollow}
            disabled={followBusy}
            className={`rounded-md border px-4 py-2 text-sm font-semibold disabled:opacity-50 ${
              following ? "border-slate-300 bg-white text-slate-700 hover:bg-slate-50" : "border-slate-900 bg-slate-900 text-white hover:bg-slate-800"
            }`}
          >
            {following ? "Following" : "Follow for signal alerts"}
          </button>
          {detail.rules_visibility === "public" ? (
            <button
              type="button"
              onClick={handleFork}
              disabled={forking}
              className="rounded-md bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800 disabled:opacity-50"
            >
              {forking ? "Forking…" : "Fork this strategy"}
            </button>
          ) : (
            <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-500">
              Rules private — can&apos;t be forked
            </span>
          )}
        </div>
      </div>
      <p className="mt-1 text-sm text-slate-500">
        by {detail.author_display_name ?? "a member"} · published {new Date(detail.published_at).toLocaleDateString()}
      </p>
      <p className="mt-1 text-xs text-slate-400">
        Following gets you an alert when this strategy&apos;s forward replay closes a new trade -- it does not
        place any paper or live order on your behalf.
      </p>

      {(versions?.length ?? 0) > 1 && (
        <div className="mt-2 flex flex-wrap gap-2 text-xs">
          {versions!.map((v) => (
            <Link
              key={v.id}
              href={`/strategies/published/${v.id}`}
              className={`rounded-full px-2 py-0.5 ${v.id === detail.id ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200"}`}
            >
              v{v.version}
            </Link>
          ))}
        </div>
      )}

      <div className="mt-6 rounded-xl border border-slate-200 bg-white p-5">
        <h2 className="text-sm font-semibold text-slate-900">Rules</h2>
        {detail.rules_visibility === "public" && detail.definition ? (
          <pre className="mt-2 overflow-x-auto rounded bg-slate-50 p-3 text-xs text-slate-700">
            {JSON.stringify({ entry: detail.definition.entry, exit: detail.definition.exit, exits: detail.definition.exits }, null, 2)}
          </pre>
        ) : (
          <p className="mt-2 text-sm text-slate-700">{detail.rules_summary}</p>
        )}
      </div>

      <div className="mt-4 rounded-xl border border-slate-200 bg-white p-5">
        <h2 className="text-sm font-semibold text-slate-900">Backtest (historical)</h2>
        <p className="mt-1 text-xs text-slate-400">Past prices only. Not a forecast or a recommendation.</p>
        <dl className="mt-3 grid grid-cols-2 gap-y-2 text-sm sm:grid-cols-4">
          <dt className="text-slate-500">Total return</dt>
          <dd className="font-mono font-semibold text-slate-900">{pct(detail.result.strategy?.total_return_pct)}</dd>
          <dt className="text-slate-500">CAGR</dt>
          <dd className="font-mono font-semibold text-slate-900">{pct(detail.result.strategy?.cagr_pct)}</dd>
          <dt className="text-slate-500">Max drawdown</dt>
          <dd className="font-mono text-slate-900">{pct(detail.result.strategy?.max_drawdown_pct)}</dd>
          <dt className="text-slate-500">Sharpe</dt>
          <dd className="font-mono text-slate-900">{detail.result.strategy?.sharpe?.toFixed(2) ?? "–"}</dd>
        </dl>
      </div>

      <div className="mt-4 rounded-xl border border-slate-200 bg-white p-5">
        <h2 className="text-sm font-semibold text-slate-900">Forward record</h2>
        <p className="mt-1 text-xs text-slate-400">
          A replay of the same backtest rules over the real time since this was published -- not a brokered paper
          account, and separate from the historical backtest above.
        </p>
        {forward === null && <p className="mt-2 text-sm text-slate-500">Loading…</p>}
        {forward?.length === 0 && (
          <p className="mt-2 text-sm text-slate-500">No forward reading yet -- check back a day or two after publishing.</p>
        )}
        {latestForward && (
          <p className="mt-2 text-sm">
            Since publish: <span className="font-mono font-semibold text-slate-900">{pct(latestForward.cumulative_return_pct)}</span>{" "}
            <span className="text-slate-500">over {latestForward.trades} trade{latestForward.trades === 1 ? "" : "s"}, as of {latestForward.as_of_date}</span>
          </p>
        )}
      </div>

      <div className="mt-4">
        <CommunityDiscussionPanel
          publishedStrategyId={id}
          composePlaceholder={`Ask a question or share something about ${detail.name}...`}
        />
      </div>
    </div>
  );
}
