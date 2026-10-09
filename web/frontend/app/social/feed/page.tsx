"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { ApiError, createPost, followTicker, followTopic, getSocialFeed } from "@/lib/api";
import type { FeedItem } from "@/lib/types";

function fmtPct(v: number | null): string {
  if (v === null || v === undefined) return "—";
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
}

export default function SocialFeedPage() {
  const [feed, setFeed] = useState<FeedItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [body, setBody] = useState("");
  const [ticker, setTicker] = useState("");
  const [topic, setTopic] = useState("");
  const [attachChart, setAttachChart] = useState(false);
  const [isPerformanceClaim, setIsPerformanceClaim] = useState(false);
  const [claimIdeaId, setClaimIdeaId] = useState("");
  const [posting, setPosting] = useState(false);
  const [followTickerInput, setFollowTickerInput] = useState("");
  const [followTopicInput, setFollowTopicInput] = useState("");
  const [followMsg, setFollowMsg] = useState<string | null>(null);

  async function load() {
    try {
      const res = await getSocialFeed();
      setFeed(res.feed);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load your feed.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handlePost() {
    const text = body.trim();
    if (!text) return;
    setPosting(true);
    try {
      await createPost({
        body: text, ticker: ticker.trim() || null, topic: topic.trim() || null,
        attach_chart: attachChart && !!ticker.trim(),
        post_type: isPerformanceClaim ? "performance_claim" : "note",
        claim_reference_id: isPerformanceClaim && claimIdeaId.trim() ? Number(claimIdeaId.trim()) : null,
      });
      setBody("");
      setTicker("");
      setTopic("");
      setAttachChart(false);
      setIsPerformanceClaim(false);
      setClaimIdeaId("");
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not post.");
    } finally {
      setPosting(false);
    }
  }

  async function handleFollowTicker() {
    const t = followTickerInput.trim().toUpperCase();
    if (!t) return;
    await followTicker(t);
    setFollowTickerInput("");
    setFollowMsg(`Following ${t} -- posts and ideas on it will show up here.`);
    await load();
  }

  async function handleFollowTopic() {
    const t = followTopicInput.trim();
    if (!t) return;
    await followTopic(t);
    setFollowTopicInput("");
    setFollowMsg(`Following topic "${t}".`);
    await load();
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <div className="flex items-center justify-between">
        <h1 className="font-display text-2xl font-semibold text-slate-900">Feed</h1>
        <div className="flex gap-3 text-sm">
          <Link href="/social/groups" className="font-medium text-slate-600 hover:underline">Groups</Link>
          <Link href="/social/messages" className="font-medium text-slate-600 hover:underline">Messages</Link>
        </div>
      </div>
      <p className="mt-1 text-sm text-slate-500">
        Posts, chart snapshots and ideas from people, tickers and topics you follow -- newest first. Follow authors
        from their profile or the <Link href="/community/leaderboard" className="underline">leaderboard</Link>.
      </p>

      <div className="mt-4 flex flex-col gap-2 rounded-xl border border-slate-200 bg-white p-4 sm:flex-row">
        <input value={followTickerInput} onChange={(e) => setFollowTickerInput(e.target.value)} placeholder="Follow a ticker (e.g. AAPL)" className="flex-1 rounded-md border border-slate-300 px-3 py-1.5 text-sm" />
        <button onClick={handleFollowTicker} className="rounded-md border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-50">Follow ticker</button>
        <input value={followTopicInput} onChange={(e) => setFollowTopicInput(e.target.value)} placeholder="Follow a topic (e.g. options)" className="flex-1 rounded-md border border-slate-300 px-3 py-1.5 text-sm" />
        <button onClick={handleFollowTopic} className="rounded-md border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-50">Follow topic</button>
      </div>
      {followMsg && <p className="mt-2 text-xs text-emerald-700">{followMsg}</p>}

      <div className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
        <textarea value={body} onChange={(e) => setBody(e.target.value)} placeholder="Share an update, a setup, a question..." className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm" rows={3} />
        <div className="mt-2 flex flex-wrap gap-2">
          <input value={ticker} onChange={(e) => setTicker(e.target.value)} placeholder="Ticker (optional)" className="w-32 rounded-md border border-slate-300 px-2 py-1 text-sm" />
          <input value={topic} onChange={(e) => setTopic(e.target.value)} placeholder="Topic (optional)" className="w-40 rounded-md border border-slate-300 px-2 py-1 text-sm" />
          {ticker.trim() && (
            <label className="flex items-center gap-1.5 text-sm text-slate-600">
              <input type="checkbox" checked={attachChart} onChange={(e) => setAttachChart(e.target.checked)} />
              Attach my {ticker.trim().toUpperCase()} chart
            </label>
          )}
          <label className="flex items-center gap-1.5 text-sm text-slate-600">
            <input type="checkbox" checked={isPerformanceClaim} onChange={(e) => setIsPerformanceClaim(e.target.checked)} />
            This is a performance claim
          </label>
          {isPerformanceClaim && (
            <input
              value={claimIdeaId}
              onChange={(e) => setClaimIdeaId(e.target.value)}
              placeholder="Your idea ID to verify it (optional)"
              className="w-56 rounded-md border border-slate-300 px-2 py-1 text-sm"
            />
          )}
          <button onClick={handlePost} disabled={posting || !body.trim()} className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50">Post</button>
        </div>
        <p className="mt-2 text-xs text-slate-400">
          Attaching a chart shares a copy of your saved drawings for that ticker -- a reader gets their own editable
          copy, not a live shared canvas. A performance claim only shows as &ldquo;verified&rdquo; when it links one
          of your own published ideas (find its ID on the <Link href="/community" className="underline">Idea Feed</Link>) --
          otherwise it&apos;s shown as unverified, never silently treated as fact-checked.
        </p>
      </div>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {feed === null && !error && <p className="mt-6 text-sm text-slate-500">Loading…</p>}

      <div className="mt-6 flex flex-col gap-3">
        {feed?.map((item) => (
          <div key={`${item.kind}-${item.id}`} className="rounded-xl border border-slate-200 bg-white p-4 text-sm">
            {item.kind === "post" ? (
              <>
                <div className="flex items-center justify-between">
                  <Link href={`/community/authors/${item.author_user_id}`} className="font-medium text-slate-800 hover:underline">
                    {item.display_name}
                  </Link>
                  <span className="text-xs text-slate-400">{new Date(item.created_at).toLocaleString()}</span>
                </div>
                {item.post_type === "performance_claim" && (
                  <span className={`mt-1 inline-block rounded-full px-2 py-0.5 text-[10px] font-semibold ${item.verified ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"}`}>
                    {item.verified ? "verified claim" : "unverified"}
                  </span>
                )}
                <p className="mt-1 text-slate-700">{item.body}</p>
                <div className="mt-1 flex gap-2 text-xs text-slate-400">
                  {item.ticker && <Link href={`/stock/${item.ticker}`} className="hover:underline">${item.ticker}</Link>}
                  {item.topic && <span>#{item.topic}</span>}
                </div>
              </>
            ) : (
              <>
                <div className="flex items-center justify-between">
                  <Link href={`/community/authors/${item.author_user_id ?? "model"}`} className="font-medium text-slate-800 hover:underline">
                    {item.display_name}
                  </Link>
                  <span className="text-xs text-slate-400">{new Date(item.created_at).toLocaleString()}</span>
                </div>
                <p className="mt-1 text-slate-700">
                  <span className={item.direction === "LONG" ? "text-emerald-700" : "text-red-700"}>{item.direction}</span>{" "}
                  <Link href={`/stock/${item.ticker}`} className="font-medium hover:underline">{item.ticker}</Link>
                  {item.scored_at && <span className="ml-2 text-xs text-slate-400">{item.outcome} ({fmtPct(item.realized_return_pct)})</span>}
                </p>
              </>
            )}
          </div>
        ))}
        {feed?.length === 0 && (
          <p className="text-sm text-slate-500">
            Nothing here yet -- follow a ticker, topic, or an author to start seeing posts and ideas.
          </p>
        )}
      </div>
    </div>
  );
}
