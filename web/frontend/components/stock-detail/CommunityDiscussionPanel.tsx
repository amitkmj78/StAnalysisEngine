"use client";

import { useEffect, useState } from "react";

import {
  ApiError, acceptComment, blockUser, createPost, createPostComment, getCurrentUser, getPostComments, getPosts,
  muteContent, reportContent, unacceptComment,
} from "@/lib/api";
import type { Post, PostComment } from "@/lib/types";
import DisclosureBanner from "@/components/DisclosureBanner";

/** SOC-3: every ticker page shows community discussion next to the
 * model's signal and evidence panel -- this IS the social feature's
 * posts list (web/backend/routers/social.py::list_posts), filtered by
 * ticker, not a separate discussion table.
 *
 * BEG-4: a "question" is just a post with post_type = "question"; an
 * "answer" is just a comment on it. Only the asker can accept one.
 *
 * STS-6 (comments/questions half only -- ratings deferred): a published
 * strategy's discussion reuses this same component, filtered by
 * publishedStrategyId instead of ticker. */
export default function CommunityDiscussionPanel({
  ticker, publishedStrategyId, composePlaceholder,
}: { ticker?: string; publishedStrategyId?: number; composePlaceholder?: string }) {
  const [posts, setPosts] = useState<Post[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [isQuestion, setIsQuestion] = useState(false);
  const [hasPosition, setHasPosition] = useState<boolean | null>(null);
  const [isPaidPromotion, setIsPaidPromotion] = useState(false);
  const [posting, setPosting] = useState(false);
  const [myUserId, setMyUserId] = useState<string | null>(null);
  const [openComments, setOpenComments] = useState<Record<number, PostComment[] | undefined>>({});
  const [commentDraft, setCommentDraft] = useState<Record<number, string>>({});

  async function load() {
    try {
      const res = await getPosts(ticker ? { ticker } : { published_strategy_id: publishedStrategyId });
      setPosts(res.posts);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load discussion.");
    }
  }

  useEffect(() => {
    load();
    getCurrentUser().then((u) => setMyUserId(u.id)).catch(() => setMyUserId(null));
  }, [ticker, publishedStrategyId]);

  async function handlePost() {
    const text = draft.trim();
    if (!text) return;
    if (ticker && hasPosition === null) {
      setError("Say whether you hold this ticker before posting (required -- see the community guidelines).");
      return;
    }
    setPosting(true);
    try {
      await createPost({
        body: text, ticker, published_strategy_id: publishedStrategyId,
        post_type: isQuestion ? "question" : "note",
        has_position: ticker ? (hasPosition ?? undefined) : undefined,
        is_paid_promotion: isPaidPromotion,
      });
      setDraft("");
      setIsQuestion(false);
      setHasPosition(null);
      setIsPaidPromotion(false);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not post.");
    } finally {
      setPosting(false);
    }
  }

  async function handleReport(postId: number) {
    const reason = window.prompt("Why are you reporting this post? (e.g. pump-and-dump, paid promotion, spam, harassment)");
    if (!reason || !reason.trim()) return;
    try {
      await reportContent("post", postId, reason.trim());
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not submit the report.");
    }
  }

  async function handleBlock(authorId: string) {
    if (!window.confirm("Block this user? You won't see their posts anymore, and they won't be able to message you.")) return;
    try {
      await blockUser(authorId);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not block this user.");
    }
  }

  async function handleMuteTicker(t: string) {
    try {
      await muteContent("ticker", t);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not mute this ticker.");
    }
  }

  async function refreshComments(postId: number) {
    const res = await getPostComments(postId);
    setOpenComments((prev) => ({ ...prev, [postId]: res.comments }));
  }

  async function handleAccept(postId: number, commentId: number) {
    await acceptComment(commentId);
    await refreshComments(postId);
  }

  async function handleUnaccept(postId: number, commentId: number) {
    await unacceptComment(commentId);
    await refreshComments(postId);
  }

  async function toggleComments(postId: number) {
    if (openComments[postId]) {
      setOpenComments((prev) => ({ ...prev, [postId]: undefined }));
      return;
    }
    await refreshComments(postId);
  }

  async function handleComment(postId: number) {
    const text = (commentDraft[postId] ?? "").trim();
    if (!text) return;
    await createPostComment(postId, text);
    setCommentDraft((prev) => ({ ...prev, [postId]: "" }));
    await refreshComments(postId);
  }

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-4">
      <h3 className="font-semibold text-slate-900 dark:text-slate-50">Community Discussion</h3>
      <div className="mt-1">
        <DisclosureBanner />
      </div>
      {error && <p className="mt-2 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <div className="mt-3 flex gap-2">
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={composePlaceholder ?? `Share something about ${ticker}...`}
          className="flex-1 rounded-md border border-slate-300 dark:border-slate-600 px-3 py-1.5 text-sm"
        />
        <button
          onClick={handlePost}
          disabled={posting || !draft.trim()}
          className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
        >
          Post
        </button>
      </div>
      <label className="mt-1.5 flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
        <input type="checkbox" checked={isQuestion} onChange={(e) => setIsQuestion(e.target.checked)} />
        This is a question -- you&apos;ll be able to mark the best reply as the accepted answer.
      </label>
      {ticker && (
        <div className="mt-1.5 flex flex-wrap items-center gap-3 text-xs text-slate-500 dark:text-slate-400">
          <span>
            Do you hold {ticker}? (required)
            <label className="ml-2">
              <input type="radio" name="has-position" checked={hasPosition === true} onChange={() => setHasPosition(true)} /> Yes
            </label>
            <label className="ml-2">
              <input type="radio" name="has-position" checked={hasPosition === false} onChange={() => setHasPosition(false)} /> No
            </label>
          </span>
          <label className="flex items-center gap-1.5">
            <input type="checkbox" checked={isPaidPromotion} onChange={(e) => setIsPaidPromotion(e.target.checked)} />
            This is a paid promotion
          </label>
        </div>
      )}

      {posts === null && !error && <p className="mt-4 text-sm text-slate-500 dark:text-slate-400">Loading…</p>}
      <div className="mt-4 flex flex-col gap-3">
        {posts?.map((p) => (
          <div key={p.id} className="rounded-md border border-slate-100 dark:border-slate-800 p-3 text-sm">
            <div className="flex items-center justify-between">
              <strong className="text-slate-800 dark:text-slate-100">{p.display_name}</strong>
              <span className="text-xs text-slate-400 dark:text-slate-500">{new Date(p.created_at).toLocaleString()}</span>
            </div>
            {p.post_type === "performance_claim" && (
              <span className={`mt-1 inline-block rounded-full px-2 py-0.5 text-[10px] font-semibold ${p.verified ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"}`}>
                {p.verified ? "verified claim" : "unverified"}
              </span>
            )}
            {p.post_type === "question" && (
              <span className="mt-1 inline-block rounded-full bg-[var(--pf-accent-soft)] px-2 py-0.5 text-[10px] font-semibold text-[var(--pf-accent)]">
                question
              </span>
            )}
            {p.ticker && p.has_position !== null && (
              <span className="mt-1 ml-1 inline-block rounded-full bg-slate-100 dark:bg-slate-800 px-2 py-0.5 text-[10px] font-semibold text-slate-600 dark:text-slate-300">
                {p.position_verified ? "✓ Verified position" : p.has_position ? "Holds this ticker" : "Does not hold this ticker"}
              </span>
            )}
            {p.is_paid_promotion && (
              <span className="mt-1 ml-1 inline-block rounded-full bg-amber-50 px-2 py-0.5 text-[10px] font-semibold text-amber-800">
                Paid promotion
              </span>
            )}
            <p className="mt-1 text-slate-700 dark:text-slate-200">{p.body}</p>
            <div className="mt-2 flex flex-wrap gap-3 text-xs font-medium text-slate-500 dark:text-slate-400">
              <button onClick={() => toggleComments(p.id)} className="hover:underline">
                {openComments[p.id] ? "Hide replies" : "Replies"}
              </button>
              {p.author_user_id !== myUserId && (
                <>
                  <button onClick={() => handleReport(p.id)} className="hover:text-red-700 hover:underline">Report</button>
                  <button onClick={() => handleBlock(p.author_user_id)} className="hover:text-red-700 hover:underline">Block author</button>
                </>
              )}
              {p.ticker && <button onClick={() => handleMuteTicker(p.ticker as string)} className="hover:underline">Mute {p.ticker}</button>}
            </div>
            {openComments[p.id] && (
              <div className="mt-2 flex flex-col gap-2 border-t border-slate-100 dark:border-slate-800 pt-2">
                {openComments[p.id]!.map((c) => (
                  <div key={c.id} className="flex items-start justify-between gap-2 text-xs text-slate-600 dark:text-slate-300">
                    <span>
                      {c.is_accepted && (
                        <span className="mr-1 rounded-full bg-emerald-50 px-1.5 py-0.5 font-semibold text-emerald-700">
                          ✓ Best answer
                        </span>
                      )}
                      <strong>{c.display_name ?? "User"}</strong> {c.body}
                    </span>
                    {p.post_type === "question" && p.author_user_id === myUserId && (
                      <button
                        onClick={() => (c.is_accepted ? handleUnaccept(p.id, c.id) : handleAccept(p.id, c.id))}
                        className="flex-none whitespace-nowrap text-slate-400 dark:text-slate-500 hover:text-slate-700 dark:hover:text-slate-200 hover:underline"
                      >
                        {c.is_accepted ? "Unaccept" : "Accept"}
                      </button>
                    )}
                  </div>
                ))}
                <div className="flex gap-2">
                  <input
                    value={commentDraft[p.id] ?? ""}
                    onChange={(e) => setCommentDraft((prev) => ({ ...prev, [p.id]: e.target.value }))}
                    placeholder="Reply..."
                    className="flex-1 rounded-md border border-slate-300 dark:border-slate-600 px-2 py-1 text-xs"
                  />
                  <button onClick={() => handleComment(p.id)} className="rounded-md border border-slate-300 dark:border-slate-600 px-2 py-1 text-xs hover:bg-slate-50 dark:hover:bg-slate-800">
                    Reply
                  </button>
                </div>
              </div>
            )}
          </div>
        ))}
        {posts?.length === 0 && <p className="text-sm text-slate-500 dark:text-slate-400">No discussion yet -- be the first.</p>}
      </div>
    </div>
  );
}
