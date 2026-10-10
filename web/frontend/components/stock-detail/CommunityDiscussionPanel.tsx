"use client";

import { useEffect, useState } from "react";

import {
  ApiError, acceptComment, createPost, createPostComment, getCurrentUser, getPostComments, getPosts, unacceptComment,
} from "@/lib/api";
import type { Post, PostComment } from "@/lib/types";

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
    setPosting(true);
    try {
      await createPost({
        body: text, ticker, published_strategy_id: publishedStrategyId,
        post_type: isQuestion ? "question" : "note",
      });
      setDraft("");
      setIsQuestion(false);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not post.");
    } finally {
      setPosting(false);
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
    <div className="rounded-xl border border-slate-200 bg-white p-4">
      <h3 className="font-semibold text-slate-900">Community Discussion</h3>
      {error && <p className="mt-2 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <div className="mt-3 flex gap-2">
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={composePlaceholder ?? `Share something about ${ticker}...`}
          className="flex-1 rounded-md border border-slate-300 px-3 py-1.5 text-sm"
        />
        <button
          onClick={handlePost}
          disabled={posting || !draft.trim()}
          className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
        >
          Post
        </button>
      </div>
      <label className="mt-1.5 flex items-center gap-1.5 text-xs text-slate-500">
        <input type="checkbox" checked={isQuestion} onChange={(e) => setIsQuestion(e.target.checked)} />
        This is a question -- you&apos;ll be able to mark the best reply as the accepted answer.
      </label>

      {posts === null && !error && <p className="mt-4 text-sm text-slate-500">Loading…</p>}
      <div className="mt-4 flex flex-col gap-3">
        {posts?.map((p) => (
          <div key={p.id} className="rounded-md border border-slate-100 p-3 text-sm">
            <div className="flex items-center justify-between">
              <strong className="text-slate-800">{p.display_name}</strong>
              <span className="text-xs text-slate-400">{new Date(p.created_at).toLocaleString()}</span>
            </div>
            {p.post_type === "performance_claim" && (
              <span className={`mt-1 inline-block rounded-full px-2 py-0.5 text-[10px] font-semibold ${p.verified ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"}`}>
                {p.verified ? "verified claim" : "unverified"}
              </span>
            )}
            {p.post_type === "question" && (
              <span className="mt-1 inline-block rounded-full bg-indigo-50 px-2 py-0.5 text-[10px] font-semibold text-indigo-700">
                question
              </span>
            )}
            <p className="mt-1 text-slate-700">{p.body}</p>
            <button onClick={() => toggleComments(p.id)} className="mt-2 text-xs font-medium text-slate-500 hover:underline">
              {openComments[p.id] ? "Hide replies" : "Replies"}
            </button>
            {openComments[p.id] && (
              <div className="mt-2 flex flex-col gap-2 border-t border-slate-100 pt-2">
                {openComments[p.id]!.map((c) => (
                  <div key={c.id} className="flex items-start justify-between gap-2 text-xs text-slate-600">
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
                        className="flex-none whitespace-nowrap text-slate-400 hover:text-slate-700 hover:underline"
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
                    className="flex-1 rounded-md border border-slate-300 px-2 py-1 text-xs"
                  />
                  <button onClick={() => handleComment(p.id)} className="rounded-md border border-slate-300 px-2 py-1 text-xs hover:bg-slate-50">
                    Reply
                  </button>
                </div>
              </div>
            )}
          </div>
        ))}
        {posts?.length === 0 && <p className="text-sm text-slate-500">No discussion yet -- be the first.</p>}
      </div>
    </div>
  );
}
