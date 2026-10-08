"use client";

import { useEffect, useState } from "react";

import { ApiError, deleteCommunityIdea, getReportedCommunityIdeas, restoreCommunityIdea } from "@/lib/api";
import type { CommunityIdea } from "@/lib/types";

export default function CommunityReportsAdminPage() {
  const [ideas, setIdeas] = useState<CommunityIdea[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<number | null>(null);

  async function load() {
    try {
      const res = await getReportedCommunityIdeas();
      setIdeas(res.ideas);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load reported ideas.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleRestore(id: number) {
    setBusyId(id);
    try {
      await restoreCommunityIdea(id);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not restore this idea.");
    } finally {
      setBusyId(null);
    }
  }

  async function handleDelete(id: number) {
    if (!window.confirm("Permanently delete this idea? This can't be undone.")) return;
    setBusyId(id);
    try {
      await deleteCommunityIdea(id);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not delete this idea.");
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <h1 className="font-display text-2xl font-semibold text-slate-900">Community Idea Reports</h1>
      <p className="mt-1 text-sm text-slate-500">
        COM-6: every idea with at least one report, most-reported first. An idea auto-hides from the public feed
        once it crosses 3 distinct reporters — restore it if the report was unfounded, or delete it permanently.
        There is no automated pump-and-dump/paid-promotion detection; this queue plus the publish-time
        attestation are the real moderation mechanism.
      </p>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {ideas === null && !error && <p className="mt-6 text-sm text-slate-500">Loading…</p>}

      {ideas !== null && (
        <div className="mt-6 flex flex-col gap-3">
          {ideas.map((idea) => (
            <div key={idea.id} className="rounded-md border border-slate-200 bg-white p-4 text-sm">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <strong>{idea.display_name}</strong> — {idea.direction} {idea.ticker}
                  {idea.hidden && <span className="ml-2 rounded-full bg-red-50 px-2 py-0.5 text-xs font-semibold text-red-700">HIDDEN</span>}
                  <div className="text-xs text-slate-400">{new Date(idea.created_at).toLocaleString()}</div>
                </div>
                <div className="flex shrink-0 gap-2">
                  <button
                    onClick={() => handleRestore(idea.id)}
                    disabled={busyId === idea.id}
                    className="rounded-md border border-slate-300 px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                  >
                    Restore
                  </button>
                  <button
                    onClick={() => handleDelete(idea.id)}
                    disabled={busyId === idea.id}
                    className="rounded-md border border-red-300 px-2.5 py-1 text-xs font-medium text-red-700 hover:bg-red-50 disabled:opacity-50"
                  >
                    Delete permanently
                  </button>
                </div>
              </div>
            </div>
          ))}
          {ideas.length === 0 && <p className="text-sm text-slate-500">No reported ideas.</p>}
        </div>
      )}
    </div>
  );
}
