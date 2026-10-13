"use client";

import { useEffect, useState } from "react";

import { ApiError, getFlaggedContent, resolveFlaggedContent } from "@/lib/api";
import type { FlaggedContentRow } from "@/lib/api";

export default function FlaggedContentPanel() {
  const [flags, setFlags] = useState<FlaggedContentRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<number | null>(null);

  async function load() {
    try {
      const res = await getFlaggedContent();
      setFlags(res.flags);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load flagged content.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleResolve(id: number) {
    setBusyId(id);
    try {
      await resolveFlaggedContent(id);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not resolve this flag.");
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5">
      <h2 className="font-semibold text-slate-900 dark:text-slate-50">Flagged Content (automated)</h2>
      <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
        SAF-3: mention-spike and duplicate-post detection, held for review -- never auto-removed. Only runs if the
        nightly scan is enabled (Community Safety Scan control below), off by default.
      </p>

      {error && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {flags === null && !error && <p className="mt-3 text-sm text-slate-500 dark:text-slate-400">Loading…</p>}

      {flags !== null && (
        <div className="mt-3 flex flex-col gap-2">
          {flags.map((f) => (
            <div key={f.id} className="flex items-center justify-between gap-3 rounded-md border border-slate-100 dark:border-slate-800 p-3 text-sm">
              <div>
                <span className="rounded-full bg-amber-50 px-2 py-0.5 text-xs font-medium text-amber-800">
                  {f.flag_type === "mention_spike" ? "Mention spike" : "Duplicate post"}
                </span>
                <span className="ml-2 text-slate-700 dark:text-slate-200">{f.detail}</span>
                <div className="text-xs text-slate-400 dark:text-slate-500">{new Date(f.created_at).toLocaleString()}</div>
              </div>
              <button
                onClick={() => handleResolve(f.id)}
                disabled={busyId === f.id}
                className="shrink-0 rounded-md border border-slate-300 dark:border-slate-600 px-2.5 py-1 text-xs font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800 disabled:opacity-50"
              >
                Mark reviewed
              </button>
            </div>
          ))}
          {flags.length === 0 && <p className="text-sm text-slate-500 dark:text-slate-400">No unreviewed flags.</p>}
        </div>
      )}
    </div>
  );
}
