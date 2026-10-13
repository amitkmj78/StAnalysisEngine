"use client";

import { useEffect, useState } from "react";

import { ApiError, getContentReports, getReportReviewTimeStats, resolveContentReport } from "@/lib/api";
import type { ContentReport } from "@/lib/api";

export default function ContentReportsPanel() {
  const [reports, setReports] = useState<ContentReport[] | null>(null);
  const [stats, setStats] = useState<{ month: string; reviewed_count: number; avg_review_hours: number | null }[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<number | null>(null);

  async function load() {
    try {
      const [r, s] = await Promise.all([getContentReports(false), getReportReviewTimeStats()]);
      setReports(r.reports);
      setStats(s.months);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load reports.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleResolve(id: number) {
    const resolution = window.prompt("How was this resolved? (e.g. content removed, warned author, no action needed)");
    if (!resolution || !resolution.trim()) return;
    setBusyId(id);
    try {
      await resolveContentReport(id, resolution.trim());
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not resolve this report.");
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5">
      <h2 className="font-semibold text-slate-900 dark:text-slate-50">Content Reports (posts, comments, groups, profiles)</h2>
      <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
        SAF-2: a report&apos;s &quot;reviewed&quot; timestamp feeds the review-time stats below -- this measures the
        24h target, it doesn&apos;t enforce it. Community idea reports have their own separate queue.
      </p>

      {stats && stats.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-3 text-xs text-slate-500 dark:text-slate-400">
          {stats.slice(0, 3).map((m) => (
            <span key={m.month}>
              {m.month}: {m.reviewed_count} reviewed, avg {m.avg_review_hours ?? "–"}h
            </span>
          ))}
        </div>
      )}

      {error && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {reports === null && !error && <p className="mt-3 text-sm text-slate-500 dark:text-slate-400">Loading…</p>}

      {reports !== null && (
        <div className="mt-3 flex flex-col gap-2">
          {reports.map((r) => (
            <div key={r.id} className="flex items-center justify-between gap-3 rounded-md border border-slate-100 dark:border-slate-800 p-3 text-sm">
              <div>
                <span className="rounded-full bg-slate-100 dark:bg-slate-800 px-2 py-0.5 text-xs font-medium text-slate-600 dark:text-slate-300">
                  {r.content_type} #{r.content_id}
                </span>
                <span className="ml-2 text-slate-700 dark:text-slate-200">{r.reason}</span>
                <div className="text-xs text-slate-400 dark:text-slate-500">
                  Reported by {r.reporter_display_name ?? "a member"} · {new Date(r.created_at).toLocaleString()}
                </div>
              </div>
              <button
                onClick={() => handleResolve(r.id)}
                disabled={busyId === r.id}
                className="shrink-0 rounded-md border border-slate-300 dark:border-slate-600 px-2.5 py-1 text-xs font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800 disabled:opacity-50"
              >
                Resolve
              </button>
            </div>
          ))}
          {reports.length === 0 && <p className="text-sm text-slate-500 dark:text-slate-400">No unreviewed reports.</p>}
        </div>
      )}
    </div>
  );
}
