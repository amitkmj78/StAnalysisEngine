"use client";

import { useEffect, useState } from "react";

import { ApiError, getAdminActivity } from "@/lib/api";
import type { AdminActivityRow } from "@/lib/types";

export default function ActivityLog() {
  const [rows, setRows] = useState<AdminActivityRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [emailFilter, setEmailFilter] = useState("");

  async function load() {
    setError(null);
    try {
      setRows(await getAdminActivity(200));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load activity.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  if (rows === null && !error) {
    return <p className="text-sm text-slate-500">Loading activity…</p>;
  }

  const filtered = (rows ?? []).filter((r) =>
    emailFilter.trim() ? r.email.toLowerCase().includes(emailFilter.trim().toLowerCase()) : true
  );

  return (
    <div className="flex flex-col gap-4">
      {error && <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-slate-500">Filter by email</label>
          <input
            value={emailFilter}
            onChange={(e) => setEmailFilter(e.target.value)}
            placeholder="e.g. amitkmj78@gmail.com"
            className="rounded-md border border-slate-300 px-3 py-2 text-sm"
          />
        </div>
        <button
          onClick={load}
          className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
        >
          Refresh
        </button>
      </div>

      <p className="text-xs text-slate-500">
        Showing the most recent {filtered.length} of {rows?.length ?? 0} logged requests (quota-tracked endpoints
        only).
      </p>

      <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
              <th className="px-3 py-2">User</th>
              <th className="px-3 py-2">Endpoint</th>
              <th className="px-3 py-2">Time</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={3} className="px-3 py-4 text-center text-slate-500">
                  No activity found.
                </td>
              </tr>
            ) : (
              filtered.map((r) => (
                <tr key={r.id} className="border-b border-slate-100 last:border-0">
                  <td className="px-3 py-2 text-slate-700">{r.email}</td>
                  <td className="px-3 py-2 font-mono text-xs text-slate-600">{r.endpoint}</td>
                  <td className="px-3 py-2 text-slate-500">{new Date(r.created_at).toLocaleString()}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
