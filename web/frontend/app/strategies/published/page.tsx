"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { ApiError, listPublishedStrategies } from "@/lib/api";
import type { PublishedStrategySummary } from "@/lib/types";

export default function PublishedStrategiesPage() {
  const [rows, setRows] = useState<PublishedStrategySummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listPublishedStrategies()
      .then((res) => setRows(res.published))
      .catch((err) => setError(err instanceof ApiError ? err.message : "Published strategies could not be loaded."));
  }, []);

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h1 className="font-display text-2xl font-semibold text-slate-900">Published strategies</h1>
        <Link href="/strategies/saved" className="text-sm font-medium text-slate-700 hover:underline">
          ← Your saved strategies
        </Link>
      </div>
      <p className="mt-1 text-sm text-slate-500">
        Published by other members, with their rules (or a plain-language summary when the author keeps rules
        private) and an ongoing forward paper track record from the day each was published. Backtests are past
        prices only; nothing here places an order, real or paper.
      </p>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <div className="mt-6 rounded-xl border border-slate-200 bg-white">
        {rows === null && !error && <p className="p-5 text-sm text-slate-500">Loading…</p>}
        {rows?.length === 0 && <p className="p-5 text-sm text-slate-500">Nothing published yet.</p>}
        <ul className="divide-y divide-slate-100">
          {rows?.map((row) => (
            <li key={row.id}>
              <Link href={`/strategies/published/${row.id}`} className="flex items-center justify-between gap-3 p-4 hover:bg-slate-50">
                <span>
                  <span className="font-medium text-slate-900">{row.name}</span>
                  <span className="ml-2 text-xs text-slate-400">v{row.version}</span>
                  <span className="block text-xs text-slate-500">
                    by {row.author_display_name ?? "a member"} · published {new Date(row.published_at).toLocaleDateString()}
                    {row.rules_visibility === "summary_only" && " · rules private"}
                  </span>
                </span>
                <span className="text-sm text-indigo-600">View →</span>
              </Link>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
