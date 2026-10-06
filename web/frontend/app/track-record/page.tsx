"use client";

import { useEffect, useState } from "react";

import { DisclosureBanner } from "@/components/DisclosureBanner";
import { ApiError, getPublishedSignals } from "@/lib/api";
import type { PublishedSignalsResponse } from "@/lib/types";
import LiveTab from "./LiveTab";
import HypotheticalTab from "./HypotheticalTab";

type Tab = "live" | "hypothetical";

export default function TrackRecordPage() {
  const [data, setData] = useState<PublishedSignalsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<Tab>("live");

  useEffect(() => {
    getPublishedSignals()
      .then(setData)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Failed to load the track record."))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="mx-auto max-w-2xl px-4 py-10">
      <h1 className="font-display text-2xl font-semibold text-slate-900">Track Record</h1>
      <p className="mt-2 text-sm leading-relaxed text-slate-600">
        A daily, timestamped, append-only record of one ranking rule&apos;s picks — published before each
        day&apos;s outcome is known, and never edited after the fact. This is impersonal research: the same
        content for every reader, describing what the model ranked and why, not a recommendation to buy or
        sell anything.
      </p>
      <div className="mt-3">
        <DisclosureBanner />
      </div>

      <div className="mt-6 flex gap-1 rounded-md border border-slate-300 bg-white p-1" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={tab === "live"}
          onClick={() => setTab("live")}
          className={`flex-1 rounded px-3 py-1.5 text-sm font-medium ${
            tab === "live" ? "bg-emerald-600 text-white" : "text-slate-600 hover:bg-slate-50"
          }`}
        >
          Live
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "hypothetical"}
          onClick={() => setTab("hypothetical")}
          className={`flex-1 rounded px-3 py-1.5 text-sm font-medium ${
            tab === "hypothetical" ? "bg-indigo-600 text-white" : "text-slate-600 hover:bg-slate-50"
          }`}
        >
          Hypothetical
        </button>
      </div>

      {loading && <p className="mt-6 text-sm text-slate-500">Loading…</p>}
      {error && <p className="mt-6 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {data && !loading && (tab === "live" ? <LiveTab data={data} /> : <HypotheticalTab data={data} />)}
    </div>
  );
}

export function RecordTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-3">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="mt-1 text-lg font-semibold text-slate-900">{value}</p>
    </div>
  );
}
