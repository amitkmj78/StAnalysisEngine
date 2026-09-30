"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { ApiError, dismissAlertInboxItem, getAlertInbox } from "@/lib/api";
import type { AlertInboxItem } from "@/lib/types";

const SOURCE_LABEL: Record<string, string> = {
  watchlist: "Watchlist",
  portfolio_drop: "Price drop",
  signal_change: "Signal change",
  earnings: "Earnings",
  cost_drop: "Cost-basis drop",
};

export default function AlertsInboxPage() {
  const [items, setItems] = useState<AlertInboxItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyKey, setBusyKey] = useState<string | null>(null);

  async function load() {
    setError(null);
    try {
      setItems(await getAlertInbox());
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load alerts.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleDismiss(item: AlertInboxItem) {
    const key = `${item.source}-${item.id}`;
    setBusyKey(key);
    try {
      await dismissAlertInboxItem(item.source, item.id);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Dismiss failed.");
    } finally {
      setBusyKey(null);
    }
  }

  const unseen = (items ?? []).filter((i) => !i.seen_at);
  const seen = (items ?? []).filter((i) => i.seen_at);

  function Row({ item }: { item: AlertInboxItem }) {
    const key = `${item.source}-${item.id}`;
    return (
      <div
        key={key}
        className={`flex items-center justify-between gap-3 rounded-md border px-3 py-2 text-sm ${
          item.seen_at ? "border-slate-200 bg-white text-slate-500" : "border-emerald-200 bg-emerald-50"
        }`}
      >
        <div className="min-w-0">
          <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-slate-600">
            {SOURCE_LABEL[item.source] ?? item.source}
          </span>{" "}
          {item.link ? (
            <Link href={item.link} className="font-semibold text-slate-900 hover:underline">
              {item.ticker}
            </Link>
          ) : (
            <strong>{item.ticker}</strong>
          )}{" "}
          {item.summary}
          <div className="text-xs text-slate-400">{new Date(item.event_at).toLocaleString()}</div>
        </div>
        {!item.seen_at && (
          <button
            onClick={() => handleDismiss(item)}
            disabled={busyKey === key}
            className="shrink-0 rounded-md border border-slate-300 bg-white px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
          >
            Dismiss
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold text-slate-900">Alerts</h1>
        <Link href="/alerts/settings" className="text-sm font-medium text-slate-600 hover:underline">
          Settings
        </Link>
      </div>
      <p className="mt-1 text-sm text-slate-500">
        Every triggered alert in one place — watchlist price/score targets, portfolio drops, signal changes,
        earnings, and cost-basis drops.
      </p>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {items === null && !error && <p className="mt-6 text-sm text-slate-500">Loading…</p>}

      {items !== null && (
        <div className="mt-6 flex flex-col gap-6">
          <div>
            <h2 className="font-semibold text-emerald-700">New ({unseen.length})</h2>
            {unseen.length === 0 ? (
              <p className="mt-1 text-sm text-slate-500">Nothing new.</p>
            ) : (
              <div className="mt-2 flex flex-col gap-2">
                {unseen.map((item) => (
                  <Row key={`${item.source}-${item.id}`} item={item} />
                ))}
              </div>
            )}
          </div>

          {seen.length > 0 && (
            <div>
              <h2 className="font-semibold text-slate-900">History ({seen.length})</h2>
              <div className="mt-2 flex flex-col gap-2">
                {seen.map((item) => (
                  <Row key={`${item.source}-${item.id}`} item={item} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
