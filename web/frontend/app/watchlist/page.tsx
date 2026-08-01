"use client";

import { useEffect, useState } from "react";

import CurrentPriceBadge from "@/components/CurrentPriceBadge";
import {
  ApiError,
  createWatchlistAlert,
  deleteWatchlistAlert,
  dismissWatchlistAlert,
  getWatchlistAlerts,
} from "@/lib/api";
import type { AlertConditionType, WatchlistAlert } from "@/lib/types";

export default function WatchlistPage() {
  const [alerts, setAlerts] = useState<WatchlistAlert[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<number | null>(null);

  const [ticker, setTicker] = useState("");
  const [condition, setCondition] = useState<AlertConditionType>("price_above");
  const [threshold, setThreshold] = useState("");
  const [creating, setCreating] = useState(false);

  async function load() {
    setError(null);
    try {
      setAlerts(await getWatchlistAlerts());
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load watchlist.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    const parsed = Number(threshold);
    if (!ticker.trim() || !parsed || parsed <= 0) return;
    setCreating(true);
    setError(null);
    try {
      await createWatchlistAlert(ticker.trim().toUpperCase(), condition, parsed);
      setTicker("");
      setThreshold("");
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to create alert.");
    } finally {
      setCreating(false);
    }
  }

  async function handleDelete(id: number) {
    setBusyId(id);
    try {
      await deleteWatchlistAlert(id);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Delete failed.");
    } finally {
      setBusyId(null);
    }
  }

  async function handleDismiss(id: number) {
    setBusyId(id);
    try {
      await dismissWatchlistAlert(id);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Dismiss failed.");
    } finally {
      setBusyId(null);
    }
  }

  const pending = (alerts ?? []).filter((a) => !a.triggered_at);
  const triggeredUnseen = (alerts ?? []).filter((a) => a.triggered_at && !a.seen_at);
  const triggeredSeen = (alerts ?? []).filter((a) => a.triggered_at && a.seen_at);

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">Watchlist Alerts</h1>
      <p className="mt-1 text-sm text-slate-500">
        Set a price target for a ticker and it gets checked automatically in the background — every 5 minutes,
        no need to keep this page open. Price-only for now (above/below a target).
      </p>

      <form onSubmit={handleCreate} className="mt-6 flex flex-wrap items-end gap-3 rounded-lg border border-slate-200 bg-white p-4">
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-slate-500">Ticker</label>
          <input
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            placeholder="e.g. AAPL"
            className="w-28 rounded-md border border-slate-300 px-3 py-2 text-sm uppercase"
            maxLength={10}
          />
        </div>
        <CurrentPriceBadge ticker={ticker} />
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-slate-500">Condition</label>
          <select
            value={condition}
            onChange={(e) => setCondition(e.target.value as AlertConditionType)}
            className="rounded-md border border-slate-300 px-3 py-2 text-sm"
          >
            <option value="price_above">Price rises above</option>
            <option value="price_below">Price falls below</option>
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-slate-500">Threshold ($)</label>
          <input
            value={threshold}
            onChange={(e) => setThreshold(e.target.value)}
            type="number"
            step="0.01"
            min="0"
            placeholder="0.00"
            className="w-28 rounded-md border border-slate-300 px-3 py-2 text-sm"
          />
        </div>
        <button
          type="submit"
          disabled={creating || !ticker.trim() || !threshold}
          className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
        >
          {creating ? "Adding…" : "Add Alert"}
        </button>
      </form>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {alerts === null && !error && <p className="mt-6 text-sm text-slate-500">Loading…</p>}

      {alerts !== null && (
        <div className="mt-6 flex flex-col gap-6">
          {triggeredUnseen.length > 0 && (
            <div>
              <h2 className="font-semibold text-emerald-700">
                🔔 Triggered ({triggeredUnseen.length})
              </h2>
              <div className="mt-2 flex flex-col gap-2">
                {triggeredUnseen.map((a) => (
                  <div
                    key={a.id}
                    className="flex items-center justify-between rounded-md border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm"
                  >
                    <span>
                      <strong>{a.ticker}</strong>{" "}
                      {a.condition_type === "price_above" ? "rose above" : "fell below"} ${a.threshold.toFixed(2)}{" "}
                      — now ${a.triggered_price?.toFixed(2)}{" "}
                      <span className="text-xs text-slate-500">
                        ({a.triggered_at ? new Date(a.triggered_at).toLocaleString() : ""})
                      </span>
                    </span>
                    <div className="flex gap-2">
                      <button
                        onClick={() => handleDismiss(a.id)}
                        disabled={busyId === a.id}
                        className="rounded-md border border-slate-300 bg-white px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                      >
                        Dismiss
                      </button>
                      <button
                        onClick={() => handleDelete(a.id)}
                        disabled={busyId === a.id}
                        className="rounded-md border border-red-200 bg-white px-2.5 py-1 text-xs font-medium text-red-700 hover:bg-red-50 disabled:opacity-50"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div>
            <h2 className="font-semibold text-slate-900">Watching ({pending.length})</h2>
            {pending.length === 0 ? (
              <p className="mt-1 text-sm text-slate-500">No active alerts.</p>
            ) : (
              <div className="mt-2 overflow-x-auto rounded-lg border border-slate-200 bg-white">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                      <th className="px-3 py-2">Ticker</th>
                      <th className="px-3 py-2">Condition</th>
                      <th className="px-3 py-2">Created</th>
                      <th className="px-3 py-2"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {pending.map((a) => (
                      <tr key={a.id} className="border-b border-slate-100 last:border-0">
                        <td className="px-3 py-2 font-medium text-slate-800">{a.ticker}</td>
                        <td className="px-3 py-2 text-slate-600">
                          {a.condition_type === "price_above" ? "Price above" : "Price below"} ${a.threshold.toFixed(2)}
                        </td>
                        <td className="px-3 py-2 text-slate-500">{new Date(a.created_at).toLocaleDateString()}</td>
                        <td className="px-3 py-2 text-right">
                          <button
                            onClick={() => handleDelete(a.id)}
                            disabled={busyId === a.id}
                            className="rounded-md border border-red-200 px-2.5 py-1 text-xs font-medium text-red-700 hover:bg-red-50 disabled:opacity-50"
                          >
                            Delete
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {triggeredSeen.length > 0 && (
            <div>
              <h2 className="font-semibold text-slate-900">History ({triggeredSeen.length})</h2>
              <div className="mt-2 overflow-x-auto rounded-lg border border-slate-200 bg-white">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                      <th className="px-3 py-2">Ticker</th>
                      <th className="px-3 py-2">Condition</th>
                      <th className="px-3 py-2">Triggered</th>
                      <th className="px-3 py-2"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {triggeredSeen.map((a) => (
                      <tr key={a.id} className="border-b border-slate-100 last:border-0 text-slate-500">
                        <td className="px-3 py-2">{a.ticker}</td>
                        <td className="px-3 py-2">
                          {a.condition_type === "price_above" ? "rose above" : "fell below"} ${a.threshold.toFixed(2)} —
                          hit ${a.triggered_price?.toFixed(2)}
                        </td>
                        <td className="px-3 py-2">{a.triggered_at ? new Date(a.triggered_at).toLocaleString() : "—"}</td>
                        <td className="px-3 py-2 text-right">
                          <button
                            onClick={() => handleDelete(a.id)}
                            disabled={busyId === a.id}
                            className="rounded-md border border-slate-300 px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                          >
                            Delete
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
