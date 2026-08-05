"use client";

import { useEffect, useState } from "react";

import { ApiError, dismissPortfolioDropAlert, getPortfolioDropAlerts } from "@/lib/api";
import type { PortfolioDropAlert } from "@/lib/types";

export default function PortfolioDropAlerts() {
  const [alerts, setAlerts] = useState<PortfolioDropAlert[] | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [dismissingId, setDismissingId] = useState<number | null>(null);

  async function load() {
    try {
      const res = await getPortfolioDropAlerts();
      setAlerts(res.alerts.filter((a) => a.seen_at === null));
    } catch {
      // Silent — this is a supplementary notice, not core portfolio data;
      // a failed fetch here shouldn't block or alarm about the main page.
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleDismiss(id: number) {
    setDismissingId(id);
    try {
      await dismissPortfolioDropAlert(id);
      setAlerts((prev) => (prev ?? []).filter((a) => a.id !== id));
    } catch (err) {
      // Leave the card in place if the dismiss call failed, but surface nothing
      // disruptive — the user can just try again.
      console.error(err instanceof ApiError ? err.message : err);
    } finally {
      setDismissingId(null);
    }
  }

  if (!alerts || alerts.length === 0) return null;

  return (
    <div className="mt-6 flex flex-col gap-3">
      {alerts.map((a) => {
        const expanded = expandedId === a.id;
        return (
          <div key={a.id} className="rounded-lg border border-red-200 bg-red-50/50 p-4">
            <div className="flex flex-wrap items-start justify-between gap-2">
              <div className="flex items-baseline gap-2">
                <span className="text-base font-semibold text-slate-900">{a.ticker}</span>
                <span className="rounded-full bg-red-100 px-2 py-0.5 text-xs font-semibold text-red-700">
                  {a.pct_change.toFixed(2)}% today
                </span>
                <span className="text-xs text-slate-500">
                  ${a.prev_close.toFixed(2)} &rarr; ${a.price_at_check.toFixed(2)}
                </span>
              </div>
              <button
                onClick={() => handleDismiss(a.id)}
                disabled={dismissingId === a.id}
                className="rounded-md border border-slate-300 bg-white px-2.5 py-1 text-xs font-medium text-slate-600 hover:bg-slate-100 disabled:opacity-50"
              >
                {dismissingId === a.id ? "Dismissing…" : "Dismiss"}
              </button>
            </div>

            {a.recommended_action && (
              <p className="mt-2 text-sm text-slate-800">{a.recommended_action}</p>
            )}

            {a.predicted_signal && (
              <p className="mt-2 text-xs text-slate-500">
                Quant signal: <span className="font-medium text-slate-700">{a.predicted_signal}</span>
                {a.predicted_expected_return_pct !== null && (
                  <> · expected return {a.predicted_expected_return_pct.toFixed(2)}%</>
                )}
                {a.predicted_target_price !== null && <> · target ${a.predicted_target_price.toFixed(2)}</>}
              </p>
            )}

            {a.sentiment_summary && (
              <div className="mt-2">
                <button
                  onClick={() => setExpandedId(expanded ? null : a.id)}
                  className="text-xs font-medium text-slate-500 underline hover:text-slate-700"
                >
                  {expanded ? "Hide news/earnings context" : "Show news/earnings context"}
                </button>
                {expanded && (
                  <p className="mt-2 whitespace-pre-wrap rounded-md bg-white p-3 text-xs text-slate-600">
                    {a.sentiment_summary}
                  </p>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
