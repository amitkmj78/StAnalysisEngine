"use client";

import { useEffect, useState } from "react";

import {
  ApiError,
  dismissPortfolioDropAlert,
  getDropAlertThreshold,
  getPortfolioDropAlerts,
  refreshDropAlert,
  refreshPortfolioDropAlerts,
  setDropAlertThreshold,
} from "@/lib/api";
import type { DropAlertThreshold, PortfolioDropAlert } from "@/lib/types";

export default function PortfolioDropAlerts() {
  const [alerts, setAlerts] = useState<PortfolioDropAlert[] | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [dismissingId, setDismissingId] = useState<number | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshMessage, setRefreshMessage] = useState<string | null>(null);
  const [refreshingAlertId, setRefreshingAlertId] = useState<number | null>(null);
  const [alertErrors, setAlertErrors] = useState<Record<number, string>>({});

  const [threshold, setThreshold] = useState<DropAlertThreshold | null>(null);
  const [editingThreshold, setEditingThreshold] = useState(false);
  const [thresholdInput, setThresholdInput] = useState("");
  const [savingThreshold, setSavingThreshold] = useState(false);
  const [thresholdError, setThresholdError] = useState<string | null>(null);

  async function load() {
    try {
      const res = await getPortfolioDropAlerts();
      setAlerts(res.alerts.filter((a) => a.seen_at === null));
    } catch {
      // Silent — this is a supplementary notice, not core portfolio data;
      // a failed fetch here shouldn't block or alarm about the main page.
    }
  }

  async function loadThreshold() {
    try {
      const res = await getDropAlertThreshold();
      setThreshold(res);
      setThresholdInput(String(res.threshold_pct));
    } catch {
      // Silent, same reasoning as load() above.
    }
  }

  useEffect(() => {
    load();
    loadThreshold();
  }, []);

  async function handleSaveThreshold() {
    const parsed = Number(thresholdInput);
    if (!Number.isFinite(parsed) || parsed < 0.1 || parsed > 50) {
      setThresholdError("Enter a value between 0.1 and 50.");
      return;
    }
    setSavingThreshold(true);
    setThresholdError(null);
    try {
      await setDropAlertThreshold(parsed);
      await loadThreshold();
      setEditingThreshold(false);
    } catch (err) {
      setThresholdError(err instanceof ApiError ? err.message : "Could not save threshold.");
    } finally {
      setSavingThreshold(false);
    }
  }

  async function handleResetThreshold() {
    setSavingThreshold(true);
    setThresholdError(null);
    try {
      await setDropAlertThreshold(null);
      await loadThreshold();
      setEditingThreshold(false);
    } catch (err) {
      setThresholdError(err instanceof ApiError ? err.message : "Could not reset threshold.");
    } finally {
      setSavingThreshold(false);
    }
  }

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

  async function handleRefreshAlert(id: number) {
    setRefreshingAlertId(id);
    setAlertErrors((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
    try {
      const res = await refreshDropAlert(id);
      setAlerts((prev) => (prev ?? []).map((a) => (a.id === id ? res.alert : a)));
    } catch (err) {
      setAlertErrors((prev) => ({
        ...prev,
        [id]: err instanceof ApiError ? err.message : "Could not refresh this alert.",
      }));
    } finally {
      setRefreshingAlertId(null);
    }
  }

  async function handleRefresh() {
    setRefreshing(true);
    setRefreshMessage(null);
    try {
      const res = await refreshPortfolioDropAlerts();
      await load();
      setRefreshMessage(
        res.inserted > 0
          ? `Found ${res.inserted} new alert${res.inserted === 1 ? "" : "s"}.`
          : "No new drops since your last check.",
      );
    } catch (err) {
      setRefreshMessage(err instanceof ApiError ? err.message : "Could not check for new drops.");
    } finally {
      setRefreshing(false);
    }
  }

  // Wait for the initial load to resolve before rendering anything, so the
  // refresh control doesn't flash in before we know whether there's
  // anything to report.
  if (alerts === null) return null;

  return (
    <div className="mt-6 flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-xs font-medium text-slate-500">
          {alerts.length > 0
            ? `${alerts.length} portfolio drop alert${alerts.length === 1 ? "" : "s"}`
            : "No portfolio drop alerts right now"}
        </span>
        <div className="flex items-center gap-2">
          {refreshMessage && <span className="text-xs text-slate-400">{refreshMessage}</span>}
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="rounded-md border border-slate-300 bg-white px-2.5 py-1 text-xs font-medium text-slate-600 hover:bg-slate-100 disabled:opacity-50"
          >
            {refreshing ? "Checking…" : "Check for new drops"}
          </button>
        </div>
      </div>

      {threshold && (
        <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
          {!editingThreshold ? (
            <>
              <span>
                Alert me when a holding drops{" "}
                <span className="font-medium text-slate-700">{threshold.threshold_pct}%</span> or more in a day
                {threshold.is_custom ? "" : ` (default)`}
              </span>
              <button
                onClick={() => {
                  setEditingThreshold(true);
                  setThresholdError(null);
                }}
                className="font-medium text-slate-500 underline hover:text-slate-700"
              >
                Change
              </button>
            </>
          ) : (
            <>
              <span>Alert me when a holding drops</span>
              <input
                type="number"
                min={0.1}
                max={50}
                step={0.1}
                value={thresholdInput}
                onChange={(e) => setThresholdInput(e.target.value)}
                className="w-16 rounded-md border border-slate-300 px-1.5 py-0.5 text-xs"
              />
              <span>% or more in a day</span>
              <button
                onClick={handleSaveThreshold}
                disabled={savingThreshold}
                className="rounded-md border border-slate-300 bg-white px-2 py-0.5 text-xs font-medium text-slate-600 hover:bg-slate-100 disabled:opacity-50"
              >
                {savingThreshold ? "Saving…" : "Save"}
              </button>
              {threshold.is_custom && (
                <button
                  onClick={handleResetThreshold}
                  disabled={savingThreshold}
                  className="text-xs font-medium text-slate-500 underline hover:text-slate-700 disabled:opacity-50"
                >
                  Reset to default ({threshold.default_pct}%)
                </button>
              )}
              <button
                onClick={() => {
                  setEditingThreshold(false);
                  setThresholdInput(String(threshold.threshold_pct));
                  setThresholdError(null);
                }}
                disabled={savingThreshold}
                className="text-xs text-slate-400 hover:text-slate-600"
              >
                Cancel
              </button>
              {thresholdError && <span className="text-red-600">{thresholdError}</span>}
            </>
          )}
        </div>
      )}

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
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleRefreshAlert(a.id)}
                  disabled={refreshingAlertId === a.id}
                  className="rounded-md border border-slate-300 bg-white px-2.5 py-1 text-xs font-medium text-slate-600 hover:bg-slate-100 disabled:opacity-50"
                >
                  {refreshingAlertId === a.id ? "Refreshing…" : "Refresh"}
                </button>
                <button
                  onClick={() => handleDismiss(a.id)}
                  disabled={dismissingId === a.id}
                  className="rounded-md border border-slate-300 bg-white px-2.5 py-1 text-xs font-medium text-slate-600 hover:bg-slate-100 disabled:opacity-50"
                >
                  {dismissingId === a.id ? "Dismissing…" : "Dismiss"}
                </button>
              </div>
            </div>

            {alertErrors[a.id] && (
              <p className="mt-2 text-xs text-red-600">{alertErrors[a.id]}</p>
            )}

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

            {a.updated_at && (
              <p className="mt-2 text-[11px] text-slate-400">
                Refreshed {new Date(a.updated_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}
