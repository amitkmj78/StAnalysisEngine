"use client";

import { useEffect, useState } from "react";

import {
  ApiError,
  disableHorizon1Subscriptions,
  enableHorizon1Subscriptions,
  getAdminSettings,
  getAuditLog,
  getDemandReport,
  setFreeTierLagDays,
} from "@/lib/api";
import type { AuditLogEntry, DemandReport } from "@/lib/types";

export default function HorizonSubscriptionControls() {
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [lagDays, setLagDays] = useState<number | null>(null);
  const [lagDaysInput, setLagDaysInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [confirming, setConfirming] = useState(false);

  const [report, setReport] = useState<DemandReport | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);

  const [events, setEvents] = useState<AuditLogEntry[] | null>(null);
  const [eventsError, setEventsError] = useState<string | null>(null);

  async function load() {
    setError(null);
    try {
      const settings = await getAdminSettings();
      setEnabled(settings.horizon1_subscriptions_enabled);
      setLagDays(settings.free_tier_lag_days);
      setLagDaysInput(String(settings.free_tier_lag_days));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load Horizon 1 settings.");
    }
    try {
      setReport(await getDemandReport());
    } catch (err) {
      setReportError(err instanceof ApiError ? err.message : "Failed to load demand report.");
    }
    try {
      const log = await getAuditLog({ limit: 20 });
      setEvents(log.events);
    } catch (err) {
      setEventsError(err instanceof ApiError ? err.message : "Failed to load audit log.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleToggle() {
    if (!enabled && !confirming) {
      setConfirming(true);
      return;
    }
    setBusy(true);
    setError(null);
    setConfirming(false);
    try {
      const result = enabled ? await disableHorizon1Subscriptions() : await enableHorizon1Subscriptions();
      setEnabled(result.horizon1_subscriptions_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update Horizon 1 setting.");
    } finally {
      setBusy(false);
    }
  }

  async function handleSaveLagDays() {
    const days = Number(lagDaysInput);
    if (!Number.isInteger(days) || days < 0) {
      setError("Free-tier lag days must be a non-negative integer.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await setFreeTierLagDays(days);
      setLagDays(result.free_tier_lag_days);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update lag days.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rounded-lg border border-amber-200 bg-amber-50/40 p-5">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="font-semibold text-slate-900">Horizon 1 — Impersonal Research Subscription</h2>
          <p className="mt-1 text-sm text-slate-600">
            Paid-subscription layer on top of the public track record (RS-1 through RS-6). Built and
            testable, but gated per{" "}
            <code className="rounded bg-slate-100 px-1 py-0.5 text-xs">
              docs/signal-licensing-whitelabel-requirements.md.pdf
            </code>{" "}
            — Gate 0→1 requires ≥6 months of continuous live publication and{" "}
            <strong>written counsel confirmation</strong> that this sits within the publisher&apos;s
            exclusion (CMP-03). This toggle does not and cannot verify either — it is a raw switch.
          </p>
        </div>
        {enabled !== null && (
          <span
            className={`shrink-0 rounded-full px-3 py-1 text-xs font-medium ${
              enabled ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-500"
            }`}
          >
            {enabled ? "Live" : "Off"}
          </span>
        )}
      </div>

      {error && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {confirming && !enabled && (
        <p className="mt-3 rounded-md bg-amber-100 px-3 py-2 text-sm text-amber-800">
          Click again to confirm — only do this after counsel has signed off and the track record has
          actually run ≥6 months. This is a business/legal decision, not a deploy.
        </p>
      )}

      <div className="mt-4 flex items-center gap-2">
        <button
          onClick={handleToggle}
          disabled={busy || enabled === null}
          className={`rounded-md px-4 py-2 text-sm font-medium disabled:opacity-50 ${
            enabled
              ? "border border-red-200 text-red-700 hover:bg-red-50"
              : confirming
              ? "bg-amber-600 text-white hover:bg-amber-700"
              : "bg-slate-900 text-white hover:bg-slate-800"
          }`}
        >
          {busy ? "Updating…" : enabled ? "Disable" : confirming ? "Confirm Enable" : "Enable"}
        </button>
        {confirming && !enabled && (
          <button
            onClick={() => setConfirming(false)}
            disabled={busy}
            className="rounded-md border border-slate-300 px-3 py-2 text-sm font-medium text-slate-600 hover:bg-slate-100"
          >
            Cancel
          </button>
        )}
      </div>

      <div className="mt-4 flex items-center gap-2 border-t border-amber-200 pt-4">
        <label className="text-sm text-slate-600">Free-tier lag (days):</label>
        <input
          type="number"
          min={0}
          value={lagDaysInput}
          onChange={(e) => setLagDaysInput(e.target.value)}
          className="w-20 rounded-md border border-slate-300 px-2 py-1 text-sm"
        />
        <button
          onClick={handleSaveLagDays}
          disabled={busy || String(lagDays) === lagDaysInput}
          className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
        >
          Save
        </button>
      </div>

      <div className="mt-4 border-t border-amber-200 pt-4">
        <p className="text-sm font-medium text-slate-700">Demand report (RS-5)</p>
        {reportError && <p className="mt-2 text-sm text-red-700">{reportError}</p>}
        {report && (
          <dl className="mt-2 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
            <div>
              <dt className="text-xs text-slate-500">Active paid</dt>
              <dd className="font-medium text-slate-900">{report.currently_active_subscribers}</dd>
            </div>
            <div>
              <dt className="text-xs text-slate-500">Ever paid</dt>
              <dd className="font-medium text-slate-900">{report.ever_paid_subscribers}</dd>
            </div>
            <div>
              <dt className="text-xs text-slate-500">Checkout conversion</dt>
              <dd className="font-medium text-slate-900">
                {report.checkout_conversion_rate !== null
                  ? `${(report.checkout_conversion_rate * 100).toFixed(1)}%`
                  : "—"}
              </dd>
            </div>
            <div>
              <dt className="text-xs text-slate-500">Monthly churn</dt>
              <dd className="font-medium text-slate-900">
                {report.monthly_churn_rate !== null ? `${(report.monthly_churn_rate * 100).toFixed(1)}%` : "—"}
              </dd>
            </div>
            {report.cohort_retention.map((c) => (
              <div key={c.window}>
                <dt className="text-xs text-slate-500">{c.window.replace("_", "-")} retention</dt>
                <dd className="font-medium text-slate-900">
                  {c.retention_rate !== null ? `${(c.retention_rate * 100).toFixed(1)}% (n=${c.cohort_size})` : "—"}
                </dd>
              </div>
            ))}
          </dl>
        )}
        {report && report.enquiries_by_type.length > 0 && (
          <p className="mt-2 text-xs text-slate-500">
            Enquiries: {report.enquiries_by_type.map((e) => `${e.enquiry_type} (${e.count})`).join(", ")}
          </p>
        )}
      </div>

      <div className="mt-4 border-t border-amber-200 pt-4">
        <p className="text-sm font-medium text-slate-700">Recent activity (RS-6 audit log)</p>
        {eventsError && <p className="mt-2 text-sm text-red-700">{eventsError}</p>}
        {events && events.length === 0 && <p className="mt-2 text-xs text-slate-500">No events recorded yet.</p>}
        {events && events.length > 0 && (
          <div className="mt-2 max-h-64 overflow-y-auto rounded-md border border-slate-200 bg-white">
            <table className="min-w-full text-xs">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-left uppercase tracking-wide text-slate-500">
                  <th className="px-2 py-1.5">When</th>
                  <th className="px-2 py-1.5">Event</th>
                  <th className="px-2 py-1.5">Resource</th>
                </tr>
              </thead>
              <tbody>
                {events.map((e) => (
                  <tr key={e.id} className="border-b border-slate-100 last:border-0">
                    <td className="px-2 py-1.5 text-slate-500">{new Date(e.created_at).toLocaleString()}</td>
                    <td className="px-2 py-1.5 text-slate-800">{e.event_type}</td>
                    <td className="px-2 py-1.5 text-slate-500">{e.resource ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
