"use client";

import { useEffect, useState } from "react";

import { ApiError, disableVerifyPredictions, enableVerifyPredictions, getAdminSettings } from "@/lib/api";

export default function SchedulerControls() {
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function load() {
    setError(null);
    try {
      const settings = await getAdminSettings();
      setEnabled(settings.verify_predictions_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load scheduler settings.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleToggle() {
    setBusy(true);
    setError(null);
    try {
      const result = enabled ? await disableVerifyPredictions() : await enableVerifyPredictions();
      setEnabled(result.verify_predictions_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update scheduler setting.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="font-semibold text-slate-900">Auto-Verify Saved Predictions</h2>
          <p className="mt-1 text-sm text-slate-600">
            Background job that checks saved predictions against real prices every 15 minutes.
            Disabling it stops future runs — saved predictions already verified stay as they are,
            and new ones simply won&apos;t be checked until this is turned back on.
          </p>
        </div>
        {enabled !== null && (
          <span
            className={`shrink-0 rounded-full px-3 py-1 text-xs font-medium ${
              enabled ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-500"
            }`}
          >
            {enabled ? "Enabled" : "Disabled"}
          </span>
        )}
      </div>

      {error && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <button
        onClick={handleToggle}
        disabled={busy || enabled === null}
        className={`mt-4 rounded-md px-4 py-2 text-sm font-medium disabled:opacity-50 ${
          enabled
            ? "border border-red-200 text-red-700 hover:bg-red-50"
            : "bg-slate-900 text-white hover:bg-slate-800"
        }`}
      >
        {busy ? "Updating…" : enabled ? "Disable" : "Enable"}
      </button>
    </div>
  );
}
