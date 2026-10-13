"use client";

import { useEffect, useState } from "react";

import { ApiError, disableCommunitySafetyScan, enableCommunitySafetyScan, getAdminSettings } from "@/lib/api";

export default function CommunitySafetyScanControls() {
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [confirming, setConfirming] = useState(false);

  async function load() {
    setError(null);
    try {
      const settings = await getAdminSettings();
      setEnabled(settings.community_safety_scan_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load this setting.");
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
      const result = enabled ? await disableCommunitySafetyScan() : await enableCommunitySafetyScan();
      setEnabled(result.community_safety_scan_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update this setting.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="font-semibold text-slate-900 dark:text-slate-50">Community Safety Scan (SAF-3)</h2>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
            Nightly (3am ET): flags sudden spikes in mentions of small/illiquid tickers, and near-identical posts
            from different accounts within 24h. Only flags for the admin queue below -- never hides or removes
            anything automatically.
          </p>
        </div>
        {enabled !== null && (
          <span
            className={`shrink-0 rounded-full px-3 py-1 text-xs font-medium ${
              enabled ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400"
            }`}
          >
            {enabled ? "Scanning" : "Disabled"}
          </span>
        )}
      </div>

      {error && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {confirming && !enabled && (
        <p className="mt-3 rounded-md bg-amber-100 px-3 py-2 text-sm text-amber-800">
          Click again to confirm -- this starts the nightly scan.
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
              : "bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-900 hover:bg-slate-800 dark:hover:bg-slate-200"
          }`}
        >
          {busy ? "Updating…" : enabled ? "Disable" : confirming ? "Confirm Enable" : "Enable"}
        </button>
        {confirming && !enabled && (
          <button
            onClick={() => setConfirming(false)}
            disabled={busy}
            className="rounded-md border border-slate-300 dark:border-slate-600 px-3 py-2 text-sm font-medium text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
          >
            Cancel
          </button>
        )}
      </div>
    </div>
  );
}
