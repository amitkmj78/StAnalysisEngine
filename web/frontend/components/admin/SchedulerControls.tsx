"use client";

import { useEffect, useState } from "react";

import {
  ApiError,
  disableMarketRegime,
  disableVerifyPredictions,
  enableMarketRegime,
  enableVerifyPredictions,
  getAdminSettings,
} from "@/lib/api";

function JobToggleCard({
  title,
  description,
  enabled,
  busy,
  onToggle,
}: {
  title: string;
  description: string;
  enabled: boolean | null;
  busy: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="font-semibold text-slate-900">{title}</h2>
          <p className="mt-1 text-sm text-slate-600">{description}</p>
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

      <button
        onClick={onToggle}
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

export default function SchedulerControls() {
  const [verifyEnabled, setVerifyEnabled] = useState<boolean | null>(null);
  const [regimeEnabled, setRegimeEnabled] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyKey, setBusyKey] = useState<string | null>(null);

  async function load() {
    setError(null);
    try {
      const settings = await getAdminSettings();
      setVerifyEnabled(settings.verify_predictions_enabled);
      setRegimeEnabled(settings.market_regime_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load scheduler settings.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleToggleVerify() {
    setBusyKey("verify");
    setError(null);
    try {
      const result = verifyEnabled ? await disableVerifyPredictions() : await enableVerifyPredictions();
      setVerifyEnabled(result.verify_predictions_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update scheduler setting.");
    } finally {
      setBusyKey(null);
    }
  }

  async function handleToggleRegime() {
    setBusyKey("regime");
    setError(null);
    try {
      const result = regimeEnabled ? await disableMarketRegime() : await enableMarketRegime();
      setRegimeEnabled(result.market_regime_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update scheduler setting.");
    } finally {
      setBusyKey(null);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      {error && <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <JobToggleCard
        title="Auto-Verify Saved Predictions"
        description="Background job that checks saved predictions against real prices every 15 minutes. Disabling it stops future runs — saved predictions already verified stay as they are, and new ones simply won't be checked until this is turned back on."
        enabled={verifyEnabled}
        busy={busyKey === "verify"}
        onToggle={handleToggleVerify}
      />

      <JobToggleCard
        title="Market Regime Banner"
        description="Daily job (weekdays 18:10 ET) that computes the site-wide regime reading shown in the banner on every page. This wires up a scoring engine that failed its own release-gate backtest three times — see the banner's own disclosure for the full history. Disabling it stops new daily readings; the banner falls back to showing nothing until re-enabled."
        enabled={regimeEnabled}
        busy={busyKey === "regime"}
        onToggle={handleToggleRegime}
      />
    </div>
  );
}
