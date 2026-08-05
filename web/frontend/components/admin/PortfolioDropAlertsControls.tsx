"use client";

import { useEffect, useState } from "react";

import {
  ApiError,
  disablePortfolioDropAlerts,
  enablePortfolioDropAlerts,
  getAdminSettings,
  scanPortfolioDropAlertsNow,
  setPortfolioDropThreshold,
} from "@/lib/api";

export default function PortfolioDropAlertsControls() {
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [threshold, setThreshold] = useState<number | null>(null);
  const [thresholdInput, setThresholdInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [confirming, setConfirming] = useState(false);

  const [savingThreshold, setSavingThreshold] = useState(false);
  const [thresholdError, setThresholdError] = useState<string | null>(null);
  const [thresholdSaved, setThresholdSaved] = useState(false);

  const [scanning, setScanning] = useState(false);
  const [scanResult, setScanResult] = useState<string | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);

  async function load() {
    setError(null);
    try {
      const settings = await getAdminSettings();
      setEnabled(settings.portfolio_drop_alerts_enabled);
      setThreshold(settings.portfolio_drop_threshold_pct);
      setThresholdInput(String(settings.portfolio_drop_threshold_pct));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load portfolio drop alert settings.");
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
      const result = enabled ? await disablePortfolioDropAlerts() : await enablePortfolioDropAlerts();
      setEnabled(result.portfolio_drop_alerts_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update portfolio drop alert setting.");
    } finally {
      setBusy(false);
    }
  }

  async function handleSaveThreshold() {
    const value = Number(thresholdInput);
    if (!Number.isFinite(value) || value <= 0 || value > 50) {
      setThresholdError("Enter a number greater than 0 and at most 50.");
      return;
    }
    setSavingThreshold(true);
    setThresholdError(null);
    setThresholdSaved(false);
    try {
      const res = await setPortfolioDropThreshold(value);
      setThreshold(res.portfolio_drop_threshold_pct);
      setThresholdSaved(true);
    } catch (err) {
      setThresholdError(err instanceof ApiError ? err.message : "Failed to save threshold.");
    } finally {
      setSavingThreshold(false);
    }
  }

  async function handleScanNow() {
    setScanning(true);
    setScanError(null);
    setScanResult(null);
    try {
      const res = await scanPortfolioDropAlertsNow();
      setScanResult(
        res.inserted > 0
          ? `Inserted ${res.inserted} new alert${res.inserted === 1 ? "" : "s"} just now.`
          : "Scanned all holdings — no new drops found (or already alerted today)."
      );
    } catch (err) {
      setScanError(err instanceof ApiError ? err.message : "Failed to scan.");
    } finally {
      setScanning(false);
    }
  }

  const thresholdDirty = threshold !== null && thresholdInput !== String(threshold);

  return (
    <div className="rounded-lg border border-amber-200 bg-amber-50/40 p-5">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="font-semibold text-slate-900">Portfolio Drop Alerts</h2>
          <p className="mt-1 text-sm text-slate-600">
            Every 15 minutes, scans every user&apos;s holdings for a same-day drop of{" "}
            {threshold !== null ? `${threshold}%` : "the configured threshold"} or more. For each new
            drop, fetches news/earnings sentiment and the Predict-page quant signal, asks an LLM to
            synthesize a short recommended-action note, and posts it as an in-app alert on their
            Portfolio page. One alert per user/ticker/day — a ticker already alerted today is skipped.
          </p>
          <p className="mt-2 text-xs text-slate-500">
            Enabling this starts real per-drop external API and LLM spend (Tavily search + LLM call per
            newly-dropping ticker) and user-visible notifications — not just a preview.
          </p>
        </div>
        {enabled !== null && (
          <span
            className={`shrink-0 rounded-full px-3 py-1 text-xs font-medium ${
              enabled ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-500"
            }`}
          >
            {enabled ? "Scanning" : "Disabled"}
          </span>
        )}
      </div>

      {error && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <div className="mt-4 flex flex-wrap items-end gap-2 rounded-md border border-slate-200 bg-white p-3">
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-slate-500">Drop threshold (%)</label>
          <input
            type="number"
            min={0.01}
            max={50}
            step={0.1}
            value={thresholdInput}
            onChange={(e) => {
              setThresholdInput(e.target.value);
              setThresholdSaved(false);
            }}
            className="w-28 rounded-md border border-slate-300 px-2 py-1.5 text-sm"
          />
        </div>
        <button
          onClick={handleSaveThreshold}
          disabled={savingThreshold || !thresholdDirty}
          className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
        >
          {savingThreshold ? "Saving…" : "Save Threshold"}
        </button>
        {thresholdSaved && !thresholdDirty && (
          <span className="text-xs font-medium text-emerald-700">Saved</span>
        )}
        {thresholdError && <p className="w-full text-xs text-red-600">{thresholdError}</p>}
      </div>

      {confirming && !enabled && (
        <p className="mt-3 rounded-md bg-amber-100 px-3 py-2 text-sm text-amber-800">
          Click again to confirm — this starts real API/LLM spend and notifying users.
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
        <button
          onClick={handleScanNow}
          disabled={scanning}
          className="rounded-md border border-emerald-300 bg-white px-4 py-2 text-sm font-medium text-emerald-700 hover:bg-emerald-50 disabled:opacity-50"
        >
          {scanning ? "Scanning…" : "Scan Now"}
        </button>
      </div>

      {scanResult && <p className="mt-3 text-sm text-emerald-700">{scanResult}</p>}
      {scanError && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{scanError}</p>}
    </div>
  );
}
