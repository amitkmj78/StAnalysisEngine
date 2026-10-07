"use client";

import { useEffect, useState } from "react";

import {
  ApiError,
  computeStockScoresNow,
  disableStockScoreCompute,
  enableStockScoreCompute,
  getAdminSettings,
} from "@/lib/api";

export default function StockScoreComputeControls() {
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [running, setRunning] = useState(false);
  const [runResult, setRunResult] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  async function load() {
    setError(null);
    try {
      const settings = await getAdminSettings();
      setEnabled(settings.stock_score_compute_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load stock score compute status.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleToggle() {
    setBusy(true);
    setError(null);
    try {
      const result = enabled ? await disableStockScoreCompute() : await enableStockScoreCompute();
      setEnabled(result.stock_score_compute_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update setting.");
    } finally {
      setBusy(false);
    }
  }

  async function handleRunNow() {
    setRunning(true);
    setRunError(null);
    setRunResult(null);
    try {
      const res = await computeStockScoresNow();
      setRunResult(
        res.inserted > 0
          ? `Scored ${res.inserted} ticker${res.inserted === 1 ? "" : "s"} as of ${res.as_of_date} just now.`
          : `Already scored for ${res.as_of_date} — nothing new to insert.`
      );
    } catch (err) {
      setRunError(err instanceof ApiError ? err.message : "Failed to run.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="font-semibold text-slate-900">Stock Score Compute (Phase 1 &quot;Trust&quot;)</h2>
          <p className="mt-1 text-sm text-slate-600">
            Daily job (weekdays, 4:15pm ET) that computes every ticker&apos;s short-term/long-term two-score
            and saves it to stock_scores. One ticker&apos;s or one factor&apos;s data failing degrades just
            that piece to &quot;no data&quot; rather than aborting the whole night — safe to re-run (it only
            inserts tickers still missing for the day). Defaults on, unlike most jobs on this page: it&apos;s
            internal accumulation with no external email, and most of the app depends on it having run.
          </p>
        </div>
        {enabled !== null && (
          <span
            className={`shrink-0 rounded-full px-3 py-1 text-xs font-medium ${
              enabled ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-500"
            }`}
          >
            {enabled ? "Scoring" : "Paused"}
          </span>
        )}
      </div>

      {error && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <div className="mt-4 flex items-center gap-2">
        <button
          onClick={handleToggle}
          disabled={busy || enabled === null}
          className={`rounded-md px-4 py-2 text-sm font-medium disabled:opacity-50 ${
            enabled
              ? "border border-red-200 text-red-700 hover:bg-red-50"
              : "bg-slate-900 text-white hover:bg-slate-800"
          }`}
        >
          {busy ? "Updating…" : enabled ? "Pause" : "Resume"}
        </button>
        <button
          onClick={handleRunNow}
          disabled={running}
          className="rounded-md border border-emerald-300 bg-white px-4 py-2 text-sm font-medium text-emerald-700 hover:bg-emerald-50 disabled:opacity-50"
        >
          {running ? "Scoring…" : "Run Now"}
        </button>
      </div>

      {runResult && <p className="mt-3 text-sm text-emerald-700">{runResult}</p>}
      {runError && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{runError}</p>}
    </div>
  );
}
