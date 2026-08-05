"use client";

import { useEffect, useState } from "react";

import {
  ApiError,
  capturePitPricesNow,
  disablePitPriceCapture,
  enablePitPriceCapture,
  getAdminSettings,
  getPitPriceStatus,
} from "@/lib/api";
import type { PitPriceStatus } from "@/lib/types";

export default function PitPriceControls() {
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [status, setStatus] = useState<PitPriceStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [capturing, setCapturing] = useState(false);
  const [captureResult, setCaptureResult] = useState<string | null>(null);
  const [captureError, setCaptureError] = useState<string | null>(null);

  async function load() {
    setError(null);
    try {
      const [settings, pitStatus] = await Promise.all([getAdminSettings(), getPitPriceStatus()]);
      setEnabled(settings.pit_price_capture_enabled);
      setStatus(pitStatus);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load PIT price capture status.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleToggle() {
    setBusy(true);
    setError(null);
    try {
      const result = enabled ? await disablePitPriceCapture() : await enablePitPriceCapture();
      setEnabled(result.pit_price_capture_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update PIT price capture setting.");
    } finally {
      setBusy(false);
    }
  }

  async function handleCaptureNow() {
    setCapturing(true);
    setCaptureError(null);
    setCaptureResult(null);
    try {
      const res = await capturePitPricesNow();
      setCaptureResult(
        res.inserted > 0
          ? `Captured ${res.inserted} new prices just now.`
          : "Already captured for today — nothing new."
      );
      const pitStatus = await getPitPriceStatus();
      setStatus(pitStatus);
    } catch (err) {
      setCaptureError(err instanceof ApiError ? err.message : "Failed to capture.");
    } finally {
      setCapturing(false);
    }
  }

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="font-semibold text-slate-900">Point-in-Time Price Store (TR-3 Phase 1)</h2>
          <p className="mt-1 text-sm text-slate-600">
            Daily job (weekdays, 4:05pm ET) that appends today&apos;s close for the tracked universe to an
            append-only store. Once captured, a row is never overwritten — that&apos;s what makes it a
            genuine point-in-time record rather than a snapshot that could quietly change if the data
            vendor later revises history. Nothing reads from this store yet; it exists to accumulate
            history for later phases (honest historical comparisons and PIT-aware backtests).
          </p>
        </div>
        {enabled !== null && (
          <span
            className={`shrink-0 rounded-full px-3 py-1 text-xs font-medium ${
              enabled ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-500"
            }`}
          >
            {enabled ? "Capturing" : "Paused"}
          </span>
        )}
      </div>

      {status && (
        <dl className="mt-4 grid grid-cols-2 gap-3 rounded-md bg-slate-50 p-3 text-sm sm:grid-cols-4">
          <div>
            <dt className="text-xs text-slate-500">Trading days captured</dt>
            <dd className="font-medium text-slate-900">{status.trading_days_captured}</dd>
          </div>
          <div>
            <dt className="text-xs text-slate-500">Tickers</dt>
            <dd className="font-medium text-slate-900">{status.ticker_count}</dd>
          </div>
          <div>
            <dt className="text-xs text-slate-500">Earliest date</dt>
            <dd className="font-medium text-slate-900">{status.earliest_date ?? "—"}</dd>
          </div>
          <div>
            <dt className="text-xs text-slate-500">Latest date</dt>
            <dd className="font-medium text-slate-900">{status.latest_date ?? "—"}</dd>
          </div>
        </dl>
      )}

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
          onClick={handleCaptureNow}
          disabled={capturing}
          className="rounded-md border border-emerald-300 bg-white px-4 py-2 text-sm font-medium text-emerald-700 hover:bg-emerald-50 disabled:opacity-50"
        >
          {capturing ? "Capturing…" : "Capture Now"}
        </button>
      </div>

      {captureResult && <p className="mt-3 text-sm text-emerald-700">{captureResult}</p>}
      {captureError && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{captureError}</p>}
    </div>
  );
}
