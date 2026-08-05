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
import type { PitCaptureStats, PitPriceStatus } from "@/lib/types";

function CaptureStatsGrid({
  title,
  countLabel,
  countValue,
  stats,
}: {
  title: string;
  countLabel: string;
  countValue: number;
  stats: PitCaptureStats;
}) {
  return (
    <div>
      <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">{title}</h3>
      <dl className="mt-1 grid grid-cols-2 gap-3 rounded-md bg-slate-50 p-3 text-sm sm:grid-cols-4">
        <div>
          <dt className="text-xs text-slate-500">Days captured</dt>
          <dd className="font-medium text-slate-900">{stats.days_captured}</dd>
        </div>
        <div>
          <dt className="text-xs text-slate-500">{countLabel}</dt>
          <dd className="font-medium text-slate-900">{countValue}</dd>
        </div>
        <div>
          <dt className="text-xs text-slate-500">Earliest date</dt>
          <dd className="font-medium text-slate-900">{stats.earliest_date ?? "—"}</dd>
        </div>
        <div>
          <dt className="text-xs text-slate-500">Latest date</dt>
          <dd className="font-medium text-slate-900">{stats.latest_date ?? "—"}</dd>
        </div>
      </dl>
    </div>
  );
}

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
      setError(err instanceof ApiError ? err.message : "Failed to load PIT capture status.");
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
      setError(err instanceof ApiError ? err.message : "Failed to update PIT capture setting.");
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
      const total = res.prices_inserted + res.universe_membership_inserted + res.fundamentals_inserted;
      setCaptureResult(
        total > 0
          ? `Captured ${res.prices_inserted} price, ${res.universe_membership_inserted} membership, ` +
            `${res.fundamentals_inserted} fundamentals rows just now.`
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
          <h2 className="font-semibold text-slate-900">Point-in-Time Data Store (TR-3)</h2>
          <p className="mt-1 text-sm text-slate-600">
            Daily job (weekdays, 4:05pm ET) that appends today&apos;s prices, universe membership, and
            fundamentals to three append-only stores. Once captured, a row is never overwritten — that&apos;s
            what makes it a genuine point-in-time record rather than a snapshot that could quietly change if
            the data vendor later revises history. Nothing reads from these stores yet; they exist to
            accumulate history for later phases (honest historical comparisons and PIT-aware backtests).
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
        <div className="mt-4 flex flex-col gap-3">
          <CaptureStatsGrid
            title="Prices (Phase 1)"
            countLabel="Tickers"
            countValue={status.prices.ticker_count}
            stats={status.prices}
          />
          <CaptureStatsGrid
            title="Universe membership (Phase 2)"
            countLabel="Universes"
            countValue={status.universe_membership.universe_count}
            stats={status.universe_membership}
          />
          <CaptureStatsGrid
            title="Fundamentals (Phase 3)"
            countLabel="Tickers"
            countValue={status.fundamentals.ticker_count}
            stats={status.fundamentals}
          />
        </div>
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
