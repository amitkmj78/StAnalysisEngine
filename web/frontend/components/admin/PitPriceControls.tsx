"use client";

import { useEffect, useState } from "react";

import {
  ApiError,
  capturePitAnalystRatingsNow,
  capturePitPricesNow,
  capturePitQuantSignalsNow,
  disablePitAnalystRatingCapture,
  disablePitPriceCapture,
  disablePitQuantSignalCapture,
  enablePitAnalystRatingCapture,
  enablePitPriceCapture,
  enablePitQuantSignalCapture,
  getAdminSettings,
  getPitPriceStatus,
  getPitReconciliation,
} from "@/lib/api";
import type { PitCaptureStats, PitPriceStatus, PitReconciliationReport } from "@/lib/types";

function todayIso(): string {
  return new Date().toISOString().slice(0, 10);
}

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

function PitStoreCard({
  title,
  description,
  countLabel,
  countValue,
  stats,
  enabled,
  busy,
  capturing,
  captureResult,
  captureError,
  error,
  onToggle,
  onCaptureNow,
}: {
  title: string;
  description: string;
  countLabel: string;
  countValue: number;
  stats: PitCaptureStats | null;
  enabled: boolean | null;
  busy: boolean;
  capturing: boolean;
  captureResult: string | null;
  captureError: string | null;
  error: string | null;
  onToggle: () => void;
  onCaptureNow: () => void;
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
            {enabled ? "Capturing" : "Paused"}
          </span>
        )}
      </div>

      {stats && (
        <div className="mt-4">
          <CaptureStatsGrid title={countLabel} countLabel="Tickers" countValue={countValue} stats={stats} />
        </div>
      )}

      {error && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <div className="mt-4 flex items-center gap-2">
        <button
          onClick={onToggle}
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
          onClick={onCaptureNow}
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

export default function PitPriceControls() {
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [status, setStatus] = useState<PitPriceStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [capturing, setCapturing] = useState(false);
  const [captureResult, setCaptureResult] = useState<string | null>(null);
  const [captureError, setCaptureError] = useState<string | null>(null);

  const [quantSignalEnabled, setQuantSignalEnabled] = useState<boolean | null>(null);
  const [quantSignalBusy, setQuantSignalBusy] = useState(false);
  const [quantSignalCapturing, setQuantSignalCapturing] = useState(false);
  const [quantSignalCaptureResult, setQuantSignalCaptureResult] = useState<string | null>(null);
  const [quantSignalCaptureError, setQuantSignalCaptureError] = useState<string | null>(null);
  const [quantSignalError, setQuantSignalError] = useState<string | null>(null);

  const [analystRatingEnabled, setAnalystRatingEnabled] = useState<boolean | null>(null);
  const [analystRatingBusy, setAnalystRatingBusy] = useState(false);
  const [analystRatingCapturing, setAnalystRatingCapturing] = useState(false);
  const [analystRatingCaptureResult, setAnalystRatingCaptureResult] = useState<string | null>(null);
  const [analystRatingCaptureError, setAnalystRatingCaptureError] = useState<string | null>(null);
  const [analystRatingError, setAnalystRatingError] = useState<string | null>(null);

  const [reconcileDate, setReconcileDate] = useState(todayIso());
  const [reconciling, setReconciling] = useState(false);
  const [reconcileReport, setReconcileReport] = useState<PitReconciliationReport | null>(null);
  const [reconcileError, setReconcileError] = useState<string | null>(null);

  async function handleReconcile() {
    setReconciling(true);
    setReconcileError(null);
    setReconcileReport(null);
    try {
      const report = await getPitReconciliation(reconcileDate);
      setReconcileReport(report);
    } catch (err) {
      setReconcileError(err instanceof ApiError ? err.message : "Failed to run reconciliation.");
    } finally {
      setReconciling(false);
    }
  }

  async function load() {
    setError(null);
    try {
      const [settings, pitStatus] = await Promise.all([getAdminSettings(), getPitPriceStatus()]);
      setEnabled(settings.pit_price_capture_enabled);
      setQuantSignalEnabled(settings.pit_quant_signal_capture_enabled);
      setAnalystRatingEnabled(settings.pit_analyst_rating_capture_enabled);
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

  async function handleToggleQuantSignal() {
    setQuantSignalBusy(true);
    setQuantSignalError(null);
    try {
      const result = quantSignalEnabled
        ? await disablePitQuantSignalCapture()
        : await enablePitQuantSignalCapture();
      setQuantSignalEnabled(result.pit_quant_signal_capture_enabled);
    } catch (err) {
      setQuantSignalError(err instanceof ApiError ? err.message : "Failed to update setting.");
    } finally {
      setQuantSignalBusy(false);
    }
  }

  async function handleCaptureQuantSignalNow() {
    setQuantSignalCapturing(true);
    setQuantSignalCaptureError(null);
    setQuantSignalCaptureResult(null);
    try {
      const res = await capturePitQuantSignalsNow();
      setQuantSignalCaptureResult(
        res.quant_signal_inserted > 0
          ? `Captured ${res.quant_signal_inserted} quant signal rows just now.`
          : "Already captured for today — nothing new."
      );
      const pitStatus = await getPitPriceStatus();
      setStatus(pitStatus);
    } catch (err) {
      setQuantSignalCaptureError(err instanceof ApiError ? err.message : "Failed to capture.");
    } finally {
      setQuantSignalCapturing(false);
    }
  }

  async function handleToggleAnalystRating() {
    setAnalystRatingBusy(true);
    setAnalystRatingError(null);
    try {
      const result = analystRatingEnabled
        ? await disablePitAnalystRatingCapture()
        : await enablePitAnalystRatingCapture();
      setAnalystRatingEnabled(result.pit_analyst_rating_capture_enabled);
    } catch (err) {
      setAnalystRatingError(err instanceof ApiError ? err.message : "Failed to update setting.");
    } finally {
      setAnalystRatingBusy(false);
    }
  }

  async function handleCaptureAnalystRatingNow() {
    setAnalystRatingCapturing(true);
    setAnalystRatingCaptureError(null);
    setAnalystRatingCaptureResult(null);
    try {
      const res = await capturePitAnalystRatingsNow();
      setAnalystRatingCaptureResult(
        res.analyst_rating_inserted > 0
          ? `Captured ${res.analyst_rating_inserted} analyst rating rows just now.`
          : "Already captured for today — nothing new."
      );
      const pitStatus = await getPitPriceStatus();
      setStatus(pitStatus);
    } catch (err) {
      setAnalystRatingCaptureError(err instanceof ApiError ? err.message : "Failed to capture.");
    } finally {
      setAnalystRatingCapturing(false);
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

      <div className="mt-4">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          Lookahead-safety reconciliation (Phase 4)
        </h3>
        <p className="mt-1 text-xs text-slate-500">
          Reconstructs a date&apos;s ranking purely from PIT data known as of that date, and checks it
          against what was actually published — TR-6&apos;s acceptance test. Only produces a full match
          once the PIT store has 31+ trading days of history for a 30-day lookback; before that it
          honestly reports which tickers aren&apos;t reconstructible yet, rather than a false pass.
        </p>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <input
            type="date"
            value={reconcileDate}
            onChange={(e) => setReconcileDate(e.target.value)}
            className="rounded-md border border-slate-300 px-2 py-1.5 text-sm"
          />
          <button
            onClick={handleReconcile}
            disabled={reconciling}
            className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
          >
            {reconciling ? "Checking…" : "Check Reconciliation"}
          </button>
        </div>

        {reconcileError && (
          <p className="mt-2 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{reconcileError}</p>
        )}

        {reconcileReport && (
          <div className="mt-3 rounded-md border border-slate-200 bg-slate-50 p-3">
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-medium text-slate-800">{reconcileReport.target_date}</span>
              <span
                className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
                  reconcileReport.byte_identical
                    ? "bg-emerald-50 text-emerald-700"
                    : "bg-amber-50 text-amber-700"
                }`}
              >
                {reconcileReport.byte_identical ? "Byte-identical" : "Not yet reconstructible"}
              </span>
            </div>
            <dl className="mt-2 grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
              <div>
                <dt className="text-slate-500">PIT history</dt>
                <dd className="font-medium text-slate-800">
                  {reconcileReport.pit_trading_days_available}/{reconcileReport.pit_trading_days_required} days
                </dd>
              </div>
              <div>
                <dt className="text-slate-500">Published</dt>
                <dd className="font-medium text-slate-800">{reconcileReport.published_count}</dd>
              </div>
              <div>
                <dt className="text-slate-500">Matched</dt>
                <dd className="font-medium text-slate-800">{reconcileReport.matches}</dd>
              </div>
              <div>
                <dt className="text-slate-500">Mismatched</dt>
                <dd className="font-medium text-slate-800">{reconcileReport.mismatches.length}</dd>
              </div>
            </dl>
            {reconcileReport.missing_from_pit_history.length > 0 && (
              <p className="mt-2 text-xs text-slate-500">
                {reconcileReport.missing_from_pit_history.length} ticker
                {reconcileReport.missing_from_pit_history.length === 1 ? "" : "s"} not yet reconstructible
                (not enough trailing PIT history): {reconcileReport.missing_from_pit_history.map((m) => m.ticker).join(", ")}
              </p>
            )}
            {reconcileReport.mismatches.length > 0 && (
              <p className="mt-2 text-xs text-red-600">
                Rank mismatches: {reconcileReport.mismatches.map((m) => `${m.ticker} (published #${m.published_rank}, reconstructed #${m.reconstructed_rank})`).join("; ")}
              </p>
            )}
          </div>
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
          onClick={handleCaptureNow}
          disabled={capturing}
          className="rounded-md border border-emerald-300 bg-white px-4 py-2 text-sm font-medium text-emerald-700 hover:bg-emerald-50 disabled:opacity-50"
        >
          {capturing ? "Capturing…" : "Capture Now"}
        </button>
      </div>

      {captureResult && <p className="mt-3 text-sm text-emerald-700">{captureResult}</p>}
      {captureError && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{captureError}</p>}

      <div className="mt-4 flex flex-col gap-4">
        <PitStoreCard
          title="Quant Signal (internal model)"
          description="Daily job (weekdays, 6:00pm ET, after market close) that trains the internal predict-page model per ticker and saves its signal, expected return, and target price. CPU-heavy (~500 tickers for the full S&P 500), so it's gated by its own flag independent of the price/fundamentals capture above and scheduled later to avoid resource contention with market-close activity. See Signal Stability for day-over-day flip analysis of this history."
          countLabel="Quant Signal"
          countValue={status?.quant_signal.ticker_count ?? 0}
          stats={status?.quant_signal ?? null}
          enabled={quantSignalEnabled}
          busy={quantSignalBusy}
          capturing={quantSignalCapturing}
          captureResult={quantSignalCaptureResult}
          captureError={quantSignalCaptureError}
          error={quantSignalError}
          onToggle={handleToggleQuantSignal}
          onCaptureNow={handleCaptureQuantSignalNow}
        />
        <PitStoreCard
          title="Analyst Rating (yfinance)"
          description="Daily job (weekdays, 4:07pm ET) that captures each ticker's Wall Street analyst consensus, target price, and buy percentage from Yahoo Finance. Cheap relative to the quant signal capture, so it runs right after the price/fundamentals job. Its own independent flag lets it be paused without affecting the other stores."
          countLabel="Analyst Rating"
          countValue={status?.analyst_rating.ticker_count ?? 0}
          stats={status?.analyst_rating ?? null}
          enabled={analystRatingEnabled}
          busy={analystRatingBusy}
          capturing={analystRatingCapturing}
          captureResult={analystRatingCaptureResult}
          captureError={analystRatingCaptureError}
          error={analystRatingError}
          onToggle={handleToggleAnalystRating}
          onCaptureNow={handleCaptureAnalystRatingNow}
        />
      </div>
    </div>
  );
}
