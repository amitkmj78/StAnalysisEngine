"use client";

import { useEffect, useState } from "react";

import {
  ApiError,
  disableBasketRebalance,
  disableStockFinderCachePrewarm,
  enableBasketRebalance,
  enableStockFinderCachePrewarm,
  getAdminSettings,
  scanRebalanceAlertsNow,
} from "@/lib/api";

export default function BasketRebalanceControls() {
  const [rebalanceEnabled, setRebalanceEnabled] = useState<boolean | null>(null);
  const [prewarmEnabled, setPrewarmEnabled] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<"rebalance" | "prewarm" | null>(null);
  const [confirming, setConfirming] = useState<"rebalance" | "prewarm" | null>(null);

  const [scanning, setScanning] = useState(false);
  const [scanResult, setScanResult] = useState<string | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);

  async function load() {
    setError(null);
    try {
      const settings = await getAdminSettings();
      setRebalanceEnabled(settings.basket_rebalance_enabled);
      setPrewarmEnabled(settings.stock_finder_cache_prewarm_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load Diversified Basket settings.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleToggleRebalance() {
    if (!rebalanceEnabled && confirming !== "rebalance") {
      setConfirming("rebalance");
      return;
    }
    setBusy("rebalance");
    setError(null);
    setConfirming(null);
    try {
      const result = rebalanceEnabled ? await disableBasketRebalance() : await enableBasketRebalance();
      setRebalanceEnabled(result.basket_rebalance_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update basket rebalance setting.");
    } finally {
      setBusy(null);
    }
  }

  async function handleTogglePrewarm() {
    if (!prewarmEnabled && confirming !== "prewarm") {
      setConfirming("prewarm");
      return;
    }
    setBusy("prewarm");
    setError(null);
    setConfirming(null);
    try {
      const result = prewarmEnabled ? await disableStockFinderCachePrewarm() : await enableStockFinderCachePrewarm();
      setPrewarmEnabled(result.stock_finder_cache_prewarm_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update cache prewarm setting.");
    } finally {
      setBusy(null);
    }
  }

  async function handleScanNow() {
    setScanning(true);
    setScanError(null);
    setScanResult(null);
    try {
      const res = await scanRebalanceAlertsNow();
      setScanResult(
        res.inserted > 0
          ? `Inserted/refreshed ${res.inserted} rebalance alert${res.inserted === 1 ? "" : "s"} just now.`
          : "Scanned every due basket — nothing needed a rebalance alert."
      );
    } catch (err) {
      setScanError(err instanceof ApiError ? err.message : "Failed to scan.");
    } finally {
      setScanning(false);
    }
  }

  return (
    <div className="rounded-lg border border-amber-200 bg-amber-50/40 p-5">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="font-semibold text-slate-900">Diversified Basket Rebalancing</h2>
          <p className="mt-1 text-sm text-slate-600">
            Monthly (1st of the month) and quarterly (Jan/Apr/Jul/Oct) re-ranks every saved basket&apos;s
            original universe/goal, flags weight drift past its own threshold and any holding that fell out
            of its sector&apos;s fresh top-N, and writes a review-and-act alert — never executes a trade
            automatically.
          </p>
        </div>
        {rebalanceEnabled !== null && (
          <span
            className={`shrink-0 rounded-full px-3 py-1 text-xs font-medium ${
              rebalanceEnabled ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-500"
            }`}
          >
            {rebalanceEnabled ? "Scanning" : "Disabled"}
          </span>
        )}
      </div>

      {error && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {confirming === "rebalance" && !rebalanceEnabled && (
        <p className="mt-3 rounded-md bg-amber-100 px-3 py-2 text-sm text-amber-800">
          Click again to confirm — this starts writing user-visible rebalance alerts.
        </p>
      )}

      <div className="mt-4 flex items-center gap-2">
        <button
          onClick={handleToggleRebalance}
          disabled={busy === "rebalance" || rebalanceEnabled === null}
          className={`rounded-md px-4 py-2 text-sm font-medium disabled:opacity-50 ${
            rebalanceEnabled
              ? "border border-red-200 text-red-700 hover:bg-red-50"
              : confirming === "rebalance"
              ? "bg-amber-600 text-white hover:bg-amber-700"
              : "bg-slate-900 text-white hover:bg-slate-800"
          }`}
        >
          {busy === "rebalance" ? "Updating…" : rebalanceEnabled ? "Disable" : confirming === "rebalance" ? "Confirm Enable" : "Enable"}
        </button>
        {confirming === "rebalance" && !rebalanceEnabled && (
          <button
            onClick={() => setConfirming(null)}
            disabled={busy === "rebalance"}
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

      <div className="mt-5 border-t border-amber-200 pt-4">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h3 className="font-medium text-slate-900">Stock-Finder Cache Prewarm</h3>
            <p className="mt-1 text-sm text-slate-600">
              Every ~50 minutes, rescans the &quot;All&quot; and S&amp;P 500 universes purely to keep the
              Diversified Basket page&apos;s &lt;3s generation goal met in steady state. A real, continuous
              increase in Yahoo Finance traffic — a genuine cost/rate-limit tradeoff, not free.
            </p>
          </div>
          {prewarmEnabled !== null && (
            <span
              className={`shrink-0 rounded-full px-3 py-1 text-xs font-medium ${
                prewarmEnabled ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-500"
              }`}
            >
              {prewarmEnabled ? "Prewarming" : "Disabled"}
            </span>
          )}
        </div>

        {confirming === "prewarm" && !prewarmEnabled && (
          <p className="mt-3 rounded-md bg-amber-100 px-3 py-2 text-sm text-amber-800">
            Click again to confirm — this starts a continuous background scan every ~50 minutes.
          </p>
        )}

        <div className="mt-3 flex items-center gap-2">
          <button
            onClick={handleTogglePrewarm}
            disabled={busy === "prewarm" || prewarmEnabled === null}
            className={`rounded-md px-4 py-2 text-sm font-medium disabled:opacity-50 ${
              prewarmEnabled
                ? "border border-red-200 text-red-700 hover:bg-red-50"
                : confirming === "prewarm"
                ? "bg-amber-600 text-white hover:bg-amber-700"
                : "bg-slate-900 text-white hover:bg-slate-800"
            }`}
          >
            {busy === "prewarm" ? "Updating…" : prewarmEnabled ? "Disable" : confirming === "prewarm" ? "Confirm Enable" : "Enable"}
          </button>
          {confirming === "prewarm" && !prewarmEnabled && (
            <button
              onClick={() => setConfirming(null)}
              disabled={busy === "prewarm"}
              className="rounded-md border border-slate-300 px-3 py-2 text-sm font-medium text-slate-600 hover:bg-slate-100"
            >
              Cancel
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
