"use client";

import { useEffect, useState } from "react";

import InfoModal, { type ColumnInfo } from "@/components/InfoModal";
import { ApiError, getBaselineBand } from "@/lib/api";
import type { BaselineBand } from "@/lib/types";

const HORIZONS = [10, 30, 60, 90];
const CONFIDENCES = [0.9, 0.95];

const METHOD_INFO: ColumnInfo = {
  title: "Safe Baseline Price Band — what this is",
  body: [
    "Five levels derived purely from this ticker's own historical price paths: for every day in the lookback window, \"if you'd bought here, what's the worst and best this position did over the following N trading days?\" The floor and ceiling are the (1 − confidence) and confidence percentiles of that outcome distribution — calibrated so roughly `confidence` of historical windows stayed inside them.",
    "This is not a forecast, target, or recommendation. It incorporates no earnings, guidance, corporate actions, or news — only realized historical excursions. \"Median Path\" is simply today's price, unmodified — an anchor, not a prediction.",
    "The Accumulation and Distribution zones mark a typical (25th/75th percentile) pullback or rally depth, independent of your chosen confidence level — like the box in a box plot.",
    "Different from Predict's forecast confidence interval (model-based) or Entry Signals' stop/target (an ATR heuristic) — each uses a different method, so don't expect the numbers to match across pages.",
  ],
};

export default function SafeBaselineBand({ ticker }: { ticker: string }) {
  const [horizon, setHorizon] = useState(30);
  const [confidence, setConfidence] = useState(0.9);
  const [band, setBand] = useState<BaselineBand | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [insufficientHistory, setInsufficientHistory] = useState<string | null>(null);
  const [showInfo, setShowInfo] = useState(false);

  useEffect(() => {
    if (!ticker.trim()) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setInsufficientHistory(null);
    getBaselineBand(ticker.trim().toUpperCase(), { horizon, confidence })
      .then((res) => {
        if (cancelled) return;
        setBand(res);
      })
      .catch((err) => {
        if (cancelled) return;
        setBand(null);
        if (err instanceof ApiError && err.status === 422) {
          setInsufficientHistory(err.message);
        } else if (err instanceof ApiError && err.status === 404) {
          setError(`No price history available for ${ticker.trim().toUpperCase()}.`);
        } else {
          setError(err instanceof ApiError ? err.message : "Could not load the price band.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [ticker, horizon, confidence]);

  if (!ticker.trim()) return null;

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-1.5">
          <h3 className="font-semibold text-slate-900">Safe Baseline Price Band</h3>
          <button
            type="button"
            onClick={() => setShowInfo(true)}
            title="What is the Safe Baseline Price Band?"
            className="flex h-3.5 w-3.5 items-center justify-center rounded-full border border-slate-300 text-[9px] font-normal text-slate-400 hover:border-slate-500 hover:text-slate-700"
          >
            i
          </button>
        </div>
        <div className="flex flex-wrap items-end gap-2">
          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-slate-500">Horizon</label>
            <select
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
              className="rounded-md border border-slate-300 px-2 py-1 text-xs"
            >
              {HORIZONS.map((h) => (
                <option key={h} value={h}>
                  {h}d
                </option>
              ))}
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-slate-500">Confidence</label>
            <select
              value={confidence}
              onChange={(e) => setConfidence(Number(e.target.value))}
              className="rounded-md border border-slate-300 px-2 py-1 text-xs"
            >
              {CONFIDENCES.map((c) => (
                <option key={c} value={c}>
                  {Math.round(c * 100)}%
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <p className="mt-2 text-xs text-slate-500">
        Derived from historical price paths only — not a prediction, target, or recommendation.
      </p>

      {loading && <p className="mt-4 text-sm text-slate-500">Computing band…</p>}

      {insufficientHistory && !loading && (
        <p className="mt-4 rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-800">
          Not enough price history yet for a {horizon}-day band on {ticker.trim().toUpperCase()}. Try a shorter
          horizon.
        </p>
      )}

      {error && !loading && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {band && !loading && (
        <div className="mt-4 flex flex-col gap-4">
          <BandLadder band={band} />

          {confidence >= 0.95 && (
            <p className="text-xs text-slate-500">
              At 95% confidence, the floor/ceiling are indicative rather than precisely measured — there are
              fewer independent historical windows this far into the tail.
            </p>
          )}

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <SmallStat label="Samples" value={`${band.effective_samples} indep. / ${band.samples} raw`} />
            <SmallStat
              label="Breach Rate"
              value={`${(band.breach_rate_full * 100).toFixed(1)}% (expect ${(band.expected_breach * 100).toFixed(0)}%)`}
              warn={band.calibration_warning}
            />
            <SmallStat label="Reward:Risk" value={band.rr_ratio !== null ? band.rr_ratio.toFixed(2) : "—"} />
            <SmallStat label="Upside-First Rate" value={`${(band.upside_first_rate * 100).toFixed(0)}%`} />
          </div>

          {band.calibration_warning && (
            <p className="text-xs text-amber-700">
              This band's actual historical breach rate diverges from what {Math.round(confidence * 100)}%
              confidence implies by more than 5 points — treat it as a rough guide for this ticker rather than a
              tightly calibrated one.
            </p>
          )}

          <p className="text-xs text-slate-400">As of {band.as_of} &middot; method: {band.method}</p>
        </div>
      )}

      {showInfo && <InfoModal info={METHOD_INFO} onClose={() => setShowInfo(false)} />}
    </div>
  );
}

function BandLadder({ band }: { band: BaselineBand }) {
  const levels = [
    { key: "ceiling", label: "Ceiling", price: band.ceiling, pct: band.ceiling_pct },
    { key: "distribution_zone_lo", label: "Distribution Zone", price: band.distribution_zone_lo, pct: band.distribution_zone_lo_pct },
    { key: "median_path", label: "Median Path (today's price)", price: band.median_path, pct: band.median_path_pct },
    { key: "accumulation_zone_hi", label: "Accumulation Zone", price: band.accumulation_zone_hi, pct: band.accumulation_zone_hi_pct },
    { key: "floor", label: "Floor", price: band.floor, pct: band.floor_pct },
  ];

  return (
    <div className="flex flex-col divide-y divide-slate-100 overflow-hidden rounded-md border border-slate-200">
      {levels.map((level) => (
        <div
          key={level.key}
          className={`flex items-center justify-between px-3 py-2 ${
            level.key === "median_path" ? "bg-slate-50" : ""
          }`}
        >
          <span className={`text-sm ${level.key === "median_path" ? "font-medium text-slate-900" : "text-slate-600"}`}>
            {level.label}
          </span>
          <span className="flex items-baseline gap-2">
            <span className="text-sm font-semibold text-slate-900">${level.price.toFixed(2)}</span>
            <span className={`text-xs ${level.pct > 0 ? "text-emerald-600" : level.pct < 0 ? "text-red-600" : "text-slate-400"}`}>
              {level.pct > 0 ? "+" : ""}
              {level.pct.toFixed(2)}%
            </span>
          </span>
        </div>
      ))}
    </div>
  );
}

function SmallStat({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div className={`rounded-md border p-2 ${warn ? "border-amber-200 bg-amber-50" : "border-slate-200 bg-white"}`}>
      <p className="text-[11px] text-slate-500">{label}</p>
      <p className={`mt-0.5 text-sm font-semibold ${warn ? "text-amber-800" : "text-slate-900"}`}>{value}</p>
    </div>
  );
}
