"use client";

import { useEffect, useState } from "react";

import InfoModal, { type ColumnInfo } from "@/components/InfoModal";
import { ApiError, deleteBaselineSnapshot, getBaselineBand, getBaselineHistory, saveBaselineSnapshot } from "@/lib/api";
import type { BaselineBand, SavedBaselineSnapshot } from "@/lib/types";

const HORIZONS = [10, 30, 60, 90];
const CONFIDENCES = [0.9, 0.95];

const METHOD_INFO: ColumnInfo = {
  title: "Safe Baseline Price Band — what this is",
  body: [
    "A report card on the past, not a prediction of the future. It looks at years of this ticker's real price history and asks: every time someone bought at a given price and held for N trading days, what actually happened — how far did it typically dip, and how far did it typically run?",
    "Floor and Ceiling: historically, price has rarely moved outside this range within the horizon you picked. Accumulation Zone and Distribution Zone: a \"typical\" dip and a \"typical\" rally — where a normal pullback has bottomed, or a normal run has topped out. Median Path is simply today's price, unmodified — an anchor, not a forecast.",
    "How people use it: if the price falls into the Accumulation Zone, that's historically a normal-to-deep pullback, not usually a sign something's broken. If it's already above the Distribution Zone, it's already had a historically strong run, with typically less room left before a pause.",
    "The trust-check numbers below the band tell you how much history this is built on (\"Samples\") and how well-calibrated it's been (\"Breach Rate\" — how often price has actually broken the floor vs. how often the math expected it to).",
    "The most important caveat: this only looks at price history. It knows nothing about earnings, news, or what's happening with the company right now — treat it as one input, not the whole picture.",
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

  const [history, setHistory] = useState<SavedBaselineSnapshot[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [compareId, setCompareId] = useState<number | null>(null);

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

  async function loadHistory() {
    setHistoryLoading(true);
    try {
      const res = await getBaselineHistory(ticker.trim().toUpperCase());
      setHistory(res.snapshots);
    } catch {
      // Non-fatal — history is supplementary.
    } finally {
      setHistoryLoading(false);
    }
  }

  useEffect(() => {
    if (!ticker.trim()) return;
    setCompareId(null);
    loadHistory();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticker]);

  async function handleSave() {
    if (!band) return;
    setSaving(true);
    setSaveMessage(null);
    try {
      await saveBaselineSnapshot(band);
      setSaveMessage("Saved — reload this band later (e.g. after price moves) to compare against it.");
      await loadHistory();
    } catch (err) {
      setSaveMessage(err instanceof ApiError ? err.message : "Could not save this snapshot.");
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete(id: number) {
    setDeletingId(id);
    try {
      await deleteBaselineSnapshot(id);
      setHistory((prev) => prev.filter((s) => s.id !== id));
      if (compareId === id) setCompareId(null);
    } catch {
      // Non-fatal — leave the row in place if delete failed.
    } finally {
      setDeletingId(null);
    }
  }

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

          <p className="text-sm text-slate-600">
            A typical pullback for {band.ticker} has bottomed around{" "}
            <strong className="text-slate-900">${band.accumulation_zone_hi.toFixed(2)}</strong>; a typical rally has
            topped out around <strong className="text-slate-900">${band.distribution_zone_lo.toFixed(2)}</strong>.
            Over the last {band.effective_samples} independent {band.horizon_days}-day periods, price has stayed
            within <strong className="text-slate-900">${band.floor.toFixed(2)}–${band.ceiling.toFixed(2)}</strong>{" "}
            about {Math.round((1 - band.breach_rate_full) * 100)}% of the time. This reflects price history only —
            not news, earnings, or anything specific to the company right now.
          </p>

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

          <div className="flex items-center gap-3 border-t border-slate-100 pt-3">
            <button
              onClick={handleSave}
              disabled={saving}
              className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
            >
              {saving ? "Saving…" : "Save this snapshot"}
            </button>
            {saveMessage && <span className="text-xs text-slate-500">{saveMessage}</span>}
          </div>
        </div>
      )}

      {(historyLoading || history.length > 0) && (
        <div className="mt-4 border-t border-slate-200 pt-4">
          <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Saved Snapshots for {ticker.trim().toUpperCase()}
          </h4>
          {historyLoading ? (
            <p className="mt-2 text-sm text-slate-500">Loading saved snapshots…</p>
          ) : (
            <div className="mt-2 flex flex-col gap-2">
              {history.map((s) => (
                <div
                  key={s.id}
                  className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-slate-200 px-3 py-2 text-sm"
                >
                  <span className="text-slate-700">
                    {new Date(s.saved_at).toLocaleString()} &middot; {s.horizon_days}d @ {Math.round(s.confidence * 100)}%
                    &middot; floor ${s.floor.toFixed(2)} / ceiling ${s.ceiling.toFixed(2)}
                  </span>
                  <span className="flex items-center gap-2">
                    <button
                      onClick={() => setCompareId(compareId === s.id ? null : s.id)}
                      className="rounded-md border border-slate-300 px-2 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
                    >
                      {compareId === s.id ? "Hide Compare" : "Compare"}
                    </button>
                    <button
                      onClick={() => handleDelete(s.id)}
                      disabled={deletingId === s.id}
                      className="rounded-md border border-red-200 px-2 py-1 text-xs font-medium text-red-700 hover:bg-red-50 disabled:opacity-50"
                    >
                      {deletingId === s.id ? "…" : "Delete"}
                    </button>
                  </span>
                </div>
              ))}
            </div>
          )}

          {compareId !== null && (() => {
            const saved = history.find((s) => s.id === compareId);
            if (!saved) return null;
            return <CompareSnapshot saved={saved} current={band} />;
          })()}
        </div>
      )}

      {showInfo && <InfoModal info={METHOD_INFO} onClose={() => setShowInfo(false)} />}
    </div>
  );
}

function CompareSnapshot({ saved, current }: { saved: SavedBaselineSnapshot; current: BaselineBand | null }) {
  const sameParams = current && saved.horizon_days === current.horizon_days && saved.confidence === current.confidence;

  return (
    <div className="mt-3 rounded-md border border-slate-200 bg-slate-50 p-3">
      <p className="text-xs font-semibold text-slate-500">
        Saved {new Date(saved.saved_at).toLocaleString()} ({saved.horizon_days}d @ {Math.round(saved.confidence * 100)}%,
        as of {saved.as_of}) vs. current
      </p>
      {!current ? (
        <p className="mt-2 text-sm text-slate-400">Load a band above to compare.</p>
      ) : (
        <>
          {!sameParams && (
            <p className="mt-2 text-xs text-amber-700">
              Saved at a different horizon/confidence than what&apos;s shown above — levels won&apos;t line up
              apples-to-apples.
            </p>
          )}
          <div className="mt-2 overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                  <th className="px-2 py-1.5">Level</th>
                  <th className="px-2 py-1.5">Saved</th>
                  <th className="px-2 py-1.5">Current</th>
                  <th className="px-2 py-1.5">Change</th>
                </tr>
              </thead>
              <tbody>
                {[
                  { label: "Ceiling", savedVal: saved.ceiling, currentVal: current.ceiling },
                  { label: "Distribution Zone", savedVal: saved.distribution_zone_lo, currentVal: current.distribution_zone_lo },
                  { label: "Median Path", savedVal: saved.median_path, currentVal: current.median_path },
                  { label: "Accumulation Zone", savedVal: saved.accumulation_zone_hi, currentVal: current.accumulation_zone_hi },
                  { label: "Floor", savedVal: saved.floor, currentVal: current.floor },
                ].map((row) => {
                  const delta = row.currentVal - row.savedVal;
                  const deltaPct = (delta / row.savedVal) * 100;
                  return (
                    <tr key={row.label} className="border-b border-slate-100 last:border-0">
                      <td className="px-2 py-1.5 text-slate-700">{row.label}</td>
                      <td className="px-2 py-1.5 text-slate-700">${row.savedVal.toFixed(2)}</td>
                      <td className="px-2 py-1.5 font-medium text-slate-900">${row.currentVal.toFixed(2)}</td>
                      <td className={`px-2 py-1.5 ${delta > 0 ? "text-emerald-600" : delta < 0 ? "text-red-600" : "text-slate-400"}`}>
                        {delta > 0 ? "+" : ""}
                        {delta.toFixed(2)} ({deltaPct > 0 ? "+" : ""}
                        {deltaPct.toFixed(2)}%)
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
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
