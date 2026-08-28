"use client";

import { useEffect, useState } from "react";

import { ApiError, getSignalStability } from "@/lib/api";
import type { SignalStabilityFlip, SignalStabilityReport } from "@/lib/types";

const LOOKBACK_OPTIONS = [14, 30, 60, 90];

const SIGNAL_BADGE_CLASS: Record<string, string> = {
  BUY: "bg-emerald-50 text-emerald-700",
  SELL: "bg-red-50 text-red-700",
  HOLD: "bg-slate-100 text-slate-600",
  UNKNOWN: "bg-slate-100 text-slate-400",
};

const CLASSIFICATION_INFO: Record<SignalStabilityFlip["classification"], { label: string; className: string; title: string }> = {
  boundary: {
    label: "Boundary",
    className: "bg-slate-100 text-slate-600",
    title: "expected_return_pct barely crossed the ±5% BUY/SELL cutoff — likely noise around a view that hasn't really changed.",
  },
  chase: {
    label: "Chase",
    className: "bg-amber-50 text-amber-700",
    title: "Swung sharply alongside the ticker's own big same-day price move — the model re-rating it right after the move, not forecasting it.",
  },
  model_shift: {
    label: "Model shift",
    className: "bg-sky-50 text-sky-700",
    title: "A meaningful change in the forecast that isn't explained by boundary noise or a big same-day price move.",
  },
};

function fmtPct(v: number | null): string {
  if (v === null) return "—";
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
}

function fmtPrice(v: number | null): string {
  return v === null ? "—" : `$${v.toFixed(2)}`;
}

export default function SignalStabilityPanel() {
  const [lookbackDays, setLookbackDays] = useState(30);
  const [report, setReport] = useState<SignalStabilityReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function load(days: number) {
    setLoading(true);
    setError(null);
    try {
      setReport(await getSignalStability(days));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load signal stability data.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load(lookbackDays);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lookbackDays]);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center gap-2">
        <label className="text-sm font-medium text-slate-600">Lookback</label>
        <select
          value={lookbackDays}
          onChange={(e) => setLookbackDays(Number(e.target.value))}
          className="rounded-md border border-slate-300 px-2 py-1.5 text-sm"
        >
          {LOOKBACK_OPTIONS.map((d) => (
            <option key={d} value={d}>{d} days</option>
          ))}
        </select>
        {report && (
          <span className="text-xs text-slate-500">
            Signal flips at ≥{report.buy_threshold_pct}% (BUY) / ≤{report.sell_threshold_pct}% (SELL) expected return.
          </span>
        )}
      </div>

      {error && <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {loading && !report && <p className="text-sm text-slate-500">Loading…</p>}

      {report && !loading && report.tickers.length === 0 && (
        <p className="text-sm text-slate-500">
          Not enough Quant Signal history yet — need at least two captured days to detect a flip.
        </p>
      )}

      {report && report.tickers.length > 0 && (
        <div>
          <h2 className="text-sm font-semibold text-slate-900">Flip rate by ticker</h2>
          <p className="mt-1 text-xs text-slate-500">
            Most-flipped tickers first — a high flip count relative to days captured means the signal isn&apos;t
            settling on a view.
          </p>
          <div className="mt-2 max-h-[26rem] overflow-auto rounded-lg border border-slate-200 bg-white">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="sticky top-0 border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                  <th className="px-3 py-2">Ticker</th>
                  <th className="px-3 py-2 text-right">Days Captured</th>
                  <th className="px-3 py-2 text-right">Flips</th>
                  <th className="px-3 py-2">Current Signal</th>
                  <th className="px-3 py-2 text-right">Current Streak</th>
                  <th className="px-3 py-2">Last Flip</th>
                </tr>
              </thead>
              <tbody>
                {report.tickers.map((t) => (
                  <tr key={t.ticker} className="border-b border-slate-100 last:border-0">
                    <td className="px-3 py-2 font-medium text-slate-800">{t.ticker}</td>
                    <td className="px-3 py-2 text-right text-slate-600">{t.days_captured}</td>
                    <td className="px-3 py-2 text-right text-slate-600">{t.flip_count}</td>
                    <td className="px-3 py-2">
                      <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${SIGNAL_BADGE_CLASS[t.current_signal]}`}>
                        {t.current_signal}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-right text-slate-600">{t.current_streak_days}d</td>
                    <td className="px-3 py-2 text-slate-600">{t.last_flip_date ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {report && report.recent_flips.length > 0 && (
        <div>
          <h2 className="text-sm font-semibold text-slate-900">Recent flips</h2>
          <p className="mt-1 text-xs text-slate-500">
            Most recent 50 signal changes across all tickers, with the model&apos;s forecast and the ticker&apos;s
            own price move around the flip.
          </p>
          <div className="mt-2 max-h-[26rem] overflow-auto rounded-lg border border-slate-200 bg-white">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="sticky top-0 border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                  <th className="px-3 py-2">Ticker</th>
                  <th className="px-3 py-2">Date</th>
                  <th className="px-3 py-2">Signal Change</th>
                  <th className="px-3 py-2 text-right">Expected Return</th>
                  <th className="px-3 py-2 text-right">Price Move</th>
                  <th className="px-3 py-2">Read</th>
                </tr>
              </thead>
              <tbody>
                {report.recent_flips.map((f, i) => {
                  const info = CLASSIFICATION_INFO[f.classification];
                  return (
                    <tr key={`${f.ticker}-${f.date}-${i}`} className="border-b border-slate-100 last:border-0">
                      <td className="px-3 py-2 font-medium text-slate-800">{f.ticker}</td>
                      <td className="px-3 py-2 text-slate-600">
                        {f.prev_date} → {f.date}
                      </td>
                      <td className="px-3 py-2">
                        <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${SIGNAL_BADGE_CLASS[f.prev_signal]}`}>
                          {f.prev_signal}
                        </span>
                        <span className="mx-1 text-slate-400">→</span>
                        <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${SIGNAL_BADGE_CLASS[f.signal]}`}>
                          {f.signal}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-right text-slate-600">
                        {fmtPct(f.prev_expected_return_pct)} → {fmtPct(f.expected_return_pct)}
                      </td>
                      <td className="px-3 py-2 text-right text-slate-600">
                        {fmtPrice(f.prev_close)} → {fmtPrice(f.last_close)}
                        {f.price_move_pct !== null && (
                          <span className={`ml-1 ${f.price_move_pct >= 0 ? "text-emerald-600" : "text-red-600"}`}>
                            ({fmtPct(f.price_move_pct)})
                          </span>
                        )}
                      </td>
                      <td className="px-3 py-2">
                        <span title={info.title} className={`rounded-full px-2 py-0.5 text-xs font-semibold ${info.className}`}>
                          {info.label}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
