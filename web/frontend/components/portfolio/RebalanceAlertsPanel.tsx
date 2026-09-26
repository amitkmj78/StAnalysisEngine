"use client";

import { useEffect, useState } from "react";

import { ApiError, applyRebalanceAlert, dismissRebalanceAlert, getRebalanceAlerts } from "@/lib/api";
import type { BasketRebalanceAlert, Portfolio } from "@/lib/types";

// Diversified Basket rebalance alerts (approved scope addition alongside
// DI-01 through DI-09): a scheduled job flags weight drift and picks
// that fell out of their sector's fresh top-N, but never trades on its
// own -- this panel is the review-and-act surface. First non-admin-
// gated alert UI in this app (unlike Portfolio Drop Alerts, which is
// admin-only today), so it's built fresh rather than mirrored from an
// existing user-facing pattern.

function fmtPct(v: number, digits = 1) {
  return `${v >= 0 ? "+" : ""}${v.toFixed(digits)}%`;
}

export default function RebalanceAlertsPanel({ allPortfolios }: { allPortfolios: Portfolio[] }) {
  const [alerts, setAlerts] = useState<BasketRebalanceAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<number | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const res = await getRebalanceAlerts();
      setAlerts(res.alerts);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load rebalance alerts.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleApply(id: number) {
    setBusyId(id);
    setActionError(null);
    try {
      await applyRebalanceAlert(id);
      setAlerts((prev) => prev.filter((a) => a.id !== id));
    } catch (err) {
      setActionError(err instanceof ApiError ? err.message : "Could not apply this rebalance.");
    } finally {
      setBusyId(null);
    }
  }

  async function handleDismiss(id: number) {
    setBusyId(id);
    setActionError(null);
    try {
      await dismissRebalanceAlert(id);
      setAlerts((prev) => prev.filter((a) => a.id !== id));
    } catch (err) {
      setActionError(err instanceof ApiError ? err.message : "Could not dismiss this alert.");
    } finally {
      setBusyId(null);
    }
  }

  if (loading || error || alerts.length === 0) return null;

  return (
    <div className="mt-4 flex flex-col gap-3">
      {actionError && (
        <p className="rounded-md border border-[#e4c9c5] bg-[#fbeceb] px-3 py-2 text-sm text-[#a23b34]">{actionError}</p>
      )}
      {alerts.map((alert) => {
        const portfolioName = allPortfolios.find((p) => p.id === alert.portfolio_id)?.name ?? `Portfolio #${alert.portfolio_id}`;
        return (
          <div key={alert.id} className="rounded-xl border border-[#e3cf9c] bg-[#faf3e2] p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="font-semibold text-[#1f2420]">
                Rebalance suggested — {portfolioName}
              </span>
              <span className="rounded-full bg-[#f6e5e3] px-2.5 py-1 text-xs font-semibold text-[#a23b34]">
                Max drift {alert.max_drift_pct.toFixed(1)} pts
              </span>
            </div>
            <p className="mt-1 text-xs text-[#a39b8b]">Scores as of {alert.score_as_of}</p>

            {alert.drift_summary.length > 0 && (
              <div className="mt-2">
                <p className="text-xs font-semibold uppercase tracking-wide text-[#8a6417]">Weight drift</p>
                <ul className="mt-1 space-y-0.5 text-sm text-[#514c43]">
                  {alert.drift_summary.map((d) => (
                    <li key={d.ticker}>
                      <span className="font-medium text-[#1f2420]">{d.ticker}</span>: target{" "}
                      {d.target_weight_pct.toFixed(1)}% → now {d.current_weight_pct.toFixed(1)}% ({fmtPct(d.drift_pct)})
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {alert.suggested_swaps.length > 0 && (
              <div className="mt-2">
                <p className="text-xs font-semibold uppercase tracking-wide text-[#8a6417]">Suggested swaps</p>
                <ul className="mt-1 space-y-0.5 text-sm text-[#514c43]">
                  {alert.suggested_swaps.map((s) => (
                    <li key={s.sell_ticker}>
                      Sell <span className="font-medium text-[#1f2420]">{s.sell_ticker}</span>
                      {s.buy_ticker ? (
                        <>
                          {" "}→ buy <span className="font-medium text-[#1f2420]">{s.buy_ticker}</span>
                          {s.buy_name ? ` (${s.buy_name})` : ""}
                        </>
                      ) : (
                        " — no same-sector replacement available"
                      )}
                      <span className="text-[#a39b8b]"> · {s.reason}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="mt-3 flex items-center gap-2">
              <button
                type="button"
                onClick={() => handleApply(alert.id)}
                disabled={busyId === alert.id}
                className="rounded-md bg-[#2f5d50] px-3 py-1.5 text-sm font-semibold text-[#f4f1ea] hover:bg-[#274e43] disabled:opacity-50"
              >
                {busyId === alert.id ? "Applying…" : "Apply Rebalance"}
              </button>
              <button
                type="button"
                onClick={() => handleDismiss(alert.id)}
                disabled={busyId === alert.id}
                className="rounded-md border border-[#ddd8cd] bg-white px-3 py-1.5 text-sm font-medium text-[#1f2420] hover:border-[#2f5d50] hover:text-[#2f5d50] disabled:opacity-50"
              >
                Dismiss
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
