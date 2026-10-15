"use client";

import { useEffect, useState } from "react";

import {
  ApiError,
  disableStrategyForwardPaperJob,
  disableStrategyForwardPaperKillSwitch,
  enableStrategyForwardPaperJob,
  enableStrategyForwardPaperKillSwitch,
  getAdminSettings,
} from "@/lib/api";

export default function StrategyForwardPaperControls() {
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [killSwitch, setKillSwitch] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [killBusy, setKillBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setError(null);
    try {
      const settings = await getAdminSettings();
      setEnabled(settings.strategy_forward_paper_enabled);
      setKillSwitch(settings.strategy_forward_paper_kill_switch);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load forward-paper-trading status.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleToggle() {
    setBusy(true);
    setError(null);
    try {
      const result = enabled ? await disableStrategyForwardPaperJob() : await enableStrategyForwardPaperJob();
      setEnabled(result.strategy_forward_paper_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update setting.");
    } finally {
      setBusy(false);
    }
  }

  async function handleKillSwitch() {
    setKillBusy(true);
    setError(null);
    try {
      const result = killSwitch
        ? await disableStrategyForwardPaperKillSwitch()
        : await enableStrategyForwardPaperKillSwitch();
      setKillSwitch(result.strategy_forward_paper_kill_switch);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update the kill switch.");
    } finally {
      setKillBusy(false);
    }
  }

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="font-semibold text-slate-900 dark:text-slate-50">Strategy Forward-Paper Trading (STB-6)</h2>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
            Daily job (weekdays ~15:40 ET) that runs every self-service-enabled saved strategy forward against its
            own, real (never live) Alpaca paper account -- a strategy owner links their own key pair and accepts the
            disclosure per strategy; no per-strategy admin approval, since these are the owner&apos;s own
            fully-inspectable rules, not an opaque company ruleset. Every order runs through the same pre-trade
            checks a manual paper-trading ticket does. Off by default: deploying the code must not itself start
            submitting orders.
          </p>
        </div>
        {enabled !== null && (
          <span
            className={`shrink-0 rounded-full px-3 py-1 text-xs font-medium ${
              enabled ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400"
            }`}
          >
            {enabled ? "Running" : "Off"}
          </span>
        )}
      </div>

      {error && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <button
          onClick={handleToggle}
          disabled={busy || enabled === null}
          className={`rounded-md px-4 py-2 text-sm font-medium disabled:opacity-50 ${
            enabled ? "border border-red-200 text-red-700 hover:bg-red-50" : "bg-slate-900 text-white hover:bg-slate-800"
          }`}
        >
          {busy ? "Updating…" : enabled ? "Turn off" : "Turn on"}
        </button>
        <button
          onClick={handleKillSwitch}
          disabled={killBusy || killSwitch === null}
          className={`rounded-md px-4 py-2 text-sm font-medium disabled:opacity-50 ${
            killSwitch
              ? "bg-amber-600 text-white hover:bg-amber-700"
              : "border border-amber-300 text-amber-700 hover:bg-amber-50"
          }`}
        >
          {killBusy ? "Updating…" : killSwitch ? "Kill switch engaged -- release" : "Engage global kill switch"}
        </button>
      </div>
    </div>
  );
}
