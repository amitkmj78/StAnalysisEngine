"use client";

import { Fragment, useEffect, useState } from "react";

import { ApiError, createTrade, deleteTrade, evaluateTrades, listTrades } from "@/lib/api";
import type { Trade } from "@/lib/types";

const DIRECTIONS = ["LONG", "SHORT"];
const STRATEGY_TYPES = ["Discretionary", "Breakout", "Pullback", "Swing", "Momentum"];

export default function TradeJournalPage() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [evaluating, setEvaluating] = useState(false);

  const [ticker, setTicker] = useState("");
  const [direction, setDirection] = useState("LONG");
  const [strategyType, setStrategyType] = useState("Discretionary");
  const [entryLow, setEntryLow] = useState("");
  const [entryHigh, setEntryHigh] = useState("");
  const [stopLoss, setStopLoss] = useState("");
  const [target, setTarget] = useState("");
  const [context, setContext] = useState("");
  const [riskProfile, setRiskProfile] = useState("");
  const [riskFactor, setRiskFactor] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const res = await listTrades();
      setTrades(res.trades);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load trades.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!ticker.trim() || !entryLow || !entryHigh || !stopLoss || !target) return;
    setSubmitting(true);
    setError(null);
    try {
      await createTrade({
        ticker: ticker.trim().toUpperCase(),
        direction,
        strategy_type: strategyType,
        entry_low: Number(entryLow),
        entry_high: Number(entryHigh),
        stop_loss: Number(stopLoss),
        target: Number(target),
        context,
        risk_profile: riskProfile,
        risk_factor: riskFactor ? Number(riskFactor) : null,
      });
      setTicker("");
      setEntryLow("");
      setEntryHigh("");
      setStopLoss("");
      setTarget("");
      setContext("");
      setRiskProfile("");
      setRiskFactor("");
      await refresh();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not save trade.");
    } finally {
      setSubmitting(false);
    }
  }

  async function runEvaluate() {
    setEvaluating(true);
    setError(null);
    try {
      await evaluateTrades();
      await refresh();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not evaluate trades.");
    } finally {
      setEvaluating(false);
    }
  }

  async function removeTrade(tradeId: string) {
    setError(null);
    try {
      await deleteTrade(tradeId);
      setTrades((prev) => prev.filter((t) => t.trade_id !== tradeId));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not delete trade.");
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">Trade Journal</h1>
      <p className="mt-1 text-sm text-slate-500">
        Log trade ideas with an entry range, stop, and target, then re-evaluate them against live prices.
      </p>

      <form onSubmit={submit} className="mt-6 flex flex-col gap-3 rounded-lg border border-slate-200 bg-white p-5">
        <h2 className="text-sm font-semibold text-slate-900">New Trade Idea</h2>
        <div className="flex flex-wrap items-end gap-3">
          <Field label="Ticker">
            <input value={ticker} onChange={(e) => setTicker(e.target.value)} className="input w-24 uppercase" maxLength={10} />
          </Field>
          <Field label="Direction">
            <select value={direction} onChange={(e) => setDirection(e.target.value)} className="input">
              {DIRECTIONS.map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </Field>
          <Field label="Strategy type">
            <select value={strategyType} onChange={(e) => setStrategyType(e.target.value)} className="input">
              {STRATEGY_TYPES.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </Field>
          <Field label="Entry low">
            <input type="number" step="0.01" value={entryLow} onChange={(e) => setEntryLow(e.target.value)} className="input w-24" />
          </Field>
          <Field label="Entry high">
            <input type="number" step="0.01" value={entryHigh} onChange={(e) => setEntryHigh(e.target.value)} className="input w-24" />
          </Field>
          <Field label="Stop loss">
            <input type="number" step="0.01" value={stopLoss} onChange={(e) => setStopLoss(e.target.value)} className="input w-24" />
          </Field>
          <Field label="Target">
            <input type="number" step="0.01" value={target} onChange={(e) => setTarget(e.target.value)} className="input w-24" />
          </Field>
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <Field label="Risk profile">
            <input value={riskProfile} onChange={(e) => setRiskProfile(e.target.value)} className="input w-32" placeholder="e.g. Balanced" />
          </Field>
          <Field label="Risk factor">
            <input type="number" value={riskFactor} onChange={(e) => setRiskFactor(e.target.value)} className="input w-20" />
          </Field>
          <Field label="Context / notes">
            <input value={context} onChange={(e) => setContext(e.target.value)} className="input w-64" />
          </Field>
        </div>
        <button type="submit" disabled={submitting} className="btn-primary self-start">
          {submitting ? "Saving…" : "Save Trade"}
        </button>
      </form>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <div className="mt-6 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-900">Your Trades</h2>
        <button
          onClick={runEvaluate}
          disabled={evaluating || trades.length === 0}
          className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
        >
          {evaluating ? "Evaluating…" : "Re-evaluate against live prices"}
        </button>
      </div>

      {loading ? (
        <p className="mt-4 text-sm text-slate-500">Loading trades…</p>
      ) : trades.length === 0 ? (
        <p className="mt-4 text-sm text-slate-500">No trades logged yet.</p>
      ) : (
        <div className="mt-4 overflow-x-auto rounded-lg border border-slate-200 bg-white">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                <th className="px-3 py-2">Ticker</th>
                <th className="px-3 py-2">Current Price</th>
                <th className="px-3 py-2">Dir</th>
                <th className="px-3 py-2">Entry Range</th>
                <th className="px-3 py-2">Stop</th>
                <th className="px-3 py-2">Target</th>
                <th className="px-3 py-2">R:R</th>
                <th className="px-3 py-2">Status</th>
                <th className="px-3 py-2">PnL %</th>
                <th className="px-3 py-2"></th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t) => {
                const pnl = t.realized_pnl_pct ?? t.unrealized_pnl_pct;
                const pnlIsUnrealized = t.realized_pnl_pct === null && t.unrealized_pnl_pct !== null;
                return (
                  <Fragment key={t.trade_id}>
                    <tr className="border-b border-slate-100 last:border-0">
                      <td className="px-3 py-2 font-medium text-slate-900">{t.ticker}</td>
                      <td className="px-3 py-2 text-slate-700">
                        {t.current_price !== null ? `$${t.current_price.toFixed(2)}` : "—"}
                      </td>
                      <td className="px-3 py-2 text-slate-700">{t.direction}</td>
                      <td className="px-3 py-2 text-slate-700">
                        {t.entry_low?.toFixed(2)} – {t.entry_high?.toFixed(2)}
                      </td>
                      <td className="px-3 py-2 text-slate-700">{t.stop_loss?.toFixed(2)}</td>
                      <td className="px-3 py-2 text-slate-700">{t.target?.toFixed(2)}</td>
                      <td className="px-3 py-2 text-slate-700">
                        {t.risk_reward_ratio !== null ? `${t.risk_reward_ratio.toFixed(2)}:1` : "—"}
                      </td>
                      <td className="px-3 py-2 text-slate-700">{t.status}</td>
                      <td className={`px-3 py-2 ${pnl !== null && pnl < 0 ? "text-red-600" : pnl !== null && pnl > 0 ? "text-emerald-600" : "text-slate-700"}`}>
                        {pnl !== null ? `${pnl.toFixed(2)}%${pnlIsUnrealized ? " (unrealized)" : ""}` : "—"}
                      </td>
                      <td className="px-3 py-2">
                        <button onClick={() => removeTrade(t.trade_id)} className="text-xs text-red-600 hover:underline">
                          Delete
                        </button>
                      </td>
                    </tr>
                    {t.strategy_note && (
                      <tr className="border-b border-slate-100 bg-indigo-50/40 last:border-0">
                        <td colSpan={10} className="px-3 py-1.5 text-xs text-indigo-700">
                          💡 {t.strategy_note}
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs font-medium text-slate-500">{label}</label>
      {children}
    </div>
  );
}
