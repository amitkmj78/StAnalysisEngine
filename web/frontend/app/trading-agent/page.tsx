"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import {
  adminDisableAgentUser,
  adminEnableAgentUser,
  adminSetAgentGlobal,
  adminSetAgentKill,
  ApiError,
  getCurrentUser,
  getTradingAgentJournal,
  getTradingAgentStatus,
  resetAgentBreaker,
  runAgentPlanPreview,
  setAgentKill,
  setAgentMode,
} from "@/lib/api";
import { isAdmin } from "@/lib/admin";
import type { AgentRunSummary, TradingAgentStatus } from "@/lib/types";

const PROPOSAL_EVENTS = ["proposed", "submitted", "filled", "rejected", "skipped", "stop_placed", "stop_replaced", "stop_failed", "run_failed"];

function fmtMoney(v: number | null | undefined): string {
  return v === null || v === undefined ? "—" : `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

function fmtPct(v: number | null | undefined): string {
  return v === null || v === undefined ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
}

export default function TradingAgentPage() {
  const [status, setStatus] = useState<TradingAgentStatus | null | undefined>(undefined);
  const [journal, setJournal] = useState<AgentRunSummary[]>([]);
  const [admin, setAdmin] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [adminEmail, setAdminEmail] = useState("");
  const [complianceRef, setComplianceRef] = useState("");

  function load() {
    getTradingAgentStatus()
      .then(setStatus)
      .catch((err) => {
        if (err instanceof ApiError && err.status === 403) setStatus(null);
        else setError(err instanceof ApiError ? err.message : "Could not load the trading agent.");
      });
    getTradingAgentJournal(20)
      .then((r) => setJournal(r.runs))
      .catch(() => {});
  }

  useEffect(() => {
    load();
    getCurrentUser()
      .then((u) => setAdmin(isAdmin(u.email)))
      .catch(() => {});
  }, []);

  async function act(fn: () => Promise<unknown>, success: string) {
    setBusy(true);
    setError(null);
    setNote(null);
    try {
      await fn();
      setNote(success);
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "That action failed.");
    } finally {
      setBusy(false);
    }
  }

  const todaysPlan = status?.latest_run?.events.filter((e) => PROPOSAL_EVENTS.includes(e.event_type)) ?? [];

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">Trading Agent</h1>
          <p className="mt-1 text-sm text-slate-500">
            A rule-based agent that runs this app&apos;s Buy signals on your Alpaca paper account, with risk limits it
            can&apos;t override and a journal of every decision.
          </p>
        </div>
        <Link href="/portfolio/paper-trading" className="text-sm font-medium text-slate-600 hover:underline">
          ← Paper Trading
        </Link>
      </div>

      <p className="mt-4 rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-800">
        Automated trading can lose money. Past performance doesn&apos;t predict future results.
      </p>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {note && <p className="mt-4 rounded-md bg-emerald-50 px-3 py-2 text-sm text-emerald-700">{note}</p>}

      {status === undefined && <p className="mt-6 text-sm text-slate-500">Loading…</p>}

      {status === null && (
        <p className="mt-6 rounded-md bg-slate-50 px-3 py-3 text-sm text-slate-600">
          The trading agent isn&apos;t enabled for your account. It becomes available only after an admin enables it
          following compliance review.
        </p>
      )}

      {status && status.global_kill_engaged && (
        <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">
          The agent is stopped for everyone by the global kill switch. No new orders are being placed. Existing stops
          remain in place.
        </p>
      )}

      {status && (
        <>
          <section className="mt-6 rounded-lg border border-slate-200 bg-white p-5">
            <h2 className="text-sm font-semibold text-slate-900">Mode and controls</h2>
            <div className="mt-3 flex flex-wrap items-center gap-2">
              {(["plan", "paper", "live"] as const).map((m) => (
                <button
                  key={m}
                  disabled={busy || !status.enabled || m === "live" || status.mode === m}
                  onClick={() => act(() => setAgentMode(m as "plan" | "paper"), `Mode set to ${m}.`)}
                  className={`rounded-md px-3 py-1.5 text-sm font-medium disabled:opacity-50 ${
                    status.mode === m ? "bg-slate-900 text-white" : "border border-slate-300 text-slate-700"
                  }`}
                  title={m === "live" ? status.live.reason : undefined}
                >
                  {m === "plan" ? "Plan (no orders)" : m === "paper" ? "Paper (Alpaca sandbox)" : "Live (blocked)"}
                </button>
              ))}
            </div>
            <p className="mt-2 text-xs text-slate-500">
              {status.live.reason}
            </p>

            <div className="mt-4 flex flex-wrap gap-2">
              <button
                disabled={busy || !status.enabled}
                onClick={() => act(runAgentPlanPreview, "Plan computed and journaled. No orders were placed.")}
                className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
              >
                Preview today&apos;s plan
              </button>
              <button
                disabled={busy || !status.enabled}
                onClick={() => act(() => setAgentKill(!status.kill_engaged), status.kill_engaged ? "Your agent is re-enabled." : "Your agent is stopped. Existing stops remain.")}
                className={`rounded-md px-3 py-1.5 text-sm font-medium disabled:opacity-50 ${
                  status.kill_engaged ? "border border-slate-300 text-slate-700" : "border border-red-200 text-red-700 hover:bg-red-50"
                }`}
              >
                {status.kill_engaged ? "Resume agent" : "Stop agent (kill switch)"}
              </button>
            </div>
            {status.kill_engaged && (
              <p className="mt-2 text-xs text-red-700">Your kill switch is on. No new orders are placed. Existing protective stops stay in place.</p>
            )}
          </section>

          <section className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="rounded-lg border border-slate-200 bg-white p-5">
              <h2 className="text-sm font-semibold text-slate-900">Risk state</h2>
              <p className="mt-2 text-sm text-slate-700">
                Regime: <span className="font-medium">{status.regime.label ?? "No reading"}</span> — exposure capped at{" "}
                {status.regime.exposure_cap_pct}%.
              </p>
              <p className="mt-1 text-xs text-slate-500">{status.regime.reason}</p>
              {status.breaker_latched ? (
                <div className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">
                  Drawdown circuit breaker is tripped. Exposure is capped at 20% and buys are blocked.
                  <button
                    disabled={busy}
                    onClick={() => act(resetAgentBreaker, "Breaker reset. Normal sizing resumes on the next run.")}
                    className="ml-2 font-medium underline disabled:opacity-50"
                  >
                    Reset breaker
                  </button>
                </div>
              ) : (
                <p className="mt-3 text-sm text-slate-600">Circuit breaker: not tripped.</p>
              )}
              <p className="mt-3 text-xs text-slate-500">
                Limits: {status.limits.max_position_pct}% per stock, {status.limits.max_sector_pct}% per sector, up to{" "}
                {status.limits.max_positions} positions. Daily loss limit {status.limits.daily_loss_limit_pct}%, breaker at
                -{status.limits.drawdown_breaker_pct}% from peak. Config {status.config_version}.
              </p>
              <p className="mt-2 text-xs text-slate-400">{status.regime.disclosure}</p>
            </div>

            <div className="rounded-lg border border-slate-200 bg-white p-5">
              <h2 className="text-sm font-semibold text-slate-900">Paper performance</h2>
              {status.performance_paper && status.performance_paper.days_of_data >= 2 ? (
                <dl className="mt-2 grid grid-cols-2 gap-2 text-sm">
                  <dt className="text-slate-500">Return (90d)</dt>
                  <dd className="text-right">{fmtPct(status.performance_paper.return_pct)}</dd>
                  <dt className="text-slate-500">SPY, same window</dt>
                  <dd className="text-right">{fmtPct(status.performance_paper.spy_return_pct)}</dd>
                  <dt className="text-slate-500">Volatility (ann.)</dt>
                  <dd className="text-right">{status.performance_paper.annualized_volatility_pct?.toFixed(2) ?? "—"}%</dd>
                  <dt className="text-slate-500">Max drawdown</dt>
                  <dd className="text-right">{fmtPct(status.performance_paper.max_drawdown_pct)}</dd>
                  <dt className="text-slate-500">Sharpe</dt>
                  <dd className="text-right">{status.performance_paper.sharpe ?? "—"}</dd>
                </dl>
              ) : (
                <p className="mt-2 text-sm text-slate-600">{status.performance_paper?.note ?? "No paper history yet."}</p>
              )}
              <p className="mt-3 text-xs text-slate-400">
                Paper only. Live results are never combined with these. {status.performance_paper?.costs_paid}
              </p>
            </div>
          </section>

          <section className="mt-4 rounded-lg border border-slate-200 bg-white p-5">
            <h2 className="text-sm font-semibold text-slate-900">Positions and stops (from Alpaca)</h2>
            {!status.broker ? (
              <p className="mt-2 text-sm text-slate-600">No linked paper account.</p>
            ) : "error" in status.broker ? (
              <p className="mt-2 text-sm text-red-700">Could not reach Alpaca: {status.broker.error}</p>
            ) : status.broker.positions.length === 0 ? (
              <p className="mt-2 text-sm text-slate-600">No open positions.</p>
            ) : (
              <table className="mt-2 w-full text-sm">
                <thead className="text-left text-xs uppercase text-slate-500">
                  <tr><th className="py-1">Ticker</th><th className="text-right">Shares</th><th className="text-right">Value</th><th className="text-right">Stop</th></tr>
                </thead>
                <tbody>
                  {status.broker.positions.map((p) => (
                    <tr key={p.ticker} className="border-t border-slate-100">
                      <td className="py-1.5 font-medium">{p.ticker}</td>
                      <td className="text-right">{p.qty}</td>
                      <td className="text-right">{fmtMoney(p.market_value)}</td>
                      <td className={`text-right ${p.stop ? "" : "font-medium text-red-700"}`}>
                        {p.stop ? `trailing ${p.stop.trail_percent}%` : "NO STOP"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>

          <section className="mt-4 rounded-lg border border-slate-200 bg-white p-5">
            <h2 className="text-sm font-semibold text-slate-900">Today&apos;s plan</h2>
            {todaysPlan.length === 0 ? (
              <p className="mt-2 text-sm text-slate-600">No plan yet. Preview one, or wait for the 15:45 ET run.</p>
            ) : (
              <ul className="mt-2 flex flex-col gap-2 text-sm">
                {todaysPlan.map((e, i) => (
                  <li key={i} className="rounded-md bg-slate-50 px-3 py-2">
                    <span className="font-medium">{e.event_type.replace(/_/g, " ")}</span>
                    {e.ticker && <span> · {e.ticker}</span>}
                    {e.side && <span> {e.side}</span>}
                    {e.qty ? <span> × {e.qty}</span> : null}
                    <p className="mt-0.5 text-xs text-slate-600">{e.reason}</p>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="mt-4 rounded-lg border border-slate-200 bg-white p-5">
            <h2 className="text-sm font-semibold text-slate-900">Journal</h2>
            <p className="mt-1 text-xs text-slate-500">Every run and order, with its plain-language reason. Append-only.</p>
            {journal.length === 0 ? (
              <p className="mt-2 text-sm text-slate-600">No runs recorded yet.</p>
            ) : (
              <div className="mt-3 flex flex-col gap-3">
                {journal.map((r) => (
                  <details key={r.id} className="rounded-md border border-slate-200 px-3 py-2 text-sm">
                    <summary className="cursor-pointer">
                      <span className="font-medium">{new Date(r.created_at).toLocaleString()}</span>
                      <span className="text-slate-500"> · {r.mode} · {r.status}</span>
                      {r.risk_state && <span className="text-slate-500"> · {r.risk_state}</span>}
                      {r.reason && <span className="block text-xs text-slate-500">{r.reason}</span>}
                    </summary>
                    <ul className="mt-2 flex flex-col gap-1.5">
                      {r.events.map((e, i) => (
                        <li key={i} className="text-xs">
                          <span className="font-medium">{e.event_type.replace(/_/g, " ")}</span>
                          {e.ticker && <span> · {e.ticker}</span>}
                          {e.side && <span> {e.side}</span>}
                          {e.qty ? <span> × {e.qty}</span> : null}
                          <span className="block text-slate-600">{e.reason}</span>
                        </li>
                      ))}
                    </ul>
                  </details>
                ))}
              </div>
            )}
          </section>
        </>
      )}

      {admin && (
        <section className="mt-8 rounded-lg border border-indigo-200 bg-indigo-50 p-5">
          <h2 className="text-sm font-semibold text-slate-900">Admin: agent enablement</h2>
          <p className="mt-1 text-xs text-slate-600">
            Enabling a user requires a compliance sign-off reference (AGT-35). The owner account is the only one enabled
            until legal review is complete.
          </p>
          <div className="mt-3 flex flex-col gap-2">
            <input
              type="email"
              value={adminEmail}
              onChange={(e) => setAdminEmail(e.target.value)}
              placeholder="User email"
              className="rounded-md border border-slate-300 px-3 py-1.5 text-sm"
            />
            <input
              value={complianceRef}
              onChange={(e) => setComplianceRef(e.target.value)}
              placeholder="Compliance sign-off reference (required)"
              className="rounded-md border border-slate-300 px-3 py-1.5 text-sm"
            />
            <div className="flex flex-wrap gap-2">
              <button
                disabled={busy || !adminEmail || complianceRef.trim().length < 3}
                onClick={() => act(() => adminEnableAgentUser(adminEmail, complianceRef), `Enabled ${adminEmail} in plan mode.`)}
                className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
              >
                Enable user
              </button>
              <button
                disabled={busy || !adminEmail}
                onClick={() => act(() => adminDisableAgentUser(adminEmail), `Disabled ${adminEmail}.`)}
                className="rounded-md border border-slate-300 px-3 py-1.5 text-sm text-slate-700 disabled:opacity-50"
              >
                Disable user
              </button>
              <button
                disabled={busy}
                onClick={() => act(() => adminSetAgentGlobal(!(status?.global_enabled ?? false)), "Global scheduled run toggled.")}
                className="rounded-md border border-slate-300 px-3 py-1.5 text-sm text-slate-700 disabled:opacity-50"
              >
                {status?.global_enabled ? "Turn off scheduled runs" : "Turn on scheduled runs"}
              </button>
              <button
                disabled={busy}
                onClick={() => act(() => adminSetAgentKill(!(status?.global_kill_engaged ?? false)), "Global kill switch toggled.")}
                className="rounded-md border border-red-200 px-3 py-1.5 text-sm text-red-700 disabled:opacity-50"
              >
                {status?.global_kill_engaged ? "Release global kill switch" : "Global kill switch (stop everyone)"}
              </button>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
