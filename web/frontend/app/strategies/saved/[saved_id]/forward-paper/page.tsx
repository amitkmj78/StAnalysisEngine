"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import {
  ApiError,
  acceptStrategyForwardPaperDisclosure,
  disableStrategyForwardPaper,
  enableStrategyForwardPaper,
  getStrategyForwardPaperStatus,
  getStrategyForwardPaperTrackRecord,
  linkStrategyForwardPaper,
  setStrategyForwardPaperKillSwitch,
  unlinkStrategyForwardPaper,
} from "@/lib/api";
import type { StrategyForwardPaperStatus, StrategyTrackRecord } from "@/lib/types";

function pct(v: number | null | undefined, digits = 1) {
  return v === null || v === undefined ? "–" : `${v > 0 ? "+" : ""}${v.toFixed(digits)}%`;
}
function plain(v: number | null | undefined, digits = 2) {
  return v === null || v === undefined ? "–" : v.toFixed(digits);
}

export default function StrategyForwardPaperPage() {
  const params = useParams<{ saved_id: string }>();
  const savedId = Number(params.saved_id);

  const [status, setStatus] = useState<StrategyForwardPaperStatus | null>(null);
  const [trackRecord, setTrackRecord] = useState<StrategyTrackRecord | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [apiKeyId, setApiKeyId] = useState("");
  const [apiSecretKey, setApiSecretKey] = useState("");
  const [linking, setLinking] = useState(false);
  const [busy, setBusy] = useState(false);

  function load() {
    getStrategyForwardPaperStatus(savedId)
      .then((res) => setStatus(res))
      .catch(() => setStatus(null))
      .finally(() => setLoaded(true));
    getStrategyForwardPaperTrackRecord(savedId)
      .then((res) => setTrackRecord(res))
      .catch(() => setTrackRecord(null));
  }

  useEffect(() => {
    if (Number.isFinite(savedId)) load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [savedId]);

  async function handleLink(e: React.FormEvent) {
    e.preventDefault();
    setLinking(true);
    setError(null);
    try {
      await linkStrategyForwardPaper(savedId, apiKeyId.trim(), apiSecretKey.trim());
      setApiKeyId("");
      setApiSecretKey("");
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not link this key pair.");
    } finally {
      setLinking(false);
    }
  }

  async function handleAcceptDisclosure() {
    setBusy(true);
    setError(null);
    try {
      await acceptStrategyForwardPaperDisclosure(savedId);
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not accept the disclosure.");
    } finally {
      setBusy(false);
    }
  }

  async function handleToggleEnabled() {
    setBusy(true);
    setError(null);
    try {
      if (status?.account.enabled) {
        await disableStrategyForwardPaper(savedId);
      } else {
        await enableStrategyForwardPaper(savedId);
      }
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not update this strategy's forward-paper status.");
    } finally {
      setBusy(false);
    }
  }

  async function handleKillSwitch() {
    setBusy(true);
    setError(null);
    try {
      await setStrategyForwardPaperKillSwitch(savedId, !status?.account.kill_engaged);
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not update the kill switch.");
    } finally {
      setBusy(false);
    }
  }

  async function handleUnlink() {
    if (!window.confirm("Unlink this strategy's forward-paper account? Its positions, journal and track record will be deleted.")) return;
    setBusy(true);
    setError(null);
    try {
      await unlinkStrategyForwardPaper(savedId);
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not unlink this account.");
    } finally {
      setBusy(false);
    }
  }

  const account = status?.account ?? null;

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h1 className="font-display text-2xl font-semibold text-slate-900 dark:text-slate-50">Forward paper trading</h1>
        <Link href="/strategies/saved" className="text-sm font-medium text-slate-700 dark:text-slate-200 hover:underline">
          ← Back to saved strategies
        </Link>
      </div>
      <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
        Runs this strategy&apos;s own rules forward, once a day, against a real (never live) Alpaca paper account --
        real paper orders submitted automatically on its own entry/exit signals, starting a genuine live track
        record. Awareness, not a backtest: see the disclosures below before linking.
      </p>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {!loaded && <p className="mt-6 text-sm text-slate-500 dark:text-slate-400">Loading…</p>}

      {loaded && !account && (
        <div className="mt-6 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5">
          <h2 className="font-semibold text-slate-900 dark:text-slate-50">Link an Alpaca paper account</h2>
          <ul className="mt-2 list-disc pl-5 text-sm text-slate-600 dark:text-slate-300">
            <li>
              This strategy needs its <strong>own, separate</strong> Alpaca paper-trading key pair -- independent from
              any account you&apos;ve linked to a portfolio. Create a new one free from your Alpaca dashboard.
            </li>
            <li>
              Exits (stop-loss, trailing stop, take-profit, time stop) are only checked once a day, when this job
              runs -- not continuously by the broker. A position can move further against its stop intraday before
              the next run catches it.
            </li>
            <li>This never places a live (real-money) order -- paper only, always.</li>
          </ul>
          <form onSubmit={handleLink} className="mt-4 flex flex-col gap-2">
            <label htmlFor="sfp-key-id" className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Alpaca API Key ID
            </label>
            <input
              id="sfp-key-id"
              type="text"
              value={apiKeyId}
              onChange={(e) => setApiKeyId(e.target.value)}
              required
              className="rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 px-2 py-1.5 text-sm"
            />
            <label htmlFor="sfp-secret-key" className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Alpaca API Secret Key
            </label>
            <input
              id="sfp-secret-key"
              type="password"
              value={apiSecretKey}
              onChange={(e) => setApiSecretKey(e.target.value)}
              required
              className="rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 px-2 py-1.5 text-sm"
            />
            <button
              type="submit"
              disabled={linking}
              className="mt-2 w-fit rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
            >
              {linking ? "Linking…" : "Link account"}
            </button>
          </form>
        </div>
      )}

      {loaded && account && (
        <div className="mt-6 flex flex-col gap-4">
          <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5">
            <div className="flex items-center justify-between gap-4">
              <div>
                <h2 className="font-semibold text-slate-900 dark:text-slate-50">Account</h2>
                <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
                  Alpaca key {account.api_key_id} · status {account.status}
                  {account.last_run_at ? ` · last run ${new Date(account.last_run_at).toLocaleString()} (${account.last_run_status})` : " · no run yet"}
                </p>
              </div>
              <span
                className={`shrink-0 rounded-full px-3 py-1 text-xs font-medium ${
                  account.enabled ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400"
                }`}
              >
                {account.enabled ? "Running" : "Off"}
              </span>
            </div>

            {!account.disclosure_accepted_at ? (
              <div className="mt-4 rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-800">
                <p>
                  Accept the disclosure above (separate key pair, exits checked once a day, paper only -- never live)
                  before enabling.
                </p>
                <button
                  onClick={handleAcceptDisclosure}
                  disabled={busy}
                  className="mt-2 rounded-md bg-amber-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-amber-700 disabled:opacity-50"
                >
                  {busy ? "Updating…" : "I understand -- accept"}
                </button>
              </div>
            ) : (
              <div className="mt-4 flex flex-wrap items-center gap-2">
                <button
                  onClick={handleToggleEnabled}
                  disabled={busy}
                  className={`rounded-md px-4 py-2 text-sm font-medium disabled:opacity-50 ${
                    account.enabled ? "border border-red-200 text-red-700 hover:bg-red-50" : "bg-slate-900 text-white hover:bg-slate-800"
                  }`}
                >
                  {busy ? "Updating…" : account.enabled ? "Disable" : "Enable"}
                </button>
                <button
                  onClick={handleKillSwitch}
                  disabled={busy}
                  className={`rounded-md px-4 py-2 text-sm font-medium disabled:opacity-50 ${
                    account.kill_engaged
                      ? "bg-amber-600 text-white hover:bg-amber-700"
                      : "border border-amber-300 text-amber-700 hover:bg-amber-50"
                  }`}
                >
                  {account.kill_engaged ? "Kill switch engaged -- release" : "Engage kill switch"}
                </button>
                <button onClick={handleUnlink} disabled={busy} className="ml-auto text-xs text-slate-500 dark:text-slate-400 hover:text-red-700">
                  Unlink
                </button>
              </div>
            )}
          </div>

          {status && status.positions.length > 0 && (
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5">
              <h2 className="font-semibold text-slate-900 dark:text-slate-50">Open positions</h2>
              <ul className="mt-2 flex flex-col gap-1 text-sm">
                {status.positions.map((p) => (
                  <li key={p.ticker} className="flex justify-between font-mono">
                    <span>{p.ticker}</span>
                    <span>
                      {p.qty} sh @ {plain(p.entry_price)} · held {p.days_held}d
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5">
            <div className="flex items-center gap-2">
              <h2 className="font-semibold text-slate-900 dark:text-slate-50">Track record</h2>
              <span className="rounded-full bg-[var(--pf-accent-soft)] px-2 py-0.5 text-xs font-medium text-[var(--pf-accent)]">
                Live paper
              </span>
            </div>
            {trackRecord?.available ? (
              trackRecord.days_of_data && trackRecord.days_of_data >= 2 ? (
                <dl className="mt-2 grid grid-cols-2 gap-2 text-sm sm:grid-cols-3">
                  <div><dt className="text-slate-500 dark:text-slate-400">Return</dt><dd className="font-mono">{pct(trackRecord.return_pct)}</dd></div>
                  <div><dt className="text-slate-500 dark:text-slate-400">Max drawdown</dt><dd className="font-mono">{pct(trackRecord.max_drawdown_pct)}</dd></div>
                  <div><dt className="text-slate-500 dark:text-slate-400">Sharpe</dt><dd className="font-mono">{plain(trackRecord.sharpe)}</dd></div>
                  <div><dt className="text-slate-500 dark:text-slate-400">Sortino</dt><dd className="font-mono">{plain(trackRecord.sortino)}</dd></div>
                  <div><dt className="text-slate-500 dark:text-slate-400">Calmar</dt><dd className="font-mono">{plain(trackRecord.calmar)}</dd></div>
                  <div><dt className="text-slate-500 dark:text-slate-400">Days of data</dt><dd className="font-mono">{trackRecord.days_of_data}</dd></div>
                </dl>
              ) : (
                <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">
                  Not enough days of real equity history yet ({trackRecord.days_of_data ?? 0} so far) -- check back once the
                  daily equity-capture job has run a few more times.
                </p>
              )
            ) : (
              <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">{trackRecord?.reason ?? "No track record yet."}</p>
            )}
            <p className="mt-3 text-xs text-slate-400 dark:text-slate-500">
              A real linked account&apos;s own equity curve -- not a replay, and not a forecast.
            </p>
          </div>

          {status && status.journal.length > 0 && (
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5">
              <h2 className="font-semibold text-slate-900 dark:text-slate-50">Journal</h2>
              <ul className="mt-2 flex flex-col gap-1 text-xs">
                {status.journal.map((e) => (
                  <li key={e.id} className="flex flex-wrap items-baseline gap-2">
                    <span className="font-mono text-slate-400 dark:text-slate-500">{new Date(e.created_at).toLocaleString()}</span>
                    <span className="font-medium text-slate-700 dark:text-slate-200">{e.event_type}</span>
                    {e.ticker && <span className="font-mono">{e.ticker}</span>}
                    <span className="text-slate-500 dark:text-slate-400">{e.reason}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
