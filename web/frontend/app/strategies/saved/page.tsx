"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ApiError, compareSavedStrategies, deleteSavedStrategy, listSavedStrategies, shareSavedStrategy } from "@/lib/api";
import type { StrategyCompareRow, StrategySavedRow } from "@/lib/types";

// Saved strategies: pick up to four to compare on the verdict metrics, or share one read-only.
// Nothing here places an order.

function pct(v: number | null | undefined, digits = 1) {
  return v === null || v === undefined ? "–" : `${v > 0 ? "+" : ""}${v.toFixed(digits)}%`;
}
function plain(v: number | null | undefined, digits = 2) {
  return v === null || v === undefined ? "–" : v.toFixed(digits);
}

export default function SavedStrategiesPage() {
  const [rows, setRows] = useState<StrategySavedRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [picked, setPicked] = useState<number[]>([]);
  const [compare, setCompare] = useState<StrategyCompareRow[] | null>(null);
  const [shareLinks, setShareLinks] = useState<Record<number, string>>({});
  const [busy, setBusy] = useState<string | null>(null);

  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let cancelled = false;
    listSavedStrategies()
      .then((res) => {
        if (!cancelled) setRows(res.saved);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof ApiError ? err.message : "Saved strategies could not be loaded.");
      });
    return () => {
      cancelled = true;
    };
  }, [reloadKey]);

  function toggle(id: number) {
    setCompare(null);
    setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : p.length < 4 ? [...p, id] : p));
  }

  async function handleCompare() {
    setBusy("compare");
    setError(null);
    try {
      const res = await compareSavedStrategies(picked);
      setCompare(res.rows);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "The comparison could not be run.");
    } finally {
      setBusy(null);
    }
  }

  async function handleShare(id: number) {
    setBusy(`share-${id}`);
    try {
      const res = await shareSavedStrategy(id);
      const url = `${window.location.origin}${res.path}`;
      setShareLinks((links) => ({ ...links, [id]: url }));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "The link could not be created.");
    } finally {
      setBusy(null);
    }
  }

  async function handleDelete(id: number) {
    if (!window.confirm("Delete this saved strategy?")) return;
    try {
      await deleteSavedStrategy(id);
      setPicked((p) => p.filter((x) => x !== id));
      setReloadKey((k) => k + 1);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "It could not be deleted.");
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h1 className="font-display text-2xl font-semibold text-slate-900">Saved strategies</h1>
        <Link href="/strategies/builder" className="text-sm font-medium text-slate-700 hover:underline">← Back to the builder</Link>
      </div>
      <p className="mt-1 text-sm text-slate-500">Pick up to four to compare. Each saved run keeps its rules, stocks, costs and results as they were when saved.</p>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <div className="mt-6 rounded-xl border border-slate-200 bg-white">
        {rows === null && !error && <p className="p-5 text-sm text-slate-500">Loading…</p>}
        {rows?.length === 0 && <p className="p-5 text-sm text-slate-500">Nothing saved yet. Run a backtest in the builder and choose Save this run.</p>}
        <ul className="divide-y divide-slate-100">
          {rows?.map((row) => (
            <li key={row.id} className="flex flex-col gap-2 p-4 sm:flex-row sm:items-center sm:justify-between">
              <label className="flex items-start gap-3">
                <input type="checkbox" checked={picked.includes(row.id)} onChange={() => toggle(row.id)} className="mt-1" aria-label={`Compare ${row.name}`} />
                <span>
                  <span className="font-medium text-slate-900">{row.name}</span>
                  <span className="block text-xs text-slate-500">
                    Saved {new Date(row.created_at).toLocaleDateString()} · data to {row.data_end ?? "–"} · excess CAGR vs same stocks {pct(row.summary.excess_cagr_vs_basket_pct)} · Sharpe vs SPY {plain(row.summary.sharpe_vs_spy)}
                  </span>
                </span>
              </label>
              <div className="flex flex-wrap items-center gap-3 text-xs">
                <button type="button" onClick={() => handleShare(row.id)} disabled={busy === `share-${row.id}`} className="font-medium text-slate-700 hover:underline disabled:opacity-50">
                  {shareLinks[row.id] ? "Link created" : "Share (read-only link)"}
                </button>
                <button type="button" onClick={() => handleDelete(row.id)} className="text-slate-500 hover:text-red-700">
                  Delete
                </button>
              </div>
              {shareLinks[row.id] && (
                <div className="flex w-full flex-wrap items-center gap-2 rounded bg-slate-50 px-2 py-1 sm:basis-full">
                  <a
                    href={shareLinks[row.id]}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="break-all font-mono text-xs text-indigo-700 underline underline-offset-2"
                  >
                    {shareLinks[row.id]}
                  </a>
                  <button
                    type="button"
                    onClick={() => void navigator.clipboard?.writeText(shareLinks[row.id])}
                    className="rounded border border-slate-300 px-2 py-0.5 text-xs text-slate-700 hover:bg-slate-100"
                  >
                    Copy
                  </button>
                </div>
              )}
            </li>
          ))}
        </ul>
      </div>

      {picked.length >= 2 && (
        <div className="mt-4 flex items-center gap-3">
          <button type="button" onClick={handleCompare} disabled={busy === "compare"} className="rounded-md bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800 disabled:opacity-50">
            Compare {picked.length} saved
          </button>
          <span className="text-xs text-slate-500">Up to four at a time.</span>
        </div>
      )}

      {compare && (
        <div className="mt-6 overflow-x-auto rounded-xl border border-slate-200 bg-white p-5">
          <table className="w-full min-w-[40rem] text-right text-sm">
            <thead className="text-xs text-slate-400">
              <tr>
                <th className="py-1 text-left font-medium">Strategy</th>
                <th className="font-medium">Total</th>
                <th className="font-medium">CAGR</th>
                <th className="font-medium">Excess vs same stocks</th>
                <th className="font-medium">Sharpe vs SPY</th>
                <th className="font-medium">Max DD</th>
                <th className="font-medium">Trades</th>
                <th className="font-medium">Churn</th>
                <th className="font-medium">Checks failed</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono text-xs">
              {compare.map((c) => (
                <tr key={c.id}>
                  <td className="py-2 text-left font-sans text-sm font-medium text-slate-800">{c.name}</td>
                  <td>{pct(c.total_return_pct)}</td>
                  <td>{pct(c.cagr_pct)}</td>
                  <td>{pct(c.excess_cagr_vs_basket_pct)}</td>
                  <td>{plain(c.sharpe_vs_spy)}</td>
                  <td>{pct(c.max_drawdown_pct)}</td>
                  <td>{c.trades ?? "–"}</td>
                  <td>{c.churn_pct != null ? `${plain(c.churn_pct, 1)}%` : "–"}</td>
                  <td>{c.checks_failed}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 text-xs text-slate-500">Each run is compared as saved. Past results, not a forecast or a recommendation.</p>
        </div>
      )}
    </div>
  );
}
