"use client";

import { useEffect, useState } from "react";

import { ApiError, getMomentumOptions, getTopPerformers } from "@/lib/api";
import type { TopPerformerRow } from "@/lib/types";

const WINDOW_LABELS: Record<number, string> = {
  10: "10 Days",
  30: "30 Days",
  60: "60 Days",
  90: "90 Days",
};

export default function TopPerformersPage() {
  const [windows, setWindows] = useState<number[]>([10, 30, 60, 90]);
  const [stockUniverses, setStockUniverses] = useState<string[]>([]);
  const [fundCategories, setFundCategories] = useState<string[]>([]);

  const [window, setWindow] = useState(30);
  const [stockUniverse, setStockUniverse] = useState("All");
  const [fundCategory, setFundCategory] = useState("All");

  const [stocks, setStocks] = useState<TopPerformerRow[] | null>(null);
  const [funds, setFunds] = useState<TopPerformerRow[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getMomentumOptions()
      .then((res) => {
        setWindows(res.windows);
        setStockUniverses(res.stock_universes);
        setFundCategories(res.fund_categories);
        setStockUniverse(res.stock_universes[0] ?? "All");
        setFundCategory(res.fund_categories[0] ?? "All");
      })
      .catch(() => {});
  }, []);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const [stockRes, fundRes] = await Promise.all([
        getTopPerformers(window, "Stock", stockUniverse, 15),
        getTopPerformers(window, "Fund", fundCategory, 15),
      ]);
      setStocks(stockRes.results);
      setFunds(fundRes.results);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">Top Performers</h1>
      <p className="mt-1 text-sm text-slate-500">
        Stocks and funds ranked by raw trailing price return over a fixed trading-day window — a simple momentum
        read, separate from the weighted scoring on Best Stock Finder / Best Index Fund.
      </p>

      <div className="mt-6 flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-slate-500">Window</label>
          <div className="flex gap-1 rounded-md border border-slate-300 p-1">
            {windows.map((w) => (
              <button
                key={w}
                onClick={() => setWindow(w)}
                className={`rounded px-3 py-1.5 text-sm font-medium ${
                  window === w ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-100"
                }`}
              >
                {WINDOW_LABELS[w] ?? `${w} Days`}
              </button>
            ))}
          </div>
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-slate-500">Stock universe</label>
          <select value={stockUniverse} onChange={(e) => setStockUniverse(e.target.value)} className="input">
            {stockUniverses.map((u) => (
              <option key={u} value={u}>
                {u}
              </option>
            ))}
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-slate-500">Fund category</label>
          <select value={fundCategory} onChange={(e) => setFundCategory(e.target.value)} className="input">
            {fundCategories.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>
        <button onClick={load} disabled={loading} className="btn-primary">
          {loading ? "Loading…" : "Refresh"}
        </button>
      </div>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <TopPerformersTable title="Top Stocks" rows={stocks} loading={loading} windowLabel={WINDOW_LABELS[window] ?? `${window} Days`} />
        <TopPerformersTable title="Top Funds" rows={funds} loading={loading} windowLabel={WINDOW_LABELS[window] ?? `${window} Days`} />
      </div>

      <div className="mt-6 rounded-lg border border-slate-200 bg-white p-5">
        <h3 className="font-semibold text-slate-900">How To Read This Page</h3>
        <p className="mt-2 text-sm text-slate-600">
          Each return is just <code className="rounded bg-slate-100 px-1 py-0.5 text-xs">(price now / price {window} trading days ago) − 1</code>,
          nothing weighted or model-scored — pure recent price momentum. Useful for spotting what&apos;s hot right
          now; not the same as the multi-factor ranking used elsewhere in this app, and says nothing about
          whether a run continues.
        </p>
      </div>
    </div>
  );
}

function TopPerformersTable({
  title,
  rows,
  loading,
  windowLabel,
}: {
  title: string;
  rows: TopPerformerRow[] | null;
  loading: boolean;
  windowLabel: string;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white">
      <div className="border-b border-slate-200 px-4 py-3">
        <h2 className="font-semibold text-slate-900">{title}</h2>
        <p className="text-xs text-slate-500">Ranked by {windowLabel} trailing return</p>
      </div>
      {loading && !rows ? (
        <p className="px-4 py-6 text-sm text-slate-500">Loading…</p>
      ) : !rows || rows.length === 0 ? (
        <p className="px-4 py-6 text-sm text-slate-500">No data available right now.</p>
      ) : (
        <table className="min-w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
              <th className="px-3 py-2">#</th>
              <th className="px-3 py-2">Ticker</th>
              <th className="px-3 py-2">Name</th>
              <th className="px-3 py-2 text-right">Return</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={r.ticker} className="border-b border-slate-100 last:border-0">
                <td className="px-3 py-2 text-slate-400">{i + 1}</td>
                <td className="px-3 py-2 font-medium text-slate-800">{r.ticker}</td>
                <td className="px-3 py-2 text-slate-600">{r.name}</td>
                <td className={`px-3 py-2 text-right font-medium ${r.return_pct >= 0 ? "text-emerald-600" : "text-red-600"}`}>
                  {r.return_pct >= 0 ? "+" : ""}
                  {r.return_pct.toFixed(2)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
