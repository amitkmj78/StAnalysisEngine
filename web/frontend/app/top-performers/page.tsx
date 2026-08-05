"use client";

import { useEffect, useState } from "react";

import { ApiError, getMomentumBacktest, getMomentumOptions, getTopPerformers } from "@/lib/api";
import type { MomentumBacktestResponse, TopPerformerRow } from "@/lib/types";

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

  const [backtest, setBacktest] = useState<MomentumBacktestResponse | null>(null);
  const [backtestLoading, setBacktestLoading] = useState(false);
  const [backtestError, setBacktestError] = useState<string | null>(null);

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

  async function runBacktest() {
    setBacktestLoading(true);
    setBacktestError(null);
    try {
      setBacktest(await getMomentumBacktest("Stock", stockUniverse, window, 5, 3));
    } catch (err) {
      setBacktestError(err instanceof ApiError ? err.message : "Backtest failed.");
    } finally {
      setBacktestLoading(false);
    }
  }

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

      <hr className="mt-8 border-slate-200" />
      <div className="mt-6 rounded-lg border border-indigo-200 bg-indigo-50/40 p-5">
        <h2 className="font-semibold text-slate-900">3-Year Test: Does This Ranking Actually Work?</h2>
        <p className="mt-1 text-sm text-slate-600">
          We rewound the clock 3 years and pretended to invest for real. About once a month, we picked the 5{" "}
          {stockUniverse} stocks with the best recent run (the same {WINDOW_LABELS[window] ?? `${window}-day`}{" "}
          ranking you see above), using only what would have been known at that moment — never a peek at what
          happened next. We held those 5 until the next check-in, then compared the result to simply owning
          every stock in the universe equally over that same stretch. No cherry-picking: it&apos;s the identical
          rule applied consistently, so it can be re-checked honestly at any past date.
        </p>
        <p className="mt-2 text-xs text-slate-500">
          Trading costs are included by default, not left out to flatter the number: 5 bps slippage plus
          commission each time a position turns over. Every return below is net of those costs unless labeled
          otherwise.
        </p>
        <button
          onClick={runBacktest}
          disabled={backtestLoading}
          className="mt-3 rounded-md border border-indigo-300 bg-white px-3 py-1.5 text-sm font-medium text-indigo-700 hover:bg-indigo-50 disabled:opacity-50"
        >
          {backtestLoading ? "Running 3-year backtest…" : backtest ? "Re-run Backtest" : "Run 3-Year Backtest"}
        </button>

        {backtestError && (
          <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{backtestError}</p>
        )}

        {backtest && !backtestLoading && (
          <div className="mt-4">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
              <BacktestTile label="Times We Checked" hint="how many monthly rounds" value={String(backtest.num_periods)} />
              <BacktestTile
                label="Win Rate"
                hint="rounds our picks beat the market"
                value={backtest.hit_rate_pct !== null ? `${backtest.hit_rate_pct.toFixed(1)}%` : "N/A"}
              />
              <BacktestTile
                label="Our Picks, Total"
                hint={`compounded over ${backtest.years} yrs`}
                value={backtest.strategy_cumulative_return_pct !== null ? `${backtest.strategy_cumulative_return_pct.toFixed(1)}%` : "N/A"}
                positive={backtest.strategy_cumulative_return_pct !== null && backtest.strategy_cumulative_return_pct >= 0}
              />
              <BacktestTile
                label="Own Everything, Total"
                hint="the no-skill comparison"
                value={backtest.benchmark_cumulative_return_pct !== null ? `${backtest.benchmark_cumulative_return_pct.toFixed(1)}%` : "N/A"}
                positive={backtest.benchmark_cumulative_return_pct !== null && backtest.benchmark_cumulative_return_pct >= 0}
              />
              <BacktestTile
                label="Beat the Market?"
                hint="picks vs. owning everything"
                value={
                  backtest.strategy_cumulative_return_pct !== null && backtest.benchmark_cumulative_return_pct !== null
                    ? backtest.strategy_cumulative_return_pct > backtest.benchmark_cumulative_return_pct
                      ? "Yes"
                      : "No"
                    : "N/A"
                }
              />
            </div>

            <p className="mt-3 text-xs text-slate-500">
              In plain terms: <strong>Win Rate</strong> is how often, round by round, our 5 picks did better than
              just owning everything — close to 50% means it&apos;s basically a coin flip each time, even if the
              overall total still comes out ahead (or behind) over the full stretch. The two &quot;Total&quot;
              numbers show what happened if you strung every round&apos;s result together in sequence over all{" "}
              {backtest.years} years — that&apos;s why they can look bigger than any single round&apos;s win or
              loss.
            </p>

            <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
              <BacktestTile label="CAGR" hint="annualized growth rate" value={backtest.cagr_pct !== null ? `${backtest.cagr_pct.toFixed(1)}%` : "N/A"} positive={backtest.cagr_pct !== null && backtest.cagr_pct >= 0} />
              <BacktestTile label="Volatility" hint="annualized" value={backtest.volatility_pct !== null ? `${backtest.volatility_pct.toFixed(1)}%` : "N/A"} />
              <BacktestTile label="Sharpe" hint="return per unit of risk" value={backtest.sharpe_ratio !== null ? backtest.sharpe_ratio.toFixed(2) : "N/A"} positive={backtest.sharpe_ratio !== null && backtest.sharpe_ratio >= 0} />
              <BacktestTile label="Sortino" hint="penalizes only downside" value={backtest.sortino_ratio !== null ? backtest.sortino_ratio.toFixed(2) : "N/A"} positive={backtest.sortino_ratio !== null && backtest.sortino_ratio >= 0} />
              <BacktestTile label="Max Drawdown" hint="worst peak-to-trough" value={backtest.max_drawdown_pct !== null ? `${backtest.max_drawdown_pct.toFixed(1)}%` : "N/A"} positive={backtest.max_drawdown_pct !== null && backtest.max_drawdown_pct >= -0.01} />
              <BacktestTile label="Avg Turnover" hint="per rebalance round" value={backtest.avg_turnover_pct !== null ? `${backtest.avg_turnover_pct.toFixed(0)}%` : "N/A"} />
            </div>
            <p className="mt-2 text-xs text-slate-500">
              {backtest.capacity_estimate_usd !== null && (
                <>
                  Rough capacity estimate: about{" "}
                  <strong>${(backtest.capacity_estimate_usd / 1_000_000).toFixed(1)}M</strong> before this
                  strategy&apos;s own trading would likely start moving the least-liquid pick&apos;s price — a
                  simple 1%-of-average-daily-volume heuristic, not a real market-impact model. Costs assumed:{" "}
                  {backtest.slippage_bps} bps slippage, {backtest.commission_bps} bps commission per trade.
                </>
              )}
            </p>

            <details className="mt-3 text-xs text-slate-600">
              <summary className="cursor-pointer font-medium text-slate-700">
                Show all {backtest.periods.length} rounds, one by one
              </summary>
              <div className="mt-2 max-h-80 overflow-y-auto overflow-x-auto rounded-md border border-slate-200 bg-white">
                <table className="min-w-full text-xs">
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50 text-left uppercase tracking-wide text-slate-500">
                      <th className="px-2 py-1.5">Date</th>
                      <th className="px-2 py-1.5">Our 5 Picks</th>
                      <th className="px-2 py-1.5 text-right">Picks Return (Net)</th>
                      <th className="px-2 py-1.5 text-right">Picks Return (Gross)</th>
                      <th className="px-2 py-1.5 text-right">Own-Everything Return</th>
                      <th className="px-2 py-1.5 text-right">Turnover</th>
                    </tr>
                  </thead>
                  <tbody>
                    {backtest.periods.map((p) => (
                      <tr key={p.date} className="border-b border-slate-100 last:border-0">
                        <td className="px-2 py-1.5 text-slate-600">{p.date}</td>
                        <td className="px-2 py-1.5 text-slate-600">{p.picks.join(", ")}</td>
                        <td className="px-2 py-1.5 text-right text-slate-700">
                          {p.strategy_return_pct !== null ? `${p.strategy_return_pct.toFixed(2)}%` : "—"}
                        </td>
                        <td className="px-2 py-1.5 text-right text-slate-400">
                          {p.strategy_return_gross_pct !== null ? `${p.strategy_return_gross_pct.toFixed(2)}%` : "—"}
                        </td>
                        <td className="px-2 py-1.5 text-right text-slate-700">
                          {p.benchmark_return_pct !== null ? `${p.benchmark_return_pct.toFixed(2)}%` : "—"}
                        </td>
                        <td className="px-2 py-1.5 text-right text-slate-500">{p.turnover_pct.toFixed(0)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>

            <p className="mt-2 text-xs text-slate-400">Run #{backtest.run_id} — retrievable later at this exact result.</p>
          </div>
        )}
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

function BacktestTile({
  label,
  value,
  hint,
  positive,
}: {
  label: string;
  value: string;
  hint?: string;
  positive?: boolean;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <p className="text-xs font-medium text-slate-700">{label}</p>
      <p
        className={`mt-1 text-lg font-semibold ${
          positive === undefined ? "text-slate-900" : positive ? "text-emerald-600" : "text-red-600"
        }`}
      >
        {value}
      </p>
      {hint && <p className="mt-0.5 text-[11px] leading-tight text-slate-400">{hint}</p>}
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
              <th className="px-3 py-2 text-right">Price</th>
              <th className="px-3 py-2 text-right">Return</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={r.ticker} className="border-b border-slate-100 last:border-0">
                <td className="px-3 py-2 text-slate-400">{i + 1}</td>
                <td className="px-3 py-2 font-medium text-slate-800">{r.ticker}</td>
                <td className="px-3 py-2 text-slate-600">{r.name}</td>
                <td className="px-3 py-2 text-right text-slate-600">
                  {r.price !== null ? `$${r.price.toFixed(2)}` : "—"}
                </td>
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
