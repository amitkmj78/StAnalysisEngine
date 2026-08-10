"use client";

import { useEffect, useState } from "react";

import InfoModal, { type ColumnInfo } from "@/components/InfoModal";
import MonthlyChart from "@/components/monthly/MonthlyChart";
import { ApiError, getMonthlyPlanOptions, getMonthlyPlanSummary } from "@/lib/api";
import type { MonthlyPlanResponse } from "@/lib/types";

// Same ranking engine (and same weights) as /stock-finder and /index-fund —
// this page's "Score" is that Score, not a separate computation, so the
// explanation is kept in sync with those pages' own COLUMN_INFO.Score text.
const SCORE_INFO: Record<"Stock" | "Fund", ColumnInfo> = {
  Stock: {
    title: "Score",
    body: [
      "A 0–100 blend of several metrics, each normalized against the other tickers in the current universe (the best value in the current list scores highest on that metric, the worst scores lowest) — it's a relative ranking within this run, not an absolute grade. Re-running with a different universe can change a ticker's score even if nothing about the ticker itself changed.",
      "The metrics and their weights depend on the Goal you picked:",
      "\"Short Term\": 3-month return (30%), 1-month return (25%), RSI balance (15%), MACD signal strength (15%), volume strength (10%), 6-month volatility (5%, lower is better).",
      "\"Long Term\": 1-year return (28%), 3-year annualized return (20%), 6-month return (12%), revenue growth (12%), earnings growth (10%), forward P/E (8%, lower is better), 1-year max drawdown (10%, lower is better).",
    ],
  },
  Fund: {
    title: "Score",
    body: [
      "A 0–100 blend of several metrics, each normalized against the other funds in the current category (the best value in the current list scores highest on that metric, the worst scores lowest) — it's a relative ranking within this run, not an absolute grade. Re-running with a different category can change a fund's score even if nothing about the fund itself changed.",
      "The metrics and their weights depend on the Goal you picked:",
      "\"Balanced Core\": 1-year return (35%), 3-year annualized return (25%), expense ratio (20%, lower is better), 1-year volatility (10%, lower is better), 3-year max drawdown (10%, lower is better).",
      "\"Lowest Cost\": expense ratio (65%, lower is better), 3-year annualized return (20%), 1-year volatility (10%, lower is better), fund assets (5%).",
      "\"Best Growth\": 1-year return (50%), 3-year annualized return (35%), 1-year volatility (10%, lower is better), expense ratio (5%, lower is better).",
      "\"Most Stable\": 1-year volatility (45%, lower is better), 3-year max drawdown (30%, lower is better), expense ratio (15%, lower is better), 3-year annualized return (10%).",
    ],
  },
};

export default function MonthlyPlanPage() {
  const [fundGoals, setFundGoals] = useState<string[]>([]);
  const [fundCategories, setFundCategories] = useState<string[]>([]);
  const [stockGoals, setStockGoals] = useState<string[]>([]);
  const [stockUniverses, setStockUniverses] = useState<string[]>([]);

  const [fundGoal, setFundGoal] = useState("Balanced Core");
  const [fundCategory, setFundCategory] = useState("All");
  const [stockGoal, setStockGoal] = useState("Short Term");
  const [stockUniverse, setStockUniverse] = useState("All");

  const [monthlyAmount, setMonthlyAmount] = useState(1000);
  const [years, setYears] = useState(5);

  const [fundData, setFundData] = useState<MonthlyPlanResponse | null>(null);
  const [stockData, setStockData] = useState<MonthlyPlanResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getMonthlyPlanOptions()
      .then((res) => {
        setFundGoals(res.fund_goals);
        setFundCategories(res.fund_categories);
        setStockGoals(res.stock_goals);
        setStockUniverses(res.stock_universes);
        setFundGoal(res.fund_goals[0] ?? "Balanced Core");
        setFundCategory(res.fund_categories[0] ?? "All");
        setStockGoal(res.stock_goals[0] ?? "Short Term");
        setStockUniverse(res.stock_universes[0] ?? "All");
      })
      .catch(() => {});
  }, []);

  async function runPlan(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setFundData(null);
    setStockData(null);
    try {
      const [fundResult, stockResult] = await Promise.allSettled([
        getMonthlyPlanSummary("Fund", fundGoal, fundCategory, monthlyAmount, years),
        getMonthlyPlanSummary("Stock", stockGoal, stockUniverse, monthlyAmount, years),
      ]);

      if (fundResult.status === "fulfilled") setFundData(fundResult.value);
      if (stockResult.status === "fulfilled") setStockData(stockResult.value);

      if (fundResult.status === "rejected" && stockResult.status === "rejected") {
        const err = fundResult.reason;
        setError(err instanceof ApiError ? err.message : "Something went wrong.");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">Monthly Investing Plan</h1>
      <p className="mt-1 text-sm text-slate-500">
        See what investing a fixed amount every month into the top-ranked fund <em>and</em> the top-ranked stock
        could look like, side by side — so you can compare before picking one.
      </p>

      <form onSubmit={runPlan} className="mt-6 flex flex-wrap items-end gap-3">
        <Field label="Monthly amount">
          <input
            type="number"
            min={100}
            max={10000}
            step={100}
            value={monthlyAmount}
            onChange={(e) => setMonthlyAmount(Number(e.target.value))}
            className="input w-28"
          />
        </Field>
        <Field label="Plan length (yrs)">
          <input
            type="number"
            min={1}
            max={15}
            value={years}
            onChange={(e) => setYears(Number(e.target.value))}
            className="input w-20"
          />
        </Field>
        <Field label="Fund goal">
          <select value={fundGoal} onChange={(e) => setFundGoal(e.target.value)} className="input">
            {fundGoals.map((g) => (
              <option key={g} value={g}>
                {g}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Fund category">
          <select value={fundCategory} onChange={(e) => setFundCategory(e.target.value)} className="input">
            {fundCategories.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Stock goal">
          <select value={stockGoal} onChange={(e) => setStockGoal(e.target.value)} className="input">
            {stockGoals.map((g) => (
              <option key={g} value={g}>
                {g}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Stock universe">
          <select value={stockUniverse} onChange={(e) => setStockUniverse(e.target.value)} className="input">
            {stockUniverses.map((u) => (
              <option key={u} value={u}>
                {u}
              </option>
            ))}
          </select>
        </Field>
        <button type="submit" disabled={loading} className="btn-primary">
          {loading ? "Building…" : "Build Plan"}
        </button>
      </form>

      {loading && <p className="mt-4 text-sm text-slate-500">Building both plans…</p>}
      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {!loading && (fundData || stockData) && (
        <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
          <PlanResultCard title="Top Fund Pick" goalLabel={fundGoal} data={fundData} monthlyAmount={monthlyAmount} years={years} />
          <PlanResultCard title="Top Stock Pick" goalLabel={stockGoal} data={stockData} monthlyAmount={monthlyAmount} years={years} />
        </div>
      )}

      {!loading && (fundData || stockData) && (
        <div className="mt-6 rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="font-semibold text-slate-900">Final Note</h3>
          <p className="mt-2 text-sm text-slate-600">
            This planner is a simple dollar-cost averaging view. It does not account for taxes, fees, dividends, or
            future regime changes, so use it as a planning tool rather than a promise. Comparing a diversified fund
            against a single stock this way is illustrative — a single stock carries materially more risk than a
            fund holding hundreds of positions, even when its trailing return looks better.
          </p>
        </div>
      )}
    </div>
  );
}

function PlanResultCard({
  title,
  goalLabel,
  data,
  monthlyAmount,
  years,
}: {
  title: string;
  goalLabel: string;
  data: MonthlyPlanResponse | null;
  monthlyAmount: number;
  years: number;
}) {
  const [activeInfo, setActiveInfo] = useState<ColumnInfo | null>(null);

  if (!data) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-5">
        <h2 className="text-lg font-semibold text-slate-900">{title}</h2>
        <p className="mt-2 text-sm text-slate-500">Could not load a pick for this side right now.</p>
      </div>
    );
  }

  if (!data.recommendation) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-5">
        <h2 className="text-lg font-semibold text-slate-900">{title}</h2>
        <p className="mt-2 text-sm text-slate-500">No ranked pick was available right now. Try a different filter.</p>
      </div>
    );
  }

  const { recommendation } = data;
  const scoreInfo = SCORE_INFO[recommendation.asset_type === "Fund" ? "Fund" : "Stock"];

  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-lg border border-slate-200 bg-white p-5">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-400">{title}</p>
        <h2 className="mt-1 text-lg font-semibold text-slate-900">
          {recommendation.ticker} — {recommendation.name}
        </h2>
        <p className="mt-1 text-sm text-slate-600">
          This {recommendation.asset_type.toLowerCase()} ranked highest for <strong>{goalLabel}</strong> in the
          current screen.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <MetricTile label="Score" value={`${recommendation.score.toFixed(1)}/100`} onInfoClick={() => setActiveInfo(scoreInfo)} />
        <MetricTile label="Monthly Invest" value={`$${monthlyAmount.toLocaleString()}`} />
        <MetricTile label="Plan Length" value={`${years} years`} />
        <MetricTile
          label="Expected Annual Return"
          value={recommendation.expected_return_pct !== null ? `${recommendation.expected_return_pct.toFixed(2)}%` : "N/A"}
        />
      </div>

      {data.summary && data.history ? (
        <>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <MetricTile label="Total Invested" value={`$${data.summary.total_invested.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
            <MetricTile label="Historical Ending Value" value={`$${data.summary.ending_value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
            <MetricTile
              label="Historical Gain"
              value={`$${data.summary.gain.toLocaleString(undefined, { maximumFractionDigits: 0 })} (${data.summary.gain_pct.toFixed(1)}%)`}
            />
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-4">
            <MonthlyChart ticker={recommendation.ticker} history={data.history} />
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <h3 className="font-semibold text-slate-900">Forward Estimate</h3>
            <p className="mt-2 text-sm text-slate-600">
              Projected portfolio value:{" "}
              <strong>{data.projected_value !== null ? `$${data.projected_value.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : "N/A"}</strong>{" "}
              — uses the asset&apos;s trailing annualized return as a simple forward estimate, useful for
              planning but not guaranteed.
            </p>
          </div>
        </>
      ) : (
        <p className="text-sm text-slate-500">Not enough price history was available to simulate this plan.</p>
      )}

      {activeInfo && <InfoModal info={activeInfo} onClose={() => setActiveInfo(null)} />}
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

function MetricTile({
  label,
  value,
  onInfoClick,
}: {
  label: string;
  value: string;
  onInfoClick?: () => void;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <p className="flex items-center gap-1 text-xs text-slate-500">
        {label}
        {onInfoClick && (
          <button
            type="button"
            onClick={onInfoClick}
            title={`What is ${label}?`}
            className="flex h-4 w-4 items-center justify-center rounded-full border border-slate-300 text-[10px] font-normal normal-case text-slate-400 hover:border-slate-500 hover:text-slate-700"
          >
            i
          </button>
        )}
      </p>
      <p className="mt-1 text-lg font-semibold text-slate-900">{value}</p>
    </div>
  );
}
