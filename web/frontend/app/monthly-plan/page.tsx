"use client";

import { useEffect, useState } from "react";

import MonthlyChart from "@/components/monthly/MonthlyChart";
import { ApiError, getMonthlyPlanOptions, getMonthlyPlanSummary } from "@/lib/api";
import type { MonthlyPlanResponse } from "@/lib/types";

export default function MonthlyPlanPage() {
  const [assetType, setAssetType] = useState<"Fund" | "Stock">("Fund");
  const [fundGoals, setFundGoals] = useState<string[]>([]);
  const [fundCategories, setFundCategories] = useState<string[]>([]);
  const [stockGoals, setStockGoals] = useState<string[]>([]);
  const [stockUniverses, setStockUniverses] = useState<string[]>([]);

  const [goal, setGoal] = useState("Balanced Core");
  const [selection, setSelection] = useState("All");
  const [monthlyAmount, setMonthlyAmount] = useState(1000);
  const [years, setYears] = useState(5);

  const [data, setData] = useState<MonthlyPlanResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getMonthlyPlanOptions()
      .then((res) => {
        setFundGoals(res.fund_goals);
        setFundCategories(res.fund_categories);
        setStockGoals(res.stock_goals);
        setStockUniverses(res.stock_universes);
        setGoal(res.fund_goals[0] ?? "Balanced Core");
        setSelection(res.fund_categories[0] ?? "All");
      })
      .catch(() => {});
  }, []);

  function switchAssetType(next: "Fund" | "Stock") {
    setAssetType(next);
    if (next === "Fund") {
      setGoal(fundGoals[0] ?? "Balanced Core");
      setSelection(fundCategories[0] ?? "All");
    } else {
      setGoal(stockGoals[0] ?? "Short Term");
      setSelection(stockUniverses[0] ?? "All");
    }
  }

  async function runPlan(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await getMonthlyPlanSummary(assetType, goal, selection, monthlyAmount, years);
      setData(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong.");
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  const goalOptions = assetType === "Fund" ? fundGoals : stockGoals;
  const selectionOptions = assetType === "Fund" ? fundCategories : stockUniverses;

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">Monthly Investing Plan</h1>
      <p className="mt-1 text-sm text-slate-500">
        See what investing a fixed amount every month into a top-ranked fund or stock could look like over time.
      </p>

      <form onSubmit={runPlan} className="mt-6 flex flex-wrap items-end gap-3">
        <Field label="Asset type">
          <select value={assetType} onChange={(e) => switchAssetType(e.target.value as "Fund" | "Stock")} className="input">
            <option value="Fund">Fund</option>
            <option value="Stock">Stock</option>
          </select>
        </Field>
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
        <Field label={assetType === "Fund" ? "Fund goal" : "Stock goal"}>
          <select value={goal} onChange={(e) => setGoal(e.target.value)} className="input">
            {goalOptions.map((g) => (
              <option key={g} value={g}>
                {g}
              </option>
            ))}
          </select>
        </Field>
        <Field label={assetType === "Fund" ? "Fund category" : "Stock universe"}>
          <select value={selection} onChange={(e) => setSelection(e.target.value)} className="input">
            {selectionOptions.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </Field>
        <button type="submit" disabled={loading} className="btn-primary">
          {loading ? "Building…" : "Build Plan"}
        </button>
      </form>

      {loading && <p className="mt-4 text-sm text-slate-500">Building your plan…</p>}
      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {data && !loading && data.recommendation && (
        <div className="mt-6 flex flex-col gap-6">
          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <h2 className="text-lg font-semibold text-slate-900">
              Suggested Pick: {data.recommendation.ticker} — {data.recommendation.name}
            </h2>
            <p className="mt-1 text-sm text-slate-600">
              This {data.recommendation.asset_type.toLowerCase()} ranked highest for <strong>{goal}</strong> in the
              current screen.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricTile label="Score" value={`${data.recommendation.score.toFixed(1)}/100`} />
            <MetricTile label="Monthly Invest" value={`$${monthlyAmount.toLocaleString()}`} />
            <MetricTile label="Plan Length" value={`${years} years`} />
            <MetricTile
              label="Expected Annual Return"
              value={data.recommendation.expected_return_pct !== null ? `${data.recommendation.expected_return_pct.toFixed(2)}%` : "N/A"}
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
                <MonthlyChart ticker={data.recommendation.ticker} history={data.history} />
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
            <p className="text-sm text-slate-500">Not enough price history was available to simulate the monthly plan.</p>
          )}

          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <h3 className="font-semibold text-slate-900">Final Note</h3>
            <p className="mt-2 text-sm text-slate-600">
              This planner is a simple dollar-cost averaging view. It does not account for taxes, fees, dividends, or
              future regime changes, so use it as a planning tool rather than a promise.
            </p>
          </div>
        </div>
      )}

      {data && !loading && !data.recommendation && (
        <p className="mt-4 text-sm text-slate-500">No ranked pick was available right now. Try a different filter.</p>
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

function MetricTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="mt-1 text-lg font-semibold text-slate-900">{value}</p>
    </div>
  );
}
