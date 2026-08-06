"use client";

import { useEffect, useState } from "react";

import InfoModal, { type ColumnInfo } from "@/components/InfoModal";
import PlanChart from "@/components/strategies/PlanChart";
import {
  ApiError,
  deleteStrategyPlan,
  getPortfolioSummary,
  getStrategiesOptions,
  getStrategiesSummary,
  getStrategyPlans,
  saveStrategyPlan,
} from "@/lib/api";
import type { SavedStrategyPlan, StrategiesSummaryResponse, StrategyPickRow } from "@/lib/types";

const KPI_INFO: Record<string, ColumnInfo> = {
  historic_return: {
    title: "Historic Annualized Return",
    body: [
      "This is the pick's own trailing 3-year annualized return (CAGR), computed straight from price history — how much it actually grew per year, on average, over the last 3 years.",
      "It is a raw historical fact about that one ticker, not the weighted Ranking Score below it — a pick can have a huge historic return but a middling score if it scores poorly on the other factors (cost, valuation, drawdown, etc).",
      "Past performance like this does not guarantee future results, especially for a single stock rather than a diversified fund.",
    ],
  },
  monthly_needed: {
    title: "Monthly Needed",
    body: [
      "The monthly contribution required to reach your target amount by your target year, assuming this pick's historic annualized return holds steady for the entire period, compounding monthly.",
      "It uses the same annuity math as the 'What It Takes' table above, just plugging in this specific pick's own historic return instead of a hypothetical return case.",
      "This is best read as 'what it would have taken if the past continued exactly' — not a promise about what it will actually take.",
    ],
  },
  projected_value: {
    title: "Projected Value",
    body: [
      "The total portfolio value you'd end up with after contributing the 'Monthly Needed' amount every month for the full time horizon, assuming this pick's historic annualized return holds the whole time.",
      "It compounds the monthly contributions at that return rate — it does not account for taxes, fees, dividends, or the return rate changing year to year.",
      "The further a pick's historic return is from a realistic long-run average, the less reliable this number is as an actual forecast.",
    ],
  },
};

function scoreInfo(pick: StrategyPickRow): ColumnInfo {
  const rows = pick.score_basis.map((f) => {
    const valueStr = f.value === null || f.value === undefined ? "N/A" : `${f.value.toFixed(2)}${f.unit ? ` ${f.unit}` : ""}`;
    const direction = f.lower_is_better ? "lower is better" : "higher is better";
    return `${f.metric} — weighted ${f.weight_pct}% of the score, ${direction}. ${pick.ticker}'s actual value: ${valueStr}.`;
  });
  return {
    title: `How "${pick.label}" Was Scored`,
    body: [
      `${pick.ticker} was ranked #1 among all ${pick.asset_type.toLowerCase()}s for the "${pick.label}" strategy using a weighted composite of these factors, each normalized 0-100 relative to every other candidate in the current universe:`,
      ...rows,
      "Ranking Score is this weighted blend (0-100) — it can differ from any single metric like historic return because it balances return against cost, valuation, and risk factors specific to this strategy.",
    ],
  };
}

export default function StrategiesPage() {
  const [fundCategories, setFundCategories] = useState<string[]>([]);
  const [stockUniverses, setStockUniverses] = useState<string[]>([]);

  const [targetAmount, setTargetAmount] = useState(1_000_000);
  const [years, setYears] = useState(5);
  const [startingCapital, setStartingCapital] = useState(0);
  const [customReturn, setCustomReturn] = useState(10);
  const [topN, setTopN] = useState(1);
  const [fundCategory, setFundCategory] = useState("All");
  const [stockUniverse, setStockUniverse] = useState("All");

  const [data, setData] = useState<StrategiesSummaryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeKpiInfo, setActiveKpiInfo] = useState<ColumnInfo | null>(null);

  const [startingCapitalTouched, setStartingCapitalTouched] = useState(false);

  const [plans, setPlans] = useState<SavedStrategyPlan[] | null>(null);
  const [plansError, setPlansError] = useState<string | null>(null);
  const [planName, setPlanName] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  async function loadPlans() {
    try {
      const res = await getStrategyPlans();
      setPlans(res.plans);
    } catch (err) {
      setPlansError(err instanceof ApiError ? err.message : "Failed to load saved goals.");
    }
  }

  useEffect(() => {
    getStrategiesOptions()
      .then((res) => {
        setFundCategories(res.fund_categories);
        setStockUniverses(res.stock_universes);
        setFundCategory(res.fund_categories[0] ?? "All");
        setStockUniverse(res.stock_universes[0] ?? "All");
        setTargetAmount(res.defaults.target_amount);
        setYears(res.defaults.years);
      })
      .catch(() => {});

    getPortfolioSummary()
      .then((res) => {
        if (!startingCapitalTouched && res.summary.total_value > 0) {
          setStartingCapital(Math.round(res.summary.total_value));
        }
      })
      .catch(() => {});

    loadPlans();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleSavePlan() {
    setSaving(true);
    setSaveError(null);
    setSaveMessage(null);
    try {
      await saveStrategyPlan({
        name: planName.trim() || undefined,
        target_amount: targetAmount,
        years,
        starting_capital: startingCapital,
        annual_return_pct: customReturn,
      });
      setPlanName("");
      setSaveMessage("Goal saved — see it below under My Goals.");
      await loadPlans();
    } catch (err) {
      setSaveError(err instanceof ApiError ? err.message : "Could not save this goal.");
    } finally {
      setSaving(false);
    }
  }

  async function handleDeletePlan(id: number) {
    setDeletingId(id);
    try {
      await deleteStrategyPlan(id);
      setPlans((prev) => (prev ?? []).filter((p) => p.id !== id));
    } catch (err) {
      setPlansError(err instanceof ApiError ? err.message : "Could not delete this goal.");
    } finally {
      setDeletingId(null);
    }
  }

  async function runPlan(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await getStrategiesSummary({
        target_amount: String(targetAmount),
        years: String(years),
        starting_capital: String(startingCapital),
        min_return: "6",
        max_return: "15",
        return_step: "2",
        custom_return: String(customReturn),
        fund_category: fundCategory,
        stock_universe: stockUniverse,
        top_n: String(topN),
      });
      setData(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong.");
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">Strategies</h1>
      <p className="mt-1 text-sm text-slate-500">
        Build a target-based investing strategy and see the best fund and stock candidates that can help build it.
      </p>

      {plansError && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{plansError}</p>}

      {plans !== null && plans.length > 0 && (
        <div className="mt-6">
          <h2 className="text-lg font-semibold text-slate-900">My Goals</h2>
          <p className="mt-1 text-xs text-slate-500">
            Each goal locked in a required monthly contribution when saved. Progress compares what you&apos;d have
            if you&apos;d actually contributed that amount every month since then against your real portfolio value
            today — it assumes the contribution was made, not a verified ledger of it.
          </p>
          <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
            {plans.map((plan) => (
              <div key={plan.id} className="rounded-lg border border-slate-200 bg-white p-4">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="font-semibold text-slate-900">
                      {plan.name || `$${plan.target_amount.toLocaleString()} in ${plan.years}y`}
                    </p>
                    <p className="text-xs text-slate-500">
                      ${plan.monthly_contribution.toLocaleString(undefined, { maximumFractionDigits: 0 })}/mo at{" "}
                      {plan.annual_return_pct.toFixed(1)}% · saved {new Date(plan.created_at).toLocaleDateString()}
                    </p>
                  </div>
                  <span
                    className={`shrink-0 rounded-full px-2 py-0.5 text-xs font-semibold ${
                      plan.progress.on_track ? "bg-emerald-50 text-emerald-700" : "bg-red-50 text-red-700"
                    }`}
                  >
                    {plan.progress.on_track ? "On track" : "Behind pace"}
                  </span>
                </div>
                <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
                  <div>
                    <p className="text-xs text-slate-500">Expected by now</p>
                    <p className="font-medium text-slate-800">
                      ${plan.progress.expected_value.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-500">Your portfolio now</p>
                    <p className="font-medium text-slate-800">
                      ${plan.progress.actual_value.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </p>
                  </div>
                </div>
                <p className={`mt-2 text-xs font-medium ${plan.progress.on_track ? "text-emerald-600" : "text-red-600"}`}>
                  {plan.progress.diff >= 0 ? "+" : ""}
                  ${plan.progress.diff.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  {plan.progress.diff_pct !== null && ` (${plan.progress.diff_pct >= 0 ? "+" : ""}${plan.progress.diff_pct.toFixed(1)}%)`}
                  {" "}
                  vs. plan · {plan.progress.months_elapsed} mo in
                </p>
                <button
                  onClick={() => handleDeletePlan(plan.id)}
                  disabled={deletingId === plan.id}
                  className="mt-3 rounded-md border border-slate-300 px-2.5 py-1 text-xs font-medium text-slate-600 hover:bg-slate-100 disabled:opacity-50"
                >
                  {deletingId === plan.id ? "Removing…" : "Remove"}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      <form onSubmit={runPlan} className="mt-6 flex flex-wrap items-end gap-3">
        <Field label="Target amount">
          <input type="number" min={50000} max={10000000} step={50000} value={targetAmount} onChange={(e) => setTargetAmount(Number(e.target.value))} className="input w-32" />
        </Field>
        <Field label="Years to goal">
          <input type="number" min={1} max={20} value={years} onChange={(e) => setYears(Number(e.target.value))} className="input w-20" />
        </Field>
        <Field label="Starting capital">
          <input
            type="number"
            min={0}
            max={10000000}
            step={1000}
            value={startingCapital}
            onChange={(e) => {
              setStartingCapital(Number(e.target.value));
              setStartingCapitalTouched(true);
            }}
            className="input w-28"
          />
        </Field>
        <Field label="Custom return %">
          <input type="number" min={4} max={20} value={customReturn} onChange={(e) => setCustomReturn(Number(e.target.value))} className="input w-20" />
        </Field>
        <Field label="Picks per strategy">
          <input type="number" min={1} max={5} value={topN} onChange={(e) => setTopN(Number(e.target.value))} className="input w-16" />
        </Field>
        <Field label="Fund category source">
          <select value={fundCategory} onChange={(e) => setFundCategory(e.target.value)} className="input">
            {fundCategories.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </Field>
        <Field label="Stock universe source">
          <select value={stockUniverse} onChange={(e) => setStockUniverse(e.target.value)} className="input">
            {stockUniverses.map((u) => (
              <option key={u} value={u}>{u}</option>
            ))}
          </select>
        </Field>
        <button type="submit" disabled={loading} className="btn-primary">
          {loading ? "Building…" : "Build Plan"}
        </button>
      </form>

      {loading && <p className="mt-4 text-sm text-slate-500">Building your plan…</p>}
      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {data && !loading && (
        <div className="mt-6 flex flex-col gap-6">
          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <h2 className="text-lg font-semibold text-slate-900">What It Takes</h2>
            <p className="mt-1 text-sm text-slate-600">
              To target <strong>${targetAmount.toLocaleString()}</strong> in <strong>{years} years</strong>, the
              required monthly contribution depends heavily on return assumptions and starting capital.
            </p>
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <MetricTile label="Target" value={`$${targetAmount.toLocaleString()}`} />
            <MetricTile label="Time Horizon" value={`${years} years`} />
            <MetricTile label={`Needed at ${customReturn}%`} value={`$${data.custom_monthly.toLocaleString(undefined, { maximumFractionDigits: 0 })}/mo`} />
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-4">
            <p className="text-sm font-medium text-slate-700">Save this goal to track your progress over time</p>
            <p className="mt-1 text-xs text-slate-500">
              Saves the {customReturn}% case above — target ${targetAmount.toLocaleString()} in {years} years,
              ${data.custom_monthly.toLocaleString(undefined, { maximumFractionDigits: 0 })}/mo — and starts
              comparing it against your real portfolio value every time you visit.
            </p>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <input
                type="text"
                placeholder="Optional name, e.g. Retirement"
                value={planName}
                onChange={(e) => setPlanName(e.target.value)}
                className="input w-56"
                maxLength={100}
              />
              <button
                type="button"
                onClick={handleSavePlan}
                disabled={saving}
                className="btn-primary"
              >
                {saving ? "Saving…" : "Save This Goal"}
              </button>
            </div>
            {saveMessage && <p className="mt-2 text-xs text-emerald-700">{saveMessage}</p>}
            {saveError && <p className="mt-2 text-xs text-red-600">{saveError}</p>}
          </div>

          <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                  <th className="px-3 py-2">Strategy</th>
                  <th className="px-3 py-2">Annual Return %</th>
                  <th className="px-3 py-2">Required Monthly</th>
                  <th className="px-3 py-2">Total Contributions</th>
                  <th className="px-3 py-2">Projected Value</th>
                </tr>
              </thead>
              <tbody>
                {data.plan_table.map((row) => (
                  <tr key={row.Strategy} className="border-b border-slate-100 last:border-0">
                    <td className="px-3 py-2 text-slate-700">{row.Strategy}</td>
                    <td className="px-3 py-2 text-slate-700">{row["Annual Return %"].toFixed(1)}%</td>
                    <td className="px-3 py-2 text-slate-700">${row["Required Monthly Invest"].toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                    <td className="px-3 py-2 text-slate-700">${row["Total Contributions"].toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                    <td className="px-3 py-2 text-slate-700">${row["Projected Value"].toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {data.plan_table.length > 0 && (
            <div className="rounded-lg border border-slate-200 bg-white p-4">
              <PlanChart targetAmount={targetAmount} years={years} planTable={data.plan_table} />
            </div>
          )}

          <div>
            <h2 className="text-lg font-semibold text-slate-900">Best Builders Right Now</h2>
            <p className="mt-1 text-sm text-slate-500">Built live from the current top-ranked fund and stock results.</p>
          </div>

          {data.picks.length === 0 ? (
            <p className="text-sm text-slate-500">No ranked picks were available right now.</p>
          ) : (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {data.picks.map((pick) => {
                const historicMetrics = pick.score_basis.filter((f) => f.metric.toLowerCase().includes("return") && f.value !== null);
                return (
                  <div key={`${pick.label}-${pick.ticker}`} className="rounded-lg border border-slate-200 bg-white p-5">
                    <h3 className="font-semibold text-slate-900">{pick.label}</h3>
                    <p className="text-sm text-slate-700">{pick.ticker} — {pick.name}</p>
                    <p className="mt-1 text-sm text-slate-600">Type: {pick.asset_type}</p>

                    <KpiLine
                      label="Ranking score"
                      value={`${pick.score.toFixed(1)}/100`}
                      onInfoClick={() => setActiveKpiInfo(scoreInfo(pick))}
                    />
                    <KpiLine
                      label="Historic annualized return"
                      value={pick.annual_return_pct !== null ? `${pick.annual_return_pct.toFixed(2)}%` : "N/A"}
                      onInfoClick={() => setActiveKpiInfo(KPI_INFO.historic_return)}
                    />
                    <KpiLine
                      label="Monthly needed"
                      value={pick.implied_monthly !== null ? `$${pick.implied_monthly.toLocaleString(undefined, { maximumFractionDigits: 0 })}/mo` : "N/A"}
                      onInfoClick={() => setActiveKpiInfo(KPI_INFO.monthly_needed)}
                    />
                    <KpiLine
                      label="Projected value"
                      value={pick.projected_value !== null ? `$${pick.projected_value.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : "N/A"}
                      onInfoClick={() => setActiveKpiInfo(KPI_INFO.projected_value)}
                    />

                    {historicMetrics.length > 0 && (
                      <div className="mt-3 flex flex-wrap gap-1.5 border-t border-slate-100 pt-3">
                        {historicMetrics.map((f) => (
                          <span
                            key={f.metric}
                            className="rounded-full bg-slate-50 px-2 py-0.5 text-xs text-slate-600"
                            title={`${f.metric}: weighted ${f.weight_pct}% of the ranking score`}
                          >
                            {f.metric}: {f.value !== null ? f.value.toFixed(2) : "N/A"}{f.unit ? f.unit : ""}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <h3 className="font-semibold text-slate-900">Final Note</h3>
            <p className="mt-2 text-sm text-slate-600">
              This page is a planning calculator, not a promise. It uses simplified compounding math and current
              ranking outputs, and does not account for taxes, slippage, dividends, or changing market regimes.
            </p>
          </div>
        </div>
      )}

      {activeKpiInfo && <InfoModal info={activeKpiInfo} onClose={() => setActiveKpiInfo(null)} />}
    </div>
  );
}

function KpiLine({ label, value, onInfoClick }: { label: string; value: string; onInfoClick: () => void }) {
  return (
    <p className="flex items-center gap-1.5 text-sm text-slate-600">
      <span>
        {label}: {value}
      </span>
      <button
        type="button"
        onClick={onInfoClick}
        title={`What is ${label}?`}
        className="flex h-4 w-4 items-center justify-center rounded-full border border-slate-300 text-[10px] font-normal text-slate-400 hover:border-slate-500 hover:text-slate-700"
      >
        i
      </button>
    </p>
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
