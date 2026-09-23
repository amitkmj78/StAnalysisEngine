"use client";

import { useEffect, useMemo, useState } from "react";

import InfoModal, { type ColumnInfo } from "@/components/InfoModal";
import MonteCarloChart from "@/components/strategies/MonteCarloChart";
import {
  ApiError,
  deleteStrategyPlan,
  getPortfolioSummary,
  getStrategiesOptions,
  getStrategiesSummary,
  getStrategyPlans,
  saveStrategyPlan,
} from "@/lib/api";
import type {
  AccountType,
  DollarsMode,
  GoalPlan,
  ReturnAssumptionRow,
  SavedStrategyPlan,
  SolveMode,
  StrategiesSummaryResponse,
  StrategyPickRow,
} from "@/lib/types";

const KPI_INFO: Record<string, ColumnInfo> = {
  historic_return: {
    title: "Historic Annualized Return",
    body: [
      "This is the pick's own trailing 3-year annualized return (CAGR), computed straight from price history — how much it actually grew per year, on average, over the last 3 years.",
      "It is a raw historical fact about that one ticker, not the weighted Ranking Score below it — a pick can have a huge historic return but a middling score if it scores poorly on the other factors (cost, valuation, drawdown, etc).",
      "Past performance like this does not guarantee future results, especially for a single stock rather than a diversified fund.",
    ],
  },
};

const SOLVE_MODE_LABELS: Record<SolveMode, string> = {
  required_return: "Required return",
  required_contribution: "Required contribution",
  time_to_goal: "Time to goal",
  achievable_amount: "Achievable amount",
};

// Mirrors services/million_plan_service.py's FUND_CATEGORY_RISK_TIER --
// small, fixed vocabulary (this page's own coarse 7-category list, not the
// Fund Screener's finer-grained one) -- kept here so the horizon-conflict
// warning can update live as the user changes Years/Category, before they
// click "Build Plan" (the backend's own check runs again after, as
// confirmation/fallback).
const FUND_CATEGORY_RISK_TIER: Record<string, "cash_short" | "growth" | null> = {
  All: null,
  Bond: "cash_short",
  "US Large Blend": "growth",
  "US Total Market": "growth",
  "US Growth": "growth",
  "US Small Cap": "growth",
  International: "growth",
};

function horizonConflictWarnings(years: number, fundCategory: string): string[] {
  const warnings: string[] = [];
  const tier = FUND_CATEGORY_RISK_TIER[fundCategory];
  if (years < 3) {
    if (tier === "growth") {
      warnings.push(
        `A ${years}-year horizon calls for cash and short-duration bonds -- "${fundCategory}" is an equity category and carries meaningfully more risk than this timeframe usually allows for.`,
      );
    }
    warnings.push(`Individual stock picks are equity risk, which is generally not appropriate for a ${years}-year horizon.`);
  } else if (years < 7) {
    if (tier === "growth") {
      warnings.push(
        `A ${years}-year horizon calls for a conservative mix with equity capped -- "${fundCategory}" is a full growth category; consider a more conservative source or capping how much of the plan it drives.`,
      );
    }
    warnings.push(`Individual stock picks are equity risk; at a ${years}-year horizon, consider limiting how much of the plan relies on them.`);
  }
  return warnings;
}

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

function fmtMoney(v: number | null | undefined, opts: Intl.NumberFormatOptions = {}) {
  if (v === null || v === undefined) return "N/A";
  return `$${v.toLocaleString(undefined, { maximumFractionDigits: 0, ...opts })}`;
}

// Rounds a solved value for display in a disabled input -- the raw API
// figure can carry many decimals (e.g. years=14.625, contribution=537.99),
// which is precise but not what a "solved" field should visually show.
function round2(v: number) {
  return Math.round(v * 100) / 100;
}

export default function StrategiesPage() {
  const [fundCategories, setFundCategories] = useState<string[]>([]);
  const [stockUniverses, setStockUniverses] = useState<string[]>([]);
  const [accountTypes, setAccountTypes] = useState<AccountType[]>(["Taxable", "Traditional", "Roth"]);
  const [taxDragByAccount, setTaxDragByAccount] = useState<Record<string, number>>({ Taxable: 0.5, Traditional: 0, Roth: 0 });

  const [mode, setMode] = useState<SolveMode>("required_return");
  const [targetAmount, setTargetAmount] = useState(1_000_000);
  const [dollarsMode, setDollarsMode] = useState<DollarsMode>("today");
  const [inflationPct, setInflationPct] = useState(2.5);
  const [years, setYears] = useState(5);
  const [startingCapital, setStartingCapital] = useState(0);
  const [monthlyContribution, setMonthlyContribution] = useState(500);
  const [annualIncreasePct, setAnnualIncreasePct] = useState(0);
  const [annualReturnPct, setAnnualReturnPct] = useState(8);
  const [accountType, setAccountType] = useState<AccountType>("Taxable");
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
        setAccountTypes(res.account_types);
        setTaxDragByAccount(res.tax_drag_pct_by_account);
        setFundCategory(res.fund_categories[0] ?? "All");
        setStockUniverse(res.stock_universes[0] ?? "All");
        setTargetAmount(res.defaults.target_amount);
        setYears(res.defaults.years);
        setInflationPct(res.defaults.inflation_pct);
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

  const preflightHorizonWarnings = useMemo(() => horizonConflictWarnings(years, fundCategory), [years, fundCategory]);

  async function handleSavePlan() {
    if (!data) return;
    setSaving(true);
    setSaveError(null);
    setSaveMessage(null);
    try {
      await saveStrategyPlan({
        name: planName.trim() || undefined,
        target_amount: data.plan.target_future_dollars,
        years: data.plan.years,
        starting_capital: startingCapital,
        annual_return_pct: data.plan.gross_return_pct ?? 0,
        monthly_contribution: data.plan.monthly_contribution,
        annual_contribution_increase_pct: annualIncreasePct,
        account_type: accountType,
        inflation_pct: inflationPct,
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
      const params: Record<string, string> = {
        mode,
        dollars_mode: dollarsMode,
        starting_capital: String(startingCapital),
        annual_contribution_increase_pct: String(annualIncreasePct),
        inflation_pct: String(inflationPct),
        account_type: accountType,
        fund_category: fundCategory,
        stock_universe: stockUniverse,
        top_n: String(topN),
      };
      if (mode !== "achievable_amount") params.target_amount = String(targetAmount);
      if (mode !== "time_to_goal") params.years = String(years);
      if (mode !== "required_contribution") params.monthly_contribution = String(monthlyContribution);
      if (mode !== "required_return") params.annual_return_pct = String(annualReturnPct);

      const res = await getStrategiesSummary(params);
      setData(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong.");
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  const plan = data?.plan ?? null;

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">Strategies</h1>
      <p className="mt-1 text-sm text-slate-500">
        Build a feasible plan — pick what to solve for, see whether it's realistic, and get the candidates behind it.
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
            {plans.map((p) => (
              <div key={p.id} className="rounded-lg border border-slate-200 bg-white p-4">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="font-semibold text-slate-900">
                      {p.name || `$${p.target_amount.toLocaleString()} in ${p.years}y`}
                    </p>
                    <p className="text-xs text-slate-500">
                      ${p.monthly_contribution.toLocaleString(undefined, { maximumFractionDigits: 0 })}/mo at{" "}
                      {p.annual_return_pct.toFixed(1)}% · {p.account_type}
                      {p.annual_contribution_increase_pct > 0 && ` · +${p.annual_contribution_increase_pct}%/yr contribution step-up`}
                      {" · saved "}
                      {new Date(p.created_at).toLocaleDateString()}
                    </p>
                  </div>
                  <span
                    className={`shrink-0 rounded-full px-2 py-0.5 text-xs font-semibold ${
                      p.progress.on_track ? "bg-emerald-50 text-emerald-700" : "bg-red-50 text-red-700"
                    }`}
                  >
                    {p.progress.on_track ? "On track" : "Behind pace"}
                  </span>
                </div>
                <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
                  <div>
                    <p className="text-xs text-slate-500">Expected by now</p>
                    <p className="font-medium text-slate-800">{fmtMoney(p.progress.expected_value)}</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-500">Your portfolio now</p>
                    <p className="font-medium text-slate-800">{fmtMoney(p.progress.actual_value)}</p>
                  </div>
                </div>
                <p className={`mt-2 text-xs font-medium ${p.progress.on_track ? "text-emerald-600" : "text-red-600"}`}>
                  {p.progress.diff >= 0 ? "+" : ""}
                  {fmtMoney(p.progress.diff)}
                  {p.progress.diff_pct !== null && ` (${p.progress.diff_pct >= 0 ? "+" : ""}${p.progress.diff_pct.toFixed(1)}%)`}
                  {" "}
                  vs. plan · {p.progress.months_elapsed} mo in
                </p>
                <button
                  onClick={() => handleDeletePlan(p.id)}
                  disabled={deletingId === p.id}
                  className="mt-3 rounded-md border border-slate-300 px-2.5 py-1 text-xs font-medium text-slate-600 hover:bg-slate-100 disabled:opacity-50"
                >
                  {deletingId === p.id ? "Removing…" : "Remove"}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      <form onSubmit={runPlan} className="mt-6 flex flex-col gap-3">
        <div className="flex flex-wrap items-end gap-3">
          <Field label="Solve for">
            <select value={mode} onChange={(e) => setMode(e.target.value as SolveMode)} className="input">
              {(Object.keys(SOLVE_MODE_LABELS) as SolveMode[]).map((m) => (
                <option key={m} value={m}>{SOLVE_MODE_LABELS[m]}</option>
              ))}
            </select>
          </Field>

          <Field label={mode === "achievable_amount" ? "Target amount (solved)" : "Target amount"}>
            <input
              type="number" min={1} max={100000000} step="any"
              value={
                mode === "achievable_amount" && plan
                  ? round2(dollarsMode === "today" ? plan.target_today_dollars : plan.target_future_dollars)
                  : targetAmount
              }
              onChange={(e) => setTargetAmount(Number(e.target.value))}
              disabled={mode === "achievable_amount"}
              className="input w-32 disabled:bg-slate-50 disabled:text-slate-400"
            />
          </Field>
          <Field label="In">
            <select value={dollarsMode} onChange={(e) => setDollarsMode(e.target.value as DollarsMode)} className="input" disabled={mode === "achievable_amount"}>
              <option value="today">Today&apos;s dollars</option>
              <option value="future">Future dollars</option>
            </select>
          </Field>

          <Field label={mode === "time_to_goal" ? "Years to goal (solved)" : "Years to goal"}>
            <input
              type="number" min={1} max={20}
              value={mode === "time_to_goal" && plan ? round2(plan.years) : years}
              onChange={(e) => setYears(Number(e.target.value))}
              disabled={mode === "time_to_goal"}
              className="input w-20 disabled:bg-slate-50 disabled:text-slate-400"
            />
          </Field>

          <Field label="Starting capital">
            <input
              type="number" min={0} max={10000000} step="any"
              value={startingCapital}
              onChange={(e) => {
                setStartingCapital(Number(e.target.value));
                setStartingCapitalTouched(true);
              }}
              className="input w-28"
            />
          </Field>
        </div>

        <div className="flex flex-wrap items-end gap-3">
          <Field label={mode === "required_contribution" ? "Monthly contribution (solved)" : "Monthly contribution"}>
            <input
              type="number" min={0} max={1000000} step={50}
              value={mode === "required_contribution" && plan ? round2(plan.monthly_contribution) : monthlyContribution}
              onChange={(e) => setMonthlyContribution(Number(e.target.value))}
              disabled={mode === "required_contribution"}
              className="input w-28 disabled:bg-slate-50 disabled:text-slate-400"
            />
          </Field>
          <Field label="Annual contribution increase %">
            <input
              type="number" min={0} max={20} step={0.5}
              value={annualIncreasePct}
              onChange={(e) => setAnnualIncreasePct(Number(e.target.value))}
              className="input w-20"
            />
          </Field>
          <Field label={mode === "required_return" ? "Annual return % (solved)" : "Annual return %"}>
            <input
              type="number" min={-20} max={50} step={0.5}
              value={mode === "required_return" && plan && plan.gross_return_pct !== null ? round2(plan.gross_return_pct) : annualReturnPct}
              onChange={(e) => setAnnualReturnPct(Number(e.target.value))}
              disabled={mode === "required_return"}
              className="input w-24 disabled:bg-slate-50 disabled:text-slate-400"
            />
          </Field>
          <Field label="Inflation %">
            <input type="number" min={0} max={15} step={0.1} value={inflationPct} onChange={(e) => setInflationPct(Number(e.target.value))} className="input w-20" />
          </Field>
          <Field label="Account type">
            <select value={accountType} onChange={(e) => setAccountType(e.target.value as AccountType)} className="input">
              {accountTypes.map((a) => (
                <option key={a} value={a}>{a}</option>
              ))}
            </select>
          </Field>
        </div>

        <p className="text-xs text-slate-500">
          {accountType} accounts assume a {(taxDragByAccount[accountType] ?? 0).toFixed(1)}%/year tax drag on returns during
          accumulation{(taxDragByAccount[accountType] ?? 0) === 0 ? " (tax-advantaged, no drag modeled)." : " (dividend/turnover taxation)."}
        </p>

        <div className="flex flex-wrap items-end gap-3">
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
        </div>

        {preflightHorizonWarnings.length > 0 && (
          <div className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
            {preflightHorizonWarnings.map((w) => (
              <p key={w}>{w}</p>
            ))}
          </div>
        )}
      </form>

      {loading && <p className="mt-4 text-sm text-slate-500">Building your plan…</p>}
      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {plan && !loading && (
        <div className="mt-6 flex flex-col gap-6">
          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <h2 className="text-lg font-semibold text-slate-900">{plan.solved_field_label}</h2>
            <p className="mt-1 text-2xl font-semibold text-slate-900">
              {plan.mode === "required_return"
                ? plan.solved_value !== null ? `${plan.solved_value.toFixed(1)}%` : "Not reachable"
                : plan.mode === "required_contribution"
                ? fmtMoney(plan.solved_value) + "/mo"
                : plan.mode === "time_to_goal"
                ? plan.solved_value !== null ? `${plan.solved_value.toFixed(1)} years` : "Not within 60 years"
                : fmtMoney(plan.solved_value)}
            </p>
            <p className="mt-2 text-sm text-slate-600">
              Target: {fmtMoney(plan.target_today_dollars)} in today&apos;s dollars ·{" "}
              {fmtMoney(plan.target_future_dollars)} in future dollars (at {plan.years.toFixed(1)}y, {plan.inflation_pct}% inflation)
            </p>
          </div>

          <MonteCarloPanel plan={plan} />

          {plan.feasibility_level === "warning" && (
            <div className="rounded-md bg-amber-50 px-4 py-3 text-sm text-amber-800">
              <p>{plan.feasibility_message}</p>
              {plan.return_assumption_table && <ReturnAssumptionTable rows={plan.return_assumption_table} tone="amber" />}
            </div>
          )}

          {plan.feasibility_level === "blocked" && (
            <div className="rounded-lg border border-red-200 bg-red-50 p-4">
              <p className="text-sm font-medium text-red-800">{plan.feasibility_message}</p>
              {plan.fixes && (
                <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
                  {plan.fixes.map((fix) => (
                    <div key={fix.type} className="rounded-md border border-red-200 bg-white p-3">
                      <p className="text-xs font-semibold uppercase tracking-wide text-red-700">{fix.label}</p>
                      <p className="mt-1 text-sm text-slate-800">
                        {fix.type === "more_time" && (fix.years_needed != null ? `${fix.years_needed} years instead of ${plan.years.toFixed(1)}` : "N/A")}
                        {fix.type === "more_contribution" &&
                          (fix.monthly_contribution_needed != null
                            ? `${fmtMoney(fix.monthly_contribution_needed)}/mo instead of ${fmtMoney(plan.monthly_contribution)}/mo`
                            : "N/A")}
                        {fix.type === "lower_target" && `${fmtMoney(fix.achievable_target_future_dollars)} instead of ${fmtMoney(plan.target_future_dollars)}`}
                      </p>
                    </div>
                  ))}
                </div>
              )}
              {plan.return_assumption_table && <ReturnAssumptionTable rows={plan.return_assumption_table} tone="red" />}
            </div>
          )}

          {plan.horizon_warnings.length > 0 && (
            <div className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
              {plan.horizon_warnings.map((w) => (
                <p key={w}>{w}</p>
              ))}
            </div>
          )}

          {plan.feasibility_level !== "blocked" && (
            <div className="rounded-lg border border-slate-200 bg-white p-4">
              <p className="text-sm font-medium text-slate-700">Save this goal to track your progress over time</p>
              <p className="mt-1 text-xs text-slate-500">
                Saves this plan — target {fmtMoney(plan.target_future_dollars)} in {plan.years.toFixed(1)} years,{" "}
                {fmtMoney(plan.monthly_contribution)}/mo at {plan.gross_return_pct?.toFixed(1)}% — and starts comparing it
                against your real portfolio value every time you visit.
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
                <button type="button" onClick={handleSavePlan} disabled={saving} className="btn-primary">
                  {saving ? "Saving…" : "Save This Goal"}
                </button>
              </div>
              {saveMessage && <p className="mt-2 text-xs text-emerald-700">{saveMessage}</p>}
              {saveError && <p className="mt-2 text-xs text-red-600">{saveError}</p>}
            </div>
          )}

          {plan.feasibility_level !== "blocked" && (
            <>
              <div>
                <h2 className="text-lg font-semibold text-slate-900">Best Builders Right Now</h2>
                <p className="mt-1 text-sm text-slate-500">Built live from the current top-ranked fund and stock results.</p>
              </div>

              {!data?.picks || data.picks.length === 0 ? (
                <p className="text-sm text-slate-500">No ranked picks were available right now.</p>
              ) : (
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                  {data.picks.map((pick) => {
                    // Every metric this pick's own goal actually weights --
                    // e.g. "Lowest Cost" surfaces expense ratio, "Most
                    // Stable" surfaces volatility/max drawdown, not just
                    // return figures. "3-Year Annualized Return" is
                    // excluded here since the dedicated KpiLine below
                    // already shows that exact number -- showing it twice
                    // was the reported duplicate.
                    const weightedMetrics = pick.score_basis.filter((f) => f.value !== null && f.metric !== "3-Year Annualized Return");
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
                          label="3-year annualized return"
                          value={pick.annual_return_pct !== null ? `${pick.annual_return_pct.toFixed(2)}%` : "N/A"}
                          onInfoClick={() => setActiveKpiInfo(KPI_INFO.historic_return)}
                        />

                        {weightedMetrics.length > 0 && (
                          <div className="mt-3 flex flex-wrap gap-1.5 border-t border-slate-100 pt-3">
                            {weightedMetrics.map((f) => (
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
            </>
          )}

          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <h3 className="font-semibold text-slate-900">Final Note</h3>
            <p className="mt-2 text-sm text-slate-600">
              This page is a planning calculator, not a promise. Projections are hypothetical and not predictive of
              actual results; past performance is not indicative of future results. It uses simplified compounding
              math and current ranking outputs, and does not account for fees, slippage, dividends, or changing
              market regimes beyond the stated {accountType} tax-drag assumption. Not investment advice.
            </p>
          </div>
        </div>
      )}

      {activeKpiInfo && <InfoModal info={activeKpiInfo} onClose={() => setActiveKpiInfo(null)} />}
    </div>
  );
}

function ReturnAssumptionTable({ rows, tone }: { rows: ReturnAssumptionRow[]; tone: "amber" | "red" }) {
  const borderColor = tone === "amber" ? "border-amber-200" : "border-red-200";
  const headColor = tone === "amber" ? "text-amber-700" : "text-red-700";
  return (
    <div className={`mt-3 overflow-hidden rounded-md border ${borderColor} bg-white`}>
      <table className="w-full text-sm">
        <thead>
          <tr className={`border-b ${borderColor} text-left text-xs font-semibold uppercase tracking-wide ${headColor}`}>
            <th className="px-3 py-1.5">If the real return is</th>
            <th className="px-3 py-1.5 text-right">Monthly contribution needed</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.annual_return_pct} className={`border-b ${borderColor} text-slate-700 last:border-0`}>
              <td className="px-3 py-1.5">{row.annual_return_pct.toFixed(0)}%</td>
              <td className="px-3 py-1.5 text-right">
                {row.monthly_contribution_needed != null ? `${fmtMoney(row.monthly_contribution_needed)}/mo` : "N/A"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const PROB_TONE_CLASSES: Record<"emerald" | "amber" | "red", string> = {
  emerald: "text-emerald-600",
  amber: "text-amber-600",
  red: "text-red-600",
};

function MonteCarloPanel({ plan }: { plan: GoalPlan }) {
  const mc = plan.monte_carlo;
  if (!mc) return null;
  const prob = mc.probability_of_success_pct;
  const tone: "emerald" | "amber" | "red" | null = prob === null ? null : prob >= 70 ? "emerald" : prob >= 40 ? "amber" : "red";

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5">
      <h2 className="text-lg font-semibold text-slate-900">Probability of Success</h2>
      <p className="mt-1 text-sm text-slate-500">
        {mc.assumptions.num_paths.toLocaleString()} randomly sampled possible futures, built from real historical
        market returns — not just the one average-return number above.
      </p>

      {prob !== null && tone !== null ? (
        <>
          <p className={`mt-3 text-4xl font-semibold ${PROB_TONE_CLASSES[tone]}`}>{prob.toFixed(1)}%</p>
          <p className="mt-1 text-sm text-slate-600">
            of simulated paths reached {fmtMoney(plan.target_future_dollars)} or more by year {plan.years.toFixed(1)}.
          </p>
        </>
      ) : (
        <p className="mt-3 text-sm text-slate-600">
          This mode has no fixed target to hit, so there&apos;s no single probability to show — the spread below is
          the range of likely ending balances instead.
        </p>
      )}

      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">10th percentile</p>
          <p className="mt-1 text-lg font-semibold text-slate-800">{fmtMoney(mc.p10_ending_balance)}</p>
          <p className="mt-1 text-xs text-slate-500">A rough outcome — 90% of simulations did better.</p>
        </div>
        <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Median</p>
          <p className="mt-1 text-lg font-semibold text-slate-800">{fmtMoney(mc.median_ending_balance)}</p>
          <p className="mt-1 text-xs text-slate-500">The middle outcome — half did better, half worse.</p>
        </div>
        <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">90th percentile</p>
          <p className="mt-1 text-lg font-semibold text-slate-800">{fmtMoney(mc.p90_ending_balance)}</p>
          <p className="mt-1 text-xs text-slate-500">A strong outcome — only 10% of simulations did better.</p>
        </div>
      </div>

      {mc.percentile_bands.length > 0 && (
        <div className="mt-4">
          <MonteCarloChart result={mc} startingCapital={plan.starting_capital} />
        </div>
      )}

      <details className="mt-4 rounded-md border border-slate-200 bg-slate-50 p-3 text-sm text-slate-600">
        <summary className="cursor-pointer font-medium text-slate-700">Simulation assumptions</summary>
        <ul className="mt-2 list-disc space-y-1 pl-5">
          <li>{mc.assumptions.return_distribution_method}.</li>
          <li>{mc.assumptions.num_paths.toLocaleString()} simulated paths.</li>
          <li>
            {mc.assumptions.sequence_of_returns_modeled
              ? "Sequence-of-returns risk is modeled — the order returns arrive in varies per simulated path, not just the long-run average."
              : "Sequence-of-returns risk is not modeled."}
          </li>
          <li>Rebalancing: {mc.assumptions.rebalancing_frequency}.</li>
          <li>Correlation across holdings: {mc.assumptions.sleeve_correlation_model}.</li>
        </ul>
      </details>
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
