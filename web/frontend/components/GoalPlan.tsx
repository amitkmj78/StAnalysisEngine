"use client";

import { useEffect, useState } from "react";

import {
  ApiError,
  deleteSavedGoal,
  getGoalPlan,
  getSavedGoals,
  getStockUniverses,
  saveGoal,
} from "@/lib/api";
import type { GoalPlanResponse, SavedGoal } from "@/lib/types";

function fmtMoney(n: number) {
  return `$${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

export default function GoalPlan({ portfolioId }: { portfolioId: number | null }) {
  const [targetAmount, setTargetAmount] = useState("100000");
  const [targetDate, setTargetDate] = useState("");
  const [useOwnAmount, setUseOwnAmount] = useState(false);
  const [monthlyAmount, setMonthlyAmount] = useState("500");
  const [compareToBest, setCompareToBest] = useState(false);
  const [universes, setUniverses] = useState<string[]>([]);
  const [compareUniverse, setCompareUniverse] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [plan, setPlan] = useState<GoalPlanResponse | null>(null);

  const [goalName, setGoalName] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [savedGoals, setSavedGoals] = useState<SavedGoal[] | null>(null);
  const [savedGoalsError, setSavedGoalsError] = useState<string | null>(null);
  const [deletingGoalId, setDeletingGoalId] = useState<number | null>(null);

  useEffect(() => {
    getStockUniverses()
      .then((res) => {
        setUniverses(res.universes);
        if (res.universes.length > 0) setCompareUniverse((prev) => prev || res.universes[0]);
      })
      .catch(() => {
        // Silent — the comparison toggle just won't have options to pick from.
      });
  }, []);

  async function loadSavedGoals() {
    if (portfolioId === null) return;
    try {
      const res = await getSavedGoals(portfolioId);
      setSavedGoals(res.goals);
    } catch (err) {
      setSavedGoalsError(err instanceof ApiError ? err.message : "Could not load saved goals.");
    }
  }

  useEffect(() => {
    setSavedGoals(null);
    setSavedGoalsError(null);
    loadSavedGoals();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [portfolioId]);

  async function runCalculation(
    amount: number,
    date: string,
    monthly: number | undefined,
    universe: string | undefined,
  ) {
    if (portfolioId === null) return;
    setLoading(true);
    setError(null);
    try {
      const res = await getGoalPlan(amount, date, monthly, portfolioId, universe);
      setPlan(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not build a plan.");
      setPlan(null);
    } finally {
      setLoading(false);
    }
  }

  async function handleCalculate() {
    const amount = Number(targetAmount);
    if (!Number.isFinite(amount) || amount <= 0) {
      setError("Enter a target amount greater than 0.");
      return;
    }
    if (!targetDate) {
      setError("Pick a target date.");
      return;
    }
    const monthly = useOwnAmount ? Number(monthlyAmount) : undefined;
    if (useOwnAmount && (!Number.isFinite(monthly!) || monthly! < 0)) {
      setError("Enter a monthly amount of 0 or more.");
      return;
    }
    await runCalculation(amount, targetDate, monthly, compareToBest ? compareUniverse : undefined);
  }

  async function handleSave() {
    if (portfolioId === null || !plan) return;
    setSaving(true);
    setSaveMessage(null);
    try {
      const monthly = useOwnAmount ? Number(monthlyAmount) : undefined;
      await saveGoal(
        goalName.trim() || "Goal",
        Number(targetAmount),
        targetDate,
        monthly,
        compareToBest ? compareUniverse : undefined,
        portfolioId,
      );
      setSaveMessage("Saved.");
      setGoalName("");
      await loadSavedGoals();
    } catch (err) {
      setSaveMessage(err instanceof ApiError ? err.message : "Could not save this goal.");
    } finally {
      setSaving(false);
    }
  }

  function handleLoadSavedGoal(g: SavedGoal) {
    setTargetAmount(String(g.target_amount));
    setTargetDate(g.target_date);
    setUseOwnAmount(g.monthly_amount !== null);
    setMonthlyAmount(g.monthly_amount !== null ? String(g.monthly_amount) : monthlyAmount);
    setCompareToBest(g.compare_universe !== null);
    if (g.compare_universe !== null) setCompareUniverse(g.compare_universe);
    runCalculation(
      g.target_amount,
      g.target_date,
      g.monthly_amount ?? undefined,
      g.compare_universe ?? undefined,
    );
  }

  async function handleDeleteSavedGoal(id: number) {
    setDeletingGoalId(id);
    try {
      await deleteSavedGoal(id);
      await loadSavedGoals();
    } catch (err) {
      setSavedGoalsError(err instanceof ApiError ? err.message : "Could not delete this goal.");
    } finally {
      setDeletingGoalId(null);
    }
  }

  if (portfolioId === null) return null;

  return (
    <div className="mt-6 rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="text-sm font-semibold text-slate-900">Goal-Based Investing Plan</h2>
      <p className="mt-1 text-xs text-slate-500">
        Set a target amount and date — this figures out how much to invest each month, and what
        percentage of it should go to each of this portfolio&apos;s holdings, tilted toward
        whichever currently has the strongest signal.
      </p>

      <div className="mt-3 flex flex-wrap items-end gap-3 text-xs">
        <label className="flex flex-col gap-1">
          <span className="font-medium text-slate-600">Target amount</span>
          <input
            type="number"
            min={1}
            value={targetAmount}
            onChange={(e) => setTargetAmount(e.target.value)}
            className="w-32 rounded-md border border-slate-300 px-2 py-1"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="font-medium text-slate-600">Target date</span>
          <input
            type="date"
            value={targetDate}
            onChange={(e) => setTargetDate(e.target.value)}
            className="rounded-md border border-slate-300 px-2 py-1"
          />
        </label>
        <label className="flex items-center gap-1.5 pb-1.5">
          <input
            type="checkbox"
            checked={useOwnAmount}
            onChange={(e) => setUseOwnAmount(e.target.checked)}
            className="h-3.5 w-3.5 rounded border-slate-300"
          />
          <span className="text-slate-600">I already invest a fixed amount</span>
        </label>
        {useOwnAmount && (
          <label className="flex flex-col gap-1">
            <span className="font-medium text-slate-600">Monthly amount</span>
            <input
              type="number"
              min={0}
              value={monthlyAmount}
              onChange={(e) => setMonthlyAmount(e.target.value)}
              className="w-28 rounded-md border border-slate-300 px-2 py-1"
            />
          </label>
        )}
        <button
          onClick={handleCalculate}
          disabled={loading}
          className="rounded-md bg-slate-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-700 disabled:opacity-50"
        >
          {loading ? "Calculating…" : "Calculate Plan"}
        </button>
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-3 text-xs">
        <label className="flex items-center gap-1.5">
          <input
            type="checkbox"
            checked={compareToBest}
            onChange={(e) => setCompareToBest(e.target.checked)}
            className="h-3.5 w-3.5 rounded border-slate-300"
          />
          <span className="text-slate-600">Compare to buying the best-ranked stock instead</span>
        </label>
        {compareToBest && (
          <select
            value={compareUniverse}
            onChange={(e) => setCompareUniverse(e.target.value)}
            className="rounded-md border border-slate-300 px-2 py-1 text-xs"
          >
            {universes.map((u) => (
              <option key={u} value={u}>
                {u}
              </option>
            ))}
          </select>
        )}
      </div>

      {error && <p className="mt-2 text-xs text-red-600">{error}</p>}

      {plan && (
        <div className="mt-4 border-t border-slate-100 pt-4">
          {plan.warnings.map((w, i) => (
            <p key={i} className="mb-2 rounded-md bg-amber-50 px-2.5 py-1.5 text-xs text-amber-800">
              {w}
            </p>
          ))}

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div>
              <div className="text-[11px] uppercase tracking-wide text-slate-400">Current Value</div>
              <div className="text-sm font-semibold text-slate-800">{fmtMoney(plan.current_value)}</div>
            </div>
            <div>
              <div className="text-[11px] uppercase tracking-wide text-slate-400">Months to Go</div>
              <div className="text-sm font-semibold text-slate-800">{plan.months_remaining}</div>
            </div>
            <div>
              <div className="text-[11px] uppercase tracking-wide text-slate-400">
                Current Holdings, Trailing Annualized
              </div>
              <div className="text-sm font-semibold text-slate-800">
                {plan.current_holdings_annualized_return_pct !== null
                  ? `${plan.current_holdings_annualized_return_pct >= 0 ? "+" : ""}${plan.current_holdings_annualized_return_pct.toFixed(1)}%`
                  : "—"}
              </div>
            </div>
            <div>
              <div className="text-[11px] uppercase tracking-wide text-slate-400">
                {plan.required_monthly_contribution === 0 ? "On Track — Needed" : "Required Monthly"}
              </div>
              <div
                className={`text-sm font-semibold ${
                  plan.required_monthly_contribution === 0 ? "text-emerald-600" : "text-slate-800"
                }`}
              >
                {plan.required_monthly_contribution !== null
                  ? fmtMoney(plan.required_monthly_contribution)
                  : "—"}
              </div>
            </div>
          </div>

          {plan.projected_value_with_given_contribution !== undefined && plan.gap_vs_target !== undefined && (
            <p className="mt-3 text-xs text-slate-600">
              At {fmtMoney(Number(monthlyAmount))}/month, you&apos;re projected to have{" "}
              <span className="font-semibold text-slate-800">
                {fmtMoney(plan.projected_value_with_given_contribution)}
              </span>{" "}
              by {plan.target_date} —{" "}
              <span className={`font-semibold ${plan.gap_vs_target >= 0 ? "text-emerald-600" : "text-red-600"}`}>
                {plan.gap_vs_target >= 0 ? "a surplus of " : "a shortfall of "}
                {fmtMoney(Math.abs(plan.gap_vs_target))}
              </span>{" "}
              vs. your {fmtMoney(plan.target_amount)} target.
            </p>
          )}

          {plan.best_stock_comparison && (
            <div className="mt-4 rounded-lg border border-indigo-100 bg-indigo-50/40 p-3">
              <div className="text-xs font-semibold text-slate-800">
                What if you bought {plan.best_stock_comparison.ticker} ({plan.best_stock_comparison.name})
                instead?
              </div>
              <p className="mt-1 text-xs text-slate-600">
                Top-ranked pick in {plan.best_stock_comparison.universe} right now, trailing{" "}
                <span className="font-medium text-slate-800">
                  {plan.best_stock_comparison.annualized_return_pct >= 0 ? "+" : ""}
                  {plan.best_stock_comparison.annualized_return_pct.toFixed(1)}%
                </span>{" "}
                annualized — vs. your portfolio&apos;s{" "}
                {plan.current_holdings_annualized_return_pct !== null
                  ? `${plan.current_holdings_annualized_return_pct >= 0 ? "+" : ""}${plan.current_holdings_annualized_return_pct.toFixed(1)}%`
                  : "—"}
                .
              </p>
              <div className="mt-2 grid grid-cols-2 gap-3 sm:grid-cols-3">
                <div>
                  <div className="text-[11px] uppercase tracking-wide text-slate-400">Required Monthly</div>
                  <div className="text-sm font-semibold text-slate-800">
                    {plan.best_stock_comparison.required_monthly_contribution !== null
                      ? fmtMoney(plan.best_stock_comparison.required_monthly_contribution)
                      : "—"}
                  </div>
                </div>
                {plan.required_monthly_contribution !== null &&
                  plan.best_stock_comparison.required_monthly_contribution !== null && (
                    <div>
                      <div className="text-[11px] uppercase tracking-wide text-slate-400">Vs. Your Plan</div>
                      <div
                        className={`text-sm font-semibold ${
                          plan.best_stock_comparison.required_monthly_contribution <=
                          plan.required_monthly_contribution
                            ? "text-emerald-600"
                            : "text-red-600"
                        }`}
                      >
                        {fmtMoney(
                          Math.abs(
                            plan.best_stock_comparison.required_monthly_contribution -
                              plan.required_monthly_contribution,
                          ),
                        )}{" "}
                        {plan.best_stock_comparison.required_monthly_contribution <=
                        plan.required_monthly_contribution
                          ? "less/month"
                          : "more/month"}
                      </div>
                    </div>
                  )}
                {plan.best_stock_comparison.gap_vs_target !== undefined && (
                  <div>
                    <div className="text-[11px] uppercase tracking-wide text-slate-400">
                      Gap at Your Monthly Amount
                    </div>
                    <div
                      className={`text-sm font-semibold ${
                        plan.best_stock_comparison.gap_vs_target >= 0 ? "text-emerald-600" : "text-red-600"
                      }`}
                    >
                      {plan.best_stock_comparison.gap_vs_target >= 0 ? "+" : ""}
                      {fmtMoney(plan.best_stock_comparison.gap_vs_target)}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          <div className="mt-4 flex flex-wrap items-end gap-2 border-t border-slate-100 pt-3">
            <label className="flex flex-col gap-1 text-xs">
              <span className="font-medium text-slate-600">Save this goal as</span>
              <input
                type="text"
                value={goalName}
                onChange={(e) => setGoalName(e.target.value)}
                placeholder="e.g. Retirement by 2035"
                className="w-56 rounded-md border border-slate-300 px-2 py-1"
              />
            </label>
            <button
              onClick={handleSave}
              disabled={saving}
              className="rounded-md border border-slate-300 px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
            >
              {saving ? "Saving…" : "Save Goal"}
            </button>
            {saveMessage && <span className="text-xs text-slate-500">{saveMessage}</span>}
          </div>

          <div className="mt-4 overflow-x-auto rounded-lg border border-slate-200">
            <table className="min-w-full text-xs">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-left uppercase tracking-wide text-slate-400">
                  <th className="px-3 py-2 font-medium">Ticker</th>
                  <th className="px-3 py-2 font-medium">Signal</th>
                  <th className="px-3 py-2 text-right font-medium">Trailing Annualized</th>
                  <th className="px-3 py-2 text-right font-medium">Allocation</th>
                  <th className="px-3 py-2 text-right font-medium">Monthly Amount</th>
                </tr>
              </thead>
              <tbody>
                {plan.allocation
                  .slice()
                  .sort((a, b) => b.weight_pct - a.weight_pct)
                  .map((a) => (
                    <tr key={a.ticker} className="border-t border-slate-100">
                      <td className="px-3 py-1.5 font-medium text-slate-800">{a.ticker}</td>
                      <td className="px-3 py-1.5">
                        {a.signal ? (
                          <span
                            className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${
                              a.signal === "BUY"
                                ? "bg-emerald-50 text-emerald-700"
                                : a.signal === "SELL"
                                ? "bg-red-50 text-red-700"
                                : "bg-slate-100 text-slate-600"
                            }`}
                          >
                            {a.signal}
                          </span>
                        ) : (
                          <span className="text-slate-400">—</span>
                        )}
                      </td>
                      <td className="px-3 py-1.5 text-right text-slate-600">
                        {a.annualized_return_pct !== null ? `${a.annualized_return_pct.toFixed(1)}%` : "—"}
                      </td>
                      <td className="px-3 py-1.5 text-right font-medium text-slate-700">
                        {a.weight_pct.toFixed(1)}%
                      </td>
                      <td className="px-3 py-1.5 text-right text-slate-700">{fmtMoney(a.monthly_amount)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {savedGoals !== null && savedGoals.length > 0 && (
        <div className="mt-4 border-t border-slate-100 pt-3">
          <div className="text-xs font-semibold text-slate-700">Saved Goals</div>
          {savedGoalsError && <p className="mt-1 text-xs text-red-600">{savedGoalsError}</p>}
          <ul className="mt-2 flex flex-col gap-1.5">
            {savedGoals.map((g) => (
              <li
                key={g.id}
                className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-slate-200 px-2.5 py-1.5 text-xs"
              >
                <span className="text-slate-700">
                  <span className="font-medium">{g.name}</span> — {fmtMoney(g.target_amount)} by{" "}
                  {g.target_date}
                  {g.monthly_amount !== null && (
                    <span className="text-slate-400"> · {fmtMoney(g.monthly_amount)}/mo</span>
                  )}
                  {g.compare_universe && (
                    <span className="text-slate-400"> · vs. best in {g.compare_universe}</span>
                  )}
                </span>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleLoadSavedGoal(g)}
                    className="rounded-md border border-slate-300 px-2 py-0.5 font-medium text-slate-600 hover:bg-slate-100"
                  >
                    Load
                  </button>
                  <button
                    onClick={() => handleDeleteSavedGoal(g.id)}
                    disabled={deletingGoalId === g.id}
                    className="rounded-md border border-red-200 px-2 py-0.5 font-medium text-red-700 hover:bg-red-50 disabled:opacity-50"
                  >
                    {deletingGoalId === g.id ? "Deleting…" : "Delete"}
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
