"use client";

import { useState } from "react";

import { ApiError, getGoalPlan } from "@/lib/api";
import type { GoalPlanResponse } from "@/lib/types";

function fmtMoney(n: number) {
  return `$${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

export default function GoalPlan({ portfolioId }: { portfolioId: number | null }) {
  const [targetAmount, setTargetAmount] = useState("100000");
  const [targetDate, setTargetDate] = useState("");
  const [useOwnAmount, setUseOwnAmount] = useState(false);
  const [monthlyAmount, setMonthlyAmount] = useState("500");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [plan, setPlan] = useState<GoalPlanResponse | null>(null);

  async function handleCalculate() {
    if (portfolioId === null) return;
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
    setLoading(true);
    setError(null);
    try {
      const res = await getGoalPlan(amount, targetDate, monthly, portfolioId);
      setPlan(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not build a plan.");
      setPlan(null);
    } finally {
      setLoading(false);
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
    </div>
  );
}
