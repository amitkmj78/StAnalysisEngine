"use client";

import { useEffect, useState } from "react";

import {
  ApiError,
  getPortfolioHealthConcentration,
  getPortfolioHealthIncomeFees,
  getPortfolioHealthOverlap,
  getPortfolioHealthRisk,
} from "@/lib/api";
import type {
  PortfolioHealthConcentrationResponse,
  PortfolioHealthIncomeFeesResponse,
  PortfolioHealthOverlapResponse,
  PortfolioHealthRiskResponse,
  PortfolioRiskWindow,
} from "@/lib/types";
import PortfolioSwitcher from "@/components/PortfolioSwitcher";
import { useUrlState } from "@/lib/useUrlState";

function SkeletonBlock({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded-md bg-slate-200 ${className}`} />;
}

function Card({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5">
      <h2 className="text-sm font-semibold text-slate-900">{title}</h2>
      {subtitle && <p className="mt-1 text-xs text-slate-500">{subtitle}</p>}
      <div className="mt-3">{children}</div>
    </div>
  );
}

function fmtPct(v: number | null, decimals = 1): string {
  return v === null || v === undefined ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(decimals)}%`;
}

export default function PortfolioHealthPage() {
  const [{ p: portfolioParam }, setUrlState] = useUrlState({ p: "" });
  const [selectedPortfolioId, setSelectedPortfolioId] = useState<number | null>(
    portfolioParam ? Number(portfolioParam) : null,
  );

  const [concentration, setConcentration] = useState<PortfolioHealthConcentrationResponse | null>(null);
  const [concentrationLoading, setConcentrationLoading] = useState(true);
  const [concentrationError, setConcentrationError] = useState<string | null>(null);

  const [risk, setRisk] = useState<PortfolioHealthRiskResponse | null>(null);
  const [riskLoading, setRiskLoading] = useState(true);
  const [riskError, setRiskError] = useState<string | null>(null);

  const [overlap, setOverlap] = useState<PortfolioHealthOverlapResponse | null>(null);
  const [overlapLoading, setOverlapLoading] = useState(true);
  const [overlapError, setOverlapError] = useState<string | null>(null);

  const [incomeFees, setIncomeFees] = useState<PortfolioHealthIncomeFeesResponse | null>(null);
  const [incomeFeesLoading, setIncomeFeesLoading] = useState(true);
  const [incomeFeesError, setIncomeFeesError] = useState<string | null>(null);

  function handlePortfolioChange(id: number) {
    setSelectedPortfolioId(id);
    setUrlState({ p: String(id) });
  }

  useEffect(() => {
    if (selectedPortfolioId === null) return;

    setConcentrationLoading(true);
    setConcentrationError(null);
    getPortfolioHealthConcentration(selectedPortfolioId)
      .then(setConcentration)
      .catch((err) => setConcentrationError(err instanceof ApiError ? err.message : "Couldn't load concentration."))
      .finally(() => setConcentrationLoading(false));

    setRiskLoading(true);
    setRiskError(null);
    getPortfolioHealthRisk(selectedPortfolioId)
      .then(setRisk)
      .catch((err) => setRiskError(err instanceof ApiError ? err.message : "Couldn't load risk."))
      .finally(() => setRiskLoading(false));

    setOverlapLoading(true);
    setOverlapError(null);
    getPortfolioHealthOverlap(selectedPortfolioId)
      .then(setOverlap)
      .catch((err) => setOverlapError(err instanceof ApiError ? err.message : "Couldn't load fund overlap."))
      .finally(() => setOverlapLoading(false));

    setIncomeFeesLoading(true);
    setIncomeFeesError(null);
    getPortfolioHealthIncomeFees(selectedPortfolioId)
      .then(setIncomeFees)
      .catch((err) => setIncomeFeesError(err instanceof ApiError ? err.message : "Couldn't load income & fees."))
      .finally(() => setIncomeFeesLoading(false));
  }, [selectedPortfolioId]);

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">Portfolio Health Check</h1>
          <p className="mt-1 text-sm text-slate-500">
            Concentration, fund overlap, risk, income, fees, and tax-loss harvesting for your real holdings — each
            section loads independently, so a slow one never blocks the rest.
          </p>
        </div>
      </div>

      <div className="mt-6">
        <PortfolioSwitcher
          selectedPortfolioId={selectedPortfolioId}
          onChange={handlePortfolioChange}
          initialPreferId={portfolioParam ? Number(portfolioParam) : undefined}
        />
      </div>

      <div className="mt-6 flex flex-col gap-6">
        <Card title="Concentration & Sector Weights" subtitle="Largest positions, and your sector mix vs. the S&P 500.">
          {concentrationLoading && (
            <div className="flex flex-col gap-2">
              <SkeletonBlock className="h-20" />
              <SkeletonBlock className="h-32" />
            </div>
          )}
          {concentrationError && <p className="text-sm text-red-700">{concentrationError}</p>}
          {concentration && !concentrationLoading && (
            <div className="flex flex-col gap-6">
              {concentration.largest_positions.length === 0 ? (
                <p className="text-sm text-slate-500">No positions yet.</p>
              ) : (
                <div>
                  <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Largest positions</p>
                  <div className="mt-2 overflow-x-auto">
                    <table className="min-w-full text-sm">
                      <thead>
                        <tr className="border-b border-slate-200 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                          <th className="px-2 py-1.5">Ticker</th>
                          <th className="px-2 py-1.5">Weight</th>
                          <th className="px-2 py-1.5">Concentrated</th>
                        </tr>
                      </thead>
                      <tbody>
                        {concentration.largest_positions.map((p) => (
                          <tr key={p.ticker} className="border-b border-slate-100 last:border-0">
                            <td className="px-2 py-1.5 font-medium text-slate-900">{p.ticker}</td>
                            <td className="px-2 py-1.5 text-slate-700">{p.weight_pct.toFixed(1)}%</td>
                            <td className="px-2 py-1.5">
                              {p.concentrated ? (
                                <span className="rounded-full bg-amber-50 px-2 py-0.5 text-xs font-medium text-amber-700">
                                  25%+ of portfolio
                                </span>
                              ) : (
                                <span className="text-slate-400">—</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              <div>
                <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
                  Sector weight vs. S&amp;P 500
                </p>
                <p className="mt-0.5 text-xs text-slate-400">
                  S&amp;P 500 side is an approximation (market-cap share across the index, not real fund weights).
                </p>
                {concentration.sector_comparison.length === 0 ? (
                  <p className="mt-2 text-sm text-slate-500">No sector data available.</p>
                ) : (
                  <div className="mt-2 overflow-x-auto">
                    <table className="min-w-full text-sm">
                      <thead>
                        <tr className="border-b border-slate-200 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                          <th className="px-2 py-1.5">Sector</th>
                          <th className="px-2 py-1.5">Your Portfolio</th>
                          <th className="px-2 py-1.5">S&amp;P 500</th>
                          <th className="px-2 py-1.5">Gap</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[...concentration.sector_comparison]
                          .sort((a, b) => b.portfolio_weight_pct - a.portfolio_weight_pct)
                          .map((row) => (
                            <tr key={row.sector} className="border-b border-slate-100 last:border-0">
                              <td className="px-2 py-1.5 text-slate-900">{row.sector}</td>
                              <td className="px-2 py-1.5 text-slate-700">{row.portfolio_weight_pct.toFixed(1)}%</td>
                              <td className="px-2 py-1.5 text-slate-700">{row.sp500_weight_pct.toFixed(1)}%</td>
                              <td
                                className={`px-2 py-1.5 font-medium ${
                                  row.gap_pct >= 0 ? "text-emerald-700" : "text-red-700"
                                }`}
                              >
                                {fmtPct(row.gap_pct)}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          )}
        </Card>

        <Card title="Risk" subtitle="Volatility, beta, correlation to the S&P 500, and max drawdown, using daily data.">
          {riskLoading && (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <SkeletonBlock className="h-32" />
              <SkeletonBlock className="h-32" />
            </div>
          )}
          {riskError && <p className="text-sm text-red-700">{riskError}</p>}
          {risk && !riskLoading && (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <RiskWindowTile label="1 Year" window={risk.windows["1Y"]} />
              <RiskWindowTile label="3 Years" window={risk.windows["3Y"]} />
            </div>
          )}
        </Card>

        <Card
          title="Fund Overlap (ETF Look-Through)"
          subtitle="A stock held directly and through a fund, combined into one exposure figure."
        >
          {overlapLoading && (
            <div className="flex flex-col gap-2">
              <SkeletonBlock className="h-8 w-2/3" />
              <SkeletonBlock className="h-40" />
            </div>
          )}
          {overlapError && <p className="text-sm text-red-700">{overlapError}</p>}
          {overlap && !overlapLoading && (
            <div className="flex flex-col gap-4">
              <p className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">{overlap.disclosure}</p>

              {Object.keys(overlap.fund_coverage_pct).length > 0 && (
                <div className="flex flex-wrap gap-2 text-xs text-slate-500">
                  {Object.entries(overlap.fund_coverage_pct).map(([ticker, pct]) => (
                    <span key={ticker} className="rounded-full border border-slate-200 px-2 py-0.5">
                      {ticker}: top 10 = {pct.toFixed(1)}% of fund
                    </span>
                  ))}
                </div>
              )}

              {overlap.combined_exposure.length === 0 ? (
                <p className="text-sm text-slate-500">No positions yet.</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full text-sm">
                    <thead>
                      <tr className="border-b border-slate-200 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                        <th className="px-2 py-1.5">Ticker</th>
                        <th className="px-2 py-1.5">Direct</th>
                        <th className="px-2 py-1.5">Via Funds</th>
                        <th className="px-2 py-1.5">Combined</th>
                        <th className="px-2 py-1.5">Weight</th>
                      </tr>
                    </thead>
                    <tbody>
                      {overlap.combined_exposure.slice(0, 20).map((row) => (
                        <tr key={row.ticker} className="border-b border-slate-100 last:border-0">
                          <td className="px-2 py-1.5 font-medium text-slate-900">{row.ticker}</td>
                          <td className="px-2 py-1.5 text-slate-700">${row.direct_value.toLocaleString()}</td>
                          <td className="px-2 py-1.5 text-slate-700">
                            {row.look_through_value > 0 ? (
                              <span title={row.via_funds.map((f) => `${f.fund_ticker}: $${f.dollars}`).join(", ")}>
                                ${row.look_through_value.toLocaleString()}
                              </span>
                            ) : (
                              "—"
                            )}
                          </td>
                          <td className="px-2 py-1.5 font-medium text-slate-900">
                            ${row.combined_value.toLocaleString()}
                          </td>
                          <td className="px-2 py-1.5 text-slate-700">{row.combined_weight_pct.toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {overlap.sector_comparison.length > 0 && (
                <div>
                  <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
                    Sector weight vs. S&amp;P 500 (look-through-adjusted)
                  </p>
                  <div className="mt-2 overflow-x-auto">
                    <table className="min-w-full text-sm">
                      <thead>
                        <tr className="border-b border-slate-200 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                          <th className="px-2 py-1.5">Sector</th>
                          <th className="px-2 py-1.5">Your Portfolio</th>
                          <th className="px-2 py-1.5">S&amp;P 500</th>
                          <th className="px-2 py-1.5">Gap</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[...overlap.sector_comparison]
                          .sort((a, b) => b.portfolio_weight_pct - a.portfolio_weight_pct)
                          .map((row) => (
                            <tr key={row.sector} className="border-b border-slate-100 last:border-0">
                              <td className="px-2 py-1.5 text-slate-900">{row.sector}</td>
                              <td className="px-2 py-1.5 text-slate-700">{row.portfolio_weight_pct.toFixed(1)}%</td>
                              <td className="px-2 py-1.5 text-slate-700">{row.sp500_weight_pct.toFixed(1)}%</td>
                              <td
                                className={`px-2 py-1.5 font-medium ${
                                  row.gap_pct >= 0 ? "text-emerald-700" : "text-red-700"
                                }`}
                              >
                                {fmtPct(row.gap_pct)}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </Card>

        <Card title="Income & Fees" subtitle="Dividend income (trailing and projected) and annual fund fee drag, in dollars.">
          {incomeFeesLoading && (
            <div className="flex flex-col gap-2">
              <SkeletonBlock className="h-16" />
              <SkeletonBlock className="h-16" />
            </div>
          )}
          {incomeFeesError && <p className="text-sm text-red-700">{incomeFeesError}</p>}
          {incomeFees && !incomeFeesLoading && (
            <div className="flex flex-col gap-6">
              <div>
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                    <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Trailing (last 12mo)</p>
                    <p className="mt-1 text-xl font-semibold text-slate-900">
                      {incomeFees.dividends.total_trailing_income !== null
                        ? `$${incomeFees.dividends.total_trailing_income.toLocaleString()}`
                        : "—"}
                    </p>
                  </div>
                  <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                    <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Projected (annual)</p>
                    <p className="mt-1 text-xl font-semibold text-slate-900">
                      {incomeFees.dividends.total_projected_income !== null
                        ? `$${incomeFees.dividends.total_projected_income.toLocaleString()}`
                        : "—"}
                    </p>
                  </div>
                </div>
                {incomeFees.dividends.by_ticker.filter((r) => r.trailing_income !== null || r.projected_income !== null)
                  .length > 0 && (
                  <div className="mt-3 overflow-x-auto">
                    <table className="min-w-full text-sm">
                      <thead>
                        <tr className="border-b border-slate-200 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                          <th className="px-2 py-1.5">Ticker</th>
                          <th className="px-2 py-1.5">Trailing</th>
                          <th className="px-2 py-1.5">Projected</th>
                        </tr>
                      </thead>
                      <tbody>
                        {incomeFees.dividends.by_ticker
                          .filter((r) => r.trailing_income !== null || r.projected_income !== null)
                          .map((r) => (
                            <tr key={r.ticker} className="border-b border-slate-100 last:border-0">
                              <td className="px-2 py-1.5 font-medium text-slate-900">{r.ticker}</td>
                              <td className="px-2 py-1.5 text-slate-700">
                                {r.trailing_income !== null ? `$${r.trailing_income.toLocaleString()}` : "—"}
                              </td>
                              <td className="px-2 py-1.5 text-slate-700">
                                {r.projected_income !== null ? `$${r.projected_income.toLocaleString()}` : "—"}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              <div>
                <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
                  Fund fee drag — total $
                  {incomeFees.fee_drag.total_annual_fee_drag_dollars !== null
                    ? incomeFees.fee_drag.total_annual_fee_drag_dollars.toLocaleString()
                    : "—"}
                  /year
                </p>
                {incomeFees.fee_drag.by_fund.length === 0 ? (
                  <p className="mt-2 text-sm text-slate-500">No fund holdings.</p>
                ) : (
                  <div className="mt-2 overflow-x-auto">
                    <table className="min-w-full text-sm">
                      <thead>
                        <tr className="border-b border-slate-200 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                          <th className="px-2 py-1.5">Fund</th>
                          <th className="px-2 py-1.5">Value</th>
                          <th className="px-2 py-1.5">Expense Ratio</th>
                          <th className="px-2 py-1.5">Annual Drag</th>
                        </tr>
                      </thead>
                      <tbody>
                        {incomeFees.fee_drag.by_fund.map((r) => (
                          <tr key={r.ticker} className="border-b border-slate-100 last:border-0">
                            <td className="px-2 py-1.5 font-medium text-slate-900">{r.ticker}</td>
                            <td className="px-2 py-1.5 text-slate-700">${r.market_value.toLocaleString()}</td>
                            <td className="px-2 py-1.5 text-slate-700">
                              {r.expense_ratio_pct !== null ? `${r.expense_ratio_pct.toFixed(2)}%` : "—"}
                            </td>
                            <td className="px-2 py-1.5 text-slate-700">
                              {r.annual_fee_drag_dollars !== null ? `$${r.annual_fee_drag_dollars.toLocaleString()}` : "—"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}

function RiskWindowTile({ label, window }: { label: string; window: PortfolioRiskWindow }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-slate-700">{label}</span>
        {window.data_start && window.data_end && (
          <span className="text-xs text-slate-400">
            {window.data_start} – {window.data_end}
          </span>
        )}
      </div>
      <dl className="mt-3 grid grid-cols-2 gap-y-2 text-sm">
        <dt className="text-slate-500">Volatility</dt>
        <dd className="text-right text-slate-900">
          {window.volatility_pct !== null ? `${window.volatility_pct.toFixed(1)}%` : "—"}
        </dd>
        <dt className="text-slate-500">Beta (vs. SPY)</dt>
        <dd className="text-right text-slate-900">{window.beta_to_spy !== null ? window.beta_to_spy.toFixed(2) : "—"}</dd>
        <dt className="text-slate-500">Correlation (vs. SPY)</dt>
        <dd className="text-right text-slate-900">
          {window.correlation_to_spy !== null ? window.correlation_to_spy.toFixed(2) : "—"}
        </dd>
        <dt className="text-slate-500">Max drawdown</dt>
        <dd className="text-right text-slate-900">
          {window.max_drawdown_pct !== null ? `${window.max_drawdown_pct.toFixed(1)}%` : "—"}
        </dd>
      </dl>
      {window.excluded_from_risk.length > 0 && (
        <p className="mt-2 text-xs text-slate-400">
          Excluded (no price history for this window): {window.excluded_from_risk.join(", ")}
        </p>
      )}
    </div>
  );
}
