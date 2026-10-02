"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

import {
  ApiError,
  previewDiversifiedBasket,
  saveDiversifiedBasket,
  getStockUniverseDetails,
} from "@/lib/api";
import type {
  BasketHolding,
  DiversifiedBasketPreview,
  RebalanceFrequency,
  SectorWeighting,
  UniverseDetail,
} from "@/lib/types";

const GOALS = ["Short Term", "Long Term"];
const SP500_UNIVERSE_NAME = "US - S&P 500";

// Same localStorage key PortfolioSwitcher.tsx reads on mount -- setting
// it before navigating is how a freshly-saved basket's own portfolio
// becomes the selected one on the Portfolio page (DI-07's "lands on the
// new portfolio's page"), without adding new URL-param plumbing.
const SELECTED_PORTFOLIO_STORAGE_KEY = "stanalysisengine.selectedPortfolioId";

function fmtMoney(v: number | null | undefined) {
  if (v === null || v === undefined) return "N/A";
  return `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

function fmtPct(v: number | null | undefined, digits = 1) {
  if (v === null || v === undefined) return "N/A";
  return `${v.toFixed(digits)}%`;
}

// Pre-generation concentration hint (DI-09), from the selected universe's
// raw ticker-per-sector counts -- an approximation of what the actual
// generated basket will look like (picks-per-sector selects top-N per
// sector, so a universe skewed by ticker count skews the basket too).
// The authoritative check runs server-side on the real basket after
// generation (preview.concentration_warning below).
function universeConcentrationWarning(sectorCounts: Record<string, number>): string | null {
  const entries = Object.entries(sectorCounts);
  if (entries.length === 0) return null;
  if (entries.length < 5) {
    return `This universe only spans ${entries.length} sector(s); the basket will be less diversified.`;
  }
  const total = entries.reduce((sum, [, c]) => sum + c, 0);
  if (!total) return null;
  const [topSector, topCount] = entries.reduce((a, b) => (b[1] > a[1] ? b : a));
  if ((topCount / total) * 100 > 40) {
    return `This universe is concentrated in ${topSector}; the basket will be less diversified.`;
  }
  return null;
}

export default function BuildDiversifiedBasketPage() {
  const router = useRouter();

  const [universeDetails, setUniverseDetails] = useState<UniverseDetail[]>([]);
  const [universeDetailsError, setUniverseDetailsError] = useState<string | null>(null);

  const [goal, setGoal] = useState("Long Term");
  const [universe, setUniverse] = useState(SP500_UNIVERSE_NAME);
  const [picksPerSector, setPicksPerSector] = useState(2);
  const [maxStocks, setMaxStocks] = useState("");
  const [totalAmount, setTotalAmount] = useState("10000");
  const [fractionalShares, setFractionalShares] = useState(false);
  const [sectorWeighting, setSectorWeighting] = useState<SectorWeighting>("equal_dollar");

  const [generating, setGenerating] = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);
  const [preview, setPreview] = useState<DiversifiedBasketPreview | null>(null);
  const [excludedTickers, setExcludedTickers] = useState<string[]>([]);
  const [showExclusions, setShowExclusions] = useState(false);

  const [portfolioName, setPortfolioName] = useState("");
  const [portfolioNameTouched, setPortfolioNameTouched] = useState(false);
  const [rebalanceFrequency, setRebalanceFrequency] = useState<RebalanceFrequency>("none");
  const [driftThresholdPct, setDriftThresholdPct] = useState(5);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    getStockUniverseDetails()
      .then((res) => setUniverseDetails(res.universes))
      .catch((err) => setUniverseDetailsError(err instanceof ApiError ? err.message : "Could not load universe details."));
  }, []);

  const selectedUniverseDetail = universeDetails.find((u) => u.key === universe) ?? null;
  const preflightWarning = selectedUniverseDetail ? universeConcentrationWarning(selectedUniverseDetail.sector_counts) : null;

  const defaultPortfolioName = useMemo(
    () => `Diversified – ${universe} – ${goal} – ${new Date().toLocaleDateString()}`,
    [universe, goal],
  );

  async function runGenerate(e: React.FormEvent, excluded: string[] = []) {
    e.preventDefault();
    setGenerating(true);
    setGenerateError(null);
    try {
      const amount = Number(totalAmount);
      const cap = Number(maxStocks);
      const result = await previewDiversifiedBasket({
        goal,
        universe,
        picks_per_sector: picksPerSector,
        max_stocks: cap > 0 ? cap : null,
        total_amount: amount,
        fractional_shares: fractionalShares,
        sector_weighting: sectorWeighting,
        excluded_tickers: excluded,
      });
      setPreview(result);
      setExcludedTickers(excluded);
    } catch (err) {
      setGenerateError(err instanceof ApiError ? err.message : "Could not generate a basket for this universe.");
      setPreview(null);
    } finally {
      setGenerating(false);
    }
  }

  async function handleRemove(ticker: string) {
    const next = [...excludedTickers, ticker];
    // DI-06: the replacement must come from the server's own algorithm
    // (same eligibility + sub-industry cap + round-robin trim), not a
    // client-side reimplementation that could drift from it and break
    // "regenerate is deterministic."
    setGenerating(true);
    setGenerateError(null);
    try {
      const amount = Number(totalAmount);
      const cap = Number(maxStocks);
      const result = await previewDiversifiedBasket({
        goal,
        universe,
        picks_per_sector: picksPerSector,
        max_stocks: cap > 0 ? cap : null,
        total_amount: amount,
        fractional_shares: fractionalShares,
        sector_weighting: sectorWeighting,
        excluded_tickers: next,
      });
      setPreview(result);
      setExcludedTickers(next);
    } catch (err) {
      setGenerateError(err instanceof ApiError ? err.message : "Could not remove that ticker.");
    } finally {
      setGenerating(false);
    }
  }

  async function handleSave() {
    if (!preview) return;
    const name = (portfolioNameTouched ? portfolioName : defaultPortfolioName).trim();
    if (!name) {
      setSaveError("Name this portfolio first.");
      return;
    }
    setSaving(true);
    setSaveError(null);
    try {
      const cap = Number(maxStocks);
      const res = await saveDiversifiedBasket({
        name,
        goal,
        universe,
        picks_per_sector: picksPerSector,
        max_stocks: cap > 0 ? cap : null,
        total_amount: Number(totalAmount),
        fractional_shares: fractionalShares,
        sector_weighting: sectorWeighting,
        as_of_date: preview.as_of_date || null,
        holdings: preview.holdings,
        rebalance_frequency: rebalanceFrequency,
        drift_threshold_pct: driftThresholdPct,
      });
      localStorage.setItem(SELECTED_PORTFOLIO_STORAGE_KEY, String(res.id));
      router.push("/portfolio");
    } catch (err) {
      setSaveError(err instanceof ApiError ? err.message : "Could not save this basket as a portfolio.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">Build a Diversified Basket</h1>
          <p className="mt-1 text-sm text-slate-500">
            Generates a custom basket of individual stocks spread across sectors — the top-scoring, sector-
            diversified tickers from the universe you pick, then saves it as a new portfolio. A saved basket
            with fixed holdings isn&apos;t an index fund; it&apos;s a one-time snapshot (or, if you turn on
            rebalancing below, a recipe re-checked on a schedule). For ranking existing index ETFs instead,
            see the <Link href="/index-fund" className="underline">Fund Screener</Link>.
          </p>
        </div>
        <Link href="/portfolio" className="text-sm font-medium text-slate-600 hover:underline">
          ← Back to Portfolio
        </Link>
      </div>

      {universeDetailsError && (
        <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{universeDetailsError}</p>
      )}

      <form onSubmit={runGenerate} className="mt-6 flex flex-col gap-3">
        <div className="flex flex-wrap items-end gap-3">
          <Field label="Goal">
            <select value={goal} onChange={(e) => setGoal(e.target.value)} className="input">
              {GOALS.map((g) => (
                <option key={g} value={g}>{g}</option>
              ))}
            </select>
          </Field>
          <Field label="Universe">
            <select value={universe} onChange={(e) => setUniverse(e.target.value)} className="input w-56">
              {universeDetails.map((u) => (
                <option key={u.key} value={u.key}>{u.key} ({u.stock_count})</option>
              ))}
            </select>
          </Field>
          <Field label="Picks per sector">
            <input
              type="number" min={1} max={10}
              value={picksPerSector}
              onChange={(e) => setPicksPerSector(Number(e.target.value))}
              className="input w-20"
            />
          </Field>
          <Field label="Max stocks (optional)">
            <input
              type="number" min={1} max={100}
              placeholder="No limit"
              value={maxStocks}
              onChange={(e) => setMaxStocks(e.target.value)}
              className="input w-28"
            />
          </Field>
          <Field label="Total to invest ($)">
            <input
              type="number" min={100} max={10000000} step="100"
              value={totalAmount}
              onChange={(e) => setTotalAmount(e.target.value)}
              className="input w-32"
            />
          </Field>
        </div>

        <div className="flex flex-wrap items-end gap-3">
          <Field label="Sector weighting">
            <select
              value={sectorWeighting}
              onChange={(e) => setSectorWeighting(e.target.value as SectorWeighting)}
              className="input w-52"
            >
              <option value="equal_dollar">Equal dollars per stock</option>
              <option value="market_cap_by_sector">Market-cap weighted by sector</option>
            </select>
          </Field>
          <label className="flex items-center gap-2 pb-2 text-sm text-slate-700">
            <input
              type="checkbox"
              checked={fractionalShares}
              onChange={(e) => setFractionalShares(e.target.checked)}
              className="h-4 w-4 rounded border-slate-300"
            />
            Fractional shares
          </label>
          <button type="submit" disabled={generating} className="btn-primary">
            {generating ? "Generating…" : "Generate Basket"}
          </button>
          {selectedUniverseDetail?.as_of_date && (
            <span className="pb-2 text-xs text-slate-500">
              Scores as of {selectedUniverseDetail.as_of_date}
            </span>
          )}
        </div>

        {selectedUniverseDetail?.description && (
          <p className="text-xs text-slate-500">{selectedUniverseDetail.description}</p>
        )}
        {preflightWarning && (
          <div className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">{preflightWarning}</div>
        )}
      </form>

      {generateError && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{generateError}</p>}
      {generating && !preview && (
        <p className="mt-4 text-sm text-slate-500">Scanning the universe — first run for a universe can take a moment.</p>
      )}

      {preview && (
        <div className="mt-6 flex flex-col gap-4">
          {preview.concentration_warning && (
            <div className="rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-800">{preview.concentration_warning}</div>
          )}
          {preview.warnings.map((w) => (
            <div key={w} className="rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-800">{w}</div>
          ))}
          {preview.sector_notes.map((n) => (
            <p key={n} className="text-xs text-slate-500">{n}</p>
          ))}
          {preview.trim_notes.map((n) => (
            <p key={n} className="text-xs text-slate-500">{n}</p>
          ))}

          <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600">
            {preview.totals.holding_count} stocks · invested {fmtMoney(preview.totals.invested)} · leftover cash{" "}
            {fmtMoney(preview.totals.leftover_cash)} · as of {preview.as_of_date || "N/A"}
          </div>

          <div className="max-h-[28rem] overflow-auto rounded-lg border border-slate-200 bg-white">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="sticky top-0 border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                  <th className="px-3 py-2"></th>
                  <th className="px-3 py-2">Ticker</th>
                  <th className="px-3 py-2">Name</th>
                  <th className="px-3 py-2">Sector</th>
                  <th className="px-3 py-2 text-right">Score</th>
                  <th className="px-3 py-2 text-right">Price</th>
                  <th className="px-3 py-2 text-right">Shares</th>
                  <th className="px-3 py-2 text-right">Amount</th>
                  <th className="px-3 py-2 text-right">Weight %</th>
                </tr>
              </thead>
              <tbody>
                {preview.holdings.map((h: BasketHolding) => (
                  <tr key={h.Ticker} className="border-b border-slate-100 last:border-0">
                    <td className="px-3 py-2">
                      <button
                        type="button"
                        onClick={() => handleRemove(h.Ticker)}
                        disabled={generating}
                        title="Remove this stock"
                        className="rounded-md border border-slate-300 px-2 py-0.5 text-xs font-medium text-slate-600 hover:bg-slate-50 disabled:opacity-50"
                      >
                        Remove
                      </button>
                    </td>
                    <td className="px-3 py-2 font-medium text-slate-800">{h.Ticker}</td>
                    <td className="px-3 py-2 text-slate-600">{h.Name}</td>
                    <td className="px-3 py-2 text-slate-600">{h["GICS Sector"]}</td>
                    <td className="px-3 py-2 text-right text-slate-600">{h.Score.toFixed(1)}</td>
                    <td className="px-3 py-2 text-right text-slate-600">${h.Price.toFixed(2)}</td>
                    <td className="px-3 py-2 text-right text-slate-600">{h.Shares}</td>
                    <td className="px-3 py-2 text-right text-slate-600">{fmtMoney(h.Amount)}</td>
                    <td className="px-3 py-2 text-right text-slate-600">{fmtPct(h.Weight_pct, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {preview.excluded.length > 0 && (
            <div className="rounded-lg border border-slate-200 bg-white p-3">
              <button
                type="button"
                onClick={() => setShowExclusions((v) => !v)}
                className="text-sm font-medium text-slate-700"
              >
                {showExclusions ? "▾" : "▸"} {preview.excluded.length} ticker{preview.excluded.length === 1 ? "" : "s"} excluded
              </button>
              {showExclusions && (
                <ul className="mt-2 space-y-1 text-xs text-slate-500">
                  {preview.excluded.map((e) => (
                    <li key={e.ticker}>
                      <span className="font-medium text-slate-700">{e.ticker}</span>: {e.reason}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}

          <div>
            <h3 className="font-semibold text-slate-900">Sector Summary</h3>
            <div className="mt-2 overflow-hidden rounded-lg border border-slate-200 bg-white">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                    <th className="px-3 py-2">Sector</th>
                    <th className="px-3 py-2 text-right">Count</th>
                    <th className="px-3 py-2 text-right">Weight %</th>
                    <th className="px-3 py-2 text-right">SPY sector mix (approx.)</th>
                  </tr>
                </thead>
                <tbody>
                  {preview.sector_summary.map((s) => (
                    <tr key={s.Sector} className="border-b border-slate-100 last:border-0">
                      <td className="px-3 py-2 text-slate-700">{s.Sector}</td>
                      <td className="px-3 py-2 text-right text-slate-600">{s.Count}</td>
                      <td className="px-3 py-2 text-right text-slate-600">{fmtPct(s.Weight_pct, 1)}</td>
                      <td className="px-3 py-2 text-right text-slate-500">{fmtPct(s.Spy_Approx_Weight_pct, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-1 text-xs text-slate-500">
              &quot;SPY sector mix&quot; is approximated from the aggregate market cap of S&amp;P 500 constituents
              by sector, not SPY&apos;s real published holdings — this app has no source for that.
            </p>
          </div>

          <div>
            <h3 className="font-semibold text-slate-900">Risk Preview</h3>
            <p className="mt-1 text-xs text-slate-500">
              Estimated as if today&apos;s picks and weights had been held, unchanged, for the past{" "}
              {preview.risk_preview.lookback} — a retroactive approximation, not a real trade-by-trade backtest.{" "}
              <Link href="/guides/diversification" className="text-indigo-600 hover:underline">
                Learn more →
              </Link>
            </p>
            <div className="mt-2 grid grid-cols-2 gap-3 sm:grid-cols-5">
              <RiskTile label="Volatility (ann.)" value={fmtPct(preview.risk_preview.annualized_volatility_pct)} />
              <RiskTile
                label="Beta to SPY"
                value={preview.risk_preview.beta_to_spy !== null ? preview.risk_preview.beta_to_spy.toFixed(2) : "N/A"}
              />
              <RiskTile label="Max drawdown" value={fmtPct(preview.risk_preview.max_drawdown_pct)} />
              <RiskTile label="Largest stock" value={fmtPct(preview.risk_preview.largest_single_stock_weight_pct)} />
              <RiskTile label="Largest sector" value={fmtPct(preview.risk_preview.largest_single_sector_weight_pct)} />
            </div>
            {preview.risk_preview.excluded_from_risk.length > 0 && (
              <p className="mt-1 text-xs text-slate-500">
                Excluded from this estimate (insufficient history): {preview.risk_preview.excluded_from_risk.join(", ")}
              </p>
            )}
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-4">
            <p className="text-sm font-medium text-slate-700">Save as a new portfolio</p>
            <div className="mt-2 flex flex-wrap items-end gap-3">
              <Field label="Portfolio name">
                <input
                  value={portfolioNameTouched ? portfolioName : defaultPortfolioName}
                  onChange={(e) => {
                    setPortfolioName(e.target.value);
                    setPortfolioNameTouched(true);
                  }}
                  className="input w-72"
                  maxLength={100}
                />
              </Field>
              <Field label="Rebalance">
                <select
                  value={rebalanceFrequency}
                  onChange={(e) => setRebalanceFrequency(e.target.value as RebalanceFrequency)}
                  className="input"
                >
                  <option value="none">One-time snapshot</option>
                  <option value="monthly">Monthly</option>
                  <option value="quarterly">Quarterly</option>
                </select>
              </Field>
              {rebalanceFrequency !== "none" && (
                <Field label="Drift threshold %">
                  <input
                    type="number" min={1} max={50} step={0.5}
                    value={driftThresholdPct}
                    onChange={(e) => setDriftThresholdPct(Number(e.target.value))}
                    className="input w-24"
                  />
                </Field>
              )}
              <button type="button" onClick={handleSave} disabled={saving} className="btn-primary">
                {saving ? "Saving…" : "Save as New Portfolio"}
              </button>
            </div>
            {rebalanceFrequency !== "none" && (
              <p className="mt-2 text-xs text-slate-500">
                {rebalanceFrequency === "monthly" ? "Every 1st of the month" : "Every Jan/Apr/Jul/Oct 1st"}, this
                basket is re-ranked and checked for drift past {driftThresholdPct}% — you&apos;ll see a review-
                and-act alert on the Portfolio page with suggested swaps; nothing trades automatically.
              </p>
            )}
            {saveError && <p className="mt-2 text-xs text-red-600">{saveError}</p>}
          </div>
        </div>
      )}

      <div className="mt-8 rounded-lg border border-slate-200 bg-white p-5">
        <h3 className="font-semibold text-slate-900">Final Note</h3>
        <p className="mt-2 text-sm text-slate-600">
          This basket is generated from the same ranking/scoring methodology as the Strategies and Stock
          Finder pages — a weighted composite of return, cost, valuation, and risk factors, not personalized
          investment advice. Projections and risk estimates are hypothetical and not predictive of actual
          results; past performance is not indicative of future results. Sector-mix and risk figures are
          approximations (see notes above), not real index holdings data or a trade-by-trade backtest. Not
          investment advice.
        </p>
      </div>
    </div>
  );
}

function RiskTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-slate-200 bg-white p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-1 text-lg font-semibold text-slate-800">{value}</p>
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
