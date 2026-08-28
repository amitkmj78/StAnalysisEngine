"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";

import { ApiError, createPortfolio, getDiversifiedBasket, getStockUniverses, submitManualPositions } from "@/lib/api";

const GOALS = ["Short Term", "Long Term"];

interface BasketRow {
  Ticker: string;
  Name: string;
  Sector: string;
  Price: number;
  Score: number;
  shares: number;
  excluded: boolean;
}

export default function BuildIndexPage() {
  const [goal, setGoal] = useState("Long Term");
  const [universes, setUniverses] = useState<string[]>(["All"]);
  const [universe, setUniverse] = useState("All");
  const [picksPerSector, setPicksPerSector] = useState(2);
  const [totalAmount, setTotalAmount] = useState("10000");

  const [generating, setGenerating] = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);
  const [basket, setBasket] = useState<BasketRow[]>([]);

  const [portfolioName, setPortfolioName] = useState("My Diversified Index");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    getStockUniverses()
      .then((res) => setUniverses(res.universes))
      .catch(() => {
        // Non-fatal: fall back to "All" already in state.
      });
  }, []);

  const includedCount = useMemo(() => basket.filter((r) => !r.excluded).length, [basket]);
  const amountPerStock = useMemo(() => {
    const amount = Number(totalAmount);
    return includedCount > 0 && amount > 0 ? amount / includedCount : 0;
  }, [totalAmount, includedCount]);

  function recomputeShares(rows: BasketRow[], perStock: number): BasketRow[] {
    return rows.map((r) => ({
      ...r,
      shares: !r.excluded && r.Price > 0 ? Number((perStock / r.Price).toFixed(4)) : r.shares,
    }));
  }

  async function generate(e: React.FormEvent) {
    e.preventDefault();
    setGenerating(true);
    setGenerateError(null);
    setSaved(false);
    try {
      const res = await getDiversifiedBasket(goal, universe, picksPerSector);
      const amount = Number(totalAmount);
      const perStock = res.results.length > 0 && amount > 0 ? amount / res.results.length : 0;
      setBasket(
        res.results.map((r) => ({
          ...r,
          excluded: false,
          shares: r.Price > 0 ? Number((perStock / r.Price).toFixed(4)) : 0,
        }))
      );
    } catch (err) {
      setGenerateError(err instanceof ApiError ? err.message : "Could not generate a basket for this universe.");
      setBasket([]);
    } finally {
      setGenerating(false);
    }
  }

  function toggleExclude(ticker: string) {
    setBasket((prev) => {
      const next = prev.map((r) => (r.Ticker === ticker ? { ...r, excluded: !r.excluded } : r));
      const included = next.filter((r) => !r.excluded).length;
      const amount = Number(totalAmount);
      const perStock = included > 0 && amount > 0 ? amount / included : 0;
      return recomputeShares(next, perStock);
    });
  }

  function handleAmountBlur() {
    setBasket((prev) => recomputeShares(prev, amountPerStock));
  }

  async function saveAsPortfolio() {
    const name = portfolioName.trim();
    if (!name) {
      setSaveError("Name this portfolio first.");
      return;
    }
    const rows = basket.filter((r) => !r.excluded && r.shares > 0);
    if (rows.length === 0) {
      setSaveError("No positions to save — generate a basket and keep at least one stock.");
      return;
    }
    setSaving(true);
    setSaveError(null);
    try {
      const portfolio = await createPortfolio(name);
      await submitManualPositions(
        rows.map((r) => ({
          ticker: r.Ticker,
          name: r.Name,
          shares: r.shares,
          avg_cost: r.Price,
          current_price: r.Price,
        })),
        "Balanced",
        5,
        portfolio.id,
      );
      setSaved(true);
    } catch (err) {
      setSaveError(err instanceof ApiError ? err.message : "Could not save this basket as a portfolio.");
    } finally {
      setSaving(false);
    }
  }

  const sectorCount = useMemo(() => new Set(basket.map((r) => r.Sector)).size, [basket]);

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">Build a Diversified Index</h1>
          <p className="mt-1 text-sm text-slate-500">
            Generates a custom basket of individual stocks spread across sectors — the top-scoring tickers from
            each sector in the universe you pick, equal-dollar weighted — then saves it as a new portfolio. For
            ranking existing index ETFs instead, see the <Link href="/index-fund" className="underline">Fund Screener</Link>.
          </p>
        </div>
        <Link href="/portfolio" className="text-sm font-medium text-slate-600 hover:underline">
          ← Back to Portfolio
        </Link>
      </div>

      <form onSubmit={generate} className="mt-6 flex flex-wrap items-end gap-3">
        <Field label="Goal">
          <select value={goal} onChange={(e) => setGoal(e.target.value)} className="input">
            {GOALS.map((g) => (
              <option key={g} value={g}>{g}</option>
            ))}
          </select>
        </Field>
        <Field label="Universe">
          <select value={universe} onChange={(e) => setUniverse(e.target.value)} className="input">
            {universes.map((u) => (
              <option key={u} value={u}>{u}</option>
            ))}
          </select>
        </Field>
        <Field label="Picks per sector">
          <input
            type="number"
            min={1}
            max={10}
            value={picksPerSector}
            onChange={(e) => setPicksPerSector(Number(e.target.value))}
            className="input w-20"
          />
        </Field>
        <Field label="Total to invest ($)">
          <input
            type="number"
            min={0}
            step="100"
            value={totalAmount}
            onChange={(e) => setTotalAmount(e.target.value)}
            onBlur={handleAmountBlur}
            className="input w-32"
          />
        </Field>
        <button type="submit" disabled={generating} className="btn-primary">
          {generating ? "Generating…" : "Generate Basket"}
        </button>
      </form>

      {generateError && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{generateError}</p>}
      {generating && <p className="mt-4 text-sm text-slate-500">Scanning the universe — first run for a universe can take a moment.</p>}

      {basket.length > 0 && !generating && (
        <>
          <div className="mt-6 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600">
            {includedCount} of {basket.length} stocks across {sectorCount} sectors ·{" "}
            ${amountPerStock.toLocaleString(undefined, { maximumFractionDigits: 0 })} per stock
          </div>

          <div className="mt-3 max-h-[28rem] overflow-auto rounded-lg border border-slate-200 bg-white">
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
                  <th className="px-3 py-2 text-right">Allocation</th>
                </tr>
              </thead>
              <tbody>
                {basket.map((r) => (
                  <tr key={r.Ticker} className={`border-b border-slate-100 last:border-0 ${r.excluded ? "opacity-40" : ""}`}>
                    <td className="px-3 py-2">
                      <button
                        type="button"
                        onClick={() => toggleExclude(r.Ticker)}
                        title={r.excluded ? "Include this stock" : "Exclude this stock"}
                        className="rounded-md border border-slate-300 px-2 py-0.5 text-xs font-medium text-slate-600 hover:bg-slate-50"
                      >
                        {r.excluded ? "Include" : "Remove"}
                      </button>
                    </td>
                    <td className="px-3 py-2 font-medium text-slate-800">{r.Ticker}</td>
                    <td className="px-3 py-2 text-slate-600">{r.Name}</td>
                    <td className="px-3 py-2 text-slate-600">{r.Sector}</td>
                    <td className="px-3 py-2 text-right text-slate-600">{r.Score}</td>
                    <td className="px-3 py-2 text-right text-slate-600">${r.Price.toFixed(2)}</td>
                    <td className="px-3 py-2 text-right text-slate-600">{r.excluded ? "—" : r.shares}</td>
                    <td className="px-3 py-2 text-right text-slate-600">
                      {r.excluded ? "—" : `$${(r.shares * r.Price).toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-4 flex flex-wrap items-end gap-2 rounded-lg border border-slate-200 bg-white p-4">
            <Field label="Save as portfolio named">
              <input
                value={portfolioName}
                onChange={(e) => setPortfolioName(e.target.value)}
                className="input w-64"
              />
            </Field>
            <button type="button" onClick={saveAsPortfolio} disabled={saving} className="btn-primary">
              {saving ? "Saving…" : "Save as New Portfolio"}
            </button>
            {saveError && <p className="w-full text-xs text-red-600">{saveError}</p>}
            {saved && (
              <p className="w-full text-sm text-emerald-700">
                Saved. <Link href="/portfolio" className="underline">View your portfolio</Link>
              </p>
            )}
          </div>
        </>
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
