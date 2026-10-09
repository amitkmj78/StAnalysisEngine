"use client";

import { useState } from "react";
import { ApiError, getTradeImpact } from "@/lib/api";
import type { TradeImpactResponse } from "@/lib/types";

// DIF-6: how a hypothetical trade would change the portfolio, shown before anything is placed.
// Numbers are calculated from the stored holdings and the app's own measures. The portfolio
// score is a model output, not advice.

function fmt(v: number | null | undefined, digits = 1, suffix = "") {
  return v === null || v === undefined ? "n/a" : `${v.toFixed(digits)}${suffix}`;
}

function signed(v: number | null | undefined, digits = 1, suffix = "") {
  if (v === null || v === undefined) return "n/a";
  return `${v > 0 ? "+" : ""}${v.toFixed(digits)}${suffix}`;
}

export default function TradeImpactCard({
  ticker,
  portfolios,
  defaultSide = "buy",
}: {
  ticker: string;
  portfolios: { id: number; name: string }[];
  defaultSide?: "buy" | "sell";
}) {
  const [side, setSide] = useState<"buy" | "sell">(defaultSide);
  const [shares, setShares] = useState("");
  const [portfolioId, setPortfolioId] = useState<number | null>(portfolios[0]?.id ?? null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TradeImpactResponse | null>(null);

  async function handlePreview(e: React.FormEvent) {
    e.preventDefault();
    const qty = Number(shares);
    if (!ticker || !(qty > 0)) {
      setError("Enter a number of shares above zero.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await getTradeImpact({
        portfolio_id: portfolioId ?? undefined,
        ticker,
        side,
        shares: qty,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "The preview could not be calculated.");
    } finally {
      setLoading(false);
    }
  }

  const movedSectors = result
    ? [...result.sectors].filter((s) => Math.abs(s.change_pct) >= 0.05).sort((a, b) => Math.abs(b.change_pct) - Math.abs(a.change_pct)).slice(0, 5)
    : [];

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-sm font-semibold text-slate-900">Portfolio impact preview</h3>
        <p className="text-xs text-slate-500">Hypothetical. Nothing is placed.</p>
      </div>

      <form onSubmit={handlePreview} className="mt-3 flex flex-wrap items-end gap-3 text-sm">
        <div className="flex gap-1">
          {(["buy", "sell"] as const).map((s) => (
            <button
              key={s}
              type="button"
              aria-pressed={side === s}
              onClick={() => setSide(s)}
              className={`rounded-md border px-3 py-1.5 font-medium capitalize ${
                side === s ? "border-slate-900 bg-slate-900 text-white" : "border-slate-200 text-slate-600 hover:bg-slate-50"
              }`}
            >
              {s}
            </button>
          ))}
        </div>
        <label className="flex flex-col gap-1 text-xs text-slate-500">
          Shares
          <input
            value={shares}
            onChange={(e) => setShares(e.target.value)}
            inputMode="decimal"
            placeholder="e.g. 10"
            className="input w-28 py-1 text-sm"
          />
        </label>
        {portfolios.length > 1 && (
          <label className="flex flex-col gap-1 text-xs text-slate-500">
            Portfolio
            <select
              value={portfolioId ?? ""}
              onChange={(e) => setPortfolioId(Number(e.target.value))}
              className="input py-1 text-sm"
            >
              {portfolios.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </label>
        )}
        <button type="submit" disabled={loading} className="btn-primary disabled:opacity-50">
          {loading ? "Calculating…" : "Preview"}
        </button>
      </form>

      {error && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {result && (
        <div className="mt-4">
          <p className="text-xs text-slate-500">
            {result.side === "buy" ? "Buy" : "Sell"} {result.shares} {result.ticker} at ${fmt(result.trade_price, 2)} (
            ${fmt(result.trade_price * result.shares, 0)}).
          </p>
          <div className="overflow-x-auto">
            <table className="mt-2 w-full text-left text-sm">
              <thead className="text-xs uppercase tracking-wide text-slate-400">
              <tr>
                <th className="py-1 font-medium">Measure</th>
                <th className="py-1 font-medium">Before</th>
                <th className="py-1 font-medium">After</th>
                <th className="py-1 font-medium">Change</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              <tr>
                <td className="py-1.5 text-slate-700">Largest position</td>
                <td>{fmt(result.before.concentration.largest_position_pct, 1, "%")}</td>
                <td>{fmt(result.after.concentration.largest_position_pct, 1, "%")}</td>
                <td>{signed(result.changes.largest_position_pct, 1, " pts")}</td>
              </tr>
              <tr>
                <td className="py-1.5 text-slate-700">Top 5 positions</td>
                <td>{fmt(result.before.concentration.top5_pct, 1, "%")}</td>
                <td>{fmt(result.after.concentration.top5_pct, 1, "%")}</td>
                <td>{signed(result.changes.top5_pct, 1, " pts")}</td>
              </tr>
              <tr>
                <td className="py-1.5 text-slate-700">Beta to SPY</td>
                <td>{fmt(result.before.beta, 2)}</td>
                <td>{fmt(result.after.beta, 2)}</td>
                <td>{signed(result.changes.beta, 2)}</td>
              </tr>
              <tr>
                <td className="py-1.5 text-slate-700">Portfolio score (short-term, model output)</td>
                <td>
                  {fmt(result.before.portfolio_score.score, 1)}
                  <span className="block text-xs text-slate-400">covers {fmt(result.before.portfolio_score.coverage_pct, 0, "%")}</span>
                </td>
                <td>
                  {fmt(result.after.portfolio_score.score, 1)}
                  <span className="block text-xs text-slate-400">covers {fmt(result.after.portfolio_score.coverage_pct, 0, "%")}</span>
                </td>
                <td>{signed(result.changes.portfolio_score, 1)}</td>
              </tr>
            </tbody>
          </table>
          </div>

          {movedSectors.length > 0 && (
            <div className="mt-4">
              <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Sector weights that move</p>
              <ul className="mt-1 flex flex-col gap-1 text-sm">
                {movedSectors.map((s) => (
                  <li key={s.sector} className="flex justify-between gap-3">
                    <span className="text-slate-700">{s.sector}</span>
                    <span className="font-mono text-xs text-slate-600">
                      {fmt(s.before_pct, 1, "%")} → {fmt(s.after_pct, 1, "%")} ({signed(s.change_pct, 1, " pts")})
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          <p className="mt-3 text-xs text-slate-400">{result.note}</p>
        </div>
      )}
    </div>
  );
}
