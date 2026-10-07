"use client";

import { useEffect, useState } from "react";
import {
  ApiError,
  addGoalLink,
  getGoalLinks,
  getPortfolioPositions,
  getPortfolios,
  removeGoalLink,
} from "@/lib/api";
import type { GoalLinksResponse, Portfolio, PortfolioPosition } from "@/lib/types";

// STRAT-10: the holdings that fund one goal, their target mix, drift against it, and where the next contribution goes.
// Buys only. Nothing here sells a holding, and nothing places an order.

function money(v: number | null | undefined) {
  return v === null || v === undefined ? "–" : `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

export default function GoalLinks({ planId }: { planId: number }) {
  const [data, setData] = useState<GoalLinksResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [portfolioId, setPortfolioId] = useState<number | null>(null);
  const [positions, setPositions] = useState<PortfolioPosition[]>([]);
  const [ticker, setTicker] = useState("");
  const [sharePct, setSharePct] = useState(100);
  const [role, setRole] = useState<"core" | "pick">("core");
  const [saving, setSaving] = useState(false);

  async function reload() {
    try {
      setData(await getGoalLinks(planId));
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Linked holdings could not be loaded.");
    }
  }

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    getGoalLinks(planId)
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof ApiError ? err.message : "Linked holdings could not be loaded.");
      });
    return () => {
      cancelled = true;
    };
  }, [open, planId]);

  useEffect(() => {
    if (!open || portfolios.length > 0) return;
    getPortfolios()
      .then((res) => {
        setPortfolios(res.portfolios);
        if (res.portfolios[0]) setPortfolioId(res.portfolios[0].id);
      })
      .catch(() => setError("Your portfolios could not be loaded."));
  }, [open, portfolios.length]);

  useEffect(() => {
    if (portfolioId === null) return;
    getPortfolioPositions(portfolioId)
      .then((res) => setPositions(res.positions))
      .catch(() => setPositions([]));
  }, [portfolioId]);

  async function link() {
    if (portfolioId === null || !ticker) return;
    setSaving(true);
    setError(null);
    try {
      await addGoalLink(planId, { portfolio_id: portfolioId, ticker, share_pct: sharePct, role });
      await reload();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "The holding could not be linked.");
    } finally {
      setSaving(false);
    }
  }

  async function unlink(linkId: number) {
    try {
      await removeGoalLink(planId, linkId);
      await reload();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "The link could not be removed.");
    }
  }

  return (
    <div className="mt-3 border-t border-slate-100 pt-3">
      <button type="button" onClick={() => setOpen((v) => !v)} className="text-xs font-medium text-indigo-700 hover:underline">
        {open ? "Hide funded-by holdings" : "Funded by: choose holdings"}
      </button>

      {open && (
        <div className="mt-3 flex flex-col gap-3 text-sm">
          {error && <p className="rounded-md bg-red-50 px-3 py-2 text-xs text-red-700">{error}</p>}

          {data && data.links.length === 0 && (
            <p className="text-xs text-slate-500">No holdings are linked yet. Link a holding below to see its share of this goal.</p>
          )}

          {data && data.links.length > 0 && (
            <>
              <div className="overflow-x-auto">
                <table className="w-full min-w-[26rem] text-right text-xs">
                  <thead className="text-slate-400">
                    <tr>
                      <th className="py-1 text-left font-medium">Holding</th>
                      <th className="font-medium">Role</th>
                      <th className="font-medium">Share</th>
                      <th className="font-medium">Value</th>
                      <th className="font-medium">Gain since linked</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-mono">
                    {data.links.map((l) => (
                      <tr key={l.id}>
                        <td className="py-1.5 text-left font-sans font-medium text-slate-800">{l.ticker}</td>
                        <td className="font-sans">{l.role === "core" ? "Core fund" : "Stock pick"}</td>
                        <td>{l.share_pct.toFixed(0)}%</td>
                        <td>{money(l.value)}</td>
                        <td>{money(l.gain)}</td>
                        <td>
                          <button type="button" onClick={() => unlink(l.id)} className="text-slate-500 hover:text-red-700">Unlink</button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="rounded-md bg-slate-50 p-3 text-xs text-slate-700">
                <p className="font-semibold text-slate-800">Target mix for {data.targets.stock_pct.toFixed(0)}% stock, {data.targets.bonds_pct.toFixed(0)}% bonds</p>
                <p className="mt-1">
                  Core funds {data.targets.core_pct.toFixed(0)}% of the goal, stock picks up to {data.targets.picks_cap_pct.toFixed(0)}%. Shares below are of the linked stock sleeve.
                </p>
                {data.drift.rows.map((r) => (
                  <p key={r.role} className="mt-1">
                    {r.role === "core" ? "Core funds" : "Stock picks"}: {r.actual_pct.toFixed(0)}% now, target {r.target_pct.toFixed(0)}%
                  </p>
                ))}
                {data.drift.flags.map((f) => (
                  <p key={f} className="mt-1 font-medium text-amber-800">{f}</p>
                ))}
              </div>

              {data.steering && data.steering.buys.length > 0 && (
                <div className="rounded-md border border-emerald-200 bg-emerald-50 p-3 text-xs text-emerald-900">
                  <p className="font-semibold">Next {money(data.steering.monthly)}: put {money(data.steering.into_stocks)} into these holdings</p>
                  <ul className="mt-1 list-disc pl-5">
                    {data.steering.buys.map((b) => (
                      <li key={b.ticker}>
                        {money(b.amount)} to {b.ticker}{b.shares > 0 ? ` (${b.shares} share${b.shares === 1 ? "" : "s"})` : " (less than one share)"}
                      </li>
                    ))}
                  </ul>
                  <p className="mt-1">
                    {money(data.steering.toward_bonds_outside_portfolio)} belongs to the bonds share, which sits outside this portfolio.
                    New money only: nothing is sold.
                  </p>
                </div>
              )}
            </>
          )}

          <div className="rounded-md border border-slate-200 p-3">
            <p className="text-xs font-semibold text-slate-700">Link a holding</p>
            <div className="mt-2 flex flex-wrap items-end gap-2 text-xs">
              <label className="flex flex-col gap-1">
                Portfolio
                <select value={portfolioId ?? ""} onChange={(e) => setPortfolioId(Number(e.target.value))} className="input">
                  {portfolios.map((p) => (
                    <option key={p.id} value={p.id}>{p.name}</option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-1">
                Holding
                <select value={ticker} onChange={(e) => setTicker(e.target.value)} className="input">
                  <option value="">Choose…</option>
                  {positions.map((p) => (
                    <option key={p.ticker} value={p.ticker}>{p.ticker}</option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-1">
                Share of it %
                <input type="number" min={1} max={100} value={sharePct} onChange={(e) => setSharePct(Number(e.target.value))} className="input w-20" />
              </label>
              <label className="flex flex-col gap-1">
                Role
                <select value={role} onChange={(e) => setRole(e.target.value as "core" | "pick")} className="input">
                  <option value="core">Core fund</option>
                  <option value="pick">Stock pick</option>
                </select>
              </label>
              <button type="button" onClick={link} disabled={saving || !ticker || portfolioId === null} className="btn-primary">
                {saving ? "Linking…" : "Link"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
