"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import {
  ApiError,
  acceptPaperTradingDisclosure,
  createPortfolio,
  getPaperAccounts,
  getPaperOrders,
  getPortfolios,
  linkPaperAccount,
  unlinkPaperAccount,
} from "@/lib/api";
import type { PaperAccount, PaperOrder, Portfolio } from "@/lib/types";

function statusBadgeClass(status: PaperOrder["status"]): string {
  if (status === "FILLED") return "bg-emerald-50 text-emerald-700";
  if (status === "PARTIALLY_FILLED" || status === "OPEN" || status === "SUBMITTING") return "bg-blue-50 text-blue-700";
  if (status === "REJECTED" || status === "UNKNOWN") return "bg-red-50 text-red-700";
  return "bg-slate-100 text-slate-500";
}

type LinkFormState = { apiKeyId: string; apiSecretKey: string };

export default function PaperTradingPage() {
  // A user can link one Alpaca paper account per portfolio -- each
  // portfolio below shows either its linked account or a form to link one.
  const [portfolios, setPortfolios] = useState<Portfolio[] | null>(null);
  const [accounts, setAccounts] = useState<PaperAccount[] | null>(null);
  const [orders, setOrders] = useState<PaperOrder[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);

  const [linkForms, setLinkForms] = useState<Record<number, LinkFormState>>({});
  const [linkingPortfolioId, setLinkingPortfolioId] = useState<number | null>(null);
  const [acceptingPortfolioId, setAcceptingPortfolioId] = useState<number | null>(null);
  const [unlinkingPortfolioId, setUnlinkingPortfolioId] = useState<number | null>(null);

  // Offered when the user has no portfolio yet -- named explicitly here
  // (not left to silently auto-create a generic "My Portfolio" the first
  // time some other endpoint needs one), so a portfolio created for paper
  // trading is never indistinguishable from a real one by name alone.
  const [newPortfolioName, setNewPortfolioName] = useState(
    () => `Paper Trading (${new Date().toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })})`
  );
  const [creatingPortfolio, setCreatingPortfolio] = useState(false);
  const [createPortfolioError, setCreatePortfolioError] = useState<string | null>(null);

  function load() {
    getPortfolios()
      .then((res) => setPortfolios(res.portfolios))
      .catch(() => setPortfolios([]));
    getPaperAccounts()
      .then((res) => setAccounts(res.accounts))
      .catch(() => setAccounts([]));
    getPaperOrders()
      .then((res) => setOrders(res.orders))
      .catch(() => {});
  }

  useEffect(() => {
    load();
  }, []);

  function updateLinkForm(portfolioId: number, patch: Partial<LinkFormState>) {
    setLinkForms((prev) => {
      const current: LinkFormState = prev[portfolioId] ?? { apiKeyId: "", apiSecretKey: "" };
      return { ...prev, [portfolioId]: { ...current, ...patch } };
    });
  }

  async function handleLink(e: React.FormEvent, portfolioId: number) {
    e.preventDefault();
    const form = linkForms[portfolioId];
    if (!form) return;
    setLinkingPortfolioId(portfolioId);
    setError(null);
    setNote(null);
    try {
      const res = await linkPaperAccount(form.apiKeyId.trim(), form.apiSecretKey.trim(), portfolioId);
      setNote(`Linked. Synced ${res.positions_synced} position(s).`);
      setLinkForms((prev) => ({ ...prev, [portfolioId]: { apiKeyId: "", apiSecretKey: "" } }));
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not link this key pair.");
    } finally {
      setLinkingPortfolioId(null);
    }
  }

  async function handleCreatePortfolio(e: React.FormEvent) {
    e.preventDefault();
    const name = newPortfolioName.trim();
    if (!name) {
      setCreatePortfolioError("Enter a name for this portfolio.");
      return;
    }
    setCreatingPortfolio(true);
    setCreatePortfolioError(null);
    try {
      await createPortfolio(name);
      load();
    } catch (err) {
      setCreatePortfolioError(err instanceof ApiError ? err.message : "Could not create that portfolio.");
    } finally {
      setCreatingPortfolio(false);
    }
  }

  async function handleAcceptDisclosure(portfolioId: number) {
    setAcceptingPortfolioId(portfolioId);
    setError(null);
    try {
      await acceptPaperTradingDisclosure(portfolioId);
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not record disclosure acceptance.");
    } finally {
      setAcceptingPortfolioId(null);
    }
  }

  async function handleUnlink(portfolioId: number) {
    if (
      !window.confirm("Unlink this paper-trading account? This removes its positions and order history from this app.")
    ) {
      return;
    }
    setUnlinkingPortfolioId(portfolioId);
    setError(null);
    try {
      await unlinkPaperAccount(portfolioId);
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not unlink this account.");
    } finally {
      setUnlinkingPortfolioId(null);
    }
  }

  const loading = portfolios === null || accounts === null;
  const accountByPortfolio = new Map((accounts ?? []).map((a) => [a.portfolio_id, a]));
  const showPortfolioLabel = (accounts?.length ?? 0) > 1;

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h1 className="font-display text-2xl font-semibold text-slate-900">Paper Trading</h1>
          <p className="mt-1 text-sm text-slate-500">
            Practice placing orders with simulated money through Alpaca&apos;s paper-trading sandbox. No real money is
            ever involved. You can link one account per portfolio.
          </p>
          <Link href="/challenges" className="mt-1 inline-block text-sm text-indigo-600 hover:underline">
            Compete with friends using this account →
          </Link>
        </div>
        <Link href="/portfolio" className="text-sm font-medium text-slate-600 hover:underline">
          ← Back to Portfolio
        </Link>
      </div>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {note && <p className="mt-4 rounded-md bg-emerald-50 px-3 py-2 text-sm text-emerald-700">{note}</p>}

      {loading ? (
        <p className="mt-6 text-sm text-slate-500">Loading…</p>
      ) : portfolios.length === 0 ? (
        <form onSubmit={handleCreatePortfolio} className="mt-6 rounded-xl border border-slate-200 bg-white p-5">
          <h2 className="text-sm font-semibold text-slate-900">Create a portfolio for paper trading</h2>
          <p className="mt-1 text-xs text-slate-500">
            You need a portfolio before linking a paper account. Name it something you&apos;ll recognize as
            practice money -- you can always rename or create another real one later from the main Portfolio page.
          </p>
          <label className="mt-3 block text-xs font-medium text-slate-600">
            Portfolio name
            <input
              value={newPortfolioName}
              onChange={(e) => setNewPortfolioName(e.target.value)}
              className="mt-1 w-full max-w-sm rounded-md border border-slate-300 px-3 py-2 text-sm"
            />
          </label>
          <button
            type="submit"
            disabled={creatingPortfolio}
            className="mt-3 rounded-md bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800 disabled:opacity-50"
          >
            {creatingPortfolio ? "Creating…" : "Create portfolio"}
          </button>
          {createPortfolioError && <p className="mt-2 text-xs text-red-600">{createPortfolioError}</p>}
        </form>
      ) : (
        <div className="mt-6 flex flex-col gap-4">
          {portfolios.map((p) => {
            const account = accountByPortfolio.get(p.id);
            return (
              <div key={p.id} className="rounded-xl border border-slate-200 bg-white p-5">
                {portfolios.length > 1 && <h2 className="text-sm font-semibold text-slate-900">{p.name}</h2>}
                {!account ? (
                  <form onSubmit={(e) => handleLink(e, p.id)} className={portfolios.length > 1 ? "mt-3 flex flex-col gap-3" : "flex flex-col gap-3"}>
                    {portfolios.length === 1 && <h2 className="text-sm font-semibold text-slate-900">Link your Alpaca paper account</h2>}
                    <p className="text-xs text-slate-500">
                      Generate a paper-trading API key pair from your own Alpaca dashboard (paper account, not live)
                      and paste it below. The secret key is encrypted before it&apos;s stored.
                    </p>
                    <label className="text-xs font-medium text-slate-600">
                      API Key ID
                      <input
                        value={linkForms[p.id]?.apiKeyId ?? ""}
                        onChange={(e) => updateLinkForm(p.id, { apiKeyId: e.target.value })}
                        required
                        className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
                        placeholder="PK..."
                      />
                    </label>
                    <label className="text-xs font-medium text-slate-600">
                      API Secret Key
                      <input
                        value={linkForms[p.id]?.apiSecretKey ?? ""}
                        onChange={(e) => updateLinkForm(p.id, { apiSecretKey: e.target.value })}
                        required
                        type="password"
                        className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
                      />
                    </label>
                    <button
                      type="submit"
                      disabled={linkingPortfolioId === p.id}
                      className="self-start rounded-md bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800 disabled:opacity-50"
                    >
                      {linkingPortfolioId === p.id ? "Linking…" : "Link Paper Account"}
                    </button>
                  </form>
                ) : (
                  <div className={portfolios.length > 1 ? "mt-3" : ""}>
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-slate-900">Paper Account {account.account_number ?? ""}</span>
                        <span className="rounded-full bg-blue-50 px-2 py-0.5 text-xs font-semibold text-blue-700">Paper</span>
                      </div>
                      <button
                        onClick={() => handleUnlink(p.id)}
                        disabled={unlinkingPortfolioId === p.id}
                        className="rounded-md border border-red-200 px-3 py-1.5 text-sm font-medium text-red-600 hover:bg-red-50 disabled:opacity-50"
                      >
                        {unlinkingPortfolioId === p.id ? "Unlinking…" : "Unlink"}
                      </button>
                    </div>

                    {account.disclosure_accepted_at === null ? (
                      <div className="mt-4 rounded-md bg-amber-50 p-3 text-sm text-amber-800">
                        <p>
                          This is a simulated, practice-only account backed by Alpaca&apos;s paper-trading sandbox. No
                          real money is ever at risk, and orders never reach a real exchange. You must accept this
                          before placing an order.
                        </p>
                        <button
                          onClick={() => handleAcceptDisclosure(p.id)}
                          disabled={acceptingPortfolioId === p.id}
                          className="mt-2 rounded-md bg-amber-600 px-3 py-1.5 text-sm font-semibold text-white hover:bg-amber-700 disabled:opacity-50"
                        >
                          {acceptingPortfolioId === p.id ? "Saving…" : "I understand, continue"}
                        </button>
                      </div>
                    ) : (
                      <Link
                        href={`/portfolio/paper-trading/ticket?portfolio_id=${p.id}`}
                        className="mt-4 inline-flex rounded-md bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800"
                      >
                        + New Order
                      </Link>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      <h2 className="mt-8 text-sm font-semibold text-slate-900">Orders</h2>
      {orders.length === 0 ? (
        <p className="mt-2 text-sm text-slate-500">No orders yet.</p>
      ) : (
        <div className="mt-3 flex flex-col gap-2">
          {orders.map((order) => {
            const account = (accounts ?? []).find((a) => a.id === order.alpaca_paper_account_id);
            const portfolioName = account ? portfolios?.find((p) => p.id === account.portfolio_id)?.name : null;
            return (
              <div
                key={order.id}
                className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-200 bg-white p-3"
              >
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-slate-900">
                      {order.side === "buy" ? "Buy" : "Sell"} {order.qty} {order.ticker}
                    </span>
                    <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${statusBadgeClass(order.status)}`}>
                      {order.status.replace("_", " ")}
                    </span>
                    {showPortfolioLabel && portfolioName && (
                      <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-500">{portfolioName}</span>
                    )}
                  </div>
                  <p className="mt-1 text-xs text-slate-500">
                    {order.order_type === "limit" ? `Limit $${order.limit_price}` : "Market"} ·{" "}
                    {order.time_in_force.toUpperCase()}
                    {order.filled_qty > 0 &&
                      ` · Filled ${order.filled_qty}${order.filled_avg_price ? ` @ $${order.filled_avg_price}` : ""}`}
                    {order.reject_reason && ` · ${order.reject_reason}`}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
