"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import {
  ApiError,
  acceptPaperTradingDisclosure,
  getPaperAccount,
  getPaperOrders,
  linkPaperAccount,
  unlinkPaperAccount,
} from "@/lib/api";
import type { PaperAccount, PaperOrder } from "@/lib/types";

function statusBadgeClass(status: PaperOrder["status"]): string {
  if (status === "FILLED") return "bg-emerald-50 text-emerald-700";
  if (status === "PARTIALLY_FILLED" || status === "OPEN" || status === "SUBMITTING") return "bg-blue-50 text-blue-700";
  if (status === "REJECTED" || status === "UNKNOWN") return "bg-red-50 text-red-700";
  return "bg-slate-100 text-slate-500";
}

export default function PaperTradingPage() {
  const [account, setAccount] = useState<PaperAccount | null | undefined>(undefined); // undefined = loading
  const [live, setLive] = useState<Record<string, unknown> | null>(null);
  const [orders, setOrders] = useState<PaperOrder[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);

  const [apiKeyId, setApiKeyId] = useState("");
  const [apiSecretKey, setApiSecretKey] = useState("");
  const [linking, setLinking] = useState(false);
  const [accepting, setAccepting] = useState(false);
  const [unlinking, setUnlinking] = useState(false);

  function load() {
    getPaperAccount()
      .then((res) => {
        setAccount(res.account);
        setLive(res.live);
      })
      .catch((err) => {
        if (err instanceof ApiError && err.status === 404) {
          setAccount(null);
        } else {
          setError(err instanceof ApiError ? err.message : "Could not load your paper-trading account.");
        }
      });
    getPaperOrders()
      .then((res) => setOrders(res.orders))
      .catch(() => {});
  }

  useEffect(() => {
    load();
  }, []);

  async function handleLink(e: React.FormEvent) {
    e.preventDefault();
    setLinking(true);
    setError(null);
    setNote(null);
    try {
      const res = await linkPaperAccount(apiKeyId.trim(), apiSecretKey.trim());
      setNote(`Linked. Synced ${res.positions_synced} position(s).`);
      setApiKeyId("");
      setApiSecretKey("");
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not link this key pair.");
    } finally {
      setLinking(false);
    }
  }

  async function handleAcceptDisclosure() {
    setAccepting(true);
    setError(null);
    try {
      const res = await acceptPaperTradingDisclosure();
      setAccount(res.account);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not record disclosure acceptance.");
    } finally {
      setAccepting(false);
    }
  }

  async function handleUnlink() {
    if (!window.confirm("Unlink your paper-trading account? This removes its positions and order history from this app.")) {
      return;
    }
    setUnlinking(true);
    setError(null);
    try {
      await unlinkPaperAccount();
      setAccount(null);
      setOrders([]);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not unlink this account.");
    } finally {
      setUnlinking(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h1 className="font-display text-2xl font-semibold text-slate-900">Paper Trading</h1>
          <p className="mt-1 text-sm text-slate-500">
            Practice placing orders with simulated money through Alpaca&apos;s paper-trading sandbox. No real money is
            ever involved.
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

      {account === undefined ? (
        <p className="mt-6 text-sm text-slate-500">Loading…</p>
      ) : account === null ? (
        <form onSubmit={handleLink} className="mt-6 rounded-xl border border-slate-200 bg-white p-5">
          <h2 className="text-sm font-semibold text-slate-900">Link your Alpaca paper account</h2>
          <p className="mt-1 text-xs text-slate-500">
            Generate a paper-trading API key pair from your own Alpaca dashboard (paper account, not live) and paste
            it below. The secret key is encrypted before it&apos;s stored.
          </p>
          <div className="mt-4 flex flex-col gap-3">
            <label className="text-xs font-medium text-slate-600">
              API Key ID
              <input
                value={apiKeyId}
                onChange={(e) => setApiKeyId(e.target.value)}
                required
                className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
                placeholder="PK..."
              />
            </label>
            <label className="text-xs font-medium text-slate-600">
              API Secret Key
              <input
                value={apiSecretKey}
                onChange={(e) => setApiSecretKey(e.target.value)}
                required
                type="password"
                className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
              />
            </label>
          </div>
          <button
            type="submit"
            disabled={linking}
            className="mt-4 rounded-md bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800 disabled:opacity-50"
          >
            {linking ? "Linking…" : "Link Paper Account"}
          </button>
        </form>
      ) : (
        <>
          <div className="mt-6 rounded-xl border border-slate-200 bg-white p-5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-medium text-slate-900">Paper Account {account.account_number ?? ""}</span>
                  <span className="rounded-full bg-blue-50 px-2 py-0.5 text-xs font-semibold text-blue-700">Paper</span>
                </div>
                {live && (
                  <p className="mt-1 text-xs text-slate-500">
                    Buying power ${Number(live.buying_power ?? 0).toLocaleString()} · Equity $
                    {Number(live.equity ?? 0).toLocaleString()}
                  </p>
                )}
              </div>
              <button
                onClick={handleUnlink}
                disabled={unlinking}
                className="rounded-md border border-red-200 px-3 py-1.5 text-sm font-medium text-red-600 hover:bg-red-50 disabled:opacity-50"
              >
                {unlinking ? "Unlinking…" : "Unlink"}
              </button>
            </div>

            {account.disclosure_accepted_at === null ? (
              <div className="mt-4 rounded-md bg-amber-50 p-3 text-sm text-amber-800">
                <p>
                  This is a simulated, practice-only account backed by Alpaca&apos;s paper-trading sandbox. No real
                  money is ever at risk, and orders never reach a real exchange. You must accept this before placing
                  an order.
                </p>
                <button
                  onClick={handleAcceptDisclosure}
                  disabled={accepting}
                  className="mt-2 rounded-md bg-amber-600 px-3 py-1.5 text-sm font-semibold text-white hover:bg-amber-700 disabled:opacity-50"
                >
                  {accepting ? "Saving…" : "I understand, continue"}
                </button>
              </div>
            ) : (
              <Link
                href="/portfolio/paper-trading/ticket"
                className="mt-4 inline-flex rounded-md bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800"
              >
                + New Order
              </Link>
            )}
          </div>

          <h2 className="mt-8 text-sm font-semibold text-slate-900">Orders</h2>
          {orders.length === 0 ? (
            <p className="mt-2 text-sm text-slate-500">No orders yet.</p>
          ) : (
            <div className="mt-3 flex flex-col gap-2">
              {orders.map((order) => (
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
                    </div>
                    <p className="mt-1 text-xs text-slate-500">
                      {order.order_type === "limit" ? `Limit $${order.limit_price}` : "Market"} · {order.time_in_force.toUpperCase()}
                      {order.filled_qty > 0 && ` · Filled ${order.filled_qty}${order.filled_avg_price ? ` @ $${order.filled_avg_price}` : ""}`}
                      {order.reject_reason && ` · ${order.reject_reason}`}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
