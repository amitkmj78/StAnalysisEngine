"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import { ApiError, disconnectPlaidItem, getPlaidItems, syncPlaidItem } from "@/lib/api";
import type { PlaidItem } from "@/lib/types";

function statusBadgeClass(status: PlaidItem["status"]): string {
  if (status === "active") return "bg-emerald-50 text-emerald-700";
  if (status === "login_required") return "bg-amber-50 text-amber-700";
  if (status === "error") return "bg-red-50 text-red-700";
  return "bg-slate-100 text-slate-500";
}

function statusLabel(status: PlaidItem["status"]): string {
  if (status === "login_required") return "Needs re-login";
  if (status === "active") return "Connected";
  if (status === "error") return "Error";
  return status;
}

export default function LinkedAccountsPage() {
  const [items, setItems] = useState<PlaidItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [syncingId, setSyncingId] = useState<number | null>(null);
  const [disconnectingId, setDisconnectingId] = useState<number | null>(null);
  const [note, setNote] = useState<string | null>(null);

  function load() {
    getPlaidItems()
      .then((res) => setItems(res.items))
      .catch((err) => setError(err instanceof ApiError ? err.message : "Could not load linked accounts."));
  }

  useEffect(() => {
    load();
  }, []);

  async function handleSync(item: PlaidItem) {
    setSyncingId(item.id);
    setError(null);
    setNote(null);
    try {
      const res = await syncPlaidItem(item.id);
      if (res.status === "success") {
        setNote(`Synced ${item.institution_name ?? "this account"}: ${res.positions_upserted} position(s).`);
      } else if (res.status === "login_required") {
        setError(`${item.institution_name ?? "This account"} needs you to reconnect — its login has expired.`);
      } else {
        setError(`Sync failed for ${item.institution_name ?? "this account"}.`);
      }
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Sync failed.");
    } finally {
      setSyncingId(null);
    }
  }

  async function handleDisconnect(item: PlaidItem) {
    if (
      !window.confirm(
        `Disconnect ${item.institution_name ?? "this account"}? This removes every position it imported from your portfolio.`,
      )
    ) {
      return;
    }
    setDisconnectingId(item.id);
    setError(null);
    try {
      await disconnectPlaidItem(item.id);
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not disconnect this account.");
    } finally {
      setDisconnectingId(null);
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">Linked Accounts</h1>
          <p className="mt-1 text-sm text-slate-500">
            Brokerage accounts connected through Plaid. Each syncs into its own set of positions — disconnecting
            one only removes what it imported, never anything you added manually or by CSV.
          </p>
        </div>
        <Link href="/portfolio" className="text-sm font-medium text-slate-600 hover:underline">
          ← Back to Portfolio
        </Link>
      </div>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {note && <p className="mt-4 rounded-md bg-emerald-50 px-3 py-2 text-sm text-emerald-700">{note}</p>}

      {items === null ? (
        <p className="mt-6 text-sm text-slate-500">Loading…</p>
      ) : items.length === 0 ? (
        <div className="mt-6 rounded-lg border border-slate-200 bg-white p-5 text-sm text-slate-500">
          No brokerage accounts linked yet.{" "}
          <Link href="/portfolio/add?mode=plaid" className="text-slate-700 underline">
            Connect one
          </Link>
          .
        </div>
      ) : (
        <div className="mt-6 flex flex-col gap-3">
          {items.map((item) => (
            <div
              key={item.id}
              className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-slate-200 bg-white p-4"
            >
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-medium text-slate-900">{item.institution_name ?? "Unnamed connection"}</span>
                  <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${statusBadgeClass(item.status)}`}>
                    {statusLabel(item.status)}
                  </span>
                </div>
                <p className="mt-1 text-xs text-slate-500">
                  {item.last_sync_at ? `Last synced ${new Date(item.last_sync_at).toLocaleString()}` : "Never synced"}
                  {item.last_sync_error && item.status !== "active" ? ` — ${item.last_sync_error}` : ""}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleSync(item)}
                  disabled={syncingId === item.id}
                  className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
                >
                  {syncingId === item.id ? "Syncing…" : "Sync Now"}
                </button>
                <button
                  onClick={() => handleDisconnect(item)}
                  disabled={disconnectingId === item.id}
                  className="rounded-md border border-red-200 px-3 py-1.5 text-sm font-medium text-red-600 hover:bg-red-50 disabled:opacity-50"
                >
                  {disconnectingId === item.id ? "Disconnecting…" : "Disconnect"}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <Link
        href="/portfolio/add?mode=plaid"
        className="mt-6 inline-flex rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-100"
      >
        + Connect Another Account
      </Link>
    </div>
  );
}
