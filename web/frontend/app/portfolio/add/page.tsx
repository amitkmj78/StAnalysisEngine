"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { usePlaidLink } from "react-plaid-link";

import {
  ApiError,
  createPlaidLinkToken,
  exchangePlaidPublicToken,
  getCurrentPrice,
  importPortfolioCsv,
  submitManualPositions,
} from "@/lib/api";
import type { ManualPositionInput } from "@/lib/types";
import PortfolioSwitcher from "@/components/PortfolioSwitcher";
import TickerSearchInput from "@/components/TickerSearchInput";

const RISK_PROFILES = ["Conservative", "Balanced", "Aggressive"];

type Mode = "manual" | "csv" | "plaid";

const EMPTY_ROW: ManualPositionInput = {
  name: "",
  ticker: "",
  shares: 0,
  current_price: 0,
  avg_cost: 0,
  acquired_at: null,
};

export default function AddPositionsPage() {
  const searchParams = useSearchParams();
  const [selectedPortfolioId, setSelectedPortfolioId] = useState<number | null>(null);
  const [mode, setMode] = useState<Mode>(searchParams.get("mode") === "plaid" ? "plaid" : "manual");
  const [riskProfile, setRiskProfile] = useState("Balanced");
  const [riskFactor, setRiskFactor] = useState(5);

  const [rows, setRows] = useState<ManualPositionInput[]>([{ ...EMPTY_ROW }]);
  const [file, setFile] = useState<File | null>(null);
  const [priceFetchingRow, setPriceFetchingRow] = useState<number | null>(null);

  const [plaidLinkToken, setPlaidLinkToken] = useState<string | null>(null);
  const [plaidConnecting, setPlaidConnecting] = useState(false);
  const [plaidPositionsImported, setPlaidPositionsImported] = useState<number | null>(null);

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [watchlistNote, setWatchlistNote] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  function updateRow(i: number, patch: Partial<ManualPositionInput>) {
    setRows((prev) => prev.map((r, idx) => (idx === i ? { ...r, ...patch } : r)));
  }

  function addRow() {
    setRows((prev) => [...prev, { ...EMPTY_ROW }]);
  }

  function removeRow(i: number) {
    setRows((prev) => prev.filter((_, idx) => idx !== i));
  }

  async function populateCurrentPrice(i: number, ticker: string) {
    const trimmed = ticker.trim().toUpperCase();
    if (!trimmed) return;
    setPriceFetchingRow(i);
    try {
      const res = await getCurrentPrice(trimmed);
      if (res.price !== null) {
        updateRow(i, { current_price: res.price });
      }
    } catch {
      // Lookup failed (bad ticker, no data) — leave whatever's there so the
      // user can still type a price in by hand.
    } finally {
      setPriceFetchingRow((cur) => (cur === i ? null : cur));
    }
  }

  function noteWatchlist(count: number) {
    if (count > 0) {
      setWatchlistNote(
        `${count} watchlist alert${count === 1 ? "" : "s"} set from your strategies' upside targets and stops.`
      );
    }
  }

  async function submitManual(e: React.FormEvent) {
    e.preventDefault();
    const valid = rows.filter((r) => r.ticker.trim() && r.shares > 0);
    if (valid.length === 0) {
      setError("Add at least one position with a ticker and share count.");
      return;
    }
    setSubmitting(true);
    setError(null);
    setWatchlistNote(null);
    setSaved(false);
    try {
      const res = await submitManualPositions(
        valid.map((r) => ({ ...r, ticker: r.ticker.trim().toUpperCase() })),
        riskProfile,
        riskFactor,
        selectedPortfolioId ?? undefined,
      );
      setRows([{ ...EMPTY_ROW }]);
      noteWatchlist(res.watchlist_alerts_created);
      setSaved(true);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not save positions.");
    } finally {
      setSubmitting(false);
    }
  }

  async function submitCsv(e: React.FormEvent) {
    e.preventDefault();
    if (!file) {
      setError("Choose a Robinhood activity CSV first.");
      return;
    }
    setSubmitting(true);
    setError(null);
    setWatchlistNote(null);
    setSaved(false);
    try {
      const res = await importPortfolioCsv(file, riskProfile, riskFactor, selectedPortfolioId ?? undefined);
      setFile(null);
      noteWatchlist(res.watchlist_alerts_created);
      setSaved(true);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not process CSV.");
    } finally {
      setSubmitting(false);
    }
  }

  const { open: openPlaidLink, ready: plaidLinkReady } = usePlaidLink({
    token: plaidLinkToken,
    onSuccess: async (publicToken, metadata) => {
      // Plaid's own type allows null here, but Link only calls onSuccess
      // after a real successful connection -- there's no real flow where
      // this fires with no token, just a defensive guard against the
      // wider type.
      if (!publicToken) return;
      setSubmitting(true);
      setError(null);
      setPlaidPositionsImported(null);
      setSaved(false);
      try {
        const res = await exchangePlaidPublicToken(
          publicToken,
          selectedPortfolioId ?? undefined,
          metadata.institution?.institution_id ?? undefined,
          metadata.institution?.name ?? undefined,
        );
        setPlaidPositionsImported(res.sync.positions_upserted);
        setSaved(res.sync.status === "success");
        if (res.sync.status === "login_required" || res.sync.status === "error") {
          setError(
            "Connected, but the first sync didn't complete. Try \"Sync Now\" from Linked Accounts in a moment.",
          );
        }
      } catch (err) {
        setError(err instanceof ApiError ? err.message : "Could not finish connecting this account.");
      } finally {
        setSubmitting(false);
        setPlaidLinkToken(null);
      }
    },
    onExit: () => {
      setPlaidConnecting(false);
      setPlaidLinkToken(null);
    },
  });

  useEffect(() => {
    if (plaidLinkToken && plaidLinkReady) {
      openPlaidLink();
    }
  }, [plaidLinkToken, plaidLinkReady, openPlaidLink]);

  async function startPlaidConnect() {
    setError(null);
    setPlaidConnecting(true);
    try {
      const res = await createPlaidLinkToken();
      setPlaidLinkToken(res.link_token);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not start connecting a brokerage account.");
      setPlaidConnecting(false);
    }
  }

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">Add Positions</h1>
          <p className="mt-1 text-sm text-slate-500">
            Import a Robinhood activity CSV or enter positions manually. Each save also sets watchlist alerts by
            default at the suggested upside target and stop for every position. For a CSV import, the date you
            actually bought is pulled from the file itself; for manual entry, leave &quot;Date acquired&quot;
            blank to default to today.
          </p>
        </div>
        <Link href="/portfolio" className="text-sm font-medium text-slate-600 hover:underline">
          ← Back to Portfolio
        </Link>
      </div>

      <div className="mt-6">
        <PortfolioSwitcher selectedPortfolioId={selectedPortfolioId} onChange={setSelectedPortfolioId} />
      </div>

      <div className="mt-6 flex flex-wrap items-end gap-3">
        <Field label="Risk profile">
          <select value={riskProfile} onChange={(e) => setRiskProfile(e.target.value)} className="input">
            {RISK_PROFILES.map((r) => (
              <option key={r} value={r}>{r}</option>
            ))}
          </select>
        </Field>
        <Field label="Risk factor (1-10)">
          <input type="number" min={1} max={10} value={riskFactor} onChange={(e) => setRiskFactor(Number(e.target.value))} className="input w-20" />
        </Field>
      </div>

      <div className="mt-4 flex gap-2">
        <button
          onClick={() => setMode("manual")}
          className={`rounded-md px-3 py-1.5 text-sm font-medium ${mode === "manual" ? "bg-slate-900 text-white" : "border border-slate-300 text-slate-700 hover:bg-slate-100"}`}
        >
          Manual Entry
        </button>
        <button
          onClick={() => setMode("csv")}
          className={`rounded-md px-3 py-1.5 text-sm font-medium ${mode === "csv" ? "bg-slate-900 text-white" : "border border-slate-300 text-slate-700 hover:bg-slate-100"}`}
        >
          Import Robinhood CSV
        </button>
        <button
          onClick={() => setMode("plaid")}
          className={`rounded-md px-3 py-1.5 text-sm font-medium ${mode === "plaid" ? "bg-slate-900 text-white" : "border border-slate-300 text-slate-700 hover:bg-slate-100"}`}
        >
          Connect Brokerage
        </button>
      </div>

      {mode === "manual" ? (
        <form onSubmit={submitManual} className="mt-4 flex flex-col gap-3 rounded-lg border border-slate-200 bg-white p-5">
          {rows.map((row, i) => (
            <div key={i} className="flex flex-wrap items-end gap-2">
              <Field label="Ticker">
                <TickerSearchInput
                  value={row.ticker}
                  onChange={(v) => updateRow(i, { ticker: v })}
                  onSelect={(v) => populateCurrentPrice(i, v)}
                  onBlurValue={(v) => populateCurrentPrice(i, v)}
                  className="input w-40 uppercase"
                />
              </Field>
              <Field label="Name">
                <input value={row.name} onChange={(e) => updateRow(i, { name: e.target.value })} className="input w-32" />
              </Field>
              <Field label="Shares">
                <input type="number" step="0.0001" value={row.shares || ""} onChange={(e) => updateRow(i, { shares: Number(e.target.value) })} className="input w-24" />
              </Field>
              <Field label="Avg cost">
                <input type="number" step="0.01" value={row.avg_cost || ""} onChange={(e) => updateRow(i, { avg_cost: Number(e.target.value) })} className="input w-24" />
              </Field>
              <Field label="Current price">
                <input
                  type="number"
                  step="0.01"
                  placeholder={priceFetchingRow === i ? "Fetching…" : undefined}
                  value={row.current_price || ""}
                  onChange={(e) => updateRow(i, { current_price: Number(e.target.value) })}
                  className="input w-24"
                />
              </Field>
              <Field label="Date acquired (optional)">
                <input
                  type="date"
                  value={row.acquired_at ?? ""}
                  onChange={(e) => updateRow(i, { acquired_at: e.target.value || null })}
                  className="input w-36"
                />
              </Field>
              {rows.length > 1 && (
                <button type="button" onClick={() => removeRow(i)} className="text-xs text-red-600 hover:underline">
                  Remove
                </button>
              )}
            </div>
          ))}
          <div className="flex items-center gap-3">
            <button type="button" onClick={addRow} className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-100">
              + Add Position
            </button>
            <button type="submit" disabled={submitting} className="btn-primary">
              {submitting ? "Saving…" : "Save Positions"}
            </button>
          </div>
        </form>
      ) : mode === "csv" ? (
        <form onSubmit={submitCsv} className="mt-4 flex flex-col gap-3 rounded-lg border border-slate-200 bg-white p-5">
          <Field label="Robinhood activity CSV">
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="text-sm text-slate-700"
            />
          </Field>
          <button type="submit" disabled={submitting} className="btn-primary self-start">
            {submitting ? "Processing…" : "Import CSV"}
          </button>
        </form>
      ) : (
        <div className="mt-4 flex flex-col items-start gap-3 rounded-lg border border-slate-200 bg-white p-5">
          <p className="text-sm text-slate-600">
            Link a real brokerage account through Plaid — your holdings import automatically and stay in sync.
            Your login credentials go directly to Plaid&apos;s secure widget; this app never sees them.
          </p>
          <button
            type="button"
            onClick={startPlaidConnect}
            disabled={plaidConnecting || submitting}
            className="btn-primary"
          >
            {plaidConnecting ? "Opening…" : "Connect Brokerage"}
          </button>
          {plaidPositionsImported !== null && (
            <p className="text-sm text-emerald-700">
              Imported {plaidPositionsImported} position{plaidPositionsImported === 1 ? "" : "s"}.{" "}
              <Link href="/portfolio/linked-accounts" className="underline">
                Manage linked accounts
              </Link>
            </p>
          )}
        </div>
      )}

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {watchlistNote && (
        <p className="mt-4 rounded-md bg-emerald-50 px-3 py-2 text-sm text-emerald-700">
          {watchlistNote}{" "}
          <Link href="/watchlist" className="underline">
            View watchlist
          </Link>
        </p>
      )}
      {saved && (
        <p className="mt-4 rounded-md bg-emerald-50 px-3 py-2 text-sm text-emerald-700">
          Saved.{" "}
          <Link href="/portfolio" className="underline">
            View your portfolio
          </Link>
        </p>
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
