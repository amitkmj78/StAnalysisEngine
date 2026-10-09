"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import { ApiError, getEarningsCalendar } from "@/lib/api";
import type { EarningsCalendarResponse } from "@/lib/types";

// "Ledger" direction (warm paper, Fraunces for headings/numbers, IBM Plex
// for body/UI text) is now the whole app's shared look -- see app/
// layout.tsx + app/globals.css's --pf-* tokens, loaded globally there.
const PF = {
  page: "bg-[var(--pf-bg)]",
  ink: "text-[var(--pf-text)]",
  muted: "text-slate-500",
  line: "border-[var(--pf-border)]",
  card: "rounded-xl border border-[var(--pf-border)] bg-white",
  surface2: "bg-slate-100",
  chip: "inline-flex items-center gap-1 rounded-full border border-[var(--pf-border)] bg-white px-2.5 py-1 text-xs font-medium",
  btn: "rounded-md border border-[var(--pf-border)] bg-white px-3 py-1.5 text-sm font-medium text-slate-900 hover:border-[var(--pf-accent)] hover:text-[var(--pf-accent)]",
};

export default function EarningsCalendarPage() {
  const [data, setData] = useState<EarningsCalendarResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    load();
  }, []);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      setData(await getEarningsCalendar());
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load the earnings calendar.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={`${PF.page} ${PF.ink} min-h-screen`}>
      <div className="mx-auto max-w-4xl px-4 py-8">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h1 className="font-display text-3xl font-semibold" style={{ fontFamily: "var(--font-pf-display)" }}>
            Earnings Calendar
          </h1>
          <button onClick={load} disabled={loading} className={`${PF.btn} disabled:opacity-50`}>
            {loading ? "Refreshing…" : "Refresh"}
          </button>
        </div>
        <p className={`mt-1 max-w-2xl text-sm ${PF.muted}`}>
          Upcoming earnings for every stock you hold or have watchlisted, over the next {data?.window_days ?? 30}{" "}
          days.
        </p>

        {loading && <p className={`mt-6 text-sm ${PF.muted}`}>Loading…</p>}
        {error && <p className="mt-6 rounded-md bg-[var(--pf-down-soft)] px-3 py-2 text-sm text-[var(--pf-down)]">{error}</p>}

        {data && !loading && (
          <div className={`mt-6 ${PF.card} p-5`}>
            {data.entries.length === 0 ? (
              <p className={`text-sm ${PF.muted}`}>
                No held or watchlisted stocks have earnings in the next {data.window_days} days.
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className={`border-b ${PF.line} text-left text-xs font-medium uppercase tracking-wide ${PF.muted}`}>
                      <th className="px-2 py-1.5">Ticker</th>
                      <th className="px-2 py-1.5">Date</th>
                      <th className="px-2 py-1.5">Market Timing</th>
                      <th className="px-2 py-1.5">EPS Estimate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.entries.map((e) => (
                      <tr key={e.ticker} className={`border-b ${PF.line} last:border-0`}>
                        <td className="px-2 py-1.5 font-medium">
                          <Link href={`/stock/${encodeURIComponent(e.ticker)}`} className="hover:underline">
                            {e.ticker}
                          </Link>
                          <span className="ml-2 inline-flex gap-1">
                            {e.owned && <span className={PF.chip}>Owned</span>}
                            {e.watchlisted && <span className={PF.chip}>Watchlisted</span>}
                          </span>
                        </td>
                        <td className="px-2 py-1.5" style={{ fontFamily: "var(--font-pf-mono)" }}>
                          {e.date}
                        </td>
                        <td className="px-2 py-1.5">
                          {e.market_timing === "before_market" ? "Before Market" : "After Market"}
                        </td>
                        <td className="px-2 py-1.5" style={{ fontFamily: "var(--font-pf-mono)" }}>
                          {e.eps_estimate !== null ? `$${e.eps_estimate.toFixed(2)}` : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className={`mt-3 text-xs ${PF.muted}`}>
                  Market timing is inferred from the report&apos;s timestamp, not a confirmed before/after-market
                  flag from the data provider.
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
