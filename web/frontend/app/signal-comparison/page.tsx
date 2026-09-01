"use client";

import { Fragment, useEffect, useMemo, useState } from "react";

import InfoModal, { ColumnInfo } from "@/components/InfoModal";
import { ApiError, getQuantSignalNarrative, getQuantVsAnalyst } from "@/lib/api";
import type { QuantVsAnalystResponse, QuantVsAnalystRow } from "@/lib/types";

type SignalFilter = "ALL" | "BUY" | "SELL" | "HOLD";
type StabilityFilter = "ALL" | "STABLE" | "UNSTABLE";
type SortKey = "ticker" | "quant_expected_return_pct" | "analyst_buy_pct" | "analyst_count" | "signal_flip_count";

const COLUMN_INFO: Record<string, ColumnInfo> = {
  quant_signal: {
    title: "Quant Signal",
    body: [
      "The app's own model's BUY/HOLD/SELL call, based on a 10-trading-day price forecast.",
      "BUY requires an expected return of at least +5%, SELL at most -5%, both measured after the raw " +
        "model output is deliberately damped toward zero — the model's raw predictions lost to a simple " +
        "\"no change\" forecast in backtesting, so the damping and wide neutral band are intentional, not a bug. " +
        "Most tickers land in HOLD by design.",
    ],
  },
  quant_expected_return_pct: {
    title: "Quant Expected Return",
    body: ["The model's forecasted return over the next 10 trading days, after damping. This is what the BUY/SELL threshold is measured against."],
  },
  quant_target_price: {
    title: "Quant Target Price",
    body: ["The model's forecasted price 10 trading days out, implied by the expected return above."],
  },
  analyst_consensus: {
    title: "Analyst Consensus",
    body: ["Real, third-party Wall Street analyst consensus rating for this ticker (e.g. Buy, Hold, Sell), sourced from Yahoo Finance — independent of this app's own model."],
  },
  analyst_buy_pct: {
    title: "Analyst Buy %",
    body: ["Share of covering analysts rating this ticker a Buy or Strong Buy."],
  },
  analyst_target_mean: {
    title: "Analyst Target (Mean)",
    body: ["The average of covering analysts' individual price targets."],
  },
  signal_flip_count: {
    title: "Flips",
    body: [
      "How many times this ticker's Quant Signal has changed (BUY/HOLD/SELL) over its trailing captured history (up to 30 days).",
      "\"Unstable\" means it's flipped 3+ times — treat today's signal with less confidence, since it hasn't settled on a view.",
    ],
  },
};

function signalBadgeClass(signal: string): string {
  if (signal === "BUY") return "bg-emerald-50 text-emerald-700";
  if (signal === "SELL") return "bg-red-50 text-red-700";
  if (signal === "HOLD") return "bg-slate-100 text-slate-600";
  return "bg-slate-100 text-slate-400";
}

function consensusBadgeClass(consensus: string | null): string {
  if (!consensus) return "bg-slate-100 text-slate-400";
  const c = consensus.toLowerCase();
  if (c.includes("buy")) return "bg-emerald-50 text-emerald-700";
  if (c.includes("sell")) return "bg-red-50 text-red-700";
  return "bg-slate-100 text-slate-600";
}

interface NarrativeState {
  status: "loading" | "error" | "ok";
  text?: string;
  plainEnglish?: string | null;
  error?: string;
}

export default function SignalComparisonPage() {
  const [data, setData] = useState<QuantVsAnalystResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [search, setSearch] = useState("");
  const [signalFilter, setSignalFilter] = useState<SignalFilter>("ALL");
  const [stabilityFilter, setStabilityFilter] = useState<StabilityFilter>("ALL");
  const [sortKey, setSortKey] = useState<SortKey>("quant_expected_return_pct");
  const [sortDesc, setSortDesc] = useState(true);

  const [openInfo, setOpenInfo] = useState<string | null>(null);
  const [narratives, setNarratives] = useState<Record<string, NarrativeState>>({});

  useEffect(() => {
    getQuantVsAnalyst()
      .then(setData)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Failed to load signal data."))
      .finally(() => setLoading(false));
  }, []);

  async function loadNarrative(row: QuantVsAnalystRow) {
    setNarratives((prev) => ({ ...prev, [row.ticker]: { status: "loading" } }));
    try {
      const res = await getQuantSignalNarrative(
        row.ticker,
        row.quant_signal,
        row.quant_expected_return_pct,
        row.quant_target_price,
        row.last_close,
      );
      setNarratives((prev) => ({
        ...prev,
        [row.ticker]: { status: "ok", text: res.narrative, plainEnglish: res.plain_english },
      }));
    } catch (err) {
      setNarratives((prev) => ({
        ...prev,
        [row.ticker]: { status: "error", error: err instanceof ApiError ? err.message : "Failed to generate context." },
      }));
    }
  }

  function handleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDesc((d) => !d);
    } else {
      setSortKey(key);
      setSortDesc(true);
    }
  }

  const counts = useMemo(() => {
    const c = { BUY: 0, SELL: 0, HOLD: 0 };
    for (const r of data?.rows ?? []) {
      if (r.quant_signal === "BUY" || r.quant_signal === "SELL" || r.quant_signal === "HOLD") c[r.quant_signal]++;
    }
    return c;
  }, [data]);

  const rows = useMemo(() => {
    let filtered = data?.rows ?? [];
    if (signalFilter !== "ALL") {
      filtered = filtered.filter((r) => r.quant_signal === signalFilter);
    }
    if (stabilityFilter !== "ALL") {
      filtered = filtered.filter((r) => (stabilityFilter === "UNSTABLE" ? r.signal_unstable : !r.signal_unstable));
    }
    if (search.trim()) {
      const q = search.trim().toUpperCase();
      filtered = filtered.filter((r) => r.ticker.includes(q));
    }
    const sorted = [...filtered].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === "string" && typeof bv === "string") {
        return sortDesc ? bv.localeCompare(av) : av.localeCompare(bv);
      }
      const an = Number(av);
      const bn = Number(bv);
      return sortDesc ? bn - an : an - bn;
    });
    return sorted;
  }, [data, signalFilter, stabilityFilter, search, sortKey, sortDesc]);

  function SortableTh({ label, sortKeyName, info }: { label: string; sortKeyName: SortKey; info?: string }) {
    return (
      <th className="sticky top-0 z-10 bg-slate-50 px-3 py-2 text-left">
        <div className="flex items-center gap-1">
          <button onClick={() => handleSort(sortKeyName)} className="hover:text-slate-900">
            {label}
            {sortKey === sortKeyName && <span className="ml-1">{sortDesc ? "↓" : "↑"}</span>}
          </button>
          {info && (
            <button
              onClick={() => setOpenInfo(info)}
              className="text-slate-400 hover:text-slate-600"
              aria-label={`About ${label}`}
            >
              ⓘ
            </button>
          )}
        </div>
      </th>
    );
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-10">
      <h1 className="text-2xl font-semibold text-slate-900">Quant Signal vs. Analyst Rating</h1>
      <p className="mt-2 text-sm leading-relaxed text-slate-600">
        The internal quant model&apos;s BUY/HOLD/SELL call next to the real Wall Street analyst consensus, for
        every ticker captured that day. Both are point-in-time snapshots — independent of each other, shown
        side by side for comparison, not blended into one score.
      </p>

      {loading && <p className="mt-6 text-sm text-slate-500">Loading…</p>}
      {error && <p className="mt-6 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {data && !loading && (
        <>
          <div className="mt-6 flex flex-wrap items-center gap-3">
            <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600">
              As of <span className="font-medium text-slate-900">{data.as_of_date ?? "—"}</span> ·{" "}
              {data.ticker_count} tickers · <span className="text-emerald-700">{counts.BUY} BUY</span> ·{" "}
              <span className="text-red-700">{counts.SELL} SELL</span> ·{" "}
              <span className="text-slate-500">{counts.HOLD} HOLD</span>
            </div>
            <input
              type="text"
              placeholder="Search ticker…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="rounded-md border border-slate-300 px-3 py-2 text-sm"
            />
            <div className="flex gap-1 rounded-md border border-slate-300 bg-white p-1">
              {(["ALL", "BUY", "SELL", "HOLD"] as SignalFilter[]).map((f) => (
                <button
                  key={f}
                  onClick={() => setSignalFilter(f)}
                  className={`rounded px-3 py-1 text-sm font-medium ${
                    signalFilter === f ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-100"
                  }`}
                >
                  {f}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-1 rounded-md border border-slate-300 bg-white p-1">
              {(["ALL", "STABLE", "UNSTABLE"] as StabilityFilter[]).map((f) => (
                <button
                  key={f}
                  onClick={() => setStabilityFilter(f)}
                  title={f === "UNSTABLE" ? "Signal has flipped 3+ times in its trailing history" : undefined}
                  className={`rounded px-3 py-1 text-sm font-medium ${
                    stabilityFilter === f ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-100"
                  }`}
                >
                  {f === "ALL" ? "All Signals" : f === "STABLE" ? "Stable Only" : "Unstable Only"}
                </button>
              ))}
            </div>
          </div>

          {rows.length === 0 ? (
            <div className="mt-6 rounded-lg border border-slate-200 bg-white p-5 text-sm text-slate-500">
              No tickers match this filter.
            </div>
          ) : (
            <div className="mt-6 max-h-[70vh] overflow-auto rounded-lg border border-slate-200 bg-white">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                    <SortableTh label="Ticker" sortKeyName="ticker" />
                    <th className="sticky top-0 z-10 bg-slate-50 px-3 py-2 text-left">
                      <div className="flex items-center gap-1">
                        Signal
                        <button onClick={() => setOpenInfo("quant_signal")} className="text-slate-400 hover:text-slate-600" aria-label="About Signal">
                          ⓘ
                        </button>
                      </div>
                    </th>
                    <SortableTh label="Exp. Return" sortKeyName="quant_expected_return_pct" info="quant_expected_return_pct" />
                    <SortableTh label="Flips" sortKeyName="signal_flip_count" info="signal_flip_count" />
                    <th className="sticky top-0 z-10 bg-slate-50 px-3 py-2 text-right">Target</th>
                    <th className="sticky top-0 z-10 bg-slate-50 px-3 py-2 text-right">Last Close</th>
                    <th className="sticky top-0 z-10 bg-slate-50 px-3 py-2 text-left">
                      <div className="flex items-center gap-1">
                        Analyst Consensus
                        <button onClick={() => setOpenInfo("analyst_consensus")} className="text-slate-400 hover:text-slate-600" aria-label="About Analyst Consensus">
                          ⓘ
                        </button>
                      </div>
                    </th>
                    <SortableTh label="Buy %" sortKeyName="analyst_buy_pct" info="analyst_buy_pct" />
                    <th className="sticky top-0 z-10 bg-slate-50 px-3 py-2 text-right">Target (Mean)</th>
                    <th className="sticky top-0 z-10 bg-slate-50 px-3 py-2 text-left">AI Context</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => {
                    const narrative = narratives[r.ticker];
                    return (
                      <Fragment key={r.ticker}>
                        <tr className="border-b border-slate-100 last:border-0">
                          <td className="px-3 py-2 font-medium text-slate-800">{r.ticker}</td>
                          <td className="px-3 py-2">
                            <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${signalBadgeClass(r.quant_signal)}`}>
                              {r.quant_signal}
                            </span>
                          </td>
                          <td
                            className={`px-3 py-2 text-right font-medium ${
                              r.quant_expected_return_pct >= 0 ? "text-emerald-600" : "text-red-600"
                            }`}
                          >
                            {r.quant_expected_return_pct >= 0 ? "+" : ""}
                            {r.quant_expected_return_pct.toFixed(2)}%
                          </td>
                          <td className="px-3 py-2 text-right">
                            <span
                              title={`${r.signal_flip_count} flip${r.signal_flip_count === 1 ? "" : "s"} over ${r.signal_days_captured} captured days`}
                              className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
                                r.signal_unstable ? "bg-amber-50 text-amber-700" : "text-slate-500"
                              }`}
                            >
                              {r.signal_flip_count}
                              {r.signal_unstable && " Unstable"}
                            </span>
                          </td>
                          <td className="px-3 py-2 text-right text-slate-600">{r.quant_target_price.toFixed(2)}</td>
                          <td className="px-3 py-2 text-right text-slate-600">{r.last_close.toFixed(2)}</td>
                          <td className="px-3 py-2">
                            {r.analyst_consensus ? (
                              <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${consensusBadgeClass(r.analyst_consensus)}`}>
                                {r.analyst_consensus}
                              </span>
                            ) : (
                              <span className="text-xs text-slate-400">No data</span>
                            )}
                          </td>
                          <td className="px-3 py-2 text-right text-slate-600">
                            {r.analyst_buy_pct !== null ? `${r.analyst_buy_pct.toFixed(0)}%` : "—"}
                          </td>
                          <td className="px-3 py-2 text-right text-slate-600">
                            {r.analyst_target_mean !== null ? r.analyst_target_mean.toFixed(2) : "—"}
                          </td>
                          <td className="px-3 py-2">
                            {!narrative && (
                              <button
                                onClick={() => loadNarrative(r)}
                                className="rounded-md border border-slate-300 bg-white px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-100"
                              >
                                Get AI Context
                              </button>
                            )}
                            {narrative?.status === "loading" && <span className="text-xs text-slate-500">Thinking…</span>}
                            {narrative?.status === "error" && (
                              <button
                                onClick={() => loadNarrative(r)}
                                className="text-xs font-medium text-red-700 hover:underline"
                              >
                                Failed — retry
                              </button>
                            )}
                            {narrative?.status === "ok" && (
                              <span className="text-xs font-medium text-emerald-700">Shown below ↓</span>
                            )}
                          </td>
                        </tr>
                        {narrative?.status === "ok" && (
                          <tr className="border-b border-slate-100 bg-slate-50/60 last:border-0">
                            <td colSpan={10} className="px-3 py-2 text-xs leading-relaxed text-slate-600">
                              {narrative.plainEnglish && (
                                <p className="mb-2">
                                  <strong className="text-slate-700">{r.ticker} — in plain terms:</strong>{" "}
                                  {narrative.plainEnglish}
                                </p>
                              )}
                              <p className="text-slate-500">
                                <strong className="text-slate-600">
                                  {r.ticker} — quant technical context:
                                </strong>{" "}
                                {narrative.text}
                              </p>
                              <p className="mt-2 text-[11px] italic text-slate-400">
                                This explains what the indicators show, not investment advice — the model&apos;s own{" "}
                                {r.quant_signal} call above is a forecast, not a guarantee.
                              </p>
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}

      {openInfo && COLUMN_INFO[openInfo] && (
        <InfoModal info={COLUMN_INFO[openInfo]} onClose={() => setOpenInfo(null)} />
      )}
    </div>
  );
}
