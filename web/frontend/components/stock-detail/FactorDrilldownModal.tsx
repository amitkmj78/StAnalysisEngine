"use client";

import { useEffect, useState } from "react";

import { ApiError, getTwoScoreFactorHistory } from "@/lib/api";
import type { TwoScoreFactorHistoryResponse, TwoScoreFactorKey } from "@/lib/types";
import PlotlyChart from "@/components/PlotlyChart";

const FACTOR_LABELS: Record<TwoScoreFactorKey, string> = {
  momentum: "Momentum",
  reversal: "Short-Term Reversal",
  value: "Value",
  growth: "Growth",
  low_vol: "Low Volatility",
};

// EXP-4: "open a factor" -- its own history (up to the last 252 trading
// days on record, honestly shorter than "1 year" until stock_scores has
// accumulated that much) plotted against the same factor's sector
// median on each of those days.
export default function FactorDrilldownModal({
  ticker,
  factor,
  onClose,
}: {
  ticker: string;
  factor: TwoScoreFactorKey;
  onClose: () => void;
}) {
  const [data, setData] = useState<TwoScoreFactorHistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    setData(null);
    getTwoScoreFactorHistory(ticker, factor)
      .then(setData)
      .catch((err) => {
        setError(
          err instanceof ApiError && err.status === 404
            ? `Not enough history yet to chart ${FACTOR_LABELS[factor]} for ${ticker}.`
            : err instanceof ApiError
            ? err.message
            : "Could not load this factor's history.",
        );
      })
      .finally(() => setLoading(false));
  }, [ticker, factor]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4" onClick={onClose}>
      <div
        className="max-h-[85vh] w-full max-w-2xl overflow-y-auto rounded-lg bg-white p-5 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="text-base font-semibold text-slate-900">{FACTOR_LABELS[factor]}</h3>
            {data && <p className="text-xs text-slate-500">{ticker} vs. {data.sector_key} sector median</p>}
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-700" aria-label="Close">
            ✕
          </button>
        </div>

        {loading && <p className="mt-4 text-sm text-slate-500">Loading…</p>}
        {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

        {data && data.history.length > 0 && (
          <>
            <PlotlyChart
              data={[
                {
                  x: data.history.map((p) => p.as_of_date),
                  y: data.history.map((p) => p.raw),
                  type: "scatter",
                  mode: "lines",
                  name: ticker,
                  line: { color: "#1F4FD1", width: 2 },
                },
                {
                  x: data.history.map((p) => p.as_of_date),
                  y: data.history.map((p) => p.sector_median),
                  type: "scatter",
                  mode: "lines",
                  name: `${data.sector_key} median`,
                  line: { color: "#94A3B8", width: 2, dash: "dash" },
                },
              ]}
              layout={{
                xaxis: { title: { text: "" } },
                yaxis: { title: { text: "" } },
                paper_bgcolor: "#ffffff",
                plot_bgcolor: "#ffffff",
                height: 280,
                margin: { t: 16, r: 16, b: 32, l: 48 },
                autosize: true,
                legend: { orientation: "h", y: -0.15 },
              }}
              style={{ width: "100%" }}
              useResizeHandler
              config={{ displayModeBar: false }}
            />
            <p className="mt-2 text-xs text-slate-400">
              {data.history.length} trading day{data.history.length === 1 ? "" : "s"} on record
              {data.history.length < 252 ? " (shorter than a year — this scoring system is still accumulating history)" : ""}.
            </p>
          </>
        )}
        {data && data.history.length === 0 && (
          <p className="mt-4 text-sm text-slate-400">No history on record yet.</p>
        )}
      </div>
    </div>
  );
}
