"use client";

import PlotlyChart from "@/components/PlotlyChart";
import type { StockPriceHistoryRange, StockPriceHistoryResponse } from "@/lib/types";

const RANGES: StockPriceHistoryRange[] = ["1D", "5D", "1M", "6M", "1Y", "5Y"];

export default function PriceHistoryChart({
  ticker,
  data,
  range,
  onRangeChange,
  loading,
}: {
  ticker: string;
  data: StockPriceHistoryResponse | null;
  range: StockPriceHistoryRange;
  onRangeChange: (range: StockPriceHistoryRange) => void;
  loading: boolean;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-900">{ticker} Price</h3>
        <div className="flex gap-1">
          {RANGES.map((r) => (
            <button
              key={r}
              onClick={() => onRangeChange(r)}
              className={`rounded-md px-2 py-1 text-xs font-medium ${
                r === range ? "bg-slate-900 text-white" : "text-slate-500 hover:bg-slate-100"
              }`}
            >
              {r}
            </button>
          ))}
        </div>
      </div>
      {loading && <p className="mt-4 text-sm text-slate-500">Loading…</p>}
      {!loading && data && data.history.length === 0 && (
        <p className="mt-4 text-sm text-slate-400">No price history available for {ticker}.</p>
      )}
      {!loading && data && data.history.length > 0 && (
        <PlotlyChart
          data={[
            {
              x: data.history.map((p) => p.date),
              y: data.history.map((p) => p.close),
              type: "scatter",
              mode: "lines",
              name: "Close",
              line: { color: "#1F4FD1", width: 2 },
            },
          ]}
          layout={{
            xaxis: {
              title: { text: "" },
              ...(range === "1D" ? { tickformat: "%-I:%M %p" } : {}),
            },
            yaxis: { title: { text: "USD" } },
            paper_bgcolor: "#ffffff",
            plot_bgcolor: "#ffffff",
            height: 320,
            margin: { t: 16, r: 24, b: 32, l: 56 },
            autosize: true,
          }}
          style={{ width: "100%" }}
          useResizeHandler
          config={{ displayModeBar: false }}
        />
      )}
    </div>
  );
}
