"use client";

import PlotlyChart from "@/components/PlotlyChart";
import type { StockPriceHistoryRange, StockPriceHistoryResponse } from "@/lib/types";

const RANGES: StockPriceHistoryRange[] = ["1D", "5D", "1M", "6M", "1Y", "5Y"];

type PastEarnings = { date: string; reported_eps: number | null };
type Dividend = { date: string; amount: number };
type SignalChange = { date: string; label: string };

// DET-4: earnings/dividend/signal-change markers, snapped onto the price
// line at their own date's close so they sit on the chart rather than
// floating at an arbitrary height. Skipped on the 1D range -- its x-axis
// is full timestamps within a single day, and a date-only event has
// nothing meaningful to snap to there.
function buildMarkerTrace(
  events: { date: string; hovertext: string }[],
  closeByDate: Map<string, number>,
  opts: { name: string; symbol: string; color: string },
) {
  const points = events
    .map((e) => ({ ...e, close: closeByDate.get(e.date) }))
    .filter((e): e is { date: string; hovertext: string; close: number } => e.close !== undefined);
  if (points.length === 0) return null;
  return {
    x: points.map((p) => p.date),
    y: points.map((p) => p.close),
    text: points.map((p) => p.hovertext),
    hovertemplate: "%{text}<extra></extra>",
    type: "scatter" as const,
    mode: "markers" as const,
    name: opts.name,
    marker: { symbol: opts.symbol, size: 10, color: opts.color, line: { color: "#ffffff", width: 1 } },
  };
}

export default function PriceHistoryChart({
  ticker,
  data,
  range,
  onRangeChange,
  loading,
  pastEarnings = [],
  recentDividends = [],
  signalChanges = [],
}: {
  ticker: string;
  data: StockPriceHistoryResponse | null;
  range: StockPriceHistoryRange;
  onRangeChange: (range: StockPriceHistoryRange) => void;
  loading: boolean;
  pastEarnings?: PastEarnings[];
  recentDividends?: Dividend[];
  signalChanges?: SignalChange[];
}) {
  const closeByDate = new Map<string, number>();
  if (range !== "1D") {
    for (const p of data?.history ?? []) closeByDate.set(p.date.slice(0, 10), p.close);
  }

  const markerTraces =
    range === "1D"
      ? []
      : [
          buildMarkerTrace(
            pastEarnings.map((e) => ({
              date: e.date,
              hovertext: e.reported_eps !== null ? `Earnings ${e.date}: EPS $${e.reported_eps.toFixed(2)}` : `Earnings reported ${e.date}`,
            })),
            closeByDate,
            { name: "Earnings", symbol: "diamond", color: "#7C3AED" },
          ),
          buildMarkerTrace(
            recentDividends.map((d) => ({ date: d.date, hovertext: `Dividend ${d.date}: $${d.amount.toFixed(4)}` })),
            closeByDate,
            { name: "Dividends", symbol: "circle", color: "#059669" },
          ),
          buildMarkerTrace(
            signalChanges.map((s) => ({ date: s.date, hovertext: `Signal change ${s.date}: ${s.label}` })),
            closeByDate,
            { name: "Signal Change", symbol: "triangle-up", color: "#D97706" },
          ),
        ].filter((t): t is NonNullable<typeof t> => t !== null);

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
            ...markerTraces,
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
            showlegend: markerTraces.length > 0,
            legend: { orientation: "h", y: -0.15 },
          }}
          style={{ width: "100%" }}
          useResizeHandler
          config={{ displayModeBar: false }}
        />
      )}
    </div>
  );
}
