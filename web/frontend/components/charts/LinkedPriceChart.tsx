"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import PlotlyChart from "@/components/PlotlyChart";
import { getStockPriceHistory } from "@/lib/api";
import type { StockPriceHistoryRange, StockPriceHistoryRow } from "@/lib/types";

// CHT-8: one chart in the grid. When linked, hovering a date here moves the crosshair on the other
// charts to the same date (see the chart grid page).

type Props = {
  ticker: string;
  range: StockPriceHistoryRange;
  chartType: "line" | "candles";
  logScale: boolean;
  linked: boolean;
  slot: number;
  hoverDate: string | null;
  hoverSource: number | null;
  onHover: (date: string | null, slot: number) => void;
};

// Loaded in the browser only: plotly.js does not build for the server.
const PlotlyHoverSync = dynamic(() => import("@/components/charts/PlotlyHoverSync"), { ssr: false });

function dayKey(date: string) {
  return date.slice(0, 10);
}

export default function LinkedPriceChart({ ticker, range, chartType, logScale, linked, slot, hoverDate, hoverSource, onHover }: Props) {
  const [rows, setRows] = useState<StockPriceHistoryRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getStockPriceHistory(ticker, range)
      .then((res) => {
        if (!cancelled) setRows(res.history);
      })
      .catch(() => {
        if (!cancelled) setError(`No price history for ${ticker}.`);
      });
    return () => {
      cancelled = true;
    };
  }, [ticker, range]);

  const dates = useMemo(() => (rows ?? []).map((r) => dayKey(r.date)), [rows]);
  const divId = `linked-chart-${slot}`;

  if (error) return <p className="rounded-md border border-dashed border-slate-300 p-6 text-center text-sm text-slate-500">{error}</p>;
  if (!rows) return <p className="rounded-md border border-slate-200 p-6 text-center text-sm text-slate-500">Loading {ticker}…</p>;

  const useCandles = chartType === "candles" && rows.every((r) => r.open !== null && r.high !== null && r.low !== null);
  const x = rows.map((r) => dayKey(r.date));
  const traces = useCandles
    ? [
        {
          x,
          open: rows.map((r) => r.open),
          high: rows.map((r) => r.high),
          low: rows.map((r) => r.low),
          close: rows.map((r) => r.close),
          type: "candlestick" as const,
          name: ticker,
          increasing: { line: { color: "#059669" } },
          decreasing: { line: { color: "#DC2626" } },
        },
      ]
    : [{ x, y: rows.map((r) => r.close), type: "scatter" as const, mode: "lines" as const, name: ticker, line: { color: "#1F4FD1", width: 2 } }];

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="mb-1 flex items-baseline justify-between">
        <p className="text-sm font-semibold text-slate-900">{ticker}</p>
        <p className="font-mono text-xs text-slate-500">{rows[rows.length - 1] ? `${rows[rows.length - 1].close.toFixed(2)}` : ""}</p>
      </div>
      <PlotlyHoverSync divId={divId} linked={linked} dates={dates} hoverDate={hoverDate} hoverSource={hoverSource} slot={slot} />
      <PlotlyChart
        divId={divId}
        data={traces}
        layout={{
          height: 260,
          margin: { t: 8, r: 12, b: 28, l: 48 },
          paper_bgcolor: "#ffffff",
          plot_bgcolor: "#ffffff",
          showlegend: false,
          hovermode: "x",
          xaxis: { rangeslider: { visible: false } },
          yaxis: { type: logScale ? "log" : "linear" },
        }}
        style={{ width: "100%" }}
        useResizeHandler
        config={{ displayModeBar: false }}
        onHover={(e: { points?: { x?: unknown }[] }) => {
          if (!linked) return;
          const x0 = e.points?.[0]?.x;
          if (typeof x0 === "string") onHover(x0, slot);
        }}
        onUnhover={() => {
          if (linked) onHover(null, slot);
        }}
      />
    </div>
  );
}
