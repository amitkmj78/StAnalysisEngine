import PlotlyChart from "@/components/PlotlyChart";
import type { ForecastOut } from "@/lib/types";

// Same CI-band Plotly technique as components/prediction/ForecastChart.tsx
// (upper-bound invisible trace + lower-bound trace with fill:"tonexty"),
// just in this page's own plain slate/orange tokens instead of the
// Ledger palette -- this page was never redesigned to Ledger.
const LINE = "#D9731A";
const CI_FILL = "rgba(217, 115, 26, 0.15)";
const INK = "#1f2937";
const MUTED = "#6b7280";
const GRID_LINE = "#e5e7eb";
const SURFACE = "#ffffff";

export default function StockForecastPanel({ ticker, forecast }: { ticker: string; forecast: ForecastOut }) {
  return (
    <PlotlyChart
      data={[
        {
          x: forecast.dates, y: forecast.predicted, type: "scatter", mode: "lines+markers",
          name: "Predicted", line: { color: LINE, width: 2 }, marker: { size: 4 },
        },
        {
          x: forecast.dates, y: forecast.upper_ci, type: "scatter", mode: "lines",
          line: { width: 0 }, showlegend: false, hoverinfo: "skip",
        },
        {
          x: forecast.dates, y: forecast.lower_ci, type: "scatter", mode: "lines",
          fill: "tonexty", fillcolor: CI_FILL, line: { width: 0 }, name: "Confidence band",
        },
      ]}
      layout={{
        title: { text: `${ticker} — ${forecast.dates.length}-Day Forecast`, font: { color: INK, size: 13 } },
        font: { color: MUTED, family: "Arial, Helvetica, sans-serif", size: 11 },
        xaxis: { gridcolor: GRID_LINE, linecolor: GRID_LINE },
        yaxis: { gridcolor: GRID_LINE, linecolor: GRID_LINE, tickprefix: "$" },
        paper_bgcolor: SURFACE,
        plot_bgcolor: SURFACE,
        height: 220,
        margin: { t: 32, r: 16, b: 32, l: 48 },
        autosize: true,
        showlegend: false,
      }}
      style={{ width: "100%" }}
      useResizeHandler
      config={{ displayModeBar: false }}
    />
  );
}
