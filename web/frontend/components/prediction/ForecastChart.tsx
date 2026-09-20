import PlotlyChart from "@/components/PlotlyChart";
import type { ForecastOut } from "@/lib/types";

// "Ledger" palette (see web/frontend/app/predict/page.tsx's PF constant)
// -- Plotly reads plain hex/rgba strings from its data/layout config, not
// CSS, so these are literal values matching that palette rather than a
// shared token.
const GOOD = "#2f6b4f";
const CI_FILL = "rgba(47, 107, 79, 0.15)";
const INK = "#1f2420";
const MUTED = "#857d6e";
const GRID_LINE = "#ddd8cd";
const SURFACE = "#ffffff";

export default function ForecastChart({ ticker, forecast }: { ticker: string; forecast: ForecastOut }) {
  return (
    <PlotlyChart
      data={[
        {
          x: forecast.dates,
          y: forecast.predicted,
          type: "scatter",
          mode: "lines+markers",
          name: "Predicted",
          line: { color: GOOD, width: 2.5 },
        },
        {
          x: forecast.dates,
          y: forecast.upper_ci,
          type: "scatter",
          mode: "lines",
          line: { width: 0 },
          showlegend: false,
          hoverinfo: "skip",
        },
        {
          x: forecast.dates,
          y: forecast.lower_ci,
          type: "scatter",
          mode: "lines",
          fill: "tonexty",
          fillcolor: CI_FILL,
          line: { width: 0 },
          name: "95% CI",
        },
      ]}
      layout={{
        title: { text: `${ticker} — ${forecast.dates.length}-Day Forecast`, font: { color: INK } },
        font: { color: MUTED, family: "IBM Plex Sans, sans-serif" },
        xaxis: { title: { text: "Date" }, gridcolor: GRID_LINE, linecolor: GRID_LINE },
        yaxis: { title: { text: "Price (USD)" }, gridcolor: GRID_LINE, linecolor: GRID_LINE },
        paper_bgcolor: SURFACE,
        plot_bgcolor: SURFACE,
        height: 420,
        margin: { t: 48, r: 24, b: 40, l: 56 },
        autosize: true,
      }}
      style={{ width: "100%" }}
      useResizeHandler
      config={{ displayModeBar: false }}
    />
  );
}
