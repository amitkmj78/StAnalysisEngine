import PlotlyChart from "@/components/PlotlyChart";
import type { MonteCarloResult } from "@/lib/types";

// Plain slate/emerald tokens -- this page was NOT redesigned to the
// "Ledger" system (see ForecastChart.tsx for that palette), so these stay
// consistent with what's already on the Strategies page instead.
const MEDIAN = "#0f172a";
const BAND_FILL = "rgba(5, 150, 105, 0.12)";
const BAND_LINE = "rgba(5, 150, 105, 0.35)";
const INK = "#0f172a";
const MUTED = "#64748b";
const GRID_LINE = "#e2e8f0";
const SURFACE = "#ffffff";

export default function MonteCarloChart({
  result,
  startingCapital,
}: {
  result: MonteCarloResult;
  startingCapital: number;
}) {
  const years = [0, ...result.percentile_bands.map((b) => b.year)];
  const p10 = [startingCapital, ...result.percentile_bands.map((b) => b.p10)];
  const p50 = [startingCapital, ...result.percentile_bands.map((b) => b.p50)];
  const p90 = [startingCapital, ...result.percentile_bands.map((b) => b.p90)];

  return (
    <PlotlyChart
      data={[
        {
          x: years,
          y: p90,
          type: "scatter",
          mode: "lines",
          line: { width: 0 },
          showlegend: false,
          hoverinfo: "skip",
        },
        {
          x: years,
          y: p10,
          type: "scatter",
          mode: "lines",
          fill: "tonexty",
          fillcolor: BAND_FILL,
          line: { width: 1, color: BAND_LINE },
          name: "10th–90th percentile",
        },
        {
          x: years,
          y: p50,
          type: "scatter",
          mode: "lines+markers",
          name: "Median",
          line: { color: MEDIAN, width: 2.5 },
        },
      ]}
      layout={{
        title: { text: "Simulated Outcome Range", font: { color: INK } },
        font: { color: MUTED, family: "Arial, Helvetica, sans-serif" },
        xaxis: { title: { text: "Years" }, gridcolor: GRID_LINE, linecolor: GRID_LINE },
        yaxis: { title: { text: "Balance (USD)" }, gridcolor: GRID_LINE, linecolor: GRID_LINE, tickprefix: "$" },
        paper_bgcolor: SURFACE,
        plot_bgcolor: SURFACE,
        height: 380,
        margin: { t: 48, r: 24, b: 40, l: 72 },
        autosize: true,
        legend: { orientation: "h", y: -0.22 },
      }}
      style={{ width: "100%" }}
      useResizeHandler
      config={{ displayModeBar: false }}
    />
  );
}
