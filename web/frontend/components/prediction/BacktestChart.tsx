import PlotlyChart from "@/components/PlotlyChart";
import type { BacktestOut } from "@/lib/types";

// "Ledger" palette -- see ForecastChart.tsx's own copy of this comment.
const ACTUAL_LINE = "#3b6ea5"; // distinct blue, kept apart from good/bad/accent
const GOOD = "#0E5C57";
const NAIVE_LINE = "#94a3b8"; // muted, "the boring baseline" -- deliberately low-contrast
const INK = "#18201D";
const MUTED = "#555F5A";
const GRID_LINE = "#ECEEEA";
const SURFACE = "#ffffff";

export default function BacktestChart({ ticker, backtest }: { ticker: string; backtest: BacktestOut }) {
  return (
    <PlotlyChart
      data={[
        {
          x: backtest.dates,
          y: backtest.actual,
          type: "scatter",
          mode: "lines+markers",
          name: "Actual",
          line: { color: ACTUAL_LINE },
        },
        {
          x: backtest.dates,
          y: backtest.predicted,
          type: "scatter",
          mode: "lines+markers",
          name: "Predicted",
          line: { color: GOOD },
        },
        {
          x: backtest.dates,
          y: backtest.naive,
          type: "scatter",
          mode: "lines",
          name: "Naive (no-change)",
          line: { color: NAIVE_LINE, dash: "dot" },
        },
      ]}
      layout={{
        title: { text: `${ticker} — Last 30 Days: Actual vs Predicted vs Naive`, font: { color: INK } },
        font: { color: MUTED, family: "IBM Plex Sans, sans-serif" },
        xaxis: { gridcolor: GRID_LINE, linecolor: GRID_LINE },
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
