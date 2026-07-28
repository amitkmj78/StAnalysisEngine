import PlotlyChart from "@/components/PlotlyChart";
import type { BacktestOut } from "@/lib/types";

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
          line: { color: "#2563EB" },
        },
        {
          x: backtest.dates,
          y: backtest.predicted,
          type: "scatter",
          mode: "lines+markers",
          name: "Predicted",
          line: { color: "#059669" },
        },
        {
          x: backtest.dates,
          y: backtest.naive,
          type: "scatter",
          mode: "lines",
          name: "Naive (no-change)",
          line: { color: "#9AA5B1", dash: "dot" },
        },
      ]}
      layout={{
        title: { text: `${ticker} — Last 30 Days: Actual vs Predicted vs Naive` },
        yaxis: { title: { text: "Price (USD)" } },
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
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
