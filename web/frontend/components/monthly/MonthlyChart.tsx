import PlotlyChart from "@/components/PlotlyChart";
import type { MonthlyHistory } from "@/lib/types";

export default function MonthlyChart({ ticker, history }: { ticker: string; history: MonthlyHistory }) {
  return (
    <PlotlyChart
      data={[
        {
          x: history.dates,
          y: history.portfolio_value,
          type: "scatter",
          mode: "lines",
          name: "Portfolio Value",
          line: { color: "#138A72", width: 3 },
        },
        {
          x: history.dates,
          y: history.total_invested,
          type: "scatter",
          mode: "lines",
          name: "Total Invested",
          line: { color: "#1c3556", width: 2, dash: "dash" },
        },
      ]}
      layout={{
        title: { text: `${ticker} Monthly Investing Path` },
        xaxis: { title: { text: "Date" } },
        yaxis: { title: { text: "USD" } },
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        height: 440,
        margin: { t: 48, r: 24, b: 40, l: 64 },
        autosize: true,
      }}
      style={{ width: "100%" }}
      useResizeHandler
      config={{ displayModeBar: false }}
    />
  );
}
