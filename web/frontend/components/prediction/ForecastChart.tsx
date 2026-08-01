import PlotlyChart from "@/components/PlotlyChart";
import type { ForecastOut } from "@/lib/types";

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
          line: { color: "#059669", width: 2.5 },
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
          fillcolor: "rgba(5, 150, 105, 0.15)",
          line: { width: 0 },
          name: "95% CI",
        },
      ]}
      layout={{
        title: { text: `${ticker} — ${forecast.dates.length}-Day Forecast` },
        xaxis: { title: { text: "Date" } },
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
