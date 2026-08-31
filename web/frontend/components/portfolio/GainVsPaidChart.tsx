import PlotlyChart from "@/components/PlotlyChart";
import type { PortfolioPerformanceRow } from "@/lib/types";

export default function GainVsPaidChart({ rows }: { rows: PortfolioPerformanceRow[] }) {
  const charted = rows
    .filter((r) => r.gain_vs_cost !== null && r.gain_vs_cost_pct !== null)
    .sort((a, b) => (b.gain_vs_cost_pct as number) - (a.gain_vs_cost_pct as number));

  if (charted.length === 0) return null;

  return (
    <div className="mt-3 rounded-lg border border-slate-200 bg-white p-3">
      <PlotlyChart
        data={[
          {
            name: "Gain vs. cost",
            x: charted.map((r) => r.ticker),
            y: charted.map((r) => r.gain_vs_cost_pct),
            type: "bar",
            marker: { color: charted.map((r) => ((r.gain_vs_cost_pct as number) >= 0 ? "#10b981" : "#ef4444")) },
            text: charted.map(
              (r) =>
                `${(r.gain_vs_cost as number) >= 0 ? "+" : ""}$${(r.gain_vs_cost as number).toLocaleString(undefined, {
                  maximumFractionDigits: 0,
                })}`,
            ),
            textposition: "outside",
            hovertemplate: "%{x}<br>%{y:.1f}%<br>%{text}<extra></extra>",
          },
          {
            name: "Equity (value now)",
            x: charted.map((r) => r.ticker),
            y: charted.map((r) => r.value_now),
            yaxis: "y2",
            type: "scatter",
            mode: "lines+markers",
            line: { color: "#3b82f6", width: 2 },
            marker: { color: "#3b82f6", size: 7 },
            hovertemplate: "%{x}<br>Equity: $%{y:,.0f}<extra></extra>",
          },
        ]}
        layout={{
          title: { text: "Gain vs. Paid & Equity, by Position" },
          yaxis: { title: { text: "% vs. average cost" }, zeroline: true, zerolinecolor: "#cbd5e1" },
          yaxis2: {
            title: { text: "Equity ($)" },
            overlaying: "y",
            side: "right",
            showgrid: false,
            tickprefix: "$",
          },
          xaxis: { title: { text: "" } },
          legend: { orientation: "h", y: 1.15 },
          paper_bgcolor: "#ffffff",
          plot_bgcolor: "#ffffff",
          height: 380,
          margin: { t: 64, r: 56, b: 40, l: 56 },
          autosize: true,
        }}
        style={{ width: "100%" }}
        useResizeHandler
        config={{ displayModeBar: false }}
      />
    </div>
  );
}
