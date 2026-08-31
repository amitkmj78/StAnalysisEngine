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
        ]}
        layout={{
          title: { text: "Gain vs. Paid, by Position" },
          yaxis: { title: { text: "% vs. average cost" }, zeroline: true, zerolinecolor: "#cbd5e1" },
          xaxis: { title: { text: "" } },
          paper_bgcolor: "#ffffff",
          plot_bgcolor: "#ffffff",
          height: 360,
          margin: { t: 48, r: 16, b: 40, l: 56 },
          autosize: true,
        }}
        style={{ width: "100%" }}
        useResizeHandler
        config={{ displayModeBar: false }}
      />
    </div>
  );
}
