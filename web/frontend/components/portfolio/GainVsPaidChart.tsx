import PlotlyChart from "@/components/PlotlyChart";
import type { PortfolioPerformanceRow } from "@/lib/types";

export default function GainVsPaidChart({ rows }: { rows: PortfolioPerformanceRow[] }) {
  const charted = rows
    .filter((r) => r.gain_vs_cost !== null && r.gain_vs_cost_pct !== null)
    .sort((a, b) => (b.gain_vs_cost_pct as number) - (a.gain_vs_cost_pct as number));

  if (charted.length === 0) return null;

  // "Ledger" palette (see web/frontend/app/portfolio/page.tsx's PF
  // constant) -- Plotly reads plain hex strings from its data/layout
  // config, not CSS, so these are literal values matching that palette
  // rather than a shared token.
  const GOOD = "#047857";
  const BAD = "#b91c1c";
  const EQUITY_LINE = "#b45309"; // distinct from good/bad, consistent with the warm palette
  const GRID_LINE = "#e2e8f0";
  const SURFACE = "#ffffff";

  return (
    <div className="mt-3 rounded-xl border border-slate-200 bg-white p-3">
      <PlotlyChart
        data={[
          {
            name: "Gain vs. cost",
            x: charted.map((r) => r.ticker),
            y: charted.map((r) => r.gain_vs_cost_pct),
            type: "bar",
            marker: { color: charted.map((r) => ((r.gain_vs_cost_pct as number) >= 0 ? GOOD : BAD)) },
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
            line: { color: EQUITY_LINE, width: 2 },
            marker: { color: EQUITY_LINE, size: 7 },
            hovertemplate: "%{x}<br>Equity: $%{y:,.0f}<extra></extra>",
          },
        ]}
        layout={{
          title: { text: "Gain vs. Paid & Equity, by Position", font: { color: "#0f172a" } },
          font: { color: "#334155", family: "IBM Plex Sans, sans-serif" },
          yaxis: { title: { text: "% vs. average cost" }, zeroline: true, zerolinecolor: GRID_LINE, gridcolor: GRID_LINE },
          yaxis2: {
            title: { text: "Equity ($)" },
            overlaying: "y",
            side: "right",
            showgrid: false,
            tickprefix: "$",
          },
          xaxis: { title: { text: "" } },
          legend: { orientation: "h", y: 1.15 },
          paper_bgcolor: SURFACE,
          plot_bgcolor: SURFACE,
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
