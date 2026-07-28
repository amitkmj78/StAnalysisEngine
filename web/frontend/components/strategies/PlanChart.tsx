import PlotlyChart from "@/components/PlotlyChart";
import type { StrategyPlanRow } from "@/lib/types";

const PALETTE = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494"];

export default function PlanChart({ targetAmount, years, planTable }: { targetAmount: number; years: number; planTable: StrategyPlanRow[] }) {
  return (
    <PlotlyChart
      data={[
        {
          x: planTable.map((r) => r.Strategy),
          y: planTable.map((r) => r["Required Monthly Invest"]),
          type: "bar",
          marker: { color: planTable.map((_, i) => PALETTE[i % PALETTE.length]) },
        },
      ]}
      layout={{
        title: { text: `Monthly Contribution Needed to Reach $${targetAmount.toLocaleString()} in ${years} Years` },
        yaxis: { title: { text: "USD per month" } },
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        height: 420,
        margin: { t: 56, r: 24, b: 60, l: 64 },
        autosize: true,
      }}
      style={{ width: "100%" }}
      useResizeHandler
      config={{ displayModeBar: false }}
    />
  );
}
