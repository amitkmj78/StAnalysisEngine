import PlotlyChart from "@/components/PlotlyChart";
import type { CompareSeriesPoint } from "@/lib/types";

// Exact palette from the compare page's own design notes: portfolio
// blue, top pick orange, S&P 500 grey dashed -- the same three colors
// used in the summary cards, legend and this chart, so a color always
// means the same series everywhere on the page.
export const COMPARE_COLORS = {
  portfolio: "#4f46e5",
  topPick: "#D9731A",
  benchmark: "#7C8794",
  gain: "#13795B",
  loss: "#B42318",
  buy: "#4f46e5",
  hold: "#7C8794",
  trim: "#D9731A",
};

function toXY(series: CompareSeriesPoint[]) {
  return { x: series.map((p) => p[0]), y: series.map((p) => p[1]) };
}

export default function CompareGrowthChart({
  portfolioSeries,
  benchmarkSeries,
  topFundSeries,
  topFundTicker,
}: {
  portfolioSeries: CompareSeriesPoint[];
  benchmarkSeries: CompareSeriesPoint[];
  topFundSeries: CompareSeriesPoint[] | null;
  topFundTicker: string | null;
}) {
  const portfolioXY = toXY(portfolioSeries);
  const benchmarkXY = toXY(benchmarkSeries);
  const topFundXY = topFundSeries ? toXY(topFundSeries) : null;

  const data: Record<string, unknown>[] = [
    {
      x: portfolioXY.x, y: portfolioXY.y, type: "scatter", mode: "lines",
      name: "Your portfolio", line: { color: COMPARE_COLORS.portfolio, width: 2.5 },
    },
    {
      x: benchmarkXY.x, y: benchmarkXY.y, type: "scatter", mode: "lines",
      name: "S&P 500", line: { color: COMPARE_COLORS.benchmark, width: 2, dash: "dash" },
    },
  ];
  if (topFundXY && topFundTicker) {
    data.push({
      x: topFundXY.x, y: topFundXY.y, type: "scatter", mode: "lines",
      name: topFundTicker, line: { color: COMPARE_COLORS.topPick, width: 2 },
    });
  }

  return (
    <PlotlyChart
      data={data}
      layout={{
        title: { text: "Growth of $10,000", font: { color: "#1f2937" } },
        font: { color: "#6b7280", family: "Arial, Helvetica, sans-serif" },
        xaxis: { title: { text: "" }, gridcolor: "#e5e7eb", linecolor: "#e5e7eb" },
        yaxis: { title: { text: "Value ($)" }, gridcolor: "#e5e7eb", linecolor: "#e5e7eb", tickprefix: "$" },
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        height: 360,
        margin: { t: 48, r: 24, b: 40, l: 64 },
        autosize: true,
        legend: { orientation: "h", y: -0.2 },
      }}
      style={{ width: "100%" }}
      useResizeHandler
      config={{ displayModeBar: false }}
    />
  );
}
