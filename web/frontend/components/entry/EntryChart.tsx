import PlotlyChart from "@/components/PlotlyChart";
import type { EntryHistory, EntryPlan } from "@/lib/types";

const LEVELS: Array<{ key: keyof EntryPlan; label: string; color: string }> = [
  { key: "ideal_entry_low", label: "Entry Low", color: "#138A72" },
  { key: "ideal_entry_high", label: "Entry High", color: "#2FAE8F" },
  { key: "breakout_entry", label: "Breakout", color: "#C76B17" },
  { key: "stop_loss", label: "Stop", color: "#B23A48" },
  { key: "first_target", label: "Target", color: "#6B8E23" },
];

export default function EntryChart({ plan, history }: { plan: EntryPlan; history: EntryHistory }) {
  const shapes = LEVELS.map((level) => ({
    type: "line" as const,
    xref: "paper" as const,
    x0: 0,
    x1: 1,
    y0: plan[level.key] as number,
    y1: plan[level.key] as number,
    line: { color: level.color, dash: "dash" as const, width: 1.5 },
  }));

  const annotations = LEVELS.map((level) => ({
    xref: "paper" as const,
    x: 0,
    xanchor: "left" as const,
    y: plan[level.key] as number,
    text: level.label,
    showarrow: false,
    font: { color: level.color, size: 11 },
    yshift: 8,
  }));

  return (
    <PlotlyChart
      data={[
        {
          x: history.dates,
          y: history.close,
          type: "scatter",
          mode: "lines",
          name: "Close",
          line: { color: "#1c3556", width: 2.5 },
        },
      ]}
      layout={{
        title: { text: `${plan.ticker} Price and Entry Levels` },
        xaxis: { title: { text: "Date" } },
        yaxis: { title: { text: "Price" } },
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        height: 460,
        margin: { t: 48, r: 24, b: 40, l: 56 },
        autosize: true,
        shapes,
        annotations,
      }}
      style={{ width: "100%" }}
      useResizeHandler
      config={{ displayModeBar: false }}
    />
  );
}
