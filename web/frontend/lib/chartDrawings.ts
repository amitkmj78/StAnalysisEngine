// CHT-5: chart drawings. A drawing is stored as points (date or timestamp label, price). This module turns the saved
// drawings into Plotly shapes and annotations so the stock chart can redraw them exactly.

import type { Annotations, Shape } from "plotly.js";

export type DrawingKind = "trend" | "horizontal" | "rectangle" | "fibonacci" | "text";

export interface DrawingPoint {
  x: string;
  y: number;
}

export interface ChartDrawing {
  id: number;
  ticker: string;
  kind: DrawingKind;
  points: DrawingPoint[];
  text: string | null;
}

export const POINTS_PER_KIND: Record<DrawingKind, number> = {
  trend: 2,
  horizontal: 1,
  rectangle: 2,
  fibonacci: 2,
  text: 1,
};

// Standard Fibonacci retracement ratios, measured from the high of the two points down to their low.
export const FIB_RATIOS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];

export function fibonacciLevels(a: DrawingPoint, b: DrawingPoint): { ratio: number; price: number }[] {
  const high = Math.max(a.y, b.y);
  const low = Math.min(a.y, b.y);
  return FIB_RATIOS.map((ratio) => ({ ratio, price: high - (high - low) * ratio }));
}

const COLOR = "#4338ca";
const FIB_COLOR = "#b45309";

export function drawingShapes(drawings: ChartDrawing[]): { shapes: Partial<Shape>[]; annotations: Partial<Annotations>[] } {
  const shapes: Partial<Shape>[] = [];
  const annotations: Partial<Annotations>[] = [];

  for (const d of drawings) {
    const [p1, p2] = d.points;
    if (d.kind === "trend" && p2) {
      shapes.push({ type: "line", xref: "x", yref: "y", x0: p1.x, y0: p1.y, x1: p2.x, y1: p2.y, line: { color: COLOR, width: 2 } });
    } else if (d.kind === "horizontal") {
      shapes.push({ type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: p1.y, y1: p1.y, line: { color: COLOR, width: 1.5, dash: "dot" } });
      annotations.push({ xref: "paper", x: 1, xanchor: "left", yref: "y", y: p1.y, text: p1.y.toFixed(2), showarrow: false, font: { size: 10, color: COLOR } });
    } else if (d.kind === "rectangle" && p2) {
      shapes.push({
        type: "rect", xref: "x", yref: "y", x0: p1.x, y0: p1.y, x1: p2.x, y1: p2.y,
        line: { color: COLOR, width: 1 }, fillcolor: "rgba(67,56,202,0.08)",
      });
    } else if (d.kind === "fibonacci" && p2) {
      for (const level of fibonacciLevels(p1, p2)) {
        shapes.push({ type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: level.price, y1: level.price, line: { color: FIB_COLOR, width: 1 } });
        annotations.push({
          xref: "paper", x: 1, xanchor: "left", yref: "y", y: level.price, showarrow: false,
          text: `${(level.ratio * 100).toFixed(1)}%  ${level.price.toFixed(2)}`, font: { size: 10, color: FIB_COLOR },
        });
      }
    } else if (d.kind === "text") {
      annotations.push({
        xref: "x", x: p1.x, yref: "y", y: p1.y, text: d.text ?? "", showarrow: false,
        font: { size: 11, color: "#1e1b4b" }, bgcolor: "rgba(255,255,255,0.85)", bordercolor: COLOR, borderwidth: 1, borderpad: 3,
      });
    }
  }
  return { shapes, annotations };
}
