"use client";

import dynamic from "next/dynamic";
import type { ComponentProps } from "react";

// Plotly.js touches `window`/`document` at import time, which breaks
// server rendering — must be loaded client-only, hence next/dynamic with
// ssr: false rather than a plain top-level import.
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type PlotProps = ComponentProps<typeof Plot>;

export default function PlotlyChart(props: PlotProps) {
  return <Plot {...props} />;
}
