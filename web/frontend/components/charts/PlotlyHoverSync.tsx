"use client";

import { useEffect } from "react";

// CHT-8: moves a chart's crosshair to a shared date. Plotly is imported here, and this component is only
// loaded in the browser (ssr: false in LinkedPriceChart), so the server build never resolves plotly.js.

type Props = {
  divId: string;
  linked: boolean;
  dates: string[];
  hoverDate: string | null;
  hoverSource: number | null;
  slot: number;
};

function dayKey(date: string) {
  return date.slice(0, 10);
}

export default function PlotlyHoverSync({ divId, linked, dates, hoverDate, hoverSource, slot }: Props) {
  useEffect(() => {
    if (!linked || dates.length === 0 || hoverSource === slot) return;
    const el = document.getElementById(divId);
    if (!el) return;
    let cancelled = false;
    (async () => {
      // Same prebuilt bundle react-plotly.js uses, so the hover runs on the chart's own instance.
      const mod = await import("plotly.js/dist/plotly");
      const Plotly = (mod.default ?? mod) as unknown as {
        Fx: {
          hover: (gd: HTMLElement, pts: { curveNumber: number; pointNumber: number }[]) => void;
          unhover: (gd: HTMLElement) => void;
        };
      };
      if (cancelled) return;
      if (hoverDate === null) {
        Plotly.Fx.unhover(el);
        return;
      }
      const index = dates.indexOf(dayKey(hoverDate));
      if (index >= 0) Plotly.Fx.hover(el, [{ curveNumber: 0, pointNumber: index }]);
      else Plotly.Fx.unhover(el);
    })();
    return () => {
      cancelled = true;
    };
  }, [hoverDate, hoverSource, linked, dates, divId, slot]);

  return null;
}
