"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import { getMarketRegime } from "@/lib/api";
import type { MarketRegimeResponse } from "@/lib/types";

// REG-1: site-wide regime summary. See services/market_regime_service.py's
// module docstring for why this ships despite a failed validation gate --
// the disclosure below is the non-negotiable condition of that decision
// and must stay visible or one click away, never collapsed by default
// into something a reader could miss entirely.
function regimeClass(regime: string | null | undefined): string {
  if (regime === "Risk-On") return "border-emerald-200 bg-emerald-50 text-emerald-800";
  if (regime === "Constructive") return "border-teal-200 bg-teal-50 text-teal-800";
  if (regime === "Cautious") return "border-amber-200 bg-amber-50 text-amber-800";
  if (regime === "Risk-Off") return "border-red-200 bg-red-50 text-red-800";
  return "border-slate-200 bg-slate-50 text-slate-700"; // Neutral, or unknown
}

function pct(value: number | null | undefined, digits = 1): string {
  return value === null || value === undefined ? "—" : `${value.toFixed(digits)}%`;
}

export default function RegimeBanner() {
  const [data, setData] = useState<MarketRegimeResponse | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getMarketRegime()
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch(() => {
        if (!cancelled) setData(null);
      })
      .finally(() => {
        if (!cancelled) setLoaded(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (!loaded || !data || !data.available) {
    return null; // not enabled yet, no data yet, or the fetch failed -- nothing honest to show
  }

  const { breadth, volatility, trend, risk_appetite } = data.components ?? {};
  const readings = data.dimensions
    ? [
        { key: "rates", label: "Rates", reading: data.dimensions.rates },
        { key: "credit", label: "Credit", reading: data.dimensions.credit },
        { key: "breadth", label: "Breadth", reading: data.dimensions.breadth },
        { key: "divergence", label: "Divergence", reading: data.dimensions.divergence },
        { key: "risk_appetite", label: "Risk appetite", reading: data.dimensions.risk_appetite },
      ].filter((r) => r.reading?.text)
    : [];

  return (
    <div className={`border-b px-4 py-2 text-xs ${regimeClass(data.regime)}`}>
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-4 gap-y-1">
        <span className="font-semibold">
          Market regime: {data.regime ?? "Unknown"}
        </span>
        <span className="text-[11px] opacity-80">as of {data.as_of_date}</span>
        <span className="text-[11px] opacity-80">
          Breadth {pct(breadth?.pct_above_50dma)} above 50-DMA
        </span>
        <span className="text-[11px] opacity-80">
          Volatility VIX {volatility?.vix ?? "—"}
          {volatility?.inverted ? " (term inverted)" : ""}
        </span>
        <span className="text-[11px] opacity-80">
          Trend {trend?.above_50dma === null || trend?.above_50dma === undefined
            ? "—"
            : trend.above_50dma
              ? "SPY above its 50-DMA"
              : "SPY below its 50-DMA"}
        </span>
        <span className="text-[11px] opacity-80">
          Risk appetite {pct(risk_appetite?.momentum_pct)}
        </span>
        <button
          type="button"
          onClick={() => setExpanded((e) => !e)}
          className="ml-auto shrink-0 underline decoration-dotted underline-offset-2"
        >
          {expanded ? "Hide readings" : "⚠ Unvalidated signal — see readings"}
        </button>
      </div>
      {expanded && (
        <div className="mx-auto mt-1 max-w-6xl text-[11px] leading-relaxed opacity-90">
          <p>
            {data.disclosure}{" "}
            <Link href="/regime/methodology" className="underline decoration-dotted underline-offset-2">
              How it was tested
            </Link>
          </p>
          {(data.takeaways?.length ?? 0) > 0 && (
            <div className="mt-2">
              <p className="font-semibold">What this suggests</p>
              <ul className="mt-1 flex list-disc flex-col gap-1 pl-4">
                {data.takeaways?.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </div>
          )}
          {readings.length > 0 && (
            <ul className="mt-2 flex flex-col gap-1">
              {readings.map((r) => (
                <li
                  key={r.key}
                  className={r.key === "divergence" && (r.reading as { flag?: boolean | null } | undefined)?.flag ? "font-semibold" : undefined}
                >
                  <span className="font-medium">{r.label}:</span> {r.reading?.text}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
      {data.regime === "Risk-Off" && (
        // REG-3: a site-wide echo of the same per-signal caution note on
        // stock/[ticker]/page.tsx's ScoreCard -- a condition, not an
        // instruction (no "sell"/"buy" language).
        <p className="mx-auto mt-1 max-w-6xl text-[11px] font-medium leading-relaxed">
          Signal confidence may read differently than usual across this app while the market is in a Risk-Off
          regime.
        </p>
      )}
    </div>
  );
}
