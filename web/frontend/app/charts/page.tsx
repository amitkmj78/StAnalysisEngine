"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import LinkedPriceChart from "@/components/charts/LinkedPriceChart";
import TickerSearchInput from "@/components/TickerSearchInput";
import { ApiError, deleteChartLayout, listChartLayouts, saveChartLayout } from "@/lib/api";
import type { ChartGridLayout, SavedChartLayout } from "@/lib/types";

// CHT-8: a grid of up to four charts. CHT-7: layouts saved per user and reloaded unchanged.
// Chart settings only. Nothing here places an order.

const SLOTS = 4;
const RANGES: ChartGridLayout["range"][] = ["1M", "6M", "1Y", "5Y"];
const DEFAULT: ChartGridLayout = {
  tickers: ["AAPL", "MSFT", "NVDA", "SPY"],
  range: "1Y",
  chart_type: "line",
  log_scale: false,
  linked_crosshair: true,
};

// ?tickers=AAPL,MSFT lets other pages (e.g. the portfolio) open the grid with those tickers in the slots.
function tickersFromQuery(value: string | null): string[] | null {
  if (!value) return null;
  const list = value
    .split(",")
    .map((t) => t.trim().toUpperCase())
    .filter((t) => /^[A-Z.\-]{1,10}$/.test(t))
    .slice(0, SLOTS);
  return list.length > 0 ? list : null;
}

export default function ChartGridPage() {
  return (
    <Suspense fallback={null}>
      <ChartGrid />
    </Suspense>
  );
}

function ChartGrid() {
  const searchParams = useSearchParams();
  const [tickers, setTickers] = useState<string[]>(() => {
    const fromQuery = tickersFromQuery(searchParams.get("tickers"));
    return fromQuery ? [...fromQuery, ...Array(SLOTS).fill("")].slice(0, SLOTS) : DEFAULT.tickers;
  });
  const [range, setRange] = useState<ChartGridLayout["range"]>(DEFAULT.range);
  const [chartType, setChartType] = useState<"line" | "candles">(DEFAULT.chart_type);
  const [logScale, setLogScale] = useState(DEFAULT.log_scale);
  const [linked, setLinked] = useState(DEFAULT.linked_crosshair);
  const [hoverDate, setHoverDate] = useState<string | null>(null);
  const [hoverSource, setHoverSource] = useState<number | null>(null);
  const [saved, setSaved] = useState<SavedChartLayout[]>([]);
  const [layoutName, setLayoutName] = useState("");
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listChartLayouts()
      .then((res) => {
        if (!cancelled) setSaved(res.layouts);
      })
      .catch(() => {
        if (!cancelled) setSaved([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function currentLayout(): ChartGridLayout {
    return {
      tickers: tickers.filter((t) => t.trim()),
      range,
      chart_type: chartType,
      log_scale: logScale,
      linked_crosshair: linked,
    };
  }

  function applyLayout(layout: ChartGridLayout) {
    setTickers([...layout.tickers, ...Array(SLOTS).fill("")].slice(0, SLOTS));
    setRange(layout.range);
    setChartType(layout.chart_type);
    setLogScale(layout.log_scale);
    setLinked(layout.linked_crosshair);
  }

  async function handleSave() {
    const name = layoutName.trim();
    if (!name) {
      setMessage("Give the layout a name first.");
      return;
    }
    try {
      await saveChartLayout(name, currentLayout());
      const res = await listChartLayouts();
      setSaved(res.layouts);
      setLayoutName("");
      setMessage(`Saved “${name}”.`);
    } catch (err) {
      setMessage(err instanceof ApiError ? err.message : "The layout could not be saved.");
    }
  }

  async function handleDelete(id: number) {
    try {
      await deleteChartLayout(id);
      setSaved((s) => s.filter((x) => x.id !== id));
    } catch (err) {
      setMessage(err instanceof ApiError ? err.message : "It could not be deleted.");
    }
  }

  function handleHover(date: string | null, slot: number) {
    setHoverDate(date);
    setHoverSource(date === null ? null : slot);
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">Chart grid</h1>
      <p className="mt-1 text-sm text-slate-500">
        Up to four charts side by side. With linked crosshairs on, hovering a date moves every chart to that date.
      </p>

      <div className="mt-6 flex flex-col gap-4 rounded-lg border border-slate-200 bg-white p-4">
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <div className="flex rounded-md border border-slate-200 p-0.5">
            {RANGES.map((r) => (
              <button
                key={r}
                type="button"
                onClick={() => setRange(r)}
                className={`rounded px-2.5 py-1 font-medium ${range === r ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-100"}`}
              >
                {r}
              </button>
            ))}
          </div>
          <div className="flex rounded-md border border-slate-200 p-0.5">
            {(["line", "candles"] as const).map((c) => (
              <button
                key={c}
                type="button"
                onClick={() => setChartType(c)}
                className={`rounded px-2.5 py-1 font-medium capitalize ${chartType === c ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-100"}`}
              >
                {c}
              </button>
            ))}
          </div>
          <label className="flex items-center gap-1.5 text-slate-600">
            <input type="checkbox" checked={logScale} onChange={(e) => setLogScale(e.target.checked)} />
            Log scale
          </label>
          <label className="flex items-center gap-1.5 text-slate-600">
            <input type="checkbox" checked={linked} onChange={(e) => setLinked(e.target.checked)} />
            Linked crosshairs
          </label>
        </div>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {Array.from({ length: SLOTS }).map((_, slot) => (
            <div key={slot} className="flex flex-col gap-1">
              <TickerSearchInput
                value={tickers[slot] ?? ""}
                onChange={(t) => setTickers(tickers.map((x, i) => (i === slot ? t : x)))}
                onSelect={(t) => setTickers(tickers.map((x, i) => (i === slot ? t.toUpperCase() : x)))}
                placeholder={`Chart ${slot + 1}: ticker or name`}
                className="input w-full py-1 text-xs"
                id={`chart-slot-${slot}`}
              />
            </div>
          ))}
        </div>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
        {Array.from({ length: SLOTS }).map((_, slot) => {
          const t = (tickers[slot] ?? "").trim().toUpperCase();
          return t ? (
            <LinkedPriceChart
              key={`${slot}-${t}-${range}-${chartType}-${logScale}-${linked}`}
              slot={slot}
              ticker={t}
              range={range}
              chartType={chartType}
              logScale={logScale}
              linked={linked}
              hoverDate={hoverDate}
              hoverSource={hoverSource}
              onHover={handleHover}
            />
          ) : (
            <div key={slot} className="flex h-64 items-center justify-center rounded-lg border border-dashed border-slate-300 text-sm text-slate-400">
              Empty chart slot
            </div>
          );
        })}
      </div>

      <section className="mt-8 rounded-lg border border-slate-200 bg-white p-5">
        <h2 className="text-sm font-semibold text-slate-900">Saved layouts</h2>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <input value={layoutName} onChange={(e) => setLayoutName(e.target.value)} placeholder="Layout name" className="input w-56 py-1 text-sm" aria-label="Layout name" />
          <button type="button" onClick={handleSave} className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-semibold text-white hover:bg-slate-800">
            Save this layout
          </button>
          {message && <span className="text-xs text-slate-500">{message}</span>}
        </div>
        {saved.length === 0 ? (
          <p className="mt-3 text-sm text-slate-500">No saved layouts yet.</p>
        ) : (
          <ul className="mt-3 divide-y divide-slate-100">
            {saved.map((s) => (
              <li key={s.id} className="flex items-center justify-between gap-3 py-2 text-sm">
                <span>
                  <span className="font-medium text-slate-800">{s.name}</span>
                  <span className="ml-2 text-xs text-slate-500">{s.layout.tickers.join(", ")} · {s.layout.range} · {s.layout.chart_type}</span>
                </span>
                <span className="flex gap-3 text-xs">
                  <button type="button" onClick={() => applyLayout(s.layout)} className="font-medium text-slate-700 hover:underline">
                    Load
                  </button>
                  <button type="button" onClick={() => handleDelete(s.id)} className="text-slate-500 hover:text-red-700">
                    Delete
                  </button>
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
