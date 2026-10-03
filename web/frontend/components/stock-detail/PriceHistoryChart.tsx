"use client";

import { useEffect, useMemo, useState } from "react";
import type { Data, Layout } from "plotly.js";
import PlotlyChart from "@/components/PlotlyChart";
import { getStockPriceHistory } from "@/lib/api";
import type {
  StockPriceHistoryRange,
  StockPriceHistoryResponse,
  StockPriceHistoryRow,
  StockSignalHistoryResponse,
} from "@/lib/types";

const RANGES: StockPriceHistoryRange[] = ["1D", "5D", "1M", "6M", "1Y", "5Y"];

type PastEarnings = { date: string; reported_eps: number | null };
type Dividend = { date: string; amount: number };
type SignalChange = { date: string; label: string };

type LineDash = "solid" | "dash" | "dot";

type ChartType = "candles" | "ohlc" | "heikin" | "line" | "area";
const CHART_TYPES: { key: ChartType; label: string }[] = [
  { key: "candles", label: "Candles" },
  { key: "ohlc", label: "OHLC" },
  { key: "heikin", label: "Heikin Ashi" },
  { key: "line", label: "Line" },
  { key: "area", label: "Area" },
];

// CHT-3: overlays draw on the price panel; panels get their own sub-chart.
const OVERLAYS = [
  { key: "sma20", label: "SMA 20" },
  { key: "sma50", label: "SMA 50" },
  { key: "sma200", label: "SMA 200" },
  { key: "ema20", label: "EMA 20" },
  { key: "bollinger", label: "Bollinger 20, 2" },
  { key: "vwap", label: "VWAP" },
] as const;
const PANELS = [
  { key: "volume", label: "Volume" },
  { key: "rsi", label: "RSI 14" },
  { key: "macd", label: "MACD 12, 26, 9" },
] as const;

type Toggle = (typeof OVERLAYS)[number]["key"] | (typeof PANELS)[number]["key"];

// CHT-4: GICS sector name (as the stock detail returns it) to its SPDR sector ETF.
export const SECTOR_ETF_BY_NAME: Record<string, string> = {
  "Information Technology": "XLK",
  "Health Care": "XLV",
  Financials: "XLF",
  "Consumer Discretionary": "XLY",
  "Communication Services": "XLC",
  Industrials: "XLI",
  "Consumer Staples": "XLP",
  Energy: "XLE",
  Utilities: "XLU",
  "Real Estate": "XLRE",
  Materials: "XLB",
};

const UP = "#059669";
const DOWN = "#DC2626";
const PRICE_BLUE = "#1F4FD1";

type Candle = { date: string; open: number; high: number; low: number; close: number };

// Heikin Ashi smooths each bar against the previous one: HA close is the
// average of the bar, HA open the midpoint of the previous HA bar.
function heikinAshi(rows: Candle[]): Candle[] {
  const out: Candle[] = [];
  rows.forEach((r, i) => {
    const close = (r.open + r.high + r.low + r.close) / 4;
    const open = i === 0 ? (r.open + r.close) / 2 : (out[i - 1].open + out[i - 1].close) / 2;
    out.push({
      date: r.date,
      open,
      high: Math.max(r.high, open, close),
      low: Math.min(r.low, open, close),
      close,
    });
  });
  return out;
}

function lineTrace(
  name: string,
  x: string[],
  y: (number | null)[],
  color: string,
  opts: { dash?: LineDash; width?: number; xaxis?: string; yaxis?: string } = {},
) {
  return {
    x,
    y,
    type: "scatter" as const,
    mode: "lines" as const,
    name,
    line: { color, width: opts.width ?? 1.5, dash: opts.dash ?? "solid" as const },
    xaxis: opts.xaxis ?? "x",
    yaxis: opts.yaxis ?? "y",
  };
}

// DET-4: earnings/dividend/signal-change markers, snapped onto the price
// line at their own date's close so they sit on the chart rather than
// floating at an arbitrary height. Skipped on the 1D range -- its x-axis
// is full timestamps within a single day, and a date-only event has
// nothing meaningful to snap to there.
function buildMarkerTrace(
  events: { date: string; hovertext: string }[],
  closeByDate: Map<string, number>,
  opts: { name: string; symbol: string; color: string },
) {
  const points = events
    .map((e) => ({ ...e, close: closeByDate.get(e.date) }))
    .filter((e): e is { date: string; hovertext: string; close: number } => e.close !== undefined);
  if (points.length === 0) return null;
  return {
    x: points.map((p) => p.date),
    y: points.map((p) => p.close),
    text: points.map((p) => p.hovertext),
    hovertemplate: "%{text}<extra></extra>",
    type: "scatter" as const,
    mode: "markers" as const,
    name: opts.name,
    marker: { symbol: opts.symbol, size: 10, color: opts.color, line: { color: "#ffffff", width: 1 } },
  };
}

export default function PriceHistoryChart({
  ticker,
  data,
  range,
  onRangeChange,
  loading,
  pastEarnings = [],
  recentDividends = [],
  signalChanges = [],
  signalHistory = null,
  sectorEtf = null,
}: {
  ticker: string;
  data: StockPriceHistoryResponse | null;
  range: StockPriceHistoryRange;
  onRangeChange: (range: StockPriceHistoryRange) => void;
  loading: boolean;
  pastEarnings?: PastEarnings[];
  recentDividends?: Dividend[];
  signalChanges?: SignalChange[];
  signalHistory?: StockSignalHistoryResponse | null;
  sectorEtf?: string | null;
}) {
  const [chartType, setChartType] = useState<ChartType>("candles");
  const [logScale, setLogScale] = useState(false);
  const [compare, setCompare] = useState(false);
  const [showScore, setShowScore] = useState(true);
  const [active, setActive] = useState<Set<Toggle>>(new Set());
  const [compareData, setCompareData] = useState<{
    spy: StockPriceHistoryResponse | null;
    sector: StockPriceHistoryResponse | null;
  } | null>(null);

  const isDaily = range !== "1D";
  const history: StockPriceHistoryRow[] = useMemo(() => data?.history ?? [], [data]);
  const indicators = isDaily ? data?.indicators ?? null : null;

  // Compare mode is only meaningful on daily bars; it swaps the price panel to % change.
  const compareActive = compare && isDaily;

  useEffect(() => {
    if (!compareActive) return;
    let cancelled = false;
    Promise.all([
      getStockPriceHistory("SPY", range).catch(() => null),
      sectorEtf ? getStockPriceHistory(sectorEtf, range).catch(() => null) : Promise.resolve(null),
    ]).then(([spy, sector]) => {
      if (!cancelled) setCompareData({ spy, sector });
    });
    return () => {
      cancelled = true;
    };
  }, [compareActive, range, sectorEtf]);

  const closeByDate = useMemo(() => {
    const m = new Map<string, number>();
    if (isDaily) for (const p of history) m.set(p.date.slice(0, 10), p.close);
    return m;
  }, [history, isDaily]);

  const ohlcRows: Candle[] = useMemo(
    () =>
      history.flatMap((p) =>
        p.open !== null && p.high !== null && p.low !== null
          ? [{ date: p.date, open: p.open, high: p.high, low: p.low, close: p.close }]
          : [],
      ),
    [history],
  );

  const hasOhlc = ohlcRows.length > 0;
  const candleType = chartType === "candles" || chartType === "ohlc" || chartType === "heikin";
  const fallbackToLine = !compareActive && candleType && !hasOhlc;
  const drawType: ChartType = compareActive || fallbackToLine ? "line" : chartType;

  const signalRows = useMemo(
    () => [...(signalHistory?.history ?? [])].sort((a, b) => a.as_of_date.localeCompare(b.as_of_date)),
    [signalHistory],
  );
  const hasScore = isDaily && signalRows.length >= 2;

  const hasVolume = history.some((p) => p.volume !== null);
  const panelOn = (key: (typeof PANELS)[number]["key"]) => !compareActive && isDaily && active.has(key);

  // Stacked sub-panels, top to bottom, sharing the price panel's date axis.
  const subPanels: ("volume" | "rsi" | "macd" | "score")[] = [];
  if (panelOn("volume") && hasVolume) subPanels.push("volume");
  if (panelOn("rsi") && indicators) subPanels.push("rsi");
  if (panelOn("macd") && indicators) subPanels.push("macd");
  if (hasScore && showScore && !compareActive) subPanels.push("score");

  const toggle = (key: Toggle) =>
    setActive((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });

  // CHT-4: rebase stock, SPY and sector ETF to 0% at the first date all three share.
  const compareSeries = useMemo(() => {
    if (!compareActive || !compareData?.spy) return null;
    const toMap = (rows: { date: string; close: number }[]) =>
      new Map(rows.map((p) => [p.date.slice(0, 10), p.close] as const));
    const series: { name: string; color: string; dash: LineDash; map: Map<string, number> }[] = [
      { name: ticker, color: PRICE_BLUE, dash: "solid", map: toMap(history) },
      { name: "SPY", color: "#64748B", dash: "dash", map: toMap(compareData.spy?.history ?? []) },
    ];
    if (compareData.sector && sectorEtf) {
      series.push({ name: sectorEtf, color: "#0D9488", dash: "dot", map: toMap(compareData.sector.history) });
    }
    if (series.some((s) => s.map.size === 0)) return { dates: [] as string[], traces: [] };
    const common = [...series[0].map.keys()].filter((d) => series.every((s) => s.map.has(d))).sort();
    if (common.length === 0) return { dates: [] as string[], traces: [] };
    const base = common[0];
    return {
      dates: common,
      traces: series.map((s) => ({
        ...s,
        y: common.map((d) => (s.map.get(d)! / s.map.get(base)! - 1) * 100),
      })),
    };
  }, [compareActive, compareData, ticker, history, sectorEtf]);

  const compareWaiting = compareActive && !compareData;

  // Price-panel traces.
  const priceTraces: Data[] = [];
  if (compareSeries) {
    for (const s of compareSeries.traces) {
      priceTraces.push(lineTrace(s.name, compareSeries.dates, s.y, s.color, { dash: s.dash, width: s.name === ticker ? 2 : 1.5 }));
    }
  } else if (drawType === "candles" || drawType === "ohlc" || drawType === "heikin") {
    const rows = drawType === "heikin" ? heikinAshi(ohlcRows) : ohlcRows;
    priceTraces.push({
      x: rows.map((r) => r.date),
      open: rows.map((r) => r.open),
      high: rows.map((r) => r.high),
      low: rows.map((r) => r.low),
      close: rows.map((r) => r.close),
      type: drawType === "ohlc" ? "ohlc" : "candlestick",
      name: drawType === "heikin" ? "Heikin Ashi" : "Price",
      increasing: { line: { color: UP } },
      decreasing: { line: { color: DOWN } },
      xaxis: "x",
      yaxis: "y",
    });
  } else {
    priceTraces.push({
      x: history.map((p) => p.date),
      y: history.map((p) => p.close),
      type: "scatter",
      mode: "lines",
      name: "Close",
      line: { color: PRICE_BLUE, width: 2 },
      fill: drawType === "area" && !logScale ? "tozeroy" : "none",
      fillcolor: "rgba(31, 79, 209, 0.12)",
      xaxis: "x",
      yaxis: "y",
    });
  }

  // Overlays (CHT-3). Null entries are warm-up bars and simply leave gaps.
  if (indicators && !compareSeries) {
    const dates = history.map((p) => p.date);
    const ov = (key: (typeof OVERLAYS)[number]["key"]) => active.has(key);
    if (ov("sma20")) priceTraces.push(lineTrace("SMA 20", dates, indicators.sma_20, "#F59E0B"));
    if (ov("sma50")) priceTraces.push(lineTrace("SMA 50", dates, indicators.sma_50, "#0EA5E9"));
    if (ov("sma200")) priceTraces.push(lineTrace("SMA 200", dates, indicators.sma_200, "#475569"));
    if (ov("ema20")) priceTraces.push(lineTrace("EMA 20", dates, indicators.ema_20, "#EC4899", { dash: "dash" }));
    if (ov("bollinger")) {
      priceTraces.push(lineTrace("Bollinger upper", dates, indicators.bollinger.upper, "#94A3B8", { dash: "dot", width: 1 }));
      priceTraces.push(lineTrace("Bollinger lower", dates, indicators.bollinger.lower, "#94A3B8", { dash: "dot", width: 1 }));
    }
    if (ov("vwap")) priceTraces.push(lineTrace("VWAP", dates, indicators.vwap, "#A855F7"));
  }

  // DIF-1: short-term signal markers, coloured by realised outcome.
  // Short-term only: the long-term signal would double the marker count.
  if (isDaily && !compareSeries && signalRows.length > 0) {
    const pts = signalRows.filter((s) => closeByDate.has(s.as_of_date));
    if (pts.length > 0) {
      const outcomeColor = (o: { outcome: "hit" | "miss" | null } | null) =>
        o?.outcome === "hit" ? UP : o?.outcome === "miss" ? DOWN : "#94A3B8";
      const symbolFor = (sig: string) => (sig === "Buy" ? "triangle-up" : sig === "Trim" ? "triangle-down" : "circle");
      const outcomeText = (o: { outcome: "hit" | "miss" | null; realized_return_pct: number } | null) =>
        o === null ? "no outcome yet" : o.outcome === null ? "pending" : `${o.outcome}, ${o.realized_return_pct.toFixed(2)}% realised`;
      priceTraces.push({
        x: pts.map((s) => s.as_of_date),
        y: pts.map((s) => closeByDate.get(s.as_of_date) ?? null),
        text: pts.map((s) => `Short-term ${s.short_signal} on ${s.as_of_date}: ${outcomeText(s.short_outcome)}`),
        hovertemplate: "%{text}<extra></extra>",
        type: "scatter",
        mode: "markers",
        name: "Short-term signal",
        marker: {
          symbol: pts.map((s) => symbolFor(s.short_signal)),
          size: 9,
          color: pts.map((s) => outcomeColor(s.short_outcome)),
          line: { color: "#ffffff", width: 1 },
        },
        xaxis: "x",
        yaxis: "y",
      });
    }
  }

  // Existing DET-4 event markers, price panel only.
  if (isDaily && !compareSeries) {
    const extra = [
      buildMarkerTrace(
        pastEarnings.map((e) => ({
          date: e.date,
          hovertext: e.reported_eps !== null ? `Earnings ${e.date}: EPS $${e.reported_eps.toFixed(2)}` : `Earnings reported ${e.date}`,
        })),
        closeByDate,
        { name: "Earnings", symbol: "diamond", color: "#7C3AED" },
      ),
      buildMarkerTrace(
        recentDividends.map((d) => ({ date: d.date, hovertext: `Dividend ${d.date}: $${d.amount.toFixed(4)}` })),
        closeByDate,
        { name: "Dividends", symbol: "circle", color: "#059669" },
      ),
      buildMarkerTrace(
        signalChanges.map((s) => ({ date: s.date, hovertext: `Signal change ${s.date}: ${s.label}` })),
        closeByDate,
        { name: "Signal Change", symbol: "triangle-up", color: "#D97706" },
      ),
    ].filter((t): t is NonNullable<typeof t> => t !== null);
    priceTraces.push(...extra);
  }

  // Sub-panel traces and layout. Panels share the price panel's x-axis via `matches`.
  const subTraces: Data[] = [];
  const subAxisIndex = new Map<string, number>();
  subPanels.forEach((p, i) => subAxisIndex.set(p, i + 2));
  const axisOf = (p: string) => subAxisIndex.get(p) ?? 1;

  if (subAxisIndex.has("volume")) {
    const n = axisOf("volume");
    subTraces.push({
      x: history.map((p) => p.date),
      y: history.map((p) => p.volume),
      type: "bar",
      name: "Volume",
      marker: {
        color: history.map((p, i) => (i > 0 && p.close >= history[i - 1].close ? UP : DOWN)),
      },
      xaxis: `x${n}`,
      yaxis: `y${n}`,
    });
  }
  if (subAxisIndex.has("rsi") && indicators) {
    const n = axisOf("rsi");
    subTraces.push(lineTrace("RSI 14", history.map((p) => p.date), indicators.rsi_14, "#1F4FD1", { xaxis: `x${n}`, yaxis: `y${n}` }));
  }
  if (subAxisIndex.has("macd") && indicators) {
    const n = axisOf("macd");
    const dates = history.map((p) => p.date);
    subTraces.push({
      x: dates,
      y: indicators.macd.histogram,
      type: "bar",
      name: "Histogram",
      marker: { color: indicators.macd.histogram.map((v) => (v !== null && v >= 0 ? UP : DOWN)) },
      xaxis: `x${n}`,
      yaxis: `y${n}`,
    });
    subTraces.push(lineTrace("MACD", dates, indicators.macd.macd, "#1F4FD1", { xaxis: `x${n}`, yaxis: `y${n}` }));
    subTraces.push(lineTrace("Signal", dates, indicators.macd.signal, "#F59E0B", { xaxis: `x${n}`, yaxis: `y${n}` }));
  }
  if (subAxisIndex.has("score")) {
    const n = axisOf("score");
    const dates = signalRows.map((s) => s.as_of_date);
    subTraces.push(lineTrace("Short-term score", dates, signalRows.map((s) => s.short_score), "#1F4FD1", { xaxis: `x${n}`, yaxis: `y${n}` }));
    subTraces.push(lineTrace("Long-term score", dates, signalRows.map((s) => s.long_score), "#7C3AED", { xaxis: `x${n}`, yaxis: `y${n}` }));
  }

  const gap = 0.05;
  const priceShare = subPanels.length > 0 ? 0.55 : 1;
  const subShare = subPanels.length > 0 ? (1 - priceShare - gap * subPanels.length) / subPanels.length : 0;
  // Axis keys are numbered per panel (xaxis2, yaxis3, ...), so this stays a plain record.
  const layoutAxes: Record<string, unknown> = {
    xaxis: {
      anchor: "y",
      rangeslider: { visible: false },
      showticklabels: subPanels.length === 0,
      ...(range === "1D" ? { tickformat: "%-I:%M %p" } : {}),
    },
    yaxis: {
      domain: [1 - priceShare, 1],
      anchor: "x",
      title: { text: compareSeries ? "% change" : "USD" },
      type: logScale && !compareSeries ? "log" : "linear",
      ticksuffix: compareSeries ? "%" : "",
      zeroline: compareSeries !== null,
    },
  };
  subPanels.forEach((p, i) => {
    const n = i + 2;
    const top = 1 - priceShare - gap - i * (subShare + gap);
    const label =
      p === "volume" ? "Volume" : p === "rsi" ? "RSI" : p === "macd" ? "MACD" : "Score";
    layoutAxes[`xaxis${n}`] = {
      anchor: `y${n}`,
      matches: "x",
      rangeslider: { visible: false },
      showticklabels: i === subPanels.length - 1,
    };
    layoutAxes[`yaxis${n}`] = {
      domain: [top - subShare, top],
      anchor: `x${n}`,
      title: { text: label },
      ...(p === "rsi" ? { range: [0, 100], tickvals: [30, 70] } : {}),
    };
  });
  const chartHeight = 320 + subPanels.length * 130;

  const showNotes = !isDaily || active.size > 0 || subPanels.length > 0 || compare;

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold text-slate-900">{ticker} Price</h3>
        <div className="flex flex-wrap gap-1">
          {RANGES.map((r) => (
            <button
              key={r}
              onClick={() => onRangeChange(r)}
              className={`rounded-md px-2 py-1 text-xs font-medium ${
                r === range ? "bg-slate-900 text-white" : "text-slate-500 hover:bg-slate-100"
              }`}
            >
              {r}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
        {CHART_TYPES.map((c) => (
          <button
            key={c.key}
            onClick={() => setChartType(c.key)}
            aria-pressed={chartType === c.key}
            disabled={compareActive}
            className={`rounded-md border px-2 py-1 font-medium disabled:opacity-40 ${
              chartType === c.key && !compareActive
                ? "border-slate-900 bg-slate-900 text-white"
                : "border-slate-200 text-slate-600 hover:bg-slate-50"
            }`}
          >
            {c.label}
          </button>
        ))}
        <button
          onClick={() => setLogScale((v) => !v)}
          aria-pressed={logScale}
          disabled={compareActive}
          className={`rounded-md border px-2 py-1 font-medium disabled:opacity-40 ${
            logScale ? "border-slate-900 bg-slate-900 text-white" : "border-slate-200 text-slate-600 hover:bg-slate-50"
          }`}
        >
          Log scale
        </button>
        <button
          onClick={() => setCompare((v) => !v)}
          aria-pressed={compare}
          disabled={!isDaily}
          className={`rounded-md border px-2 py-1 font-medium disabled:opacity-40 ${
            compare ? "border-slate-900 bg-slate-900 text-white" : "border-slate-200 text-slate-600 hover:bg-slate-50"
          }`}
        >
          Compare SPY{sectorEtf ? ` and ${sectorEtf}` : ""}
        </button>
      </div>

      {isDaily && (
        <div className="mt-2 flex flex-wrap items-center gap-1.5 text-xs">
          {OVERLAYS.map((o) => (
            <button
              key={o.key}
              onClick={() => toggle(o.key)}
              aria-pressed={active.has(o.key)}
              disabled={!indicators || compareActive}
              className={`rounded-full border px-2 py-0.5 disabled:opacity-40 ${
                active.has(o.key) ? "border-blue-600 bg-blue-50 text-blue-700" : "border-slate-200 text-slate-500 hover:bg-slate-50"
              }`}
            >
              {o.label}
            </button>
          ))}
          {PANELS.map((p) => (
            <button
              key={p.key}
              onClick={() => toggle(p.key)}
              aria-pressed={active.has(p.key)}
              disabled={compareActive || (p.key === "volume" ? !hasVolume : !indicators)}
              className={`rounded-full border px-2 py-0.5 disabled:opacity-40 ${
                active.has(p.key) ? "border-blue-600 bg-blue-50 text-blue-700" : "border-slate-200 text-slate-500 hover:bg-slate-50"
              }`}
            >
              {p.label}
            </button>
          ))}
          {hasScore && (
            <button
              onClick={() => setShowScore((v) => !v)}
              aria-pressed={showScore}
              disabled={compareActive}
              className={`rounded-full border px-2 py-0.5 disabled:opacity-40 ${
                showScore ? "border-blue-600 bg-blue-50 text-blue-700" : "border-slate-200 text-slate-500 hover:bg-slate-50"
              }`}
            >
              Score history
            </button>
          )}
        </div>
      )}

      {loading && <p className="mt-4 text-sm text-slate-500">Loading…</p>}
      {!loading && data && data.history.length === 0 && (
        <p className="mt-4 text-sm text-slate-400">No price history available for {ticker}.</p>
      )}
      {compareWaiting && !loading && <p className="mt-4 text-sm text-slate-500">Loading comparison…</p>}
      {!loading && data && data.history.length > 0 && !compareWaiting && (
        <PlotlyChart
          data={[...priceTraces, ...subTraces]}
          layout={{
            ...(layoutAxes as Partial<Layout>),
            paper_bgcolor: "#ffffff",
            plot_bgcolor: "#ffffff",
            height: chartHeight,
            margin: { t: 16, r: 24, b: 32, l: 56 },
            autosize: true,
            hovermode: "x unified",
            showlegend: true,
            legend: { orientation: "h", y: -0.15 },
          }}
          style={{ width: "100%" }}
          useResizeHandler
          config={{ displayModeBar: false }}
        />
      )}

      {!loading && compareSeries && compareSeries.dates.length === 0 && (
        <p className="mt-2 text-xs text-slate-400">No shared dates between {ticker}, SPY and the sector ETF for this range.</p>
      )}
      {!loading && fallbackToLine && (
        <p className="mt-2 text-xs text-slate-400">No open, high or low prices for this range, so the chart shows the close line.</p>
      )}
      {showNotes && (
        <div className="mt-2 space-y-1 text-xs text-slate-400">
          {!isDaily && <p>Indicators, signal markers, score history and comparison are available on the daily ranges.</p>}
          {isDaily && (active.size > 0 || subPanels.length > 0) && !compareActive && (
            <p>Indicators are calculated over the bars shown, so long averages stay blank until enough bars exist.</p>
          )}
          {compareActive && (
            <p>Comparison rebases each line to 0% at the first date all of them share. Overlays and signal markers are hidden in this mode.</p>
          )}
          {isDaily && !compareActive && signalRows.length > 0 && (
            <p>
              Short-term signal markers start {signalRows[0].as_of_date}. Green = hit, red = miss, grey = pending. Hold and Trim are marked too.
            </p>
          )}
        </div>
      )}
      {isDaily && !compareActive && (
        <p className="mt-2 text-xs">
          <a
            href={`https://www.tradingview.com/symbols/${encodeURIComponent(ticker)}/`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-700 underline-offset-2 hover:underline"
          >
            Open {ticker} in TradingView
          </a>
        </p>
      )}
    </div>
  );
}
