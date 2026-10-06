"use client";

import { Fragment, useEffect, useMemo, useState } from "react";
import type { Data, Layout, Shape } from "plotly.js";
import MetricLabel from "@/components/MetricLabel";
import PlotlyChart from "@/components/PlotlyChart";
import { CHART_CONTROL_INFO } from "@/components/stock-detail/chartControlInfo";
import { ApiError, clearChartDrawings, createChartDrawing, deleteChartDrawing, getStockPriceHistory, listChartDrawings } from "@/lib/api";
import { POINTS_PER_KIND, drawingShapes } from "@/lib/chartDrawings";
import type { ChartDrawing, DrawingKind, DrawingPoint } from "@/lib/chartDrawings";
import type {
  RegimeHistoryResponse,
  StockPriceHistoryInterval,
  StockPriceHistoryRange,
  StockPriceHistoryResponse,
  StockPriceHistoryRow,
  StockSignalHistoryResponse,
} from "@/lib/types";

const RANGES: StockPriceHistoryRange[] = ["1D", "5D", "1M", "6M", "1Y", "5Y"];

// CHT-6: bar sizes. "Auto" keeps the range's own bar size (5-minute bars on 1D, daily otherwise).
const BAR_SIZES: { key: StockPriceHistoryInterval | null; label: string }[] = [
  { key: null, label: "Auto" },
  { key: "1m", label: "1 min" },
  { key: "5m", label: "5 min" },
  { key: "15m", label: "15 min" },
  { key: "1h", label: "1 hour" },
  { key: "1D", label: "Day" },
  { key: "1W", label: "Week" },
  { key: "1M", label: "Month" },
];
const INTRADAY_BARS: StockPriceHistoryInterval[] = ["1m", "5m", "15m", "1h"];

// CHT-5: the drawing tools, in the order they appear on the chart toolbar.
const DRAW_TOOLS: { key: DrawingKind; label: string }[] = [
  { key: "trend", label: "Trend line" },
  { key: "horizontal", label: "Level" },
  { key: "rectangle", label: "Box" },
  { key: "fibonacci", label: "Fibonacci" },
  { key: "text", label: "Note" },
];

const TOOL_ICON: Record<DrawingKind, string> = {
  trend: "╱",
  horizontal: "―",
  rectangle: "▭",
  fibonacci: "F",
  text: "✎",
};

// Colour dot per drawing type, matching the lines drawn on the chart (indigo for most, amber for Fibonacci).
const KIND_DOT: Record<DrawingKind, string> = {
  trend: "bg-indigo-600",
  horizontal: "bg-indigo-400",
  rectangle: "bg-indigo-300",
  fibonacci: "bg-amber-600",
  text: "bg-emerald-600",
};

function describeDrawing(d: ChartDrawing): string {
  const name = DRAW_TOOLS.find((t) => t.key === d.kind)?.label ?? d.kind;
  if (d.kind === "text") return `${name}: ${d.text ?? ""}`;
  return `${name} ${d.points.map((p) => p.y.toFixed(2)).join(" → ")}`;
}

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

const REGIME_COLORS: Record<string, string> = {
  "Risk-On": "#16A34A",
  Constructive: "#86EFAC",
  Neutral: "#94A3B8",
  Cautious: "#F59E0B",
  "Risk-Off": "#DC2626",
};

const UP = "#059669";
const DOWN = "#DC2626";
const PRICE_BLUE = "#4f46e5";

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

// Info icon beside a control. Looked up by the control's label, lowercased.
function infoIcon(key: string) {
  const info = CHART_CONTROL_INFO[key.toLowerCase()];
  return info ? <MetricLabel info={info} /> : null;
}

export default function PriceHistoryChart({
  ticker,
  data,
  range,
  onRangeChange,
  interval = null,
  onIntervalChange,
  loading,
  pastEarnings = [],
  recentDividends = [],
  signalChanges = [],
  signalHistory = null,
  sectorEtf = null,
  regimeHistory = null,
  canDraw = false,
}: {
  ticker: string;
  data: StockPriceHistoryResponse | null;
  range: StockPriceHistoryRange;
  onRangeChange: (range: StockPriceHistoryRange) => void;
  interval?: StockPriceHistoryInterval | null;
  onIntervalChange?: (interval: StockPriceHistoryInterval | null) => void;
  loading: boolean;
  pastEarnings?: PastEarnings[];
  recentDividends?: Dividend[];
  signalChanges?: SignalChange[];
  signalHistory?: StockSignalHistoryResponse | null;
  sectorEtf?: string | null;
  regimeHistory?: RegimeHistoryResponse | null;
  // CHT-5: drawing tools, shown to signed-in users only (drawings are stored per user).
  canDraw?: boolean;
}) {
  const [chartType, setChartType] = useState<ChartType>("candles");
  const [logScale, setLogScale] = useState(false);
  const [compare, setCompare] = useState(false);
  const [shadeRegimes, setShadeRegimes] = useState(false);
  const [showScore, setShowScore] = useState(true);
  const [active, setActive] = useState<Set<Toggle>>(new Set());
  const [compareData, setCompareData] = useState<{
    spy: StockPriceHistoryResponse | null;
    sector: StockPriceHistoryResponse | null;
    extras: { ticker: string; data: StockPriceHistoryResponse | null }[];
  } | null>(null);
  // CHT-4: up to three extra tickers, typed by the user, compared alongside SPY and the sector ETF.
  const [extraInput, setExtraInput] = useState("");
  const [extraTickers, setExtraTickers] = useState<string[]>([]);
  const extraKey = extraTickers.join(",");

  // CHT-5: drawings for this stock, saved per user. Each tool takes one or two clicks on the price chart.
  const [drawings, setDrawings] = useState<ChartDrawing[]>([]);
  const [drawTool, setDrawTool] = useState<DrawingKind | null>(null);
  const [pending, setPending] = useState<DrawingPoint[]>([]);
  const [drawNote, setDrawNote] = useState<string | null>(null);
  const drawn = useMemo(() => drawingShapes(drawings), [drawings]);
  const pendingAnnotations = pending.map((p) => ({
    xref: "x" as const, x: p.x, yref: "y" as const, y: p.y, text: "●", showarrow: false, font: { size: 12, color: "#4338ca" },
  }));

  useEffect(() => {
    if (!canDraw) return;
    let cancelled = false;
    listChartDrawings(ticker)
      .then((res) => {
        if (!cancelled) setDrawings(res.drawings);
      })
      .catch(() => {
        if (!cancelled) setDrawNote("Your drawings could not be loaded.");
      });
    return () => {
      cancelled = true;
    };
  }, [canDraw, ticker]);

  function chooseTool(tool: DrawingKind | null) {
    setDrawTool(tool);
    setPending([]);
    setDrawNote(null);
  }

  async function handleDrawClick(e: { points?: Array<{ x?: unknown; y?: unknown; close?: unknown }> }) {
    if (!drawTool) return;
    const p = e.points?.[0];
    const x = typeof p?.x === "string" ? p.x : null;
    const price = typeof p?.y === "number" ? p.y : typeof p?.close === "number" ? p.close : null;
    if (x === null || price === null) {
      setDrawNote("Click on the price line or the bars.");
      return;
    }
    const next = [...pending, { x, y: Math.round(price * 100) / 100 }];
    if (next.length < POINTS_PER_KIND[drawTool]) {
      setPending(next);
      setDrawNote(null);
      return;
    }
    let text: string | undefined;
    if (drawTool === "text") {
      const entered = window.prompt("Text for the note (up to 200 characters)");
      if (!entered || !entered.trim()) {
        chooseTool(null);
        return;
      }
      text = entered.trim().slice(0, 200);
    }
    try {
      const created = await createChartDrawing(ticker, { kind: drawTool, points: next, text });
      setDrawings((prev) => [...prev, created]);
      setPending([]);
      setDrawNote(null);
    } catch (err) {
      setDrawNote(err instanceof ApiError ? err.message : "The drawing could not be saved.");
      setPending([]);
    }
  }

  async function handleDeleteDrawing(id: number) {
    try {
      await deleteChartDrawing(id);
      setDrawings((prev) => prev.filter((d) => d.id !== id));
    } catch {
      setDrawNote("That drawing could not be deleted.");
    }
  }

  async function handleClearDrawings() {
    try {
      await clearChartDrawings(ticker);
      setDrawings([]);
      setDrawNote(null);
    } catch {
      setDrawNote("The drawings could not be cleared.");
    }
  }

  // The bar size actually shown. Overlays, comparison and indicators only work on daily bars.
  const bars = interval ?? (range === "1D" ? "5m" : "1D");
  const isDaily = bars === "1D";
  const isIntraday = INTRADAY_BARS.includes(bars);
  const history: StockPriceHistoryRow[] = useMemo(() => data?.history ?? [], [data]);
  const indicators = isDaily ? data?.indicators ?? null : null;

  // Compare mode is only meaningful on daily bars; it swaps the price panel to % change.
  const compareActive = compare && isDaily;
  // Drawings are price levels, so they are hidden while the panel shows percentage comparison instead.
  const drawingsActive = canDraw && !compareActive;

  // DIF-9: shade the price panel by the stored regime label, one band per run of the same label.
  const regimeShapes = useMemo(() => {
    if (!shadeRegimes || !isDaily || !regimeHistory?.available) return [];
    const byDate = new Map(regimeHistory.history.map((r) => [r.date, r.regime] as const));
    const dates = history.map((p) => p.date.slice(0, 10)).filter((d) => byDate.has(d));
    const shapes: Partial<Shape>[] = [];
    let start = 0;
    for (let i = 1; i <= dates.length; i++) {
      const runEnds = i === dates.length || byDate.get(dates[i]) !== byDate.get(dates[start]);
      if (!runEnds) continue;
      const label = byDate.get(dates[start]) ?? "";
      shapes.push({
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: dates[start],
        x1: dates[i - 1],
        y0: 0,
        y1: 1,
        fillcolor: REGIME_COLORS[label] ?? "#CBD5E1",
        opacity: 0.12,
        line: { width: 0 },
        layer: "below",
      });
      start = i;
    }
    return shapes;
  }, [shadeRegimes, isDaily, regimeHistory, history]);

  useEffect(() => {
    if (!compareActive) return;
    let cancelled = false;
    const extras = extraKey ? extraKey.split(",") : [];
    Promise.all([
      getStockPriceHistory("SPY", range).catch(() => null),
      sectorEtf ? getStockPriceHistory(sectorEtf, range).catch(() => null) : Promise.resolve(null),
      Promise.all(extras.map((t) => getStockPriceHistory(t, range).catch(() => null))),
    ]).then(([spy, sector, extraData]) => {
      if (!cancelled)
        setCompareData({ spy, sector, extras: extras.map((t, i) => ({ ticker: t, data: extraData[i] })) });
    });
    return () => {
      cancelled = true;
    };
  }, [compareActive, range, sectorEtf, extraKey]);

  const addExtraTickers = () => {
    const parsed = extraInput
      .split(/[\s,]+/)
      .map((t) => t.trim().toUpperCase())
      .filter((t) => t && t !== ticker && t !== "SPY" && t !== sectorEtf);
    setExtraTickers(Array.from(new Set(parsed)).slice(0, 3));
  };

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

  // DIF-4: short-term score change over five recorded days, marked where it is 15 points or more.
  const weeklyMoves = useMemo(
    () =>
      signalRows.flatMap((row, i) => {
        if (i < 5) return [];
        const prev = signalRows[i - 5].short_score;
        if (row.short_score === null || prev === null) return [];
        const change = row.short_score - prev;
        return Math.abs(change) >= 15 ? [{ row: i, change }] : [];
      }),
    [signalRows],
  );

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
    const extraColors = ["#DB2777", "#CA8A04", "#4F46E5"];
    compareData.extras.forEach((x, i) => {
      if (x.data) series.push({ name: x.ticker, color: extraColors[i], dash: "solid", map: toMap(x.data.history) });
    });
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

  // Short-term signal rows that fall on a plotted price date.
  const shortMarkers = useMemo(
    () => (isDaily ? signalRows.filter((s) => closeByDate.has(s.as_of_date)) : []),
    [isDaily, signalRows, closeByDate],
  );
  const [selectedSignal, setSelectedSignal] = useState<number | null>(null);

  // DIF-1: short-term signal markers, coloured by realised outcome.
  // Short-term only: the long-term signal would double the marker count.
  if (isDaily && !compareSeries && shortMarkers.length > 0) {
    const pts = shortMarkers;
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
        customdata: pts.map((_, i) => i),
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
    subTraces.push(lineTrace("RSI 14", history.map((p) => p.date), indicators.rsi_14, "#4f46e5", { xaxis: `x${n}`, yaxis: `y${n}` }));
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
    subTraces.push(lineTrace("MACD", dates, indicators.macd.macd, "#4f46e5", { xaxis: `x${n}`, yaxis: `y${n}` }));
    subTraces.push(lineTrace("Signal", dates, indicators.macd.signal, "#F59E0B", { xaxis: `x${n}`, yaxis: `y${n}` }));
  }
  if (subAxisIndex.has("score")) {
    const n = axisOf("score");
    const dates = signalRows.map((s) => s.as_of_date);
    subTraces.push(lineTrace("Short-term score", dates, signalRows.map((s) => s.short_score), "#4f46e5", { xaxis: `x${n}`, yaxis: `y${n}` }));
    subTraces.push(lineTrace("Long-term score", dates, signalRows.map((s) => s.long_score), "#7C3AED", { xaxis: `x${n}`, yaxis: `y${n}` }));
    // DIF-4: a week is five recorded days back; a short-term move of 15 points or more is marked.
    if (weeklyMoves.length > 0) {
      const moves = weeklyMoves;
      subTraces.push({
        x: moves.map((m) => signalRows[m.row].as_of_date),
        y: moves.map((m) => signalRows[m.row].short_score),
        text: moves.map((m) => `Short-term score moved ${m.change > 0 ? "+" : ""}${m.change.toFixed(1)} points over five recorded days`),
        hovertemplate: "%{text}<extra></extra>",
        type: "scatter",
        mode: "markers",
        name: "15-point weekly move",
        marker: { symbol: "diamond", size: 9, color: "#EA580C", line: { color: "#ffffff", width: 1 } },
        xaxis: `x${n}`,
        yaxis: `y${n}`,
      });
    }
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
      ...(isIntraday ? { tickformat: "%-I:%M %p" } : {}),
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
    <div className="rounded-xl border border-slate-200 bg-white p-5">
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

      <div className="mt-2 flex flex-wrap items-center gap-1 text-xs">
        <span className="mr-1 text-slate-500">Bars:</span>
        {BAR_SIZES.map((b) => {
          const selected = (b.key ?? null) === (interval ?? null);
          // Day, week and month bars need a range longer than 1D.
          const disabled = range === "1D" && b.key !== null && !INTRADAY_BARS.includes(b.key);
          return (
            <button
              key={b.label}
              type="button"
              disabled={disabled || !onIntervalChange}
              onClick={() => onIntervalChange?.(b.key)}
              className={`rounded-md px-2 py-1 font-medium disabled:cursor-not-allowed disabled:opacity-40 ${
                selected ? "bg-slate-900 text-white" : "text-slate-500 hover:bg-slate-100"
              }`}
            >
              {b.label}
            </button>
          );
        })}
      </div>
      {isIntraday && (
        <p className="mt-1 text-xs text-slate-400">
          Intraday bars cover the last 7 days (1 min), 60 days (5 and 15 min) or 2 years (1 hour). Longer than that is not
          available from the data source.
        </p>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
        {CHART_TYPES.map((c) => (
          <Fragment key={c.key}>
          <button
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
          {infoIcon(c.label)}
          </Fragment>
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
        {infoIcon("Log scale")}
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
        <MetricLabel info={CHART_CONTROL_INFO["compare"]} />
      </div>

      {compareActive && (
        <form
          className="mt-2 flex flex-wrap items-center gap-2 text-xs"
          onSubmit={(e) => {
            e.preventDefault();
            addExtraTickers();
          }}
        >
          <label htmlFor="compare-extra" className="text-slate-500">
            Add up to 3 tickers to compare:
          </label>
          <input
            id="compare-extra"
            value={extraInput}
            onChange={(e) => setExtraInput(e.target.value)}
            placeholder="e.g. MSFT, NVDA"
            className="input w-44 py-1 text-xs"
          />
          <button type="submit" className="rounded-md border border-slate-200 px-2 py-1 font-medium text-slate-600 hover:bg-slate-50">
            Add
          </button>
          {extraTickers.length > 0 && (
            <button
              type="button"
              onClick={() => {
                setExtraInput("");
                setExtraTickers([]);
              }}
              className="text-slate-500 underline-offset-2 hover:underline"
            >
              Clear extra tickers
            </button>
          )}
        </form>
      )}

      {isDaily && (
        <div className="mt-2 flex flex-wrap items-center gap-1.5 text-xs">
          {OVERLAYS.map((o) => (
            <Fragment key={o.key}>
            <button
              onClick={() => toggle(o.key)}
              aria-pressed={active.has(o.key)}
              disabled={!indicators || compareActive}
              className={`rounded-full border px-2 py-0.5 disabled:opacity-40 ${
                active.has(o.key) ? "border-blue-600 bg-blue-50 text-blue-700" : "border-slate-200 text-slate-500 hover:bg-slate-50"
              }`}
            >
              {o.label}
            </button>
            {infoIcon(o.label)}
            </Fragment>
          ))}
          {PANELS.map((p) => (
            <Fragment key={p.key}>
            <button
              onClick={() => toggle(p.key)}
              aria-pressed={active.has(p.key)}
              disabled={compareActive || (p.key === "volume" ? !hasVolume : !indicators)}
              className={`rounded-full border px-2 py-0.5 disabled:opacity-40 ${
                active.has(p.key) ? "border-blue-600 bg-blue-50 text-blue-700" : "border-slate-200 text-slate-500 hover:bg-slate-50"
              }`}
            >
              {p.label}
            </button>
            {infoIcon(p.label)}
            </Fragment>
          ))}
          {hasScore && (
            <Fragment>
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
            <MetricLabel info={CHART_CONTROL_INFO["score history"]} />
            </Fragment>
          )}
          <button
            onClick={() => setShadeRegimes((v) => !v)}
            aria-pressed={shadeRegimes}
            disabled={!isDaily || !regimeHistory?.available || compareActive}
            className={`rounded-full border px-2 py-0.5 disabled:opacity-40 ${
              shadeRegimes ? "border-blue-600 bg-blue-50 text-blue-700" : "border-slate-200 text-slate-500 hover:bg-slate-50"
            }`}
          >
            Regime shading
          </button>
          <MetricLabel info={CHART_CONTROL_INFO["regime shading"]} />
        </div>
      )}

      {drawingsActive && (
        <div className="mt-3 flex flex-wrap items-center gap-1.5 rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-2 text-xs">
          <span className="mr-1 text-[10px] font-semibold uppercase tracking-wide text-slate-500">Draw</span>
          {DRAW_TOOLS.map((t) => {
            const selected = drawTool === t.key;
            return (
              <button
                key={t.key}
                type="button"
                onClick={() => chooseTool(selected ? null : t.key)}
                aria-pressed={selected}
                className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 font-medium transition-colors ${
                  selected
                    ? "border-indigo-600 bg-indigo-600 text-white shadow-sm"
                    : "border-slate-200 bg-white text-slate-700 hover:border-slate-300 hover:bg-slate-100"
                }`}
              >
                <span aria-hidden="true" className="w-3 text-center">{TOOL_ICON[t.key]}</span>
                {t.label}
              </button>
            );
          })}
          {drawTool && (
            <button type="button" onClick={() => chooseTool(null)} className="rounded-md px-2 py-1 text-slate-500 underline-offset-2 hover:text-slate-800 hover:underline">
              Cancel
            </button>
          )}
          <button
            type="button"
            onClick={handleClearDrawings}
            disabled={drawings.length === 0}
            className="ml-auto rounded-md px-2 py-1 text-slate-500 hover:bg-white hover:text-red-700 disabled:opacity-40"
          >
            Clear all
          </button>
        </div>
      )}
      {drawingsActive && (drawTool || drawNote) && (
        <p
          role="status"
          className={`mt-2 flex items-center gap-2 rounded-md px-3 py-1.5 text-xs ${
            drawNote ? "bg-red-50 text-red-800" : "bg-indigo-50 text-indigo-900"
          }`}
        >
          <span aria-hidden="true" className={`h-2 w-2 shrink-0 rounded-full ${drawNote ? "bg-red-500" : "bg-indigo-500"}`} />
          {drawNote ?? (pending.length ? "Now click the second point." : "Click the chart to place the first point.")}
        </p>
      )}
      {drawingsActive && drawings.length > 0 && (
        <div className="mt-2">
          <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wide text-slate-500">Your drawings ({drawings.length})</p>
          <ul className="flex flex-wrap gap-1.5 text-xs">
            {drawings.map((d) => (
              <li
                key={d.id}
                className="flex max-w-full items-center gap-1.5 rounded-full border border-slate-200 bg-white py-1 pl-2.5 pr-1 text-slate-700 shadow-sm"
              >
                <span aria-hidden="true" className={`h-2 w-2 shrink-0 rounded-full ${KIND_DOT[d.kind]}`} />
                <span className="truncate">{describeDrawing(d)}</span>
                <button
                  type="button"
                  onClick={() => handleDeleteDrawing(d.id)}
                  className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-slate-400 hover:bg-red-50 hover:text-red-700"
                  aria-label={`Delete ${describeDrawing(d)}`}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
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
          onClick={(e) => {
            if (drawingsActive && drawTool) {
              void handleDrawClick(e);
              return;
            }
            const idx = e.points[0]?.customdata;
            setSelectedSignal(typeof idx === "number" ? idx : null);
          }}
          layout={{
            ...(layoutAxes as Partial<Layout>),
            paper_bgcolor: "#ffffff",
            plot_bgcolor: "#ffffff",
            height: chartHeight,
            margin: { t: 16, r: 24, b: 32, l: 56 },
            autosize: true,
            hovermode: "x unified",
            shapes: drawingsActive ? [...regimeShapes, ...drawn.shapes] : regimeShapes,
            annotations: drawingsActive ? [...drawn.annotations, ...pendingAnnotations] : [],
            showlegend: true,
            legend: { orientation: "h", y: -0.15 },
          }}
          style={{ width: "100%" }}
          useResizeHandler
          config={{ displayModeBar: false }}
        />
      )}

      {selectedSignal !== null && shortMarkers[selectedSignal] && (
        <SignalDetail signal={shortMarkers[selectedSignal]} onClose={() => setSelectedSignal(null)} />
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
          {shadeRegimes && regimeHistory?.available && (
            <p>{regimeHistory.disclosure}</p>
          )}
          {isDaily && !compareActive && signalRows.length > 0 && (
            <p>
              Short-term signal markers start {signalRows[0].as_of_date}. Green = hit, red = miss, grey = pending. Hold and Trim are marked too. Click a marker for its confidence, reasons and outcome.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// DIF-1: what the model said on the day a marker was issued, and what happened.
// A record of past output, not a recommendation.
function SignalDetail({
  signal,
  onClose,
}: {
  signal: StockSignalHistoryResponse["history"][number];
  onClose: () => void;
}) {
  const o = signal.short_outcome;
  const outcome =
    o === null ? "No outcome yet" : o.outcome === null ? "Still pending" : `${o.outcome === "hit" ? "Hit" : "Miss"}, ${o.realized_return_pct.toFixed(2)}% realised`;
  const reasons = signal.short_reasons;
  const reasonRows = (rows: { factor: string; contribution: number }[] | undefined) =>
    rows && rows.length > 0
      ? rows.map((r) => `${FACTOR_NAMES[r.factor] ?? r.factor} ${r.contribution > 0 ? "+" : ""}${r.contribution.toFixed(1)}`).join(", ")
      : "None recorded";
  return (
    <div className="mt-4 rounded-md border border-slate-200 bg-slate-50 p-4 text-sm">
      <div className="flex items-start justify-between gap-3">
        <p className="font-semibold text-slate-900">
          Short-term {signal.short_signal} on {signal.as_of_date}
        </p>
        <button onClick={onClose} className="text-xs text-slate-500 hover:text-slate-800">
          Close
        </button>
      </div>
      <dl className="mt-2 grid grid-cols-1 gap-x-4 gap-y-1 text-xs text-slate-700 sm:grid-cols-2">
        <div>
          <dt className="text-slate-400">Score that day</dt>
          <dd>{signal.short_score?.toFixed(1) ?? "n/a"}</dd>
        </div>
        <div>
          <dt className="text-slate-400">Confidence</dt>
          <dd>
            {signal.short_confidence
              ? `${signal.short_confidence.label}${signal.short_confidence.score !== null ? ` (${signal.short_confidence.score.toFixed(0)})` : ""}`
              : "Not recorded"}
          </dd>
        </div>
        <div>
          <dt className="text-slate-400">Top drivers</dt>
          <dd>{reasonRows(reasons?.drivers)}</dd>
        </div>
        <div>
          <dt className="text-slate-400">Top drags</dt>
          <dd>{reasonRows(reasons?.drags)}</dd>
        </div>
        <div className="sm:col-span-2">
          <dt className="text-slate-400">What happened</dt>
          <dd>{outcome}</dd>
        </div>
      </dl>
    </div>
  );
}

const FACTOR_NAMES: Record<string, string> = {
  momentum: "Momentum",
  reversal: "Reversal",
  earnings_surprise: "Earnings Surprise",
  earnings_revisions: "Earnings Revisions",
  value: "Value",
  growth: "Growth",
  quality: "Quality",
  low_vol: "Low Volatility",
};
