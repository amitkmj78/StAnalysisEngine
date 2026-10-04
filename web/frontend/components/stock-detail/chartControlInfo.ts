import type { ColumnInfo } from "@/components/InfoModal";

// Plain-language explanations for the stock chart controls. These describe
// what each thing shows; none of them is a buy or sell instruction.
export const CHART_CONTROL_INFO: Record<string, ColumnInfo> = {
  candles: {
    title: "Candlestick chart",
    body: [
      "Each bar is one trading day (one 5-minute slot on the 1D range). The thick body runs from the open to the close. Green means the close was above the open, red means below.",
      "The thin lines above and below the body (wicks) show the day's high and low.",
    ],
  },
  ohlc: {
    title: "OHLC bars",
    body: [
      "Shows the same open, high, low and close as candles, drawn as a bar. A tick on the left marks the open, a tick on the right marks the close, and the vertical line spans the high to the low.",
    ],
  },
  "heikin ashi": {
    title: "Heikin Ashi",
    body: [
      "A smoothed version of candles. Each bar blends the day's prices with the previous bar, so trends look steadier and short reversals are easier to see.",
      "Heikin Ashi prices are averages, not real trade prices. Don't read them as the actual price.",
    ],
  },
  line: {
    title: "Line chart",
    body: ["Connects each day's closing price. The simplest view of the trend."],
  },
  area: {
    title: "Area chart",
    body: ["The line chart with the space below the line shaded. Available on the linear scale, not the log scale."],
  },
  "log scale": {
    title: "Log scale",
    body: [
      "The vertical axis is spaced by percentage change instead of dollar change. Equal distances mean equal percentage moves, so long periods of growth are easier to compare.",
      "Not available in comparison mode.",
    ],
  },
  compare: {
    title: "Comparison with SPY and the sector ETF",
    body: [
      "Plots the stock, the S&P 500 (SPY) and its sector ETF as % change from the first date all three share. Use it to see relative performance over the period.",
      "Overlays and signal markers are hidden in this mode.",
    ],
  },
  "sma 20": {
    title: "Simple moving average, 20 days",
    body: ["The average closing price over the last 20 trading days. It smooths out day-to-day noise, and it lags the price by design."],
  },
  "sma 50": {
    title: "Simple moving average, 50 days",
    body: ["The average closing price over the last 50 trading days. It shows the medium-term trend and lags the price by design."],
  },
  "sma 200": {
    title: "Simple moving average, 200 days",
    body: [
      "The average closing price over the last 200 trading days. It is a long-term trend line, and it stays blank until 200 bars of history are on the chart.",
    ],
  },
  "ema 20": {
    title: "Exponential moving average, 20 days",
    body: ["A 20-day average that gives recent days more weight than older ones, so it reacts faster than the simple 20-day average."],
  },
  "bollinger 20, 2": {
    title: "Bollinger Bands",
    body: [
      "A 20-day average with bands two standard deviations above and below it. Wider bands mean the price has moved more recently.",
      "Price touching a band is not a signal on its own.",
    ],
  },
  vwap: {
    title: "VWAP (volume-weighted average price)",
    body: [
      "The average price, weighted by how many shares traded at each price, starting from the first bar on this chart.",
      "Shows whether the price is above or below the level where most of the trading happened over the period.",
    ],
  },
  volume: {
    title: "Volume",
    body: ["The number of shares traded each day. Green bars closed higher than the day before, red bars closed lower."],
  },
  "rsi 14": {
    title: "Relative Strength Index (14 days)",
    body: [
      "Measures recent momentum on a scale from 0 to 100. Readings above 70 are often called overbought, and below 30 oversold.",
      "These describe recent price moves. They are not buy or sell instructions.",
    ],
  },
  "macd 12, 26, 9": {
    title: "MACD (12, 26, 9)",
    body: [
      "The gap between a 12-day and a 26-day exponential average, with a 9-day signal line drawn alongside it.",
      "The bars show MACD minus the signal line. Positive bars mean MACD is above its signal line.",
    ],
  },
  "regime shading": {
    title: "Regime shading",
    body: [
      "Shades the price chart by the market regime label stored for each day: green for Risk-On, light green for Constructive, grey for Neutral, amber for Cautious and red for Risk-Off.",
      "The regime is a condition label. It has not been validated as a signal and is not a recommendation.",
    ],
  },
  "score history": {
    title: "Score history",
    body: [
      "The short-term and long-term model scores on each day they were recorded, drawn under the price on the same dates.",
      "Recording starts on 2026-08-05, so earlier dates have no scores.",
    ],
  },
};
