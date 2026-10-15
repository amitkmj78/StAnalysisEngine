"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import { ApiError, getCurrentPrice, getPaperAccount, getPaperClock, submitPaperOrder } from "@/lib/api";
import TradeImpactCard from "@/components/portfolio/TradeImpactCard";
import type { PaperClock, Portfolio } from "@/lib/types";

// BRK-2: all six paper order types. BRK-3: risk-based sizing. BRK-4: a
// compact signal/regime readout (the DIF-6 impact preview itself is the
// existing TradeImpactCard below, unchanged). Used both as the standalone
// /portfolio/paper-trading/ticket page and embedded on the stock chart
// page -- the two render the same ticket rather than maintaining two
// separate order-submission forms.
type OrderType = "market" | "limit" | "stop" | "stop_limit" | "trailing_stop";
type TimeInForce = "day" | "gtc";
type TrailUnit = "percent" | "price";
type RiskUnit = "dollars" | "percent";

interface OrderTicketProps {
  ticker: string;
  tickerEditable?: boolean;
  portfolioId?: number;
  portfolios: Portfolio[];
  defaultSide?: "buy" | "sell";
  signal?: string | null;
  regime?: string | null;
  onSubmitted?: () => void;
}

function regimeCautionCopy(regime: string | null | undefined): string | null {
  if (regime === "Risk-Off") return "Risk-Off";
  if (regime === "Cautious") return "Cautious";
  return null;
}

export default function OrderTicket({
  ticker: initialTicker,
  tickerEditable = true,
  portfolioId,
  portfolios,
  defaultSide = "buy",
  signal,
  regime,
  onSubmitted,
}: OrderTicketProps) {
  const [ticker, setTicker] = useState(initialTicker.toUpperCase());
  const [side, setSide] = useState<"buy" | "sell">(defaultSide);
  const [orderType, setOrderType] = useState<OrderType>("market");
  const [timeInForce, setTimeInForce] = useState<TimeInForce>("day");
  const [qty, setQty] = useState("");
  const [limitPrice, setLimitPrice] = useState("");
  const [stopPrice, setStopPrice] = useState("");
  const [trailUnit, setTrailUnit] = useState<TrailUnit>("percent");
  const [trailValue, setTrailValue] = useState("");
  const [useBracket, setUseBracket] = useState(false);
  const [takeProfitPrice, setTakeProfitPrice] = useState("");
  const [stopLossPrice, setStopLossPrice] = useState("");
  const [stopLossLimitPrice, setStopLossLimitPrice] = useState("");

  // BRK-3: risk-based sizing -- independent of the order's own stop_price
  // field above (this is a planning calculator, not itself an order field).
  const [riskOpen, setRiskOpen] = useState(false);
  const [riskEntryPrice, setRiskEntryPrice] = useState("");
  const [riskStopPrice, setRiskStopPrice] = useState("");
  const [riskUnit, setRiskUnit] = useState<RiskUnit>("dollars");
  const [riskMaxLoss, setRiskMaxLoss] = useState("");
  const [riskEquity, setRiskEquity] = useState<number | null>(null);
  const [riskError, setRiskError] = useState<string | null>(null);

  const [reviewing, setReviewing] = useState(false);
  const [clock, setClock] = useState<PaperClock | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);

  useEffect(() => {
    setTicker(initialTicker.toUpperCase());
  }, [initialTicker]);

  // Prefill the risk-sizing entry price with the last trade price -- a
  // starting point the user can still override, never silently relied on.
  useEffect(() => {
    if (!ticker) return;
    let cancelled = false;
    getCurrentPrice(ticker)
      .then((res) => {
        if (!cancelled && res.price) setRiskEntryPrice(String(res.price));
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [ticker]);

  async function loadEquityForRiskPct() {
    try {
      const res = await getPaperAccount(portfolioId);
      const equity = res.live && typeof res.live.equity !== "undefined" ? Number(res.live.equity) : null;
      setRiskEquity(Number.isFinite(equity) ? equity : null);
    } catch {
      setRiskEquity(null);
    }
  }

  function handleUseRiskQuantity() {
    setRiskError(null);
    const entry = Number(riskEntryPrice);
    const stop = Number(riskStopPrice);
    const maxLossInput = Number(riskMaxLoss);
    if (!(entry > 0) || !(stop > 0) || !(maxLossInput > 0)) {
      setRiskError("Enter an entry price, a stop price, and a max loss greater than zero.");
      return;
    }
    if (entry === stop) {
      setRiskError("Entry and stop can't be the same price.");
      return;
    }
    const maxLossDollars = riskUnit === "percent"
      ? (riskEquity ? (maxLossInput / 100) * riskEquity : null)
      : maxLossInput;
    if (maxLossDollars === null) {
      setRiskError("Could not read your account equity for a %-of-equity calculation. Try the $ amount instead.");
      return;
    }
    const shares = Math.floor(maxLossDollars / Math.abs(entry - stop));
    if (shares < 1) {
      setRiskError("That max loss is too small to buy even one share at this stop distance.");
      return;
    }
    setQty(String(shares));
  }

  const qtyNum = Number(qty);
  const limitPriceNum = (orderType === "limit" || orderType === "stop_limit") ? Number(limitPrice) : null;
  const stopPriceNum = (orderType === "stop" || orderType === "stop_limit") ? Number(stopPrice) : null;
  const trailPriceNum = orderType === "trailing_stop" && trailUnit === "price" ? Number(trailValue) : null;
  const trailPercentNum = orderType === "trailing_stop" && trailUnit === "percent" ? Number(trailValue) : null;
  const takeProfitNum = useBracket ? Number(takeProfitPrice) : null;
  const stopLossNum = useBracket ? Number(stopLossPrice) : null;
  const stopLossLimitNum = useBracket && stopLossLimitPrice ? Number(stopLossLimitPrice) : null;

  const validForReview =
    ticker.trim().length > 0 &&
    qtyNum > 0 &&
    (orderType !== "limit" || (limitPriceNum !== null && limitPriceNum > 0)) &&
    (orderType !== "stop" || (stopPriceNum !== null && stopPriceNum > 0)) &&
    (orderType !== "stop_limit" || (stopPriceNum !== null && stopPriceNum > 0 && limitPriceNum !== null && limitPriceNum > 0)) &&
    (orderType !== "trailing_stop" || (trailUnit === "price" ? (trailPriceNum !== null && trailPriceNum > 0) : (trailPercentNum !== null && trailPercentNum > 0))) &&
    (!useBracket || (takeProfitNum !== null && takeProfitNum > 0 && stopLossNum !== null && stopLossNum > 0));

  async function handleReview() {
    setError(null);
    try {
      const c = await getPaperClock(portfolioId);
      setClock(c);
      setReviewing(true);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not check market status.");
    }
  }

  async function handleConfirm() {
    setSubmitting(true);
    setError(null);
    try {
      const res = await submitPaperOrder({
        ticker: ticker.trim().toUpperCase(),
        side,
        order_type: orderType,
        time_in_force: timeInForce,
        qty: qtyNum,
        limit_price: limitPriceNum ?? undefined,
        stop_price: stopPriceNum ?? undefined,
        trail_price: trailPriceNum ?? undefined,
        trail_percent: trailPercentNum ?? undefined,
        order_class: useBracket ? "bracket" : undefined,
        take_profit_price: takeProfitNum ?? undefined,
        stop_loss_price: stopLossNum ?? undefined,
        stop_loss_limit_price: stopLossLimitNum ?? undefined,
        portfolio_id: portfolioId,
      });
      setResult(res.note ?? `Order submitted: ${res.order.status}.`);
      onSubmitted?.();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not submit this order.");
      setReviewing(false);
    } finally {
      setSubmitting(false);
    }
  }

  const caution = regimeCautionCopy(regime);

  if (result) {
    return (
      <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-5 text-sm text-emerald-800">
        {result}
        <div className="mt-3 flex gap-3">
          <Link href="/portfolio/paper-trading" className="font-medium underline">
            View orders
          </Link>
          <button
            onClick={() => {
              setResult(null);
              setReviewing(false);
              setQty("");
              setLimitPrice("");
              setStopPrice("");
              setTrailValue("");
              setUseBracket(false);
              setTakeProfitPrice("");
              setStopLossPrice("");
              setStopLossLimitPrice("");
            }}
            className="font-medium underline"
          >
            Place another order
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {(signal || caution) && (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          {signal && (
            <span className="rounded-full bg-[var(--pf-accent-soft)] px-2 py-0.5 font-medium text-[var(--pf-accent)]">
              Signal: {signal}
            </span>
          )}
          {caution && (
            <span className="rounded-full bg-[var(--pf-warn-soft)] px-2 py-0.5 font-medium text-[var(--pf-warn)]">
              Regime: {caution}
            </span>
          )}
        </div>
      )}

      {error && <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {!reviewing ? (
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5">
          <div className="flex flex-col gap-3">
            <label className="text-xs font-medium text-slate-600 dark:text-slate-300">
              Ticker
              <input
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                disabled={!tickerEditable}
                className="mt-1 w-full rounded-md border border-slate-300 dark:border-slate-600 px-3 py-2 text-sm uppercase disabled:opacity-70"
                placeholder="AAPL"
              />
            </label>

            <div className="flex gap-2">
              <button
                onClick={() => setSide("buy")}
                className={`flex-1 rounded-md border px-3 py-2 text-sm font-semibold ${
                  side === "buy" ? "border-emerald-600 bg-emerald-50 text-emerald-700" : "border-slate-300 dark:border-slate-600 text-slate-600 dark:text-slate-300"
                }`}
              >
                Buy
              </button>
              <button
                onClick={() => setSide("sell")}
                className={`flex-1 rounded-md border px-3 py-2 text-sm font-semibold ${
                  side === "sell" ? "border-red-600 bg-red-50 text-red-700" : "border-slate-300 dark:border-slate-600 text-slate-600 dark:text-slate-300"
                }`}
              >
                Sell
              </button>
            </div>

            <label className="text-xs font-medium text-slate-600 dark:text-slate-300">
              Order Type
              <select
                value={orderType}
                onChange={(e) => setOrderType(e.target.value as OrderType)}
                className="mt-1 w-full rounded-md border border-slate-300 dark:border-slate-600 px-3 py-2 text-sm"
              >
                <option value="market">Market — execute at the next available price</option>
                <option value="limit">Limit — execute only at your price or better</option>
                <option value="stop">Stop — becomes a market order once triggered</option>
                <option value="stop_limit">Stop-limit — becomes a limit order once triggered</option>
                <option value="trailing_stop">Trailing stop — follows the price by a percent or $ trail</option>
              </select>
            </label>

            <label className="text-xs font-medium text-slate-600 dark:text-slate-300">
              Quantity (shares)
              <input
                value={qty}
                onChange={(e) => setQty(e.target.value)}
                type="number"
                min="0"
                step="any"
                className="mt-1 w-full rounded-md border border-slate-300 dark:border-slate-600 px-3 py-2 text-sm"
              />
            </label>

            {(orderType === "limit" || orderType === "stop_limit") && (
              <label className="text-xs font-medium text-slate-600 dark:text-slate-300">
                Limit Price
                <input
                  value={limitPrice}
                  onChange={(e) => setLimitPrice(e.target.value)}
                  type="number" min="0" step="any"
                  className="mt-1 w-full rounded-md border border-slate-300 dark:border-slate-600 px-3 py-2 text-sm"
                />
              </label>
            )}

            {(orderType === "stop" || orderType === "stop_limit") && (
              <label className="text-xs font-medium text-slate-600 dark:text-slate-300">
                Stop Price
                <input
                  value={stopPrice}
                  onChange={(e) => setStopPrice(e.target.value)}
                  type="number" min="0" step="any"
                  className="mt-1 w-full rounded-md border border-slate-300 dark:border-slate-600 px-3 py-2 text-sm"
                />
              </label>
            )}

            {orderType === "trailing_stop" && (
              <div className="flex flex-col gap-1">
                <span className="text-xs font-medium text-slate-600 dark:text-slate-300">Trail by</span>
                <div className="flex gap-2">
                  <select
                    value={trailUnit}
                    onChange={(e) => setTrailUnit(e.target.value as TrailUnit)}
                    className="rounded-md border border-slate-300 dark:border-slate-600 px-2 py-2 text-sm"
                  >
                    <option value="percent">%</option>
                    <option value="price">$</option>
                  </select>
                  <input
                    value={trailValue}
                    onChange={(e) => setTrailValue(e.target.value)}
                    type="number" min="0" step="any"
                    className="flex-1 rounded-md border border-slate-300 dark:border-slate-600 px-3 py-2 text-sm"
                  />
                </div>
              </div>
            )}

            <label className="text-xs font-medium text-slate-600 dark:text-slate-300">
              Time in Force
              <select
                value={timeInForce}
                onChange={(e) => setTimeInForce(e.target.value as TimeInForce)}
                className="mt-1 w-full rounded-md border border-slate-300 dark:border-slate-600 px-3 py-2 text-sm"
              >
                <option value="day">Day — expires at market close if unfilled</option>
                <option value="gtc">Good til cancelled</option>
              </select>
            </label>

            <label className="flex items-center gap-2 text-xs font-medium text-slate-600 dark:text-slate-300">
              <input type="checkbox" checked={useBracket} onChange={(e) => setUseBracket(e.target.checked)} />
              Attach a bracket (take-profit + stop-loss)
            </label>

            {useBracket && (
              <div className="flex flex-col gap-3 rounded-md border border-slate-200 dark:border-slate-700 p-3">
                <label className="text-xs font-medium text-slate-600 dark:text-slate-300">
                  Take-profit price
                  <input
                    value={takeProfitPrice}
                    onChange={(e) => setTakeProfitPrice(e.target.value)}
                    type="number" min="0" step="any"
                    className="mt-1 w-full rounded-md border border-slate-300 dark:border-slate-600 px-3 py-2 text-sm"
                  />
                </label>
                <label className="text-xs font-medium text-slate-600 dark:text-slate-300">
                  Stop-loss price
                  <input
                    value={stopLossPrice}
                    onChange={(e) => setStopLossPrice(e.target.value)}
                    type="number" min="0" step="any"
                    className="mt-1 w-full rounded-md border border-slate-300 dark:border-slate-600 px-3 py-2 text-sm"
                  />
                </label>
                <label className="text-xs font-medium text-slate-600 dark:text-slate-300">
                  Stop-loss limit price (optional — a plain stop otherwise)
                  <input
                    value={stopLossLimitPrice}
                    onChange={(e) => setStopLossLimitPrice(e.target.value)}
                    type="number" min="0" step="any"
                    className="mt-1 w-full rounded-md border border-slate-300 dark:border-slate-600 px-3 py-2 text-sm"
                  />
                </label>
              </div>
            )}
          </div>

          <div className="mt-4 border-t border-slate-200 dark:border-slate-700 pt-3">
            <button
              type="button"
              onClick={() => setRiskOpen((v) => !v)}
              className="text-xs font-medium text-slate-600 dark:text-slate-300 underline decoration-dotted underline-offset-2"
            >
              {riskOpen ? "Hide" : "Size this trade by risk"}
            </button>
            {riskOpen && (
              <div className="mt-3 flex flex-col gap-2 rounded-md bg-slate-50 dark:bg-slate-800/50 p-3">
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Shares = max loss ÷ |entry − stop|, rounded down to a whole share.
                </p>
                <div className="grid grid-cols-2 gap-2">
                  <label className="text-xs font-medium text-slate-600 dark:text-slate-300">
                    Entry price
                    <input
                      value={riskEntryPrice}
                      onChange={(e) => setRiskEntryPrice(e.target.value)}
                      type="number" min="0" step="any"
                      className="mt-1 w-full rounded-md border border-slate-300 dark:border-slate-600 px-2 py-1.5 text-sm"
                    />
                  </label>
                  <label className="text-xs font-medium text-slate-600 dark:text-slate-300">
                    Stop price
                    <input
                      value={riskStopPrice}
                      onChange={(e) => setRiskStopPrice(e.target.value)}
                      type="number" min="0" step="any"
                      className="mt-1 w-full rounded-md border border-slate-300 dark:border-slate-600 px-2 py-1.5 text-sm"
                    />
                  </label>
                </div>
                <div className="flex items-end gap-2">
                  <label className="flex-1 text-xs font-medium text-slate-600 dark:text-slate-300">
                    Max loss
                    <input
                      value={riskMaxLoss}
                      onChange={(e) => setRiskMaxLoss(e.target.value)}
                      type="number" min="0" step="any"
                      className="mt-1 w-full rounded-md border border-slate-300 dark:border-slate-600 px-2 py-1.5 text-sm"
                    />
                  </label>
                  <select
                    value={riskUnit}
                    onChange={(e) => {
                      const unit = e.target.value as RiskUnit;
                      setRiskUnit(unit);
                      if (unit === "percent" && riskEquity === null) loadEquityForRiskPct();
                    }}
                    className="rounded-md border border-slate-300 dark:border-slate-600 px-2 py-2 text-sm"
                  >
                    <option value="dollars">$</option>
                    <option value="percent">% of equity</option>
                  </select>
                </div>
                {riskUnit === "percent" && (
                  <p className="text-xs text-slate-400 dark:text-slate-500">
                    {riskEquity !== null ? `Account equity: $${riskEquity.toLocaleString()}` : "Loading account equity…"}
                  </p>
                )}
                {riskError && <p className="text-xs text-red-600">{riskError}</p>}
                <button
                  type="button"
                  onClick={handleUseRiskQuantity}
                  className="w-fit rounded-md border border-slate-300 dark:border-slate-600 px-3 py-1.5 text-xs font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700"
                >
                  Use this quantity
                </button>
              </div>
            )}
          </div>

          <button
            onClick={handleReview}
            disabled={!validForReview}
            className="mt-4 w-full rounded-md bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800 disabled:opacity-50"
          >
            Review Order
          </button>
        </div>
      ) : (
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5">
          <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-50">Review and Confirm</h2>
          <dl className="mt-3 grid grid-cols-2 gap-y-2 text-sm">
            <dt className="text-slate-500 dark:text-slate-400">Action</dt>
            <dd className="text-right font-medium text-slate-900 dark:text-slate-50">
              {side === "buy" ? "Buy" : "Sell"} {qtyNum} {ticker}
            </dd>
            <dt className="text-slate-500 dark:text-slate-400">Order type</dt>
            <dd className="text-right text-slate-900 dark:text-slate-50">
              {orderType} {useBracket ? "+ bracket" : ""}
            </dd>
            <dt className="text-slate-500 dark:text-slate-400">Time in force</dt>
            <dd className="text-right text-slate-900 dark:text-slate-50">{timeInForce === "day" ? "Day" : "Good til cancelled"}</dd>
            <dt className="text-slate-500 dark:text-slate-400">Market status</dt>
            <dd className="text-right text-slate-900 dark:text-slate-50">
              {clock ? (clock.is_open ? "Open" : "Closed") : "Checking…"}
            </dd>
          </dl>

          {clock && !clock.is_open && (
            <p className="mt-3 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
              The market is closed. This order will be rejected — extended-hours orders aren&apos;t supported yet.
            </p>
          )}

          <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
            This is a paper (simulated) order. Confirming sends it to your linked Alpaca paper account.
          </p>

          <div className="mt-4 flex gap-3">
            <button
              onClick={() => setReviewing(false)}
              className="rounded-md border border-slate-300 dark:border-slate-600 px-4 py-2 text-sm font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700"
            >
              Edit
            </button>
            <button
              onClick={handleConfirm}
              disabled={submitting}
              className="flex-1 rounded-md bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800 disabled:opacity-50"
            >
              {submitting ? "Submitting…" : "Confirm and Submit"}
            </button>
          </div>
        </div>
      )}

      {ticker && portfolios.length > 0 && (
        <TradeImpactCard ticker={ticker} portfolios={portfolios} defaultSide={side} />
      )}
    </div>
  );
}
