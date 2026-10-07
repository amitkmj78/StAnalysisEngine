"use client";

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";

import { ApiError, getPaperClock, getPortfolios, submitPaperOrder } from "@/lib/api";
import TradeImpactCard from "@/components/portfolio/TradeImpactCard";
import type { PaperClock, Portfolio } from "@/lib/types";

type OrderType = "market" | "limit";
type TimeInForce = "day" | "gtc";

export default function PaperTradingTicketPage() {
  const searchParams = useSearchParams();
  // Which linked paper account to trade through -- a user can have one per
  // portfolio now, so the paper-trading page links here with ?portfolio_id=.
  // Omitted (e.g. a bookmarked link from before) falls back to the
  // backend's own default: the user's oldest active portfolio.
  const portfolioIdParam = searchParams.get("portfolio_id");
  const portfolioId = portfolioIdParam ? Number(portfolioIdParam) : undefined;

  const [ticker, setTicker] = useState(searchParams.get("ticker")?.toUpperCase() ?? "");
  const [side, setSide] = useState<"buy" | "sell">(searchParams.get("side") === "sell" ? "sell" : "buy");
  const [orderType, setOrderType] = useState<OrderType>("market");
  const [timeInForce, setTimeInForce] = useState<TimeInForce>("day");
  // Quantity is never pre-filled from a signal (TRD-12) -- always starts empty.
  const [qty, setQty] = useState("");
  const [limitPrice, setLimitPrice] = useState("");

  const [reviewing, setReviewing] = useState(false);
  const [clock, setClock] = useState<PaperClock | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);

  // DIF-6: the impact preview uses the app's own portfolios, so load them once for the card.
  useEffect(() => {
    getPortfolios().then((res) => setPortfolios(res.portfolios)).catch(() => setPortfolios([]));
  }, []);

  const qtyNum = Number(qty);
  const limitPriceNum = orderType === "limit" ? Number(limitPrice) : null;
  const validForReview =
    ticker.trim().length > 0 && qtyNum > 0 && (orderType === "market" || (limitPriceNum !== null && limitPriceNum > 0));

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
        portfolio_id: portfolioId,
      });
      setResult(res.note ?? `Order submitted: ${res.order.status}.`);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not submit this order.");
      setReviewing(false);
    } finally {
      setSubmitting(false);
    }
  }

  if (result) {
    return (
      <div className="mx-auto max-w-lg px-4 py-8">
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-5 text-sm text-emerald-800">{result}</div>
        <div className="mt-4 flex gap-3">
          <Link href="/portfolio/paper-trading" className="text-sm font-medium text-slate-700 hover:underline">
            View orders
          </Link>
          <button
            onClick={() => {
              setResult(null);
              setReviewing(false);
              setQty("");
              setLimitPrice("");
            }}
            className="text-sm font-medium text-slate-700 hover:underline"
          >
            Place another order
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-lg px-4 py-8">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <h1 className="font-display text-2xl font-semibold text-slate-900">New Paper Order</h1>
        <Link href="/portfolio/paper-trading" className="text-sm font-medium text-slate-600 hover:underline">
          ← Back
        </Link>
      </div>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {!reviewing ? (
        <div className="mt-6 rounded-xl border border-slate-200 bg-white p-5">
          <div className="flex flex-col gap-3">
            <label className="text-xs font-medium text-slate-600">
              Ticker
              <input
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm uppercase"
                placeholder="AAPL"
              />
            </label>

            <div className="flex gap-2">
              <button
                onClick={() => setSide("buy")}
                className={`flex-1 rounded-md border px-3 py-2 text-sm font-semibold ${
                  side === "buy" ? "border-emerald-600 bg-emerald-50 text-emerald-700" : "border-slate-300 text-slate-600"
                }`}
              >
                Buy
              </button>
              <button
                onClick={() => setSide("sell")}
                className={`flex-1 rounded-md border px-3 py-2 text-sm font-semibold ${
                  side === "sell" ? "border-red-600 bg-red-50 text-red-700" : "border-slate-300 text-slate-600"
                }`}
              >
                Sell
              </button>
            </div>

            <label className="text-xs font-medium text-slate-600">
              Order Type
              <select
                value={orderType}
                onChange={(e) => setOrderType(e.target.value as OrderType)}
                className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
              >
                <option value="market">Market — execute at the next available price</option>
                <option value="limit">Limit — execute only at your price or better</option>
              </select>
            </label>

            <label className="text-xs font-medium text-slate-600">
              Quantity (shares)
              <input
                value={qty}
                onChange={(e) => setQty(e.target.value)}
                type="number"
                min="0"
                step="any"
                className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
              />
            </label>

            {orderType === "limit" && (
              <label className="text-xs font-medium text-slate-600">
                Limit Price
                <input
                  value={limitPrice}
                  onChange={(e) => setLimitPrice(e.target.value)}
                  type="number"
                  min="0"
                  step="any"
                  className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
                />
              </label>
            )}

            <label className="text-xs font-medium text-slate-600">
              Time in Force
              <select
                value={timeInForce}
                onChange={(e) => setTimeInForce(e.target.value as TimeInForce)}
                className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
              >
                <option value="day">Day — expires at market close if unfilled</option>
                <option value="gtc">Good til cancelled</option>
              </select>
            </label>
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
        <div className="mt-6 rounded-xl border border-slate-200 bg-white p-5">
          <h2 className="text-sm font-semibold text-slate-900">Review and Confirm</h2>
          <dl className="mt-3 grid grid-cols-2 gap-y-2 text-sm">
            <dt className="text-slate-500">Action</dt>
            <dd className="text-right font-medium text-slate-900">
              {side === "buy" ? "Buy" : "Sell"} {qtyNum} {ticker}
            </dd>
            <dt className="text-slate-500">Order type</dt>
            <dd className="text-right text-slate-900">
              {orderType === "market"
                ? "Market — fills at the next available price"
                : `Limit — fills only at $${limitPriceNum} or better`}
            </dd>
            <dt className="text-slate-500">Time in force</dt>
            <dd className="text-right text-slate-900">{timeInForce === "day" ? "Day" : "Good til cancelled"}</dd>
            <dt className="text-slate-500">Market status</dt>
            <dd className="text-right text-slate-900">
              {clock ? (clock.is_open ? "Open" : "Closed") : "Checking…"}
            </dd>
          </dl>

          {clock && !clock.is_open && (
            <p className="mt-3 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
              The market is closed. This order will be rejected — extended-hours orders aren&apos;t supported yet.
            </p>
          )}

          <p className="mt-3 text-xs text-slate-500">
            This is a paper (simulated) order. Confirming sends it to your linked Alpaca paper account.
          </p>

          <div className="mt-4 flex gap-3">
            <button
              onClick={() => setReviewing(false)}
              className="rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
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
        <div className="mt-6">
          <TradeImpactCard ticker={ticker} portfolios={portfolios} defaultSide={side} />
        </div>
      )}
    </div>
  );
}
