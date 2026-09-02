"use client";

import { useState } from "react";

import { ApiError, getPortfolioReview } from "@/lib/api";
import type { PortfolioReviewResponse } from "@/lib/types";

const SIGNAL_BADGE_CLASS: Record<string, string> = {
  BUY: "bg-emerald-50 text-emerald-700",
  SELL: "bg-red-50 text-red-700",
  HOLD: "bg-slate-100 text-slate-600",
};

const SENTIMENT_BADGE_CLASS: Record<string, string> = {
  Bullish: "bg-emerald-50 text-emerald-700",
  Bearish: "bg-red-50 text-red-700",
  Neutral: "bg-slate-100 text-slate-600",
};

export default function PortfolioReviewCard({ portfolioId }: { portfolioId: number | null }) {
  const [review, setReview] = useState<PortfolioReviewResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleReview() {
    setLoading(true);
    setError(null);
    try {
      setReview(await getPortfolioReview(portfolioId ?? undefined));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not review this portfolio right now.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mt-6 rounded-lg border border-slate-200 bg-white p-5">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Portfolio Review</h2>
          <p className="mt-1 text-xs text-slate-500">
            Flags positions worth a second look — a SELL signal, a concentrated position (single-ticker or a whole
            sector spread across several), or sentiment and the quant signal agreeing — using today&apos;s
            already-computed Signal and Sentiment plus live dollar values for each holding, then summarizes why.
            Describes what the data shows; it doesn&apos;t tell you to buy or sell.
          </p>
        </div>
        <button
          type="button"
          onClick={handleReview}
          disabled={loading}
          className="rounded-md border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
        >
          {loading ? "Reviewing…" : review ? "Review Again" : "Review My Portfolio"}
        </button>
      </div>

      {error && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {review && review.flagged.length === 0 && (
        <p className="mt-3 text-sm text-slate-500">
          Nothing stands out today — no SELL signals, no concentrated positions, and sentiment isn&apos;t
          reinforcing or conflicting with any signal in a notable way.
        </p>
      )}

      {review && review.flagged.length > 0 && (
        <>
          {review.summary && (
            <p className="mt-3 rounded-md bg-slate-50 px-3 py-2 text-sm leading-relaxed text-slate-700">
              {review.summary}
            </p>
          )}
          <ul className="mt-3 flex flex-col gap-2">
            {review.flagged.map((f) => (
              <li key={f.ticker} className="rounded-md border border-slate-200 px-3 py-2 text-sm">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium text-slate-800">{f.ticker}</span>
                  {f.signal && (
                    <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${SIGNAL_BADGE_CLASS[f.signal]}`}>
                      {f.signal}
                    </span>
                  )}
                  {f.sentiment_label && (
                    <span
                      className={`rounded-full px-2 py-0.5 text-xs font-semibold ${SENTIMENT_BADGE_CLASS[f.sentiment_label]}`}
                    >
                      {f.sentiment_label}
                    </span>
                  )}
                  {f.weight_pct !== null && (
                    <span className="text-xs text-slate-400">
                      {f.weight_pct.toFixed(1)}% of portfolio
                      {f.market_value !== null && ` (${f.market_value.toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 0 })})`}
                    </span>
                  )}
                  {f.sector && (
                    <span className="rounded-full bg-slate-50 px-2 py-0.5 text-xs text-slate-500">{f.sector}</span>
                  )}
                </div>
                <p className="mt-1 text-xs text-slate-500">{f.reasons.join(" · ")}</p>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
