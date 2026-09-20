"use client";

import { useState } from "react";

import { ApiError, getPortfolioReview } from "@/lib/api";
import type { PortfolioReviewResponse } from "@/lib/types";

const SIGNAL_BADGE_CLASS: Record<string, string> = {
  BUY: "bg-[#e3ede8] text-[#2f6b4f]",
  SELL: "bg-[#f6e5e3] text-[#a23b34]",
  HOLD: "bg-[#efece4] text-[#6b6459]",
};

const SENTIMENT_BADGE_CLASS: Record<string, string> = {
  Bullish: "bg-[#e3ede8] text-[#2f6b4f]",
  Bearish: "bg-[#f6e5e3] text-[#a23b34]",
  Neutral: "bg-[#efece4] text-[#6b6459]",
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
    <div className="mt-6 rounded-xl border border-[#ddd8cd] bg-white p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <div className="flex h-8 w-8 flex-none items-center justify-center rounded-full bg-[#eaf1ee] text-[#2f5d50]">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} className="h-4 w-4">
              <path d="M12 2l2.4 7.2H22l-6 4.6 2.3 7.2L12 16.4 5.7 21l2.3-7.2-6-4.6h7.6z" />
            </svg>
          </div>
          <div>
            <h2 className="text-base font-semibold text-[#1f2420]">Portfolio Review</h2>
            <p className="mt-1 max-w-2xl text-xs leading-relaxed text-[#857d6e]">
              Flags positions worth a second look — a SELL signal, a concentrated position (single-ticker or a whole
              sector spread across several), or sentiment and the quant signal agreeing — using today&apos;s
              already-computed Signal and Sentiment plus live dollar values for each holding, then summarizes why.
              Describes what the data shows; it doesn&apos;t tell you to buy or sell.
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={handleReview}
          disabled={loading}
          className="flex-none rounded-md border border-[#ddd8cd] bg-white px-3 py-1.5 text-xs font-semibold text-[#1f2420] hover:border-[#2f5d50] hover:text-[#2f5d50] disabled:opacity-50"
        >
          {loading ? "Reviewing…" : review ? "Review Again" : "Review My Portfolio"}
        </button>
      </div>

      {error && <p className="mt-3 rounded-md border border-[#e4c9c5] bg-[#fbeceb] px-3 py-2 text-sm text-[#a23b34]">{error}</p>}

      {review && review.flagged.length === 0 && (
        <p className="mt-3 text-sm text-[#857d6e]">
          Nothing stands out today — no SELL signals, no concentrated positions, and sentiment isn&apos;t
          reinforcing or conflicting with any signal in a notable way.
        </p>
      )}

      {review && review.flagged.length > 0 && (
        <>
          {review.summary && (
            <p className="mt-3 rounded-md bg-[#f4f1ea] px-3 py-2 text-sm leading-relaxed text-[#3a362f]">
              {review.summary}
            </p>
          )}
          <ul className="mt-3 flex flex-col gap-2">
            {review.flagged.map((f) => (
              <li key={f.ticker} className="rounded-md border border-[#ede9df] px-3 py-2 text-sm">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium text-[#1f2420]">{f.ticker}</span>
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
                    <span className="text-xs text-[#a39b8b]">
                      {f.weight_pct.toFixed(1)}% of portfolio
                      {f.market_value !== null && ` (${f.market_value.toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 0 })})`}
                    </span>
                  )}
                  {f.sector && (
                    <span className="rounded-full bg-[#f4f1ea] px-2 py-0.5 text-xs text-[#857d6e]">{f.sector}</span>
                  )}
                </div>
                <p className="mt-1 text-xs text-[#857d6e]">{f.reasons.join(" · ")}</p>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
