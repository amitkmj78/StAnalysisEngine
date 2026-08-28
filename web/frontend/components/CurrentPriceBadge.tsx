"use client";

import { useEffect, useState } from "react";

import { getCurrentPrice } from "@/lib/api";
import type { ExtendedHoursPrice } from "@/lib/types";

const POLL_INTERVAL_MS = 10000;

export default function CurrentPriceBadge({ ticker, refreshKey }: { ticker: string; refreshKey?: number | string }) {
  const [price, setPrice] = useState<number | null>(null);
  const [extendedHours, setExtendedHours] = useState<ExtendedHoursPrice | null>(null);
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const trimmed = ticker.trim().toUpperCase();
    if (!trimmed) {
      setPrice(null);
      setExtendedHours(null);
      setFailed(false);
      return;
    }

    let cancelled = false;
    let interval: ReturnType<typeof setInterval> | null = null;

    async function fetchPrice(showLoading: boolean) {
      if (showLoading) setLoading(true);
      try {
        const res = await getCurrentPrice(trimmed);
        if (!cancelled) {
          setPrice(res.price);
          setExtendedHours(res.extended_hours);
          setFailed(false);
        }
      } catch {
        if (!cancelled) setFailed(true);
      } finally {
        if (!cancelled && showLoading) setLoading(false);
      }
    }

    // `refreshKey` changing (e.g. right after an analysis finishes) skips
    // the debounce for a single deliberate refresh. Otherwise the first
    // fetch is debounced so a user still typing a ticker doesn't fire one
    // per keystroke. Once settled, poll every 10s so the price stays
    // reasonably fresh without needing any further trigger — this used to
    // poll every 1s on the (false) assumption that the underlying lookup
    // was cached 60s server-side; it's actually cached ~2-10s, so a 1s
    // poll was a real, uncached yfinance call almost every single tick,
    // across every mounted badge on /portfolio, /predict, /chat, and
    // /watchlist simultaneously — a real contributor to yfinance rate
    // limiting, not just wasted requests.
    setFailed(false);
    const delay = refreshKey !== undefined ? 0 : 400;
    const debounce = setTimeout(async () => {
      await fetchPrice(true);
      if (!cancelled) {
        interval = setInterval(() => fetchPrice(false), POLL_INTERVAL_MS);
      }
    }, delay);

    return () => {
      cancelled = true;
      clearTimeout(debounce);
      if (interval) clearInterval(interval);
    };
  }, [ticker, refreshKey]);

  if (!ticker.trim()) return null;

  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs font-medium text-slate-500">Current Price</span>
      <span className="flex min-h-[38px] flex-col justify-center text-sm text-slate-700">
        {loading && "…"}
        {!loading && failed && <span className="text-slate-400">unavailable</span>}
        {!loading && !failed && price !== null && (
          <>
            <strong>${price.toFixed(2)}</strong>
            {extendedHours && (
              <span className={`text-xs ${(extendedHours.change_pct ?? 0) >= 0 ? "text-emerald-600" : "text-red-600"}`}>
                {extendedHours.state === "POST" ? "After hours" : "Pre-market"}: ${extendedHours.price.toFixed(2)}
                {extendedHours.change_pct !== null && (
                  <> ({extendedHours.change_pct >= 0 ? "+" : ""}{extendedHours.change_pct.toFixed(2)}%)</>
                )}
              </span>
            )}
          </>
        )}
        {!loading && !failed && price === null && <span className="text-slate-400">no data</span>}
      </span>
    </div>
  );
}
