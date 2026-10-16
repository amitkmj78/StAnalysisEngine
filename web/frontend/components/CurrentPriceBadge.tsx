"use client";

import { useEffect, useState } from "react";

import { getCurrentPrice } from "@/lib/api";
import type { ExtendedHoursPrice } from "@/lib/types";
import { usePriceStream } from "@/lib/priceStream";

const POLL_INTERVAL_MS = 10000;

export default function CurrentPriceBadge({ ticker, refreshKey }: { ticker: string; refreshKey?: number | string }) {
  const trimmedTicker = ticker.trim().toUpperCase();
  // PLT-3: a lower-latency price push layered on top of the poll below, not
  // a replacement for it -- if the socket never connects, this hook's map
  // just stays empty and nothing here changes.
  const { prices: streamedPrices } = usePriceStream(trimmedTicker ? [trimmedTicker] : []);
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

  // A streamed price, when one has arrived, simply wins over the polled
  // one -- computed at render time rather than synced into state via an
  // effect, so there's no extra render lag and nothing to keep in sync.
  const streamedPrice = trimmedTicker ? streamedPrices[trimmedTicker] : undefined;
  const displayPrice = streamedPrice ?? price;
  const displayFailed = streamedPrice === undefined && failed;

  if (!ticker.trim()) return null;

  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs font-medium text-slate-500 dark:text-slate-400">Current Price</span>
      <span className="flex min-h-[38px] flex-col justify-center text-sm text-slate-700 dark:text-slate-200">
        {loading && "…"}
        {!loading && displayFailed && <span className="text-slate-400 dark:text-slate-500">unavailable</span>}
        {!loading && !displayFailed && displayPrice !== null && (
          <>
            <strong>${displayPrice.toFixed(2)}</strong>
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
        {!loading && !displayFailed && displayPrice === null && <span className="text-slate-400 dark:text-slate-500">no data</span>}
      </span>
    </div>
  );
}
