"use client";

import { useEffect, useState } from "react";

import { getCurrentPrice } from "@/lib/api";
import type { ExtendedHoursPrice } from "@/lib/types";

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

    // `refreshKey` changing (e.g. right after an analysis finishes) re-runs
    // this fetch even though `ticker` itself didn't change — so the badge
    // doesn't sit on a price fetched before a slow analysis started while
    // the actual market has since moved. Skip the debounce in that case;
    // it's a single deliberate refresh, not a user still typing a ticker.
    let cancelled = false;
    setLoading(true);
    setFailed(false);
    const delay = refreshKey !== undefined ? 0 : 400;
    const debounce = setTimeout(async () => {
      try {
        const res = await getCurrentPrice(trimmed);
        if (!cancelled) {
          setPrice(res.price);
          setExtendedHours(res.extended_hours);
        }
      } catch {
        if (!cancelled) setFailed(true);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }, delay);

    return () => {
      cancelled = true;
      clearTimeout(debounce);
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
