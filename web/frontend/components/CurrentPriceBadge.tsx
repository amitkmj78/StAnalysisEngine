"use client";

import { useEffect, useState } from "react";

import { getCurrentPrice } from "@/lib/api";

export default function CurrentPriceBadge({ ticker }: { ticker: string }) {
  const [price, setPrice] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const trimmed = ticker.trim().toUpperCase();
    if (!trimmed) {
      setPrice(null);
      setFailed(false);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setFailed(false);
    const debounce = setTimeout(async () => {
      try {
        const res = await getCurrentPrice(trimmed);
        if (!cancelled) setPrice(res.price);
      } catch {
        if (!cancelled) setFailed(true);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }, 400);

    return () => {
      cancelled = true;
      clearTimeout(debounce);
    };
  }, [ticker]);

  if (!ticker.trim()) return null;

  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs font-medium text-slate-500">Current Price</span>
      <span className="flex h-[38px] items-center text-sm text-slate-700">
        {loading && "…"}
        {!loading && failed && <span className="text-slate-400">unavailable</span>}
        {!loading && !failed && price !== null && <strong>${price.toFixed(2)}</strong>}
        {!loading && !failed && price === null && <span className="text-slate-400">no data</span>}
      </span>
    </div>
  );
}
