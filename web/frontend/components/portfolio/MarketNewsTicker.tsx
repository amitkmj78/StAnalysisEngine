"use client";

import { useEffect, useState } from "react";

import { getHotMarketNews } from "@/lib/api";
import type { MarketNewsItem } from "@/lib/types";

// Matches the backend cache (services/market_news_service.py, 5min TTL)
// — polling more often than that would just re-request the same cached
// response.
const REFRESH_INTERVAL_MS = 5 * 60 * 1000;

export default function MarketNewsTicker() {
  const [items, setItems] = useState<MarketNewsItem[]>([]);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const res = await getHotMarketNews();
        if (!cancelled) {
          setItems(res.items);
          setFailed(false);
        }
      } catch {
        if (!cancelled) setFailed(true);
      }
    }

    load();
    const interval = setInterval(load, REFRESH_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  if (failed || items.length === 0) return null;

  // Duplicated once so the CSS animation can scroll a full loop and land
  // back at an identical starting point with no visible seam/jump.
  const loopItems = [...items, ...items];

  return (
    <div className="relative mb-4 overflow-hidden rounded-lg border border-slate-200 bg-white">
      <div className="flex items-center">
        <span className="z-10 flex-shrink-0 bg-red-600 px-3 py-2 text-xs font-bold uppercase tracking-wide text-white">
          Market News
        </span>
        <div className="news-ticker-track flex flex-shrink-0 items-center gap-8 whitespace-nowrap py-2 pl-8">
          {loopItems.map((item, i) => (
            <a
              key={`${item.url}-${i}`}
              href={item.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm text-slate-700 hover:text-slate-900 hover:underline"
            >
              <span>{item.title}</span>
              <span className="text-xs text-slate-400">— {item.source}</span>
            </a>
          ))}
        </div>
      </div>
    </div>
  );
}
