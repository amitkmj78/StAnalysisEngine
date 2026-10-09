"use client";

import { useEffect, useState } from "react";

import { getHotMarketNews, getMarketOverview } from "@/lib/api";
import type { MarketIndexQuote, MarketNewsItem } from "@/lib/types";

// One combined strip instead of two separate ones (prices + headlines)
// -- fewer stacked bars across the whole app, and shown on every
// screen size (the old price-only ticker was desktop-only; this one
// isn't, so mobile gets both kinds of content too).
const PRICE_POLL_MS = 15 * 60 * 1000;
const NEWS_POLL_MS = 5 * 60 * 1000;

type TickerEntry =
  | { kind: "price"; key: string; quote: MarketIndexQuote }
  | { kind: "news"; key: string; item: MarketNewsItem };

export default function UnifiedMarketTicker() {
  const [indices, setIndices] = useState<MarketIndexQuote[]>([]);
  const [news, setNews] = useState<MarketNewsItem[]>([]);

  useEffect(() => {
    let cancelled = false;
    function load() {
      getMarketOverview()
        .then((res) => {
          if (!cancelled) setIndices(res.indices);
        })
        .catch(() => undefined); // non-fatal -- the ticker is supplementary
    }
    load();
    const interval = setInterval(load, PRICE_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    function load() {
      getHotMarketNews()
        .then((res) => {
          if (!cancelled) setNews(res.items);
        })
        .catch(() => undefined);
    }
    load();
    const interval = setInterval(load, NEWS_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const entries: TickerEntry[] = [
    ...indices.map((quote): TickerEntry => ({ kind: "price", key: `price-${quote.ticker}`, quote })),
    ...news.map((item, i): TickerEntry => ({ kind: "news", key: `news-${i}`, item })),
  ];

  if (entries.length === 0) return null;

  // Duplicated once so the CSS animation can scroll a full loop and land
  // back at an identical starting point with no visible seam/jump --
  // same idiom as the news-only ticker this replaces.
  const loopEntries = [...entries, ...entries];

  return (
    <div className="relative overflow-hidden border-b border-slate-200 bg-white">
      <div className="market-ticker-track text-strip flex flex-shrink-0 items-center gap-6 whitespace-nowrap py-1.5 pl-4">
        {loopEntries.map((entry, i) => (
          <span key={`${entry.key}-${i}`} className="flex flex-none items-center gap-1.5">
            {entry.kind === "price" ? (
              <>
                <span className="font-medium text-slate-700">{entry.quote.label}</span>
                {entry.quote.price !== null ? (
                  <>
                    <span className="text-slate-600">{entry.quote.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
                    {entry.quote.change_pct !== null && (
                      <span className={entry.quote.change_pct >= 0 ? "text-emerald-600" : "text-red-600"}>
                        {entry.quote.change_pct >= 0 ? "▲" : "▼"}
                        {Math.abs(entry.quote.change_pct).toFixed(2)}%
                      </span>
                    )}
                  </>
                ) : (
                  <span className="text-slate-400">—</span>
                )}
              </>
            ) : (
              <a
                href={entry.item.url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 text-slate-700 hover:text-slate-900 hover:underline"
              >
                <span className="text-strip-meta rounded bg-[var(--pf-accent)] px-1.5 py-0.5 font-bold uppercase tracking-wide text-white">
                  News
                </span>
                <span>{entry.item.title}</span>
              </a>
            )}
          </span>
        ))}
      </div>
    </div>
  );
}
