"use client";

import { useEffect, useState } from "react";
import { getStockNews } from "@/lib/api";
import type { StockNewsItem } from "@/lib/types";

// Signal explanation step 2: the company's recent SEC 8-K filings, newest first, each linked to the filing itself.
// A filing carries a date but no time of day, so none is shown. An empty list says so, rather than filling the space.

const EVENT_LABELS: Record<string, string> = {
  EARNINGS: "Earnings",
  EXECUTIVE_CHANGE: "Executive change",
  MATERIAL_AGREEMENT: "Material agreement",
  AGREEMENT_TERMINATED: "Agreement ended",
  ACQUISITION_OR_DISPOSAL: "Acquisition or sale",
  RESTRUCTURING: "Restructuring",
  IMPAIRMENT: "Impairment",
  LISTING_NOTICE: "Listing notice",
  RESTATEMENT: "Restatement",
  SHAREHOLDER_VOTE: "Shareholder vote",
  REG_FD: "Company disclosure",
  OTHER_EVENT: "Other event",
  EXHIBITS: "Exhibits",
  OTHER: "Filing",
};

const DAYS = 90;

export default function RecentNewsPanel({ ticker }: { ticker: string }) {
  const [items, setItems] = useState<StockNewsItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getStockNews(ticker, DAYS)
      .then((res) => {
        if (!cancelled) setItems(res.items);
      })
      .catch(() => {
        if (!cancelled) setError("Recent filings could not be loaded.");
      });
    return () => {
      cancelled = true;
    };
  }, [ticker]);

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5">
      <h3 className="text-sm font-semibold text-slate-900">Recent news</h3>
      <p className="mt-1 text-xs text-slate-500">
        Company announcements filed with the SEC (Form 8-K) in the last {DAYS} days. Filings have dates only, not times.
      </p>

      {error && <p className="mt-3 text-sm text-slate-500">{error}</p>}
      {!error && items === null && <p className="mt-3 text-sm text-slate-500">Loading filings…</p>}
      {items && items.length === 0 && (
        <p className="mt-3 text-sm text-slate-500">No 8-K filings in the last {DAYS} days.</p>
      )}
      {items && items.length > 0 && (
        <ul className="mt-3 divide-y divide-slate-100">
          {items.map((item) => (
            <li key={item.id} className="flex flex-col gap-0.5 py-2.5 text-sm">
              <div className="flex flex-wrap items-baseline gap-x-2">
                <span className="font-mono text-xs text-slate-500">{item.filed_on}</span>
                <span className="rounded bg-slate-100 px-1.5 py-0.5 text-xs font-medium text-slate-700">
                  {EVENT_LABELS[item.event_type] ?? item.event_type}
                </span>
              </div>
              <a href={item.url} target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:underline">
                {item.title}
              </a>
              <span className="text-xs text-slate-400">{item.publisher}</span>
              {item.summary && (
                <details className="mt-1 text-sm text-slate-700">
                  <summary className="cursor-pointer text-xs font-medium text-slate-600">
                    {item.event_type === "EARNINGS" ? "Press release summary" : "Summary of filing"}
                  </summary>
                  <p className="mt-1 leading-relaxed">{item.summary}</p>
                  <p className="mt-1 text-xs text-slate-400">
                    {item.event_type === "EARNINGS"
                      ? "Summarized from the press release only. Analyst questions on the call are not covered."
                      : "Written by an AI model from this filing's text. Every figure in it appears in the filing."}
                  </p>
                </details>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
