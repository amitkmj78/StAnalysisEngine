"use client";

import { useEffect, useState } from "react";

import { getEconomicCalendar } from "@/lib/api";
import type { EconomicCalendarResponse } from "@/lib/types";

// Known, publicly-scheduled macro data releases a few days out -- pure
// awareness, never a directional call (contrast services/market_internals_
// service.py's Market Direction score, which failed its own validation
// gate three times; this is a calendar of announced dates, not a
// prediction, so none of that gate applies here -- see services/economic_
// calendar_service.py's module docstring). The disclosure line below
// stays visible whenever the banner renders at all, same "non-negotiable,
// never collapsed away" posture as RegimeBanner's own disclosure.
function formatEventDate(iso: string): string {
  const d = new Date(`${iso}T00:00:00`);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function daysUntilLabel(days: number): string {
  if (days === 0) return "today";
  if (days === 1) return "tomorrow";
  return `in ${days} days`;
}

export default function EconomicCalendarBanner() {
  const [data, setData] = useState<EconomicCalendarResponse | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getEconomicCalendar()
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch(() => {
        if (!cancelled) setData(null);
      })
      .finally(() => {
        if (!cancelled) setLoaded(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const upcoming = data?.upcoming ?? [];
  if (!loaded || !data || !data.available || upcoming.length === 0) {
    return null; // not enabled, no data, fetch failed, or nothing in the window -- nothing to show
  }

  const [next, ...rest] = upcoming;

  return (
    <div className="border-b px-4 py-2 text-strip border-[var(--pf-border)] bg-[var(--pf-accent-soft)] text-[var(--pf-accent)]">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-3 gap-y-1">
        <span className="font-semibold">
          📅 {next.title} {daysUntilLabel(next.days_until)} ({formatEventDate(next.event_date)})
        </span>
        <span className="text-strip-meta opacity-80">{data.disclosure}</span>
        {rest.length > 0 && (
          <button
            type="button"
            onClick={() => setExpanded((e) => !e)}
            className="ml-auto shrink-0 underline decoration-dotted underline-offset-2"
          >
            {expanded ? "Hide" : `+${rest.length} more`}
          </button>
        )}
      </div>
      {expanded && rest.length > 0 && (
        <ul className="mx-auto mt-1 max-w-6xl text-strip-meta leading-relaxed opacity-90">
          {rest.map((event) => (
            <li key={`${event.event_type}-${event.event_date}`}>
              {event.title} {daysUntilLabel(event.days_until)} ({formatEventDate(event.event_date)})
              {event.note ? ` — ${event.note}` : ""}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
