"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { logout } from "@/app/actions";
import { getMarketOverview } from "@/lib/api";
import type { MarketIndexQuote } from "@/lib/types";

// Fetched every 15 minutes (also server-cached on that same cadence --
// see services/market_overview_service.py), not shorter: this is a
// background header ticker, not a live quote badge.
const MARKET_OVERVIEW_POLL_MS = 15 * 60 * 1000;

function MarketOverviewTicker() {
  const [indices, setIndices] = useState<MarketIndexQuote[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    function load() {
      getMarketOverview()
        .then((res) => {
          if (!cancelled) setIndices(res.indices);
        })
        .catch(() => undefined); // non-fatal -- the header ticker is supplementary
    }
    load();
    const interval = setInterval(load, MARKET_OVERVIEW_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  if (!indices) return null;

  // Duplicated once so the CSS animation can scroll a full loop and land
  // back at an identical starting point with no visible seam/jump --
  // same idiom as MarketNewsTicker.tsx's news-ticker-track.
  const loopIndices = [...indices, ...indices];

  return (
    <div className="hidden overflow-hidden border-t border-slate-100 bg-slate-50 py-1 text-xs sm:block">
      <div className="market-ticker-track flex flex-shrink-0 items-center gap-6 whitespace-nowrap px-4">
        {loopIndices.map((idx, i) => (
          <span key={`${idx.ticker}-${i}`} className="flex flex-none items-center gap-1">
            <span className="font-medium text-slate-700">{idx.label}</span>
            {idx.price !== null ? (
              <>
                <span className="text-slate-600">{idx.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
                {idx.change_pct !== null && (
                  <span className={idx.change_pct >= 0 ? "text-emerald-600" : "text-red-600"}>
                    {idx.change_pct >= 0 ? "▲" : "▼"}
                    {Math.abs(idx.change_pct).toFixed(2)}%
                  </span>
                )}
              </>
            ) : (
              <span className="text-slate-400">—</span>
            )}
          </span>
        ))}
      </div>
    </div>
  );
}

type NavItem = { href: string; label: string };
type NavEntry = { label: string; href: string } | { label: string; items: NavItem[] };

const NAV: NavEntry[] = [
  {
    label: "Research",
    items: [
      { href: "/predict", label: "Forecast" },
      { href: "/predictions", label: "Prediction History" },
      { href: "/stock-finder", label: "Stock Screener" },
      { href: "/charts", label: "Chart grid" },
      { href: "/stock/AAPL", label: "Stock Detail" },
      { href: "/signal-comparison", label: "Quant vs Analyst" },
      { href: "/web-search", label: "Web Search" },
      { href: "/index-fund", label: "Fund Screener" },
      { href: "/entry", label: "Entry Signals" },
      { href: "/top-performers", label: "Top Performers" },
      { href: "/track-record", label: "Track Record" },
    ],
  },
  {
    label: "Planning",
    items: [
      { href: "/monthly-plan", label: "Monthly Plan" },
      { href: "/strategies", label: "Strategies" },
      { href: "/trade-journal", label: "Trade Journal" },
    ],
  },
  {
    label: "Portfolio",
    items: [
      { href: "/portfolio", label: "Holdings" },
      { href: "/earnings", label: "Earnings Calendar" },
      { href: "/portfolio/health", label: "Health Check" },
      { href: "/portfolio/stress-test", label: "Stress Test" },
      { href: "/challenges", label: "Challenges" },
      { href: "/trading-agent", label: "Trading Agent" },
    ],
  },
  {
    label: "Community",
    items: [
      { href: "/community", label: "Idea Feed" },
      { href: "/community/leaderboard", label: "Leaderboard" },
    ],
  },
  {
    label: "Social",
    items: [
      { href: "/social/feed", label: "Feed" },
      { href: "/social/groups", label: "Groups" },
      { href: "/social/chat/general", label: "Chat" },
      { href: "/social/messages", label: "Messages" },
    ],
  },
  {
    label: "Alerts",
    items: [
      { href: "/alerts", label: "Inbox" },
      { href: "/watchlist", label: "Watchlist" },
      { href: "/alerts/conditions", label: "Condition Alerts" },
      { href: "/alerts/settings", label: "Settings" },
    ],
  },
  { label: "Guides", href: "/guides" },
  { label: "Assistant", href: "/chat" },
];

const ADMIN_ENTRY: NavEntry = {
  label: "Admin",
  items: [
    { href: "/admin/users", label: "Users" },
    { href: "/admin/activity", label: "Activity" },
    { href: "/admin/scheduler", label: "Scheduler" },
    { href: "/admin/signal-stability", label: "Signal Stability" },
    { href: "/admin/sql", label: "SQL" },
    { href: "/admin/integrations", label: "Integrations" },
    { href: "/admin/community-reports", label: "Community Reports" },
    { href: "/admin/deploy", label: "Deploy" },
  ],
};

function isGroup(entry: NavEntry): entry is { label: string; items: NavItem[] } {
  return "items" in entry;
}

function NavDropdown({ entry, active }: { entry: { label: string; items: NavItem[] }; active: boolean }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    function onEscape(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onClickOutside);
    document.addEventListener("keydown", onEscape);
    return () => {
      document.removeEventListener("mousedown", onClickOutside);
      document.removeEventListener("keydown", onEscape);
    };
  }, []);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="true"
        aria-expanded={open}
        className={`flex items-center gap-1 rounded-md px-2 py-1.5 transition-colors ${
          active ? "text-slate-900" : "text-slate-600 hover:text-slate-900"
        }`}
      >
        {entry.label}
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className={`h-3.5 w-3.5 transition-transform ${open ? "rotate-180" : ""}`}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M6 9l6 6 6-6" />
        </svg>
      </button>
      {open && (
        <div className="absolute left-0 top-full z-20 mt-1 min-w-[12rem] rounded-md border border-slate-200 bg-white py-1 shadow-lg">
          {entry.items.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              onClick={() => setOpen(false)}
              className="block px-3 py-2 text-sm text-slate-700 hover:bg-slate-50 hover:text-slate-900"
            >
              {item.label}
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

export default function SiteHeader({ email, isAdmin }: { email: string; isAdmin: boolean }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const pathname = usePathname();
  const entries = isAdmin ? [...NAV, ADMIN_ENTRY] : NAV;

  // "Stock Detail" links to a fixed default ticker (/stock/AAPL), but the
  // route itself is dynamic (/stock/[ticker]) -- match on the /stock/
  // prefix so the nav still highlights while browsing any other ticker.
  function matchesNavItem(href: string): boolean {
    if (href.startsWith("/stock/")) return pathname.startsWith("/stock/");
    if (href === "/guides") return pathname.startsWith("/guides");
    return pathname === href;
  }

  function isActive(entry: NavEntry): boolean {
    if (isGroup(entry)) return entry.items.some((i) => matchesNavItem(i.href));
    return matchesNavItem(entry.href);
  }

  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-3 px-4 py-3">
        <button
          type="button"
          onClick={() => setMobileOpen((v) => !v)}
          aria-expanded={mobileOpen}
          aria-label="Toggle navigation menu"
          className="flex h-9 w-9 flex-none items-center justify-center rounded-md border border-slate-300 text-slate-700 sm:hidden"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5">
            {mobileOpen ? (
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 6l12 12M6 18L18 6" />
            ) : (
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 7h16M4 12h16M4 17h16" />
            )}
          </svg>
        </button>

        <nav className="hidden flex-1 items-center gap-1 text-sm font-medium sm:flex">
          {entries.map((entry) =>
            isGroup(entry) ? (
              <NavDropdown key={entry.label} entry={entry} active={isActive(entry)} />
            ) : (
              <Link
                key={entry.href}
                href={entry.href}
                className={`rounded-md px-2 py-1.5 transition-colors ${
                  isActive(entry) ? "text-slate-900" : "text-slate-600 hover:text-slate-900"
                }`}
              >
                {entry.label}
              </Link>
            )
          )}
        </nav>

        <div className="flex flex-1 items-center justify-end gap-3 text-sm text-slate-500 sm:flex-none">
          <span className="hidden truncate sm:inline">{email}</span>
          <form action={logout}>
            <button type="submit" className="flex-none rounded-md border border-slate-300 px-2.5 py-1 hover:bg-slate-100">
              Sign out
            </button>
          </form>
        </div>
      </div>

      <MarketOverviewTicker />

      {mobileOpen && (
        <nav className="flex flex-col gap-4 border-t border-slate-200 px-4 py-3 text-sm font-medium sm:hidden">
          <p className="px-1 text-xs text-slate-500">{email}</p>
          {entries.map((entry) => (
            <div key={entry.label} className="flex flex-col gap-1">
              {isGroup(entry) ? (
                <>
                  <p className="px-2 text-xs font-semibold uppercase tracking-wide text-slate-400">{entry.label}</p>
                  {entry.items.map((item) => (
                    <Link
                      key={item.href}
                      href={item.href}
                      onClick={() => setMobileOpen(false)}
                      className={`rounded-md px-2 py-2 ${
                        pathname === item.href ? "bg-slate-100 text-slate-900" : "text-slate-700 hover:bg-slate-50"
                      }`}
                    >
                      {item.label}
                    </Link>
                  ))}
                </>
              ) : (
                <Link
                  href={entry.href}
                  onClick={() => setMobileOpen(false)}
                  className={`rounded-md px-2 py-2 ${
                    pathname === entry.href ? "bg-slate-100 text-slate-900" : "text-slate-700 hover:bg-slate-50"
                  }`}
                >
                  {entry.label}
                </Link>
              )}
            </div>
          ))}
        </nav>
      )}
    </header>
  );
}
