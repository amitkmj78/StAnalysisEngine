"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { logout } from "@/app/actions";

const NAV_LINKS = [
  { href: "/predict", label: "Price Prediction" },
  { href: "/predictions", label: "My Predictions" },
  { href: "/stock-finder", label: "Best Stock Finder" },
  { href: "/index-fund", label: "Best Index Fund" },
  { href: "/entry", label: "Best To Enter Now" },
  { href: "/top-performers", label: "Top Performers" },
  { href: "/track-record", label: "Track Record" },
  { href: "/monthly-plan", label: "Monthly Plan" },
  { href: "/strategies", label: "Strategies" },
  { href: "/trade-journal", label: "Trade Journal" },
  { href: "/portfolio", label: "Portfolio" },
  { href: "/watchlist", label: "Watchlist" },
  { href: "/chat", label: "Chat" },
];

const ADMIN_LINKS = [
  { href: "/admin/users", label: "Users" },
  { href: "/admin/activity", label: "Activity" },
  { href: "/admin/scheduler", label: "Scheduler" },
  { href: "/admin/deploy", label: "Deploy" },
];

export default function SiteHeader({ email, isAdmin }: { email: string; isAdmin: boolean }) {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();
  const links = isAdmin ? [...NAV_LINKS, ...ADMIN_LINKS] : NAV_LINKS;

  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-3 px-4 py-3">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-label="Toggle navigation menu"
          className="flex h-9 w-9 flex-none items-center justify-center rounded-md border border-slate-300 text-slate-700 sm:hidden"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5">
            {open ? (
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 6l12 12M6 18L18 6" />
            ) : (
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 7h16M4 12h16M4 17h16" />
            )}
          </svg>
        </button>

        <nav className="hidden flex-1 flex-wrap items-center gap-4 text-sm font-medium sm:flex">
          {links.map((l) => (
            <Link key={l.href} href={l.href} className="text-slate-900 hover:text-slate-600">
              {l.label}
            </Link>
          ))}
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

      {open && (
        <nav className="flex flex-col gap-1 border-t border-slate-200 px-4 py-3 text-sm font-medium sm:hidden">
          <p className="px-1 pb-2 text-xs text-slate-500">{email}</p>
          {links.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              onClick={() => setOpen(false)}
              className={`rounded-md px-2 py-2 ${
                pathname === l.href ? "bg-slate-100 text-slate-900" : "text-slate-700 hover:bg-slate-50"
              }`}
            >
              {l.label}
            </Link>
          ))}
        </nav>
      )}
    </header>
  );
}
