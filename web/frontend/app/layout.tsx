import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

import { logout } from "./actions";
import { isAdmin } from "@/lib/admin";
import { getSession } from "@/lib/session";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "StAnalysisEngine",
  description: "AI-assisted price prediction and stock ranking, with the accuracy shown next to the claim.",
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const user = await getSession();

  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col bg-slate-50 text-slate-900">
        {user && (
          <header className="border-b border-slate-200 bg-white">
            <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-3">
              <nav className="flex flex-wrap items-center gap-4 text-sm font-medium">
                <Link href="/predict" className="text-slate-900 hover:text-slate-600">
                  Price Prediction
                </Link>
                <Link href="/stock-finder" className="text-slate-900 hover:text-slate-600">
                  Best Stock Finder
                </Link>
                <Link href="/index-fund" className="text-slate-900 hover:text-slate-600">
                  Best Index Fund
                </Link>
                <Link href="/entry" className="text-slate-900 hover:text-slate-600">
                  Best To Enter Now
                </Link>
                <Link href="/monthly-plan" className="text-slate-900 hover:text-slate-600">
                  Monthly Plan
                </Link>
                <Link href="/strategies" className="text-slate-900 hover:text-slate-600">
                  Strategies
                </Link>
                <Link href="/trade-journal" className="text-slate-900 hover:text-slate-600">
                  Trade Journal
                </Link>
                <Link href="/portfolio" className="text-slate-900 hover:text-slate-600">
                  Portfolio
                </Link>
                <Link href="/chat" className="text-slate-900 hover:text-slate-600">
                  Chat
                </Link>
                {isAdmin(user.email) && (
                  <Link href="/admin/deploy" className="text-slate-900 hover:text-slate-600">
                    Deploy
                  </Link>
                )}
              </nav>
              <div className="flex items-center gap-3 text-sm text-slate-500">
                <span>{user.email}</span>
                <form action={logout}>
                  <button type="submit" className="rounded-md border border-slate-300 px-2.5 py-1 hover:bg-slate-100">
                    Sign out
                  </button>
                </form>
              </div>
            </div>
          </header>
        )}
        <main className="flex-1">{children}</main>
      </body>
    </html>
  );
}
