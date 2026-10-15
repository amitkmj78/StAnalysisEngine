"use client";

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";

import { getPortfolios } from "@/lib/api";
import OrderTicket from "@/components/portfolio/OrderTicket";
import type { Portfolio } from "@/lib/types";

export default function PaperTradingTicketPage() {
  const searchParams = useSearchParams();
  // Which linked paper account to trade through -- a user can have one per
  // portfolio now, so the paper-trading page links here with ?portfolio_id=.
  // Omitted (e.g. a bookmarked link from before) falls back to the
  // backend's own default: the user's oldest active portfolio.
  const portfolioIdParam = searchParams.get("portfolio_id");
  const portfolioId = portfolioIdParam ? Number(portfolioIdParam) : undefined;

  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);

  // DIF-6: the impact preview uses the app's own portfolios, so load them once for the card.
  useEffect(() => {
    getPortfolios().then((res) => setPortfolios(res.portfolios)).catch(() => setPortfolios([]));
  }, []);

  return (
    <div className="mx-auto max-w-lg px-4 py-8">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <h1 className="font-display text-2xl font-semibold text-slate-900 dark:text-slate-50">New Paper Order</h1>
        <Link href="/portfolio/paper-trading" className="text-sm font-medium text-slate-600 dark:text-slate-300 hover:underline">
          ← Back
        </Link>
      </div>

      <div className="mt-6">
        <OrderTicket
          ticker={searchParams.get("ticker")?.toUpperCase() ?? ""}
          portfolioId={portfolioId}
          portfolios={portfolios}
          defaultSide={searchParams.get("side") === "sell" ? "sell" : "buy"}
        />
      </div>
    </div>
  );
}
