"use client";

import { useEffect, useState } from "react";
import { usePlaidLink } from "react-plaid-link";

import { ApiError, createPlaidLinkToken, createPortfolio, exchangePlaidPublicToken } from "@/lib/api";

export interface PlaidConnectResult {
  portfolioId: number;
  portfolioName: string;
  positionsImported: number;
  syncOk: boolean;
}

/**
 * Shared brokerage-connect flow (Plaid Link) — extracted so it can be
 * triggered from anywhere with one function call (e.g. directly from the
 * Portfolio page's "More" menu) instead of only from a full page that
 * exists just to host this one button. Every connection gets its own new
 * portfolio named after the institution, same as before.
 */
export function usePlaidConnect(onImported?: (result: PlaidConnectResult) => void) {
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [linkToken, setLinkToken] = useState<string | null>(null);

  const { open: openPlaidLink, ready: plaidLinkReady } = usePlaidLink({
    token: linkToken,
    onSuccess: async (publicToken, metadata) => {
      if (!publicToken) return;
      setError(null);
      try {
        const institutionName = metadata.institution?.name ?? "Connected Brokerage";
        const newPortfolio = await createPortfolio(institutionName);
        const res = await exchangePlaidPublicToken(
          publicToken,
          newPortfolio.id,
          metadata.institution?.institution_id ?? undefined,
          metadata.institution?.name ?? undefined,
        );
        if (res.sync.status === "login_required" || res.sync.status === "error") {
          setError("Connected, but the first sync didn't complete. Try \"Sync Now\" from Linked Accounts in a moment.");
        }
        onImported?.({
          portfolioId: newPortfolio.id,
          portfolioName: institutionName,
          positionsImported: res.sync.positions_upserted,
          syncOk: res.sync.status === "success",
        });
      } catch (err) {
        setError(err instanceof ApiError ? err.message : "Could not finish connecting this account.");
      } finally {
        setConnecting(false);
        setLinkToken(null);
      }
    },
    onExit: () => {
      setConnecting(false);
      setLinkToken(null);
    },
  });

  useEffect(() => {
    if (linkToken && plaidLinkReady) openPlaidLink();
  }, [linkToken, plaidLinkReady, openPlaidLink]);

  async function connect() {
    setError(null);
    setConnecting(true);
    try {
      const res = await createPlaidLinkToken();
      setLinkToken(res.link_token);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not start connecting a brokerage account.");
      setConnecting(false);
    }
  }

  return { connect, connecting, error };
}
