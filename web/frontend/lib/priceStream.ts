"use client";

import { useEffect, useRef, useState } from "react";

// PLT-3: live price pushes over a shared WebSocket instead of each mounted
// price badge polling GET /api/v1/search/price on its own timer. Backed by
// the same server-side yfinance poll as that endpoint -- lower latency and
// fewer redundant HTTP round-trips, not a raw tick feed.
const RECONNECT_BASE_DELAY_MS = 1000;
const RECONNECT_MAX_DELAY_MS = 15000;

function wsUrl(): string {
  const base = process.env.NEXT_PUBLIC_API_BASE_URL || (typeof window !== "undefined" ? window.location.origin : "");
  const url = new URL("/api/v1/ws/prices", base);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

export type PriceStreamStatus = "connecting" | "open" | "closed";

/**
 * Subscribes to live price pushes for `tickers`. Reconnects with backoff on
 * its own, but a caller should never depend on it alone -- if the socket is
 * blocked by some network/proxy, `status` just stays "closed" forever and
 * the caller's own polling (already in place everywhere this is used) is
 * what keeps prices updating. Treat a streamed price as a bonus, lower-
 * latency update layered on top of that polling, not a replacement for it.
 */
export function usePriceStream(tickers: string[]) {
  const [prices, setPrices] = useState<Record<string, number>>({});
  const [status, setStatus] = useState<PriceStreamStatus>("connecting");
  const socketRef = useRef<WebSocket | null>(null);
  const tickersRef = useRef<string[]>(tickers);
  const tickerKey = tickers.slice().sort().join(",");

  useEffect(() => {
    tickersRef.current = tickers;
  });

  useEffect(() => {
    if (tickers.length === 0) {
      // Nothing to subscribe to. If a previous, non-empty ticker list was
      // connected, this effect's own cleanup (below) already closed that
      // socket before this run started -- status is left as-is rather than
      // forced to "closed" here, since nothing currently reads it anyway.
      return;
    }

    let cancelled = false;
    let reconnectDelay = RECONNECT_BASE_DELAY_MS;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    function connect() {
      if (cancelled) return;
      setStatus("connecting");
      const socket = new WebSocket(wsUrl());
      socketRef.current = socket;

      socket.onopen = () => {
        if (cancelled) return;
        reconnectDelay = RECONNECT_BASE_DELAY_MS;
        setStatus("open");
        socket.send(JSON.stringify({ tickers: tickersRef.current }));
      };

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data && typeof data.ticker === "string" && typeof data.price === "number") {
            setPrices((prev) => ({ ...prev, [data.ticker]: data.price }));
          }
        } catch {
          // malformed frame -- ignore, the next one will likely be fine
        }
      };

      socket.onclose = () => {
        if (cancelled) return;
        setStatus("closed");
        reconnectTimer = setTimeout(connect, reconnectDelay);
        reconnectDelay = Math.min(reconnectDelay * 2, RECONNECT_MAX_DELAY_MS);
      };

      socket.onerror = () => {
        socket.close();
      };
    }

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socketRef.current?.close();
      socketRef.current = null;
    };
    // Reconnects only when the actual set of tickers changes, not on every
    // render where the caller passes a new array literal with the same values.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tickerKey]);

  useEffect(() => {
    const socket = socketRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ tickers }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tickerKey]);

  return { prices, status };
}
