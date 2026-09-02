"use client";

import { useEffect, useRef, useState } from "react";

import { ApiError, getCrawlSearchStatus, startCrawlSearchCrawl, stopCrawlSearchCrawl } from "@/lib/api";
import type { CrawlSearchStatus } from "@/lib/types";

const POLL_INTERVAL_MS = 3000;

function fmtTime(unixSeconds: number | null): string {
  if (unixSeconds === null) return "—";
  return new Date(unixSeconds * 1000).toLocaleString();
}

export default function CrawlSearchControl() {
  const [status, setStatus] = useState<CrawlSearchStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  async function poll() {
    try {
      const res = await getCrawlSearchStatus();
      setStatus(res);
      setError(null);
    } catch (err) {
      // Most likely CRAWLSEARCH_API_URL not configured, or CrawlSearch is
      // down — surfaced once, not spammed on every poll tick.
      setError(err instanceof ApiError ? err.message : "Could not reach CrawlSearch.");
    }
  }

  useEffect(() => {
    poll();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  useEffect(() => {
    // Only poll on an interval while a crawl is actually running — no
    // point hammering the endpoint once it's idle.
    if (status?.running) {
      if (!pollRef.current) pollRef.current = setInterval(poll, POLL_INTERVAL_MS);
    } else if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, [status?.running]);

  async function handleStart() {
    setStarting(true);
    setError(null);
    try {
      const res = await startCrawlSearchCrawl();
      setStatus(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not start the crawl.");
    } finally {
      setStarting(false);
    }
  }

  async function handleStop() {
    setStopping(true);
    setError(null);
    try {
      await stopCrawlSearchCrawl();
      await poll();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not stop the crawl.");
    } finally {
      setStopping(false);
    }
  }

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h2 className="font-semibold text-slate-900">CrawlSearch — On-Demand Crawl</h2>
          <p className="mt-1 text-sm text-slate-600">
            Triggers a real crawl cycle across every enabled domain right now, instead of waiting for the
            6-hourly systemd timer. Runs on CrawlSearch&apos;s own server in the background — this page just
            starts it and polls progress; closing this tab doesn&apos;t stop it.
          </p>
        </div>
        <div className="flex gap-2">
          {status?.running ? (
            <button
              onClick={handleStop}
              disabled={stopping}
              className="rounded-md border border-red-200 px-3 py-1.5 text-xs font-medium text-red-700 hover:bg-red-50 disabled:opacity-50"
            >
              {stopping ? "Stopping…" : "Stop Crawl"}
            </button>
          ) : (
            <button
              onClick={handleStart}
              disabled={starting}
              className="rounded-md bg-slate-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-800 disabled:opacity-50"
            >
              {starting ? "Starting…" : "Crawl Now"}
            </button>
          )}
        </div>
      </div>

      {error && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {status && (
        <div className="mt-4 flex flex-col gap-2 text-sm">
          <div className="flex flex-wrap items-center gap-2">
            <span
              className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
                status.running ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-600"
              }`}
            >
              {status.running ? "Running" : "Idle"}
            </span>
            {status.cancelled && (
              <span className="rounded-full bg-amber-50 px-2 py-0.5 text-xs font-semibold text-amber-700">
                Last run was stopped early
              </span>
            )}
            {status.error && (
              <span className="rounded-full bg-red-50 px-2 py-0.5 text-xs font-semibold text-red-700">
                Last run errored
              </span>
            )}
          </div>

          {status.running && status.current_domain && (
            <p className="text-slate-600">
              Currently crawling <strong>{status.current_domain}</strong> — {status.current_domain_pages} page
              {status.current_domain_pages === 1 ? "" : "s"} so far this domain, {status.pages_crawled} total this
              run.
            </p>
          )}
          {!status.running && (
            <p className="text-slate-500">
              Started {fmtTime(status.started_at)}, finished {fmtTime(status.finished_at)} —{" "}
              {status.pages_crawled} page{status.pages_crawled === 1 ? "" : "s"} crawled.
            </p>
          )}
          {status.error && <p className="rounded-md bg-red-50 px-3 py-2 text-xs text-red-700">{status.error}</p>}

          {status.completed.length > 0 && (
            <div className="mt-2 max-h-64 overflow-auto rounded-md border border-slate-200">
              <table className="min-w-full text-xs">
                <thead>
                  <tr className="sticky top-0 border-b border-slate-200 bg-slate-50 text-left font-medium uppercase tracking-wide text-slate-500">
                    <th className="px-2 py-1.5">Domain</th>
                    <th className="px-2 py-1.5 text-right">Crawled</th>
                    <th className="px-2 py-1.5 text-right">Robots-skipped</th>
                    <th className="px-2 py-1.5 text-right">Filtered</th>
                    <th className="px-2 py-1.5 text-right">Failed</th>
                    <th className="px-2 py-1.5">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {status.completed.map((d) => (
                    <tr key={d.domain} className="border-b border-slate-100 last:border-0">
                      <td className="px-2 py-1.5 font-medium text-slate-800">{d.domain}</td>
                      <td className="px-2 py-1.5 text-right text-slate-600">{d.pages_crawled}</td>
                      <td className="px-2 py-1.5 text-right text-slate-600">{d.pages_skipped_robots}</td>
                      <td className="px-2 py-1.5 text-right text-slate-600">{d.pages_skipped_filter}</td>
                      <td className="px-2 py-1.5 text-right text-slate-600">{d.pages_failed}</td>
                      <td className="px-2 py-1.5 text-slate-500">
                        {d.disabled
                          ? "Disabled"
                          : d.skipped_domain
                          ? "Skipped (robots.txt)"
                          : d.cancelled
                          ? "Cancelled"
                          : "Done"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
