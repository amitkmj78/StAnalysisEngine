"use client";

import { useState } from "react";

import { ApiError, runWebSearch } from "@/lib/api";
import type { WebSearchResponse } from "@/lib/types";

const RESULT_COUNTS = [3, 5, 10, 15, 20];

export default function WebSearchPage() {
  const [query, setQuery] = useState("");
  const [maxResults, setMaxResults] = useState(5);
  const [includeRaw, setIncludeRaw] = useState(false);
  const [data, setData] = useState<WebSearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    setLoading(true);
    setError(null);
    try {
      setData(await runWebSearch(q, maxResults, includeRaw));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Search failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-10">
      <h1 className="text-2xl font-semibold text-slate-900">Web Search</h1>
      <p className="mt-2 text-sm leading-relaxed text-slate-600">
        Self-hosted search — DuckDuckGo finds candidate pages, then this app fetches and extracts
        real article content from each one itself (not just a one-line snippet). No third-party
        search API involved.
      </p>

      <form
        onSubmit={handleSearch}
        className="mt-6 flex flex-col gap-3 rounded-lg border border-slate-200 bg-white p-5"
      >
        <input
          type="text"
          placeholder="Search the web…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="rounded-md border border-slate-300 px-3 py-2 text-sm"
        />
        <div className="flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-slate-600">
            Results:
            <select
              value={maxResults}
              onChange={(e) => setMaxResults(Number(e.target.value))}
              className="rounded-md border border-slate-300 px-2 py-1 text-sm"
            >
              {RESULT_COUNTS.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 text-sm text-slate-600">
            <input type="checkbox" checked={includeRaw} onChange={(e) => setIncludeRaw(e.target.checked)} />
            Include full extracted content
          </label>
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="ml-auto rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
          >
            {loading ? "Searching…" : "Search"}
          </button>
        </div>
      </form>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {data && !loading && (
        <div className="mt-6">
          <p className="text-xs text-slate-500">
            {data.results.length} result{data.results.length === 1 ? "" : "s"} for &quot;{data.query}&quot; in{" "}
            {data.response_time_ms}ms
          </p>

          {data.results.length === 0 ? (
            <div className="mt-3 rounded-lg border border-slate-200 bg-white p-5 text-sm text-slate-500">
              No results found.
            </div>
          ) : (
            <div className="mt-3 flex flex-col gap-3">
              {data.results.map((r) => (
                <div key={r.url} className="rounded-lg border border-slate-200 bg-white p-4">
                  <div className="flex items-start justify-between gap-3">
                    <a
                      href={r.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="font-medium text-slate-900 hover:underline"
                    >
                      {r.title}
                    </a>
                    <span className="shrink-0 rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-500">
                      {(r.score * 100).toFixed(0)}% match
                    </span>
                  </div>
                  <a
                    href={r.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="mt-1 block truncate text-xs text-slate-400 hover:underline"
                  >
                    {r.url}
                  </a>
                  <p className="mt-2 text-sm leading-relaxed text-slate-600">{r.content}</p>
                  {r.raw_content && r.raw_content !== r.content && (
                    <details className="mt-2">
                      <summary className="cursor-pointer text-xs font-medium text-slate-500 hover:text-slate-700">
                        Show full extracted content
                      </summary>
                      <p className="mt-2 whitespace-pre-wrap text-xs leading-relaxed text-slate-600">
                        {r.raw_content}
                      </p>
                    </details>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
