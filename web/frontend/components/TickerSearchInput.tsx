"use client";

import { useEffect, useRef, useState } from "react";

import { searchTickers } from "@/lib/api";
import type { TickerSearchResult } from "@/lib/types";

export default function TickerSearchInput({
  value,
  onChange,
  placeholder = "Ticker or company name",
  className = "input w-56",
}: {
  value: string;
  onChange: (ticker: string) => void;
  placeholder?: string;
  className?: string;
}) {
  const [query, setQuery] = useState(value);
  const [results, setResults] = useState<TickerSearchResult[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => setQuery(value), [value]);

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  function handleInput(next: string) {
    setQuery(next);
    // Left as-typed, not uppercased — this doubles as the free-text ticker
    // value, and callers already uppercase at submit time. Force-casing here
    // would visibly mangle a company name while the user is still typing it.
    onChange(next.trim());

    if (debounceRef.current) clearTimeout(debounceRef.current);
    const trimmed = next.trim();
    if (trimmed.length < 2) {
      setResults([]);
      setOpen(false);
      return;
    }
    debounceRef.current = setTimeout(async () => {
      setLoading(true);
      try {
        const res = await searchTickers(trimmed);
        setResults(res.results);
        setOpen(res.results.length > 0);
      } catch {
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, 300);
  }

  function select(r: TickerSearchResult) {
    setQuery(r.symbol);
    onChange(r.symbol);
    setOpen(false);
  }

  return (
    <div ref={containerRef} className="relative">
      <input
        value={query}
        onChange={(e) => handleInput(e.target.value)}
        onFocus={() => results.length > 0 && setOpen(true)}
        placeholder={placeholder}
        className={className}
        autoComplete="off"
      />
      {open && (
        <div className="absolute z-20 mt-1 max-h-72 w-72 overflow-y-auto rounded-md border border-slate-200 bg-white shadow-lg">
          {loading && <div className="px-3 py-2 text-xs text-slate-400">Searching…</div>}
          {!loading &&
            results.map((r) => (
              <button
                key={`${r.symbol}-${r.exchange}`}
                type="button"
                onClick={() => select(r)}
                className="flex w-full flex-col items-start gap-0.5 px-3 py-2 text-left text-sm hover:bg-slate-50"
              >
                <span className="font-medium text-slate-900">
                  {r.symbol} <span className="font-normal text-slate-500">— {r.name}</span>
                </span>
                <span className="text-xs text-slate-400">
                  {r.exchange} {r.type && `· ${r.type}`}
                </span>
              </button>
            ))}
        </div>
      )}
    </div>
  );
}
