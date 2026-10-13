"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import { getMarketRegime } from "@/lib/api";

export default function RegimeMethodologyPage() {
  const [paragraphs, setParagraphs] = useState<string[] | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    getMarketRegime()
      .then((res) => setParagraphs(res.methodology ?? []))
      .catch(() => setFailed(true));
  }, []);

  return (
    <div className="mx-auto max-w-2xl px-4 py-10">
      <Link href="/predict" className="text-sm text-[var(--pf-accent)] hover:underline">
        ← Back to the app
      </Link>
      <h1 className="font-display mt-2 text-2xl font-semibold text-slate-900 dark:text-slate-50">Market regime: methodology</h1>
      <p className="mt-2 text-sm leading-relaxed text-slate-600 dark:text-slate-300">
        The statistics and test results behind the regime banner. The banner&apos;s readings are for information,
        and the regime label has not been validated.
      </p>

      {failed && <p className="mt-6 text-sm text-red-700">Could not load the methodology right now.</p>}
      {!failed && paragraphs === null && <p className="mt-6 text-sm text-slate-500 dark:text-slate-400">Loading…</p>}
      {paragraphs && paragraphs.length === 0 && (
        <p className="mt-6 text-sm text-slate-500 dark:text-slate-400">No regime reading is available yet.</p>
      )}
      {paragraphs && paragraphs.length > 0 && (
        <div className="mt-6 flex flex-col gap-4">
          {paragraphs.map((p, i) => (
            <p key={i} className="text-sm leading-relaxed text-slate-700 dark:text-slate-200">
              {p}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
