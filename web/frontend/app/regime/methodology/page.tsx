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
      <Link href="/predict" className="text-sm text-indigo-600 hover:underline">
        ← Back to the app
      </Link>
      <h1 className="mt-2 text-2xl font-semibold text-slate-900">Market regime: methodology</h1>
      <p className="mt-2 text-sm leading-relaxed text-slate-600">
        The statistics and test results behind the regime banner. The banner&apos;s readings are for information,
        and the regime label has not been validated.
      </p>

      {failed && <p className="mt-6 text-sm text-red-700">Could not load the methodology right now.</p>}
      {!failed && paragraphs === null && <p className="mt-6 text-sm text-slate-500">Loading…</p>}
      {paragraphs && paragraphs.length === 0 && (
        <p className="mt-6 text-sm text-slate-500">No regime reading is available yet.</p>
      )}
      {paragraphs && paragraphs.length > 0 && (
        <div className="mt-6 flex flex-col gap-4">
          {paragraphs.map((p, i) => (
            <p key={i} className="text-sm leading-relaxed text-slate-700">
              {p}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
