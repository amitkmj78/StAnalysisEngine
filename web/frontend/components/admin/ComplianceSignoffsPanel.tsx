"use client";

import { useEffect, useState } from "react";

import { ApiError, getComplianceSignoffs, recordComplianceSignoff } from "@/lib/api";
import type { ComplianceSignoff } from "@/lib/api";

export default function ComplianceSignoffsPanel() {
  const [signoffs, setSignoffs] = useState<ComplianceSignoff[] | null>(null);
  const [featureKey, setFeatureKey] = useState("");
  const [reference, setReference] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    try {
      const res = await getComplianceSignoffs();
      setSignoffs(res.signoffs);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load sign-offs.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleRecord() {
    if (!featureKey.trim() || !reference.trim()) return;
    setSaving(true);
    setError(null);
    try {
      await recordComplianceSignoff(featureKey.trim(), reference.trim());
      setFeatureKey("");
      setReference("");
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not record this sign-off.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5">
      <h2 className="font-semibold text-slate-900 dark:text-slate-50">Legal Sign-offs (SAF-9)</h2>
      <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
        A record of real legal review, for paid strategy subscriptions, copy trading, paid mentoring, or public
        performance advertising. This panel existing does not mean any review has happened -- only recording one
        here does, and nothing in the app is currently gated on a row existing here.
      </p>

      <div className="mt-3 flex flex-wrap items-end gap-2">
        <input
          value={featureKey}
          onChange={(e) => setFeatureKey(e.target.value)}
          placeholder="Feature key (e.g. strategy_paid_subscriptions)"
          className="w-64 rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 px-3 py-1.5 text-sm text-slate-900 dark:text-slate-50"
        />
        <input
          value={reference}
          onChange={(e) => setReference(e.target.value)}
          placeholder="Reference (counsel name, memo link, date)"
          className="w-72 rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 px-3 py-1.5 text-sm text-slate-900 dark:text-slate-50"
        />
        <button
          onClick={handleRecord}
          disabled={saving || !featureKey.trim() || !reference.trim()}
          className="rounded-md bg-slate-900 dark:bg-slate-100 px-3 py-1.5 text-sm font-medium text-white dark:text-slate-900 hover:bg-slate-800 dark:hover:bg-slate-200 disabled:opacity-50"
        >
          Record sign-off
        </button>
      </div>

      {error && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {signoffs && signoffs.length > 0 && (
        <ul className="mt-3 flex flex-col gap-1.5 text-sm">
          {signoffs.map((s) => (
            <li key={s.id} className="text-slate-700 dark:text-slate-200">
              <span className="font-medium">{s.feature_key}</span> — {s.reference} (by {s.signed_off_by},{" "}
              {new Date(s.signed_off_at).toLocaleDateString()})
            </li>
          ))}
        </ul>
      )}
      {signoffs && signoffs.length === 0 && <p className="mt-3 text-sm text-slate-500 dark:text-slate-400">No sign-offs recorded.</p>}
    </div>
  );
}
