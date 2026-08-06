"use client";

import { useEffect, useState } from "react";

import { ApiError, getAdminSettings, setDailyQuota } from "@/lib/api";

export default function DailyQuotaControls() {
  const [quota, setQuota] = useState<number | null>(null);
  const [quotaInput, setQuotaInput] = useState("");
  const [error, setError] = useState<string | null>(null);

  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  async function load() {
    setError(null);
    try {
      const settings = await getAdminSettings();
      setQuota(settings.daily_quota);
      setQuotaInput(String(settings.daily_quota));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load daily quota setting.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleSave() {
    const value = Math.trunc(Number(quotaInput));
    if (!Number.isFinite(value) || value <= 0 || value > 100000) {
      setSaveError("Enter a whole number greater than 0 and at most 100,000.");
      return;
    }
    setSaving(true);
    setSaveError(null);
    setSaved(false);
    try {
      const res = await setDailyQuota(value);
      setQuota(res.daily_quota);
      setQuotaInput(String(res.daily_quota));
      setSaved(true);
    } catch (err) {
      setSaveError(err instanceof ApiError ? err.message : "Failed to save daily quota.");
    } finally {
      setSaving(false);
    }
  }

  const dirty = quota !== null && quotaInput !== String(quota);

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5">
      <h2 className="font-semibold text-slate-900">Daily Request Limit</h2>
      <p className="mt-1 text-sm text-slate-600">
        Per-user cap on quota-gated API requests in any rolling 24-hour window (predictions, screeners,
        portfolio actions, chat, and similar) — backed by this same activity log, so it survives a
        restart. Takes effect on the very next request; no deploy needed.
      </p>

      {error && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <div className="mt-4 flex flex-wrap items-end gap-2">
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-slate-500">Requests per 24h, per user</label>
          <input
            type="number"
            min={1}
            max={100000}
            step={1}
            value={quotaInput}
            onChange={(e) => {
              setQuotaInput(e.target.value);
              setSaved(false);
            }}
            className="w-32 rounded-md border border-slate-300 px-2 py-1.5 text-sm"
          />
        </div>
        <button
          onClick={handleSave}
          disabled={saving || quota === null || !dirty}
          className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
        >
          {saving ? "Saving…" : "Save"}
        </button>
        {saved && !dirty && <span className="text-xs font-medium text-emerald-700">Saved</span>}
      </div>

      {saveError && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{saveError}</p>}
    </div>
  );
}
