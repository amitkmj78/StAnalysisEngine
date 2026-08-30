"use client";

import { useEffect, useState } from "react";

import { ApiError, getAdminSettings, setPriceDataProvider } from "@/lib/api";

const PROVIDERS = [
  { value: "yahoo" as const, label: "Yahoo (yfinance)", note: "Free, unofficial — the current default." },
  { value: "alpaca" as const, label: "Alpaca", note: "Free real-time IEX feed — needs API keys configured on the server." },
];

export default function PriceProviderControls() {
  const [provider, setProvider] = useState<"yahoo" | "alpaca" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  async function load() {
    setError(null);
    try {
      const settings = await getAdminSettings();
      setProvider(settings.price_data_provider);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load the live-price provider setting.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleSelect(next: "yahoo" | "alpaca") {
    if (next === provider) return;
    setSaving(true);
    setSaveError(null);
    setSaved(false);
    try {
      const res = await setPriceDataProvider(next);
      setProvider(res.price_data_provider);
      setSaved(true);
    } catch (err) {
      setSaveError(err instanceof ApiError ? err.message : "Failed to switch live-price provider.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5">
      <h2 className="font-semibold text-slate-900">Live Price Provider</h2>
      <p className="mt-1 text-sm text-slate-600">
        Which source get_latest_price and the extended-hours badge use. Takes effect on the very next
        price request — no restart needed. Extended-hours (pre/post-market) pricing only works on Yahoo;
        switching to Alpaca disables it rather than guessing.
      </p>

      {error && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <div className="mt-4 flex flex-wrap gap-3">
        {PROVIDERS.map((p) => (
          <label
            key={p.value}
            className={`flex cursor-pointer flex-col gap-0.5 rounded-md border px-3 py-2 text-sm ${
              provider === p.value ? "border-slate-900 bg-slate-50" : "border-slate-300"
            }`}
          >
            <span className="flex items-center gap-2 font-medium text-slate-800">
              <input
                type="radio"
                name="price-provider"
                checked={provider === p.value}
                disabled={provider === null || saving}
                onChange={() => handleSelect(p.value)}
              />
              {p.label}
            </span>
            <span className="text-xs text-slate-500">{p.note}</span>
          </label>
        ))}
      </div>

      {saving && <p className="mt-3 text-xs text-slate-500">Switching…</p>}
      {saved && !saving && <p className="mt-3 text-xs font-medium text-emerald-700">Saved</p>}
      {saveError && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{saveError}</p>}
    </div>
  );
}
