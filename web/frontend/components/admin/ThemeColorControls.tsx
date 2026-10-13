"use client";

import { useEffect, useState } from "react";

import { ApiError, getAdminSettings, setDefaultThemeAccentColor } from "@/lib/api";
import { THEME_PALETTE } from "@/lib/theme";

export default function ThemeColorControls() {
  const [current, setCurrent] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  async function load() {
    setError(null);
    try {
      const settings = await getAdminSettings();
      setCurrent(settings.default_theme_accent_color);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load the site theme color.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handlePick(accent: string) {
    setSaving(accent);
    setSaved(false);
    setError(null);
    try {
      const res = await setDefaultThemeAccentColor(accent);
      setCurrent(res.default_theme_accent_color);
      setSaved(true);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to save the site theme color.");
    } finally {
      setSaving(null);
    }
  }

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5">
      <h2 className="font-semibold text-slate-900">Site Theme Color</h2>
      <p className="mt-1 text-sm text-slate-600">
        The default accent color for every user who hasn&apos;t picked their own in Settings. A curated palette, not
        a freeform color — each option is chosen for contrast against the app&apos;s fixed background and text
        colors.
      </p>

      {error && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <div className="mt-4 flex flex-wrap gap-2">
        {THEME_PALETTE.map((c) => (
          <button
            key={c.accent}
            type="button"
            onClick={() => handlePick(c.accent)}
            disabled={saving !== null}
            aria-pressed={current === c.accent}
            title={c.name}
            className={`flex h-9 w-9 items-center justify-center rounded-full border-2 disabled:opacity-50 ${
              current === c.accent ? "border-slate-900" : "border-transparent"
            }`}
            style={{ backgroundColor: c.accent }}
          >
            {saving === c.accent && <span className="h-2 w-2 animate-pulse rounded-full bg-white" />}
          </button>
        ))}
      </div>
      {saved && <p className="mt-3 text-xs font-medium text-emerald-700">Saved — takes effect for users on their next page load.</p>}
    </div>
  );
}
