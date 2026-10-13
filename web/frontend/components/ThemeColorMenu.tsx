"use client";

import { useEffect, useRef, useState } from "react";

import { ApiError, getCurrentUser, setMyThemeColor } from "@/lib/api";
import { applyThemeAccent, DEFAULT_THEME_ACCENT, THEME_PALETTE } from "@/lib/theme";

// A quick-access theme picker next to Sign out, not buried on /settings --
// the swatch button itself always shows the currently applied color, so
// picking one gives immediate, visible confirmation right where you're
// looking instead of a change you might miss elsewhere on the page.
export default function ThemeColorMenu() {
  const [open, setOpen] = useState(false);
  const [current, setCurrent] = useState(DEFAULT_THEME_ACCENT);
  const [hasOverride, setHasOverride] = useState(false);
  const [saving, setSaving] = useState<string | "default" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getCurrentUser()
      .then((user) => {
        setCurrent(user.effective_theme_accent_color);
        setHasOverride(!!user.theme_accent_color);
        // This is the one place that applies the saved preference on page
        // load -- the swatch button's own color alone wouldn't change
        // anything else in the app.
        applyThemeAccent(user.effective_theme_accent_color, user.effective_theme_accent_soft);
      })
      .catch(() => {
        // Keep the hardcoded default -- a failed fetch should never block the header from rendering.
      });
  }, []);

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    function onEscape(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onClickOutside);
    document.addEventListener("keydown", onEscape);
    return () => {
      document.removeEventListener("mousedown", onClickOutside);
      document.removeEventListener("keydown", onEscape);
    };
  }, []);

  async function pick(color: string | null) {
    setSaving(color ?? "default");
    setError(null);
    try {
      const res = await setMyThemeColor(color);
      setCurrent(res.effective_theme_accent_color);
      setHasOverride(!!res.theme_accent_color);
      applyThemeAccent(res.effective_theme_accent_color, res.effective_theme_accent_soft);
      setOpen(false);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not save theme color.");
    } finally {
      setSaving(null);
    }
  }

  return (
    <div ref={ref} className="relative flex-none">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="true"
        aria-expanded={open}
        title="Theme color"
        className="flex h-7 w-7 items-center justify-center rounded-full border border-slate-300"
        style={{ backgroundColor: current }}
      >
        <span className="sr-only">Theme color</span>
      </button>
      {open && (
        <div className="absolute right-0 top-full z-20 mt-1 w-52 rounded-md border border-slate-200 bg-white p-3 shadow-lg">
          <p className="text-xs font-medium text-slate-500">Theme color</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {THEME_PALETTE.map((c) => (
              <button
                key={c.accent}
                type="button"
                onClick={() => pick(c.accent)}
                disabled={saving !== null}
                aria-pressed={current === c.accent}
                title={c.name}
                className={`flex h-7 w-7 items-center justify-center rounded-full border-2 disabled:opacity-50 ${
                  current === c.accent ? "border-slate-900" : "border-transparent"
                }`}
                style={{ backgroundColor: c.accent }}
              >
                {saving === c.accent && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-white" />}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={() => pick(null)}
            disabled={saving !== null || !hasOverride}
            className="mt-2 w-full rounded-md border border-slate-300 px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50 disabled:opacity-50"
          >
            {saving === "default" ? "Resetting…" : "Use site default"}
          </button>
          {error && <p className="mt-2 text-xs font-medium text-red-700">{error}</p>}
        </div>
      )}
    </div>
  );
}
