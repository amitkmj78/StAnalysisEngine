// Mirrors web/backend/app_settings.py::THEME_ACCENT_PALETTE exactly -- a
// curated set, not a freeform color picker (see that file's comment for
// why: each paired with its own hand-picked light tint rather than a
// computed one, which can come out muddy for an arbitrary hue).
export const THEME_PALETTE: { name: string; accent: string; soft: string }[] = [
  { name: "Teal", accent: "#0E5C57", soft: "#E7F1EF" },
  { name: "Blue", accent: "#1D4ED8", soft: "#E8EEFC" },
  { name: "Indigo", accent: "#4338CA", soft: "#ECEAFB" },
  { name: "Purple", accent: "#7E22CE", soft: "#F3E8FD" },
  { name: "Wine", accent: "#9D174D", soft: "#FBE7EF" },
  { name: "Forest", accent: "#166534", soft: "#E7F5EC" },
  { name: "Slate", accent: "#334155", soft: "#E9ECF0" },
  { name: "Burnt orange", accent: "#C2410C", soft: "#FDEEE4" },
];

export const DEFAULT_THEME_ACCENT = "#0E5C57";

// Light/Dark/System -- "system" is never stored as the resolved attribute
// itself (see app/layout.tsx's beforeInteractive bootstrap script, which
// always writes an explicit light/dark to <html data-theme> before paint to
// avoid a flash); it's only the localStorage value meaning "keep following
// the OS preference".
export type ThemeMode = "light" | "dark" | "system";
export const THEME_MODE_KEY = "stanalysis-theme-mode";

export function resolveMode(mode: ThemeMode): "light" | "dark" {
  if (mode === "system") {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  return mode;
}

export function getStoredThemeMode(): ThemeMode {
  try {
    const stored = localStorage.getItem(THEME_MODE_KEY);
    if (stored === "light" || stored === "dark" || stored === "system") return stored;
  } catch {
    // localStorage blocked (private mode, etc.) -- fall through to system.
  }
  return "system";
}

export function applyThemeMode(mode: ThemeMode) {
  try {
    localStorage.setItem(THEME_MODE_KEY, mode);
  } catch {
    // Best-effort -- the mode still applies for this page view even if it can't be remembered.
  }
  document.documentElement.dataset.theme = resolveMode(mode);
  refreshThemeAccentSoft();
}

// The stored palette's soft tint is a light pastel meant to sit on a white
// surface -- on a dark one it would read as a bright, out-of-place smear, so
// dark mode recomputes it as a low-opacity wash of the accent over the dark
// surface instead of using the stored hex directly.
function softForCurrentMode(accent: string, lightSoft: string): string {
  const isDark = document.documentElement.dataset.theme === "dark";
  return isDark ? `color-mix(in srgb, ${accent} 28%, #14181A)` : lightSoft;
}

export function applyThemeAccent(accent: string, soft: string) {
  const root = document.documentElement;
  root.style.setProperty("--pf-accent", accent);
  root.dataset.themeAccent = accent;
  root.dataset.themeAccentSoft = soft;
  root.style.setProperty("--pf-accent-soft", softForCurrentMode(accent, soft));
}

// Re-derives --pf-accent-soft for the mode that's active right now, without
// needing to refetch the user's saved preference -- used right after a
// Light/Dark/System switch so the accent's soft tint stays correct.
export function refreshThemeAccentSoft() {
  const root = document.documentElement;
  const accent = root.dataset.themeAccent;
  const soft = root.dataset.themeAccentSoft;
  if (accent && soft) {
    root.style.setProperty("--pf-accent-soft", softForCurrentMode(accent, soft));
  }
}
