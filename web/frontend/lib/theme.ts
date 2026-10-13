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

export function applyThemeAccent(accent: string, soft: string) {
  const root = document.documentElement;
  root.style.setProperty("--pf-accent", accent);
  root.style.setProperty("--pf-accent-soft", soft);
}
