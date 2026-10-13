"use client";

import { useEffect } from "react";

import { getCurrentUser } from "@/lib/api";
import { applyThemeAccent } from "@/lib/theme";

// Applies the signed-in user's effective accent color (their own override,
// or the admin's site-wide default -- see GET /api/v1/auth/me) on every
// page load. Renders nothing; globals.css's hardcoded --pf-accent is the
// pre-hydration default so there's no flash of an unstyled page while this
// fetch is in flight, just a possible one-frame color correction.
export default function ThemeColorProvider() {
  useEffect(() => {
    let cancelled = false;
    getCurrentUser()
      .then((user) => {
        if (cancelled) return;
        applyThemeAccent(user.effective_theme_accent_color, user.effective_theme_accent_soft);
      })
      .catch(() => {
        // Keep globals.css's hardcoded default -- a failed fetch should
        // never leave the page half-styled.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return null;
}
