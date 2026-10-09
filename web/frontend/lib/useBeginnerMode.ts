"use client";

import { useEffect, useState } from "react";

import { getCurrentUser } from "@/lib/api";

/**
 * BEG-1: whether the logged-in user has self-identified as a beginner
 * (users.experience_level, set from /settings or the community
 * author-profile page — both write the same PUT /api/v1/social/profile
 * field). Unset/null is NOT treated as beginner — this only reflects an
 * explicit choice, not a default assumption about a new user.
 */
export function useBeginnerMode() {
  const [isBeginner, setIsBeginner] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    getCurrentUser()
      .then((user) => {
        if (!cancelled) setIsBeginner(user.experience_level === "beginner");
      })
      .catch(() => {
        if (!cancelled) setIsBeginner(false);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { isBeginner, loading };
}
