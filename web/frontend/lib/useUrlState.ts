"use client";

import { useCallback, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

// No page in this app does two-way useSearchParams + router.replace sync
// today (existing pages only read a query param once on mount) -- this
// is the first, built for the compare page's bookmarkable
// ?p=...&goal=...&window=... state, small and generic enough to reuse
// elsewhere later if another page wants the same thing.
export function useUrlState<T extends Record<string, string>>(defaults: T) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [state, setState] = useState<T>(() => {
    const out = { ...defaults };
    for (const key of Object.keys(defaults)) {
      const v = searchParams.get(key);
      if (v) (out as Record<string, string>)[key] = v;
    }
    return out;
  });

  const update = useCallback(
    (patch: Partial<T>) => {
      setState((prev) => {
        const next = { ...prev, ...patch };
        const params = new URLSearchParams();
        for (const [k, v] of Object.entries(next)) {
          if (v && v !== defaults[k as keyof T]) params.set(k, v);
        }
        const query = params.toString();
        router.replace(query ? `?${query}` : "?", { scroll: false });
        return next;
      });
    },
    [router, defaults],
  );

  return [state, update] as const;
}
