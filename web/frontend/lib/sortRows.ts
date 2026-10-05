// Sorts table rows by a column. Text compares alphabetically; numbers numerically. Rows with no value
// for the column always go to the end, whichever direction is chosen, so gaps don't hide the top rows.

export type SortDirection = "asc" | "desc";

export function sortRows<T>(rows: T[], value: (row: T) => string | number | null | undefined, direction: SortDirection | null): T[] {
  if (!direction) return rows;
  const sign = direction === "asc" ? 1 : -1;
  return [...rows].sort((a, b) => {
    const va = value(a);
    const vb = value(b);
    const aMissing = va === null || va === undefined || va === "";
    const bMissing = vb === null || vb === undefined || vb === "";
    if (aMissing && bMissing) return 0;
    if (aMissing) return 1;
    if (bMissing) return -1;
    if (typeof va === "number" && typeof vb === "number") return (va - vb) * sign;
    return String(va).localeCompare(String(vb)) * sign;
  });
}

// Click cycle for a column header: unsorted → ascending → descending → unsorted.
export function nextSort<K extends string>(
  current: { key: K | null; direction: SortDirection | null },
  key: K,
): { key: K | null; direction: SortDirection | null } {
  if (current.key !== key || current.direction === null) return { key, direction: "asc" };
  if (current.direction === "asc") return { key, direction: "desc" };
  return { key: null, direction: null };
}
