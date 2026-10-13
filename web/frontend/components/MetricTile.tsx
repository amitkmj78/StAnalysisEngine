import MetricLabel from "./MetricLabel";

/**
 * Shared replacement for the three byte-identical local `MetricTile`
 * components this app used to have (app/entry, app/index-fund,
 * app/monthly-plan) -- predict/page.tsx and stock-finder/page.tsx keep
 * their own locally-themed tile wrappers (real visual differences) but use
 * MetricLabel internally too, so the tooltip logic itself is never
 * duplicated even where the tile styling is.
 */
export default function MetricTile({
  label,
  value,
  term,
  className,
}: {
  label: string;
  value: string;
  term?: string;
  className?: string;
}) {
  return (
    <div className={`rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-3 ${className ?? ""}`}>
      <p className="flex items-center gap-1 text-xs text-slate-500 dark:text-slate-400">
        <MetricLabel term={term}>{label}</MetricLabel>
      </p>
      <p className="mt-1 text-lg font-semibold text-slate-900 dark:text-slate-50">{value}</p>
    </div>
  );
}
