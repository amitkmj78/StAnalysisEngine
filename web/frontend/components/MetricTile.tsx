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
    <div className={`rounded-lg border border-slate-200 bg-white p-3 ${className ?? ""}`}>
      <p className="flex items-center gap-1 text-xs text-slate-500">
        <MetricLabel term={term}>{label}</MetricLabel>
      </p>
      <p className="mt-1 text-lg font-semibold text-slate-900">{value}</p>
    </div>
  );
}
