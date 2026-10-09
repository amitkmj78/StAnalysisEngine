import type { ReactNode } from "react";

/** Reuses the emerald/red/slate colors already consistently used app-
 * wide for these meanings (gains/LONG/correct, losses/SHORT/wrong,
 * verified) -- see the UI-consistency audit's one genuine bright spot. */
type BadgeVariant = "positive" | "negative" | "verified" | "warning" | "neutral";

const VARIANT_CLASSES: Record<BadgeVariant, string> = {
  positive: "bg-emerald-50 text-emerald-700",
  negative: "bg-red-50 text-red-700",
  verified: "bg-emerald-50 text-emerald-700",
  warning: "bg-amber-50 text-amber-700",
  neutral: "bg-slate-100 text-slate-600",
};

export default function Badge({
  variant = "neutral",
  className = "",
  children,
}: {
  variant?: BadgeVariant;
  className?: string;
  children: ReactNode;
}) {
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${VARIANT_CLASSES[variant]} ${className}`}>
      {children}
    </span>
  );
}
