import type { ReactNode } from "react";

/** Uses the shared "Price outlook" up/down/warn tokens (globals.css)
 * instead of plain Tailwind emerald/red/amber -- same meanings (gains/
 * LONG/correct, losses/SHORT/wrong, verified) the UI-consistency audit
 * already found converged well on their own, now on one shared palette. */
type BadgeVariant = "positive" | "negative" | "verified" | "warning" | "neutral";

const VARIANT_CLASSES: Record<BadgeVariant, string> = {
  positive: "bg-[var(--pf-accent-soft)] text-[var(--pf-up)]",
  negative: "bg-[var(--pf-down-soft)] text-[var(--pf-down)]",
  verified: "bg-[var(--pf-accent-soft)] text-[var(--pf-accent)]",
  warning: "bg-[var(--pf-warn-soft)] text-[var(--pf-warn)]",
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
