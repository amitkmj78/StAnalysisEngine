"use client";

/**
 * Horizon 1 (RS-4) — placeholder disclosure copy. NOT reviewed by
 * counsel. Per CMP-03/CMP-04 this exact wording must be reviewed before
 * any real paid subscriber ever sees it; this component exists to get
 * the *mechanism* right (present on every page/email, hypothetical
 * results kept structurally separate) ahead of that review, not to
 * finalize the words.
 */
export function DisclosureBanner({ variant = "standard" }: { variant?: "standard" | "hypothetical" }) {
  if (variant === "hypothetical") {
    return (
      <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
        <strong>Hypothetical / backtested results.</strong> These figures were not achieved by any
        real, out-of-sample publication — they are simulated on historical data and are shown
        separately from the live track record for that reason. Hypothetical performance has
        inherent limitations and does not reflect actual trading; results may differ materially
        from live results.
      </div>
    );
  }

  return (
    <p className="text-xs leading-relaxed text-slate-500">
      This is impersonal research: the same content is shown to every reader, describing what the
      model ranked and why — not individualized advice and not a recommendation to buy or sell any
      security. Past performance does not indicate future results.
    </p>
  );
}

export default DisclosureBanner;
