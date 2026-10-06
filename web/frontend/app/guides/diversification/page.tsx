import Link from "next/link";

export default function DiversificationGuidePage() {
  return (
    <div className="mx-auto max-w-2xl px-4 py-10">
      <Link href="/guides" className="text-sm text-indigo-600 hover:underline">
        ← All guides
      </Link>
      <h1 className="font-display mt-2 text-2xl font-semibold text-slate-900">Diversification &amp; Concentration</h1>
      <p className="mt-2 text-sm leading-relaxed text-slate-600">
        The Health Check page flags when a portfolio is concentrated. Here&apos;s what triggers that flag, and why
        it matters even when every individual holding looks fine on its own.
      </p>

      <div className="mt-6 flex flex-col gap-6">
        <section>
          <h2 className="text-base font-semibold text-slate-900">Single-position concentration</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-700">
            A position is flagged as concentrated once it passes 25% of the portfolio&apos;s total value. That
            threshold isn&apos;t about whether the stock is risky on its own — it&apos;s about how much of your
            outcome now depends on one company. Above that line, a single piece of bad news (an earnings miss, a
            product recall, a regulatory problem) can move your whole portfolio far more than it should, no matter
            how strong the underlying business looks.
          </p>
        </section>

        <section>
          <h2 className="text-base font-semibold text-slate-900">Sector concentration</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-700">
            Separately, the app flags when a single sector makes up more than 40% of the portfolio. This catches a
            quieter version of the same problem: five different tickers can still mean one bet if they&apos;re all,
            say, semiconductor companies that rise and fall together on the same industry news. Spreading across
            tickers only reduces risk if those tickers don&apos;t move for the same reasons.
          </p>
        </section>

        <section>
          <h2 className="text-base font-semibold text-slate-900">Why correlation matters more than position count</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-700">
            Holding 15 stocks sounds diversified, but if they&apos;re all highly correlated (see the{" "}
            <Link href="/guides/risk" className="text-indigo-600 hover:underline">
              Risk guide
            </Link>
            ), they behave like a smaller number of real, independent bets. True diversification comes from mixing
            positions that respond differently to the same event — not just adding more tickers.
          </p>
        </section>

        <section>
          <h2 className="text-base font-semibold text-slate-900">Build Index&apos;s risk preview</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-700">
            When you build a custom index, the Risk Preview applies these same concentration checks to the
            weights you&apos;ve chosen before you save it — so you can see a position or sector getting too heavy
            while you&apos;re still adjusting weights, rather than finding out from the Health Check afterward.
          </p>
        </section>
      </div>

      <div className="mt-8 border-t border-slate-200 pt-4">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Where this shows up in the app</p>
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-sm">
          <Link href="/portfolio/health" className="text-indigo-600 hover:underline">
            Portfolio Health Check
          </Link>
          <Link href="/portfolio/build-index" className="text-indigo-600 hover:underline">
            Build Index
          </Link>
        </div>
      </div>
    </div>
  );
}
