import Link from "next/link";

export default function RiskGuidePage() {
  return (
    <div className="mx-auto max-w-2xl px-4 py-10">
      <Link href="/guides" className="text-sm text-indigo-600 hover:underline">
        ← All guides
      </Link>
      <h1 className="font-display mt-2 text-2xl font-semibold text-slate-900">Understanding Risk</h1>
      <p className="mt-2 text-sm leading-relaxed text-slate-600">
        The Health Check and Stress Test pages show several risk numbers side by side. They measure different
        things and can disagree with each other — here&apos;s what each one is actually telling you.
      </p>

      <div className="mt-6 flex flex-col gap-6">
        <section>
          <h2 className="text-base font-semibold text-slate-900">Volatility</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-700">
            How much a position&apos;s daily returns bounce around, expressed as an annualized percentage. It&apos;s
            symmetric — it counts big up days the same as big down days — so a volatile stock isn&apos;t
            necessarily a losing one, just a less predictable one. Higher volatility means wider swings in your
            account value day to day, even if the long-run return is similar to a calmer stock.
          </p>
        </section>

        <section>
          <h2 className="text-base font-semibold text-slate-900">Beta</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-700">
            How a position tends to move relative to the S&amp;P 500. A beta of 1.5 means that, historically, when
            the market moves 1%, this position has tended to move about 1.5% in the same direction. A beta under 1
            means it&apos;s historically moved less than the market. Beta is about market-wide moves specifically —
            it says nothing about a stock&apos;s own company-specific risk (an earnings miss, a lawsuit), which is
            exactly what diversification is for.
          </p>
        </section>

        <section>
          <h2 className="text-base font-semibold text-slate-900">Correlation</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-700">
            How closely two positions&apos; returns have moved together historically, from -1 (perfectly opposite)
            to +1 (move in lockstep). This is the number that actually drives diversification, more than how many
            tickers you hold: 10 stocks that are all highly correlated with each other behave like one large bet,
            while a handful of low-correlation positions genuinely smooth out the ride. The Stress Test page uses
            this to show how a shock to one holding likely ripples through the rest of the portfolio.
          </p>
        </section>

        <section>
          <h2 className="text-base font-semibold text-slate-900">Max drawdown</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-700">
            The largest peak-to-trough decline a position or portfolio has actually experienced over the lookback
            window — not a projection, a historical fact. It answers a different question than volatility: not
            &quot;how bumpy is the ride&quot; but &quot;how far underwater has this actually gone before it
            recovered.&quot; A past drawdown isn&apos;t a ceiling — a new one can always be worse — but it&apos;s a
            real, lived lower bound for what you should be prepared to sit through.
          </p>
        </section>

        <section>
          <h2 className="text-base font-semibold text-slate-900">Stress test scenarios</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-700">
            The Stress Test page applies hypothetical shocks (e.g. a market-wide drop) to your actual current
            holdings, scaled by each position&apos;s own beta and size, to estimate portfolio impact. It&apos;s a
            what-if based on historical relationships, not a forecast that the scenario will happen or happen this
            way — correlations and betas can shift in a real crisis, often in the direction of making diversification
            look worse than it did in calmer markets.
          </p>
        </section>
      </div>

      <div className="mt-8 border-t border-slate-200 pt-4">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Where this shows up in the app</p>
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-sm">
          <Link href="/portfolio/health" className="text-indigo-600 hover:underline">
            Portfolio Health Check
          </Link>
          <Link href="/portfolio/stress-test" className="text-indigo-600 hover:underline">
            Stress Test
          </Link>
        </div>
      </div>
    </div>
  );
}
