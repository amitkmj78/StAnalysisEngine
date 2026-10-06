import Link from "next/link";

// FND-6: how the scores, signals, confidence, regime and track record are produced, and what is still unproven.
// Kept to what the code does today. Anything not yet true is said so, not described as finished.

export default function MethodologyPage() {
  return (
    <div className="mx-auto max-w-2xl px-4 py-10">
      <Link href="/guides" className="text-sm text-indigo-600 hover:underline">
        ← All guides
      </Link>
      <h1 className="mt-2 text-2xl font-semibold text-slate-900">Methodology</h1>
      <p className="mt-2 text-sm leading-relaxed text-slate-600">
        How the scores, signals, confidence, market regime and track record are produced, and which parts are not yet
        proven. This is general information about how the app works, not a recommendation to buy, hold or sell anything.
      </p>

      <div className="mt-6 flex flex-col gap-6">
        <section>
          <h2 className="text-base font-semibold text-slate-900">1. Scores (0 to 100)</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-700">
            Each stock gets two scores, recomputed each night after the close. Each score is dated by the day it was
            computed.
          </p>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm leading-relaxed text-slate-700">
            <li>
              <span className="font-medium">Short-term</span> (intended for 10 to 90 days): momentum, short-term reversal,
              earnings surprise and earnings revisions.
            </li>
            <li>
              <span className="font-medium">Long-term</span>: value, growth, low volatility and quality.
            </li>
          </ul>
        </section>

        <section>
          <h2 className="text-base font-semibold text-slate-900">2. Signals (Buy, Hold, Trim)</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-700">
            The stock-page signal is set by the score alone: <span className="font-medium">Buy</span> at 70 or above,{" "}
            <span className="font-medium">Trim</span> at 30 or below, and <span className="font-medium">Hold</span> in
            between. A score is a relative ranking of the stock on these factors, not a price forecast.
          </p>
        </section>

        <section>
          <h2 className="text-base font-semibold text-slate-900">3. Confidence</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-700">
            Each signal is shown with its score and a confidence label. The confidence is <span className="font-medium">not yet a
            calibrated probability</span> that the signal will be right. Calibrating it needs a longer out-of-sample
            history than the app has today, so that work is still open.
          </p>
        </section>

        <section>
          <h2 className="text-base font-semibold text-slate-900">4. Track record</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-700">
            Two different records are shown, and they measure different things:
          </p>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm leading-relaxed text-slate-700">
            <li>
              <span className="font-medium">Per stock</span>, under each stock chart: the short-term signal is checked 10
              trading days later and compared with SPY. It shows the hit rate, the average excess return and the worst
              miss. Until there are 30 signals for that stock, it says &quot;Not enough data yet&quot;.
            </li>
            <li>
              <span className="font-medium">Public record</span>: the published top-ranked picks, compared with SPY. This
              record does not yet score the Buy, Hold and Trim signals themselves. Changing it to do so is still open.
            </li>
          </ul>
          <p className="mt-2 text-sm leading-relaxed text-slate-700">
            Past results do not guarantee future results. Each record is also a short sample for the period it covers.
          </p>
        </section>

        <section>
          <h2 className="text-base font-semibold text-slate-900">5. Market regime</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-700">
            The regime label (for example Risk-On, Cautious or Risk-Off) is built from market data each day. Its
            validation test was run once, with the thresholds fixed in advance. It could not pass with the history
            stored so far, so the regime is shown with a caveat and is not presented as a validated signal. See the{" "}
            <Link href="/regime/methodology" className="text-indigo-600 hover:underline">
              regime methodology
            </Link>{" "}
            for the details.
          </p>
        </section>

        <section>
          <h2 className="text-base font-semibold text-slate-900">6. Data</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-700">
            Prices and company data currently come from yfinance. That source is not yet licensed for display to the
            public, and a licensed vendor is still to be chosen. The figures on this site may change when that happens.
          </p>
        </section>
      </div>
    </div>
  );
}
