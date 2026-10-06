import Link from "next/link";

export default function SignalsGuidePage() {
  return (
    <div className="mx-auto max-w-2xl px-4 py-10">
      <Link href="/guides" className="text-sm text-indigo-600 hover:underline">
        ← All guides
      </Link>
      <h1 className="font-display mt-2 text-2xl font-semibold text-slate-900">Reading Signals</h1>
      <p className="mt-2 text-sm leading-relaxed text-slate-600">
        This app shows a &quot;Signal&quot; in more than one place, and it doesn&apos;t always mean the same thing —
        that&apos;s confusing until you know which one you&apos;re looking at. Here&apos;s what each one actually
        measures.
      </p>

      <div className="mt-6 flex flex-col gap-6">
        <section>
          <h2 className="text-base font-semibold text-slate-900">BUY / HOLD / SELL (the quant forecast)</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-700">
            On the Portfolio, Stock Screener, and Quant vs. Analyst pages, this Signal comes from the app&apos;s
            forecasting model: it projects a price roughly 10 trading days out, and compares that to today&apos;s
            price. BUY means the forecast implies at least a +5% expected return; SELL means -5% or worse; HOLD is
            everything in between — which is most of the time by design, since the model&apos;s raw predictions
            lost to a simple &quot;assume no change&quot; forecast in backtesting, so the threshold is deliberately
            wide rather than chasing every small wiggle.
          </p>
          <p className="mt-2 text-sm leading-relaxed text-slate-700">
            It&apos;s a standalone statistical projection for one ticker — not a percentile rank against other
            stocks, and not investment advice. Checking the same ticker on two different days can give a different
            read as new price data comes in.
          </p>
        </section>

        <section>
          <h2 className="text-base font-semibold text-slate-900">Entry-timing labels (Buy Now, Buy on Pullback, etc.)</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-700">
            The Entry Signals page uses a completely different Signal: not a price forecast, but a read on the
            current technical setup — trend direction, RSI, and distance from recent support/resistance. &quot;Buy
            Now&quot; means a short-term uptrend with supportive momentum and price that isn&apos;t overextended.
            &quot;Watch for Reversal&quot; means the stock looks oversold and might bounce, but hasn&apos;t
            confirmed it yet. These labels describe what the chart looks like right now, not where the price is
            forecast to go.
          </p>
        </section>

        <section>
          <h2 className="text-base font-semibold text-slate-900">Short-Term Score and Long-Term Score (the two-score system)</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-700">
            A third, separate system — shown as Top Drivers/Drags and &quot;Why these scores&quot; on a stock&apos;s
            own detail page — ranks every stock in the universe on 8 factors, each measured as a percentile against
            its peers: momentum, reversal (RSI), earnings surprise, and earnings revisions make up the Short-Term
            score; value, growth, low volatility, and quality make up the Long-Term score. A factor&apos;s
            contribution is how far above or below the universe median it ranks, weighted by that factor&apos;s
            share of the score — the contributions always add back up to (score − 50). This is the only one of the
            three Signal systems that&apos;s relative to other stocks rather than an absolute call on one ticker.
          </p>
        </section>

        <section>
          <h2 className="text-base font-semibold text-slate-900">When a Signal changes</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-700">
            The quant BUY/HOLD/SELL call is checked daily, and this app keeps a record of when it flips for a
            ticker you hold or watch — the Quant vs. Analyst page shows &quot;Flips&quot; with the real captured
            history behind it, not a guess. A signal that&apos;s flipped 3 or more times recently is flagged
            unstable — worth treating with less confidence, since it hasn&apos;t settled on a view.
          </p>
        </section>
      </div>

      <div className="mt-8 border-t border-slate-200 pt-4">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Where this shows up in the app</p>
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-sm">
          <Link href="/entry" className="text-indigo-600 hover:underline">
            Entry Signals
          </Link>
          <Link href="/stock-finder" className="text-indigo-600 hover:underline">
            Stock Screener
          </Link>
          <Link href="/signal-comparison" className="text-indigo-600 hover:underline">
            Quant vs. Analyst
          </Link>
          <Link href="/portfolio" className="text-indigo-600 hover:underline">
            Portfolio Holdings
          </Link>
        </div>
      </div>
    </div>
  );
}
