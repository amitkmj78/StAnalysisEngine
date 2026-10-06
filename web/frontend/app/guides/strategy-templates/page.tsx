import Link from "next/link";

// A guide to the strategy builder's starting templates: what each one does, how to read a test of it, and what the
// test can and cannot show. General information only. Nothing here is a recommendation to buy or sell anything.

const TEMPLATES = [
  {
    name: "Trend follow (200-day)",
    idea: "Hold while the price stays above its 200-day average, which is a long-run trend line.",
    buy: "The price crosses above its 200-day average.",
    sell: "The price crosses below its 200-day average, or it falls 15% from its highest point since the buy (a trailing stop).",
    style: "Swing: positions last weeks to months.",
  },
  {
    name: "Pullback in uptrend",
    idea: "Buy a short dip in a stock that is already in an uptrend, once the dip starts to recover.",
    buy: "The 14-day RSI (a momentum gauge) crosses above 30 while the price is above its 200-day average.",
    sell: "The RSI rises above 70, or the price falls 10% from its highest point since the buy.",
    style: "Short-term: positions last days to a few weeks.",
  },
  {
    name: "52-week high breakout",
    idea: "Buy stocks that are close to their highest price of the past year, on the idea that strength tends to continue.",
    buy: "The price comes within 2% of its 52-week high.",
    sell: "The price falls 10% below its 52-week high, or it falls 8% below the buy price (a stop loss).",
    style: "Swing.",
  },
  {
    name: "20/50-day cross",
    idea: "Follow a faster average crossing a slower one, a classic short-term trend signal.",
    buy: "The 20-day average crosses above the 50-day average.",
    sell: "The 20-day average crosses back below the 50-day average, or the price falls 12% from its high since the buy.",
    style: "Swing.",
  },
  {
    name: "Calm uptrend (low volatility)",
    idea: "Hold stocks that are moving calmly and staying above their 50-day average.",
    buy: "The stock's average daily price swing (ATR) is under 2% of its price, and it crosses above its 50-day average.",
    sell: "It crosses back below its 50-day average, or it falls 6% below the buy price.",
    style: "Swing.",
  },
  {
    name: "Volume-confirmed trend",
    idea: "Buy when a rise is backed by unusually heavy trading, which some traders read as real interest.",
    buy: "Trading volume is at least 50% above its 20-day average, and the price is above its 50-day average.",
    sell: "A 7% stop loss below the buy price, or a time stop: the position closes after 20 trading days.",
    style: "Short-term.",
  },
  {
    name: "Risk-On trend",
    idea: "Only own stocks while the market's regime label says conditions are favourable.",
    buy: "The market regime is Risk-On, and the price crosses above its 50-day average.",
    sell: "The regime changes to Risk-Off, or the price falls 10% below the buy price.",
    style: "Depends on the regime changes.",
  },
  {
    name: "RSI momentum",
    idea: "Buy when momentum turns up through the middle of its range, and sell when it looks stretched.",
    buy: "The 14-day RSI crosses above 50.",
    sell: "The RSI rises above 60, or the price falls 10% from its highest point since the buy.",
    style: "Short-term: this one trades often.",
  },
];

export default function StrategyTemplatesGuidePage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-10">
      <Link href="/guides" className="text-sm text-indigo-600 hover:underline">
        ← All guides
      </Link>
      <h1 className="mt-2 text-2xl font-semibold text-slate-900">Strategies: start from a template</h1>
      <p className="mt-2 text-sm leading-relaxed text-slate-600">
        A strategy is a set of rules that says when to buy a stock and when to sell it. The strategy builder lets you
        write those rules with drop-down menus, without any code, and then test them on past prices. This guide explains
        how to start from one of the built-in templates, what each template does, and how to read what its test shows.
      </p>
      <p className="mt-2 rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-900">
        A backtest shows how a set of rules would have done on past prices. It is not a forecast, not a recommendation,
        and not an order. Nothing is bought or sold when you run one.
      </p>

      <div className="mt-6 flex flex-col gap-8">
        <section className="rounded-xl border border-slate-200 bg-white p-5">
          <h2 className="text-base font-semibold text-slate-900">1. Start from a template</h2>
          <ol className="mt-3 list-decimal space-y-2 pl-5 text-sm leading-relaxed text-slate-700">
            <li>Open <Link href="/strategies/builder" className="text-indigo-600 hover:underline">the strategy builder</Link>.</li>
            <li>Under Start from a template, pick one. The buy rules, sell rules and exit settings fill in for you.</li>
            <li>Choose the stocks to test. You can type a ticker or a company name, or pick a sector. You can also give each stock a weight, or use your portfolio&apos;s actual weights.</li>
            <li>Change anything you like. A template is only a starting point. Each rule is a field, a comparison, and a number.</li>
            <li>Run the test. Read the results, then the checks, before deciding whether the rules deserve more study.</li>
            <li>Save the test if you want to come back to it, or share it. Shared tests are read-only for the people you send them to.</li>
          </ol>
        </section>

        <section className="rounded-xl border border-slate-200 bg-white p-5">
          <h2 className="text-base font-semibold text-slate-900">2. The rules, in plain language</h2>
          <ul className="mt-3 list-disc space-y-2 pl-5 text-sm leading-relaxed text-slate-700">
            <li><span className="font-medium">Entry rules</span> say when to buy. Every entry rule has to be true on the same day for a buy to happen.</li>
            <li><span className="font-medium">Exit rules</span> say when to sell. Any one exit rule being true is enough to sell.</li>
            <li><span className="font-medium">Stop loss</span> sells if the price falls a set percentage below the buy price, so one bad trade is limited.</li>
            <li><span className="font-medium">Trailing stop</span> sells if the price falls a set percentage from its highest point since the buy. It lets a winner run, then protects part of the gain.</li>
            <li><span className="font-medium">Time stop</span> sells after a set number of trading days, whatever the price did.</li>
            <li><span className="font-medium">Cooldown</span> waits a few trading days after a sale before the same stock can be bought again, so one strong move does not create a string of trades.</li>
            <li>A trade is filled at the next day&apos;s opening price, not the price at which the rule became true. That is closer to what a real order would get.</li>
          </ul>
        </section>

        <section className="rounded-xl border border-slate-200 bg-white p-5">
          <h2 className="text-base font-semibold text-slate-900">3. The eight starting templates</h2>
          <p className="mt-2 text-sm text-slate-600">
            Each template is a complete starting point. The numbers are the defaults; you can change any of them.
          </p>
          <div className="mt-4 flex flex-col gap-4">
            {TEMPLATES.map((t) => (
              <div key={t.name} className="rounded-lg border border-slate-100 p-4">
                <h3 className="text-sm font-semibold text-slate-900">{t.name}</h3>
                <p className="mt-1 text-sm text-slate-700">{t.idea}</p>
                <dl className="mt-2 grid gap-1 text-sm text-slate-700 sm:grid-cols-[6rem_1fr]">
                  <dt className="font-medium text-slate-500">Buy when</dt>
                  <dd>{t.buy}</dd>
                  <dt className="font-medium text-slate-500">Sell when</dt>
                  <dd>{t.sell}</dd>
                  <dt className="font-medium text-slate-500">Style</dt>
                  <dd>{t.style}</dd>
                </dl>
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-xl border border-slate-200 bg-white p-5">
          <h2 className="text-base font-semibold text-slate-900">4. How to read a test</h2>
          <ul className="mt-3 list-disc space-y-2 pl-5 text-sm leading-relaxed text-slate-700">
            <li>
              <span className="font-medium">Compare with holding.</span> Every result is set beside the same stocks held
              the whole time, rebalanced monthly to equal weights. A strategy that made 20% while holding made 40% did worse.
            </li>
            <li>
              <span className="font-medium">Compare with SPY.</span> SPY is the S&amp;P 500 index fund. It shows what a
              simple broad-market investment did over the same dates.
            </li>
            <li>
              <span className="font-medium">Costs are included.</span> Each trade is charged 10 basis points (0.1%) for
              the commission and spread, and 5 basis points more for slippage, on each side of the trade.
            </li>
            <li>
              <span className="font-medium">Risk, not only return.</span> Look at the largest fall from a peak (max drawdown),
              the Sharpe ratio (return per unit of risk), the worst month. The scan also shows the share of time the money was invested.
              A strategy that earns less but falls much less can still be the better fit for some investors.
            </li>
            <li>
              <span className="font-medium">Turnover.</span> How much of the portfolio is traded each year. High turnover
              means more costs and, in many places, more tax on short-term gains.
            </li>
            <li>
              <span className="font-medium">Out-of-sample windows.</span> The test is split into 6-month windows that step
              forward, so one good year cannot carry the verdict. Each window is labelled Bull, Bear or Sideways. A
              strategy that only works in one kind of market will show it here.
            </li>
            <li>
              <span className="font-medium">Chance.</span> When many rules are tried, some will look good by luck alone.
              The chance figure estimates how likely a result is to come from luck, after accounting for how many variants you
              have tried. Treat a result the chance figure does not support with caution.
            </li>
          </ul>
        </section>

        <section className="rounded-xl border border-slate-200 bg-white p-5">
          <h2 className="text-base font-semibold text-slate-900">5. Traps to avoid</h2>
          <ul className="mt-3 list-disc space-y-2 pl-5 text-sm leading-relaxed text-slate-700">
            <li>
              <span className="font-medium">Overfitting.</span> Changing the numbers until the test looks good, then
              treating the result as proof, is the most common mistake. Decide your rules first, then test them.
            </li>
            <li>
              <span className="font-medium">Few trades.</span> A rule that trades a handful of times proves little. In the scan, a short-term template is ranked only if it made at least 30 trades.
            </li>
            <li>
              <span className="font-medium">Survivorship.</span> A test of today&apos;s index members back in time flatters
              the results, because the companies that failed have dropped out. The scan now draws its sample from index members on its start date, and it does not allow buying a stock before it joined. The holding comparison still counts some stocks outside that window, so a small flattering effect remains, and the scan says so.
            </li>
            <li>
              <span className="font-medium">One stock deciding it.</span> If a single stock drives most of the result, the
              verdict depends on that stock. The scan warns when this happens.
            </li>
          </ul>
        </section>

        <section className="rounded-xl border border-slate-200 bg-white p-5">
          <h2 className="text-base font-semibold text-slate-900">6. Scan first, then build</h2>
          <p className="mt-3 text-sm leading-relaxed text-slate-700">
            The <Link href="/strategies/scan" className="text-indigo-600 hover:underline">scan</Link> runs all eight
            templates on a random sample of S&amp;P 500 stocks, with the same test on each. It shows which templates
            beat holding on return or on risk, and which ones did not. In most samples, few or none do. That is a
            result, not a fault: it means simple rules often do worse than holding the same stocks. Use the scan to see
            the patterns, then build one rule set you understand and test it on the stocks you care about.
          </p>
        </section>

        <section className="rounded-xl border border-slate-200 bg-white p-5">
          <h2 className="text-base font-semibold text-slate-900">7. What a test cannot tell you</h2>
          <ul className="mt-3 list-disc space-y-2 pl-5 text-sm leading-relaxed text-slate-700">
            <li>How the future will go. Past prices, however carefully tested, do not guarantee similar results.</li>
            <li>Taxes, your own costs, or how a fill would have happened in a fast or thin market.</li>
            <li>Whether a stock suits you. Your goals, time horizon, and the amount you can afford to lose matter more than any test.</li>
          </ul>
          <p className="mt-3 text-sm text-slate-600">
            For a broader view of how signals and scores are produced, see the <Link href="/methodology" className="text-indigo-600 hover:underline">methodology page</Link>.
          </p>
        </section>
      </div>
    </div>
  );
}
