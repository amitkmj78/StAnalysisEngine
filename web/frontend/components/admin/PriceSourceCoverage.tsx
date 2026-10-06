// Shows which parts of the app follow the price-source switch above, and which stay on Yahoo whichever source is chosen.
// Keep this list in step with the code: the shared history function and the live quote function follow the switch;
// the direct Yahoo readers listed below do not yet.

const FOLLOWS_SWITCH = [
  "Current prices: stock pages, portfolio value, and \"Create watchlist from current prices\".",
  "Price history: stock charts, the chart grid, the strategy builder and the strategy scan.",
];

const STAYS_ON_YAHOO = [
  "Company data: fundamentals, earnings dates, analyst ratings and ownership.",
  "Company news and the market news ticker (the 8-K filings come from SEC EDGAR, not either source).",
  "Some screeners, portfolio health checks and the daily price capture still read Yahoo directly, so they do not change yet.",
];

export default function PriceSourceCoverage() {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5">
      <h2 className="text-sm font-semibold text-slate-900">What follows this switch</h2>
      <div className="mt-3 grid gap-4 sm:grid-cols-2">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-emerald-700">Follows the switch</p>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-700">
            {FOLLOWS_SWITCH.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </div>
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-amber-700">Stays on Yahoo</p>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-700">
            {STAYS_ON_YAHOO.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
