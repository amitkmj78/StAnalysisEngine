import Link from "next/link";

// Hub for everything strategy-related. Each card opens one tool.
const CARDS = [
  {
    href: "/strategies/builder",
    title: "Build & test",
    body: "Write your own rules from dropdowns, pick stocks or a portfolio, and see whether the rules beat holding the same stocks.",
  },
  {
    href: "/strategies/scan",
    title: "Scan for candidates",
    body: "Test the starting templates on a random sample of S&P 500 stocks and see which ones held up out of sample. Candidates to study, not recommendations.",
  },
  {
    href: "/strategies/saved",
    title: "Saved tests",
    body: "Compare up to four saved builder runs, or share one as a read-only link.",
  },
  {
    href: "/strategies/plans",
    title: "Your plans",
    body: "Your portfolio strategy plans and goal calculators.",
  },
];

export default function StrategiesHubPage() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <h1 className="font-display text-2xl font-semibold text-slate-900">Strategies</h1>
      <p className="mt-1 text-sm text-slate-500">Test ideas on past prices before you rely on them. Nothing here places an order.</p>
      <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2">
        {CARDS.map((c) => (
          <Link key={c.href} href={c.href} className="rounded-xl border border-slate-200 bg-white p-5 hover:border-slate-400">
            <p className="font-semibold text-slate-900">{c.title}</p>
            <p className="mt-1 text-sm text-slate-600">{c.body}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
