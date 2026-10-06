import Link from "next/link";

const GUIDES = [
  {
    href: "/guides/signals",
    title: "Reading Signals",
    description: "What BUY/HOLD/SELL and the two-score system actually mean, and where each number comes from.",
  },
  {
    href: "/guides/risk",
    title: "Understanding Risk",
    description: "Volatility, beta, correlation, and max drawdown — what each one tells you, and what it doesn't.",
  },
  {
    href: "/guides/diversification",
    title: "Diversification & Concentration",
    description: "How this app flags a position or sector as concentrated, and why that matters.",
  },
];

export default function GuidesPage() {
  return (
    <div className="mx-auto max-w-2xl px-4 py-10">
      <h1 className="font-display text-2xl font-semibold text-slate-900">Guides</h1>
      <p className="mt-2 text-sm leading-relaxed text-slate-600">
        Short, plain-language explanations of how this app reads signals, risk, and diversification — grounded in
        exactly what it computes, not general market commentary.
      </p>

      <div className="mt-6 flex flex-col gap-3">
        {GUIDES.map((g) => (
          <Link
            key={g.href}
            href={g.href}
            className="rounded-xl border border-slate-200 bg-white p-5 transition-colors hover:border-slate-300 hover:bg-slate-50"
          >
            <h2 className="font-semibold text-slate-900">{g.title}</h2>
            <p className="mt-1 text-sm text-slate-600">{g.description}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
