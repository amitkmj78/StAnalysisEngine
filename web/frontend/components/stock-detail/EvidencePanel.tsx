"use client";

// DIF-3: the short list of reasons behind the score, shown beside the chart.
// Everything here comes from the two-score response the page already loads;
// nothing is recomputed on the client. Contributions are factor points, not
// predictions, and the regime label is a condition label, not a recommendation.

type Driver = { label: string; contribution: number };

export default function EvidencePanel({
  drivers,
  drags,
  sectorKey,
  shortSectorRank,
  longSectorRank,
  regime,
  asOfDate,
  shortScore,
  longScore,
}: {
  drivers: Driver[];
  drags: Driver[];
  sectorKey: string;
  shortSectorRank: { rank: number; of: number } | null;
  longSectorRank: { rank: number; of: number } | null;
  regime: string | null;
  asOfDate: string;
  shortScore: number | null;
  longScore: number | null;
}) {
  const topDrivers = drivers.slice(0, 3);
  const topDrags = drags.slice(0, 2);
  const rankText = (r: { rank: number; of: number } | null) => (r ? `#${r.rank} of ${r.of}` : "Not ranked");

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-sm font-semibold text-slate-900">Evidence</h3>
        <p className="text-xs text-slate-500">
          As of {asOfDate}: short-term{" "}
          <span className="font-mono font-semibold text-slate-800">{shortScore?.toFixed(1) ?? "n/a"}</span>, long-term{" "}
          <span className="font-mono font-semibold text-slate-800">{longScore?.toFixed(1) ?? "n/a"}</span>
        </p>
      </div>
      <div className="mt-3 grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Pushing the score up</p>
          {topDrivers.length === 0 ? (
            <p className="mt-1 text-xs text-slate-400">No positive drivers today.</p>
          ) : (
            <ul className="mt-1 flex flex-col gap-1">
              {topDrivers.map((d) => (
                <li key={d.label} className="flex items-center justify-between gap-3 text-sm">
                  <span className="text-slate-700">{d.label}</span>
                  <span className="font-mono text-xs font-semibold text-emerald-700">+{d.contribution.toFixed(1)}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Pulling the score down</p>
          {topDrags.length === 0 ? (
            <p className="mt-1 text-xs text-slate-400">No negative drags today.</p>
          ) : (
            <ul className="mt-1 flex flex-col gap-1">
              {topDrags.map((d) => (
                <li key={d.label} className="flex items-center justify-between gap-3 text-sm">
                  <span className="text-slate-700">{d.label}</span>
                  <span className="font-mono text-xs font-semibold text-red-700">{d.contribution.toFixed(1)}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
      <div className="mt-4 grid grid-cols-1 gap-x-4 gap-y-2 border-t border-slate-100 pt-3 text-xs sm:grid-cols-3">
        <p className="text-slate-600">
          <span className="text-slate-400">Short-term rank in {sectorKey}: </span>
          {rankText(shortSectorRank)}
        </p>
        <p className="text-slate-600">
          <span className="text-slate-400">Long-term rank in {sectorKey}: </span>
          {rankText(longSectorRank)}
        </p>
        <p className="text-slate-600">
          <span className="text-slate-400">Market regime: </span>
          {regime ?? "Not available"}
        </p>
      </div>
    </div>
  );
}
