"use client";

import type { ChallengeEquityCurves } from "@/lib/types";

const PALETTE = ["#4f46e5", "#059669", "#d97706", "#db2777", "#0891b2", "#7c3aed", "#65a30d", "#ea580c"];
const WIDTH = 640;
const HEIGHT = 260;
const PAD = { left: 44, right: 12, top: 12, bottom: 26 };

export default function EquityCurveChart({ data }: { data: ChallengeEquityCurves }) {
  const series = data.members.filter((m) => m.points.length > 0);
  const hasSpy = data.spy.length > 0;
  if (series.length === 0 && !hasSpy) {
    return (
      <p className="text-sm text-slate-500">
        No equity history yet. Curves appear once the daily snapshot job has captured a few days of account values.
      </p>
    );
  }

  const allDates = Array.from(
    new Set([...data.spy.map((p) => p.date), ...series.flatMap((m) => m.points.map((p) => p.date))])
  ).sort();
  const t0 = new Date(allDates[0]).getTime();
  const t1 = new Date(allDates[allDates.length - 1]).getTime();
  const span = Math.max(t1 - t0, 1);

  const values = [100, ...data.spy.map((p) => p.value), ...series.flatMap((m) => m.points.map((p) => p.value))];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const pad = Math.max((hi - lo) * 0.08, 1);
  const yMin = lo - pad;
  const yMax = hi + pad;

  const x = (d: string) => PAD.left + ((new Date(d).getTime() - t0) / span) * (WIDTH - PAD.left - PAD.right);
  const y = (v: number) => PAD.top + (1 - (v - yMin) / (yMax - yMin)) * (HEIGHT - PAD.top - PAD.bottom);
  const path = (pts: { date: string; value: number }[]) =>
    pts.map((p, i) => `${i === 0 ? "M" : "L"}${x(p.date).toFixed(1)},${y(p.value).toFixed(1)}`).join(" ");

  const ticks = [yMin, 100, yMax].map((v) => Math.round(v * 10) / 10);

  return (
    <div>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="w-full" role="img" aria-label="Equity curves rebased to 100">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={WIDTH - PAD.right} y1={y(t)} y2={y(t)} stroke="#e2e8f0" />
            <text x={PAD.left - 6} y={y(t) + 4} textAnchor="end" fontSize="11" fill="#64748b">
              {t}
            </text>
          </g>
        ))}
        <line x1={PAD.left} x2={WIDTH - PAD.right} y1={y(100)} y2={y(100)} stroke="#cbd5e1" strokeDasharray="2 3" />
        {hasSpy && (
          <path d={path(data.spy)} fill="none" stroke="#64748b" strokeWidth="2" strokeDasharray="6 4" />
        )}
        {series.map((m, i) => (
          <path key={`${m.member}-${i}`} d={path(m.points)} fill="none" stroke={PALETTE[i % PALETTE.length]} strokeWidth="2.25" />
        ))}
        <text x={PAD.left} y={HEIGHT - 6} fontSize="11" fill="#64748b">{allDates[0]}</text>
        <text x={WIDTH - PAD.right} y={HEIGHT - 6} textAnchor="end" fontSize="11" fill="#64748b">
          {allDates[allDates.length - 1]}
        </text>
      </svg>
      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-600">
        {series.map((m, i) => (
          <li key={`${m.member}-${i}`} className="flex items-center gap-1.5">
            <span className="inline-block h-0.5 w-4" style={{ background: PALETTE[i % PALETTE.length] }} />
            {m.member}
          </li>
        ))}
        {hasSpy && (
          <li className="flex items-center gap-1.5">
            <span className="inline-block w-4 border-t-2 border-dashed border-slate-500" />
            S&amp;P 500
          </li>
        )}
      </ul>
      <p className="mt-1 text-xs text-slate-400">Each line starts at 100 on its first snapshot in this window. Raw balances are not shown.</p>
    </div>
  );
}
