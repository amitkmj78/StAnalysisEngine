// Lightweight inline SVG sparkline -- deliberately not a per-row Plotly
// chart (this renders inside a table with up to ~20 rows; that many
// Plotly instances would be far heavier than this app's existing
// per-row chart usage anywhere else).
export default function Sparkline({ values, color }: { values: number[]; color: string }) {
  if (values.length < 2) return <span className="text-xs text-slate-400">—</span>;

  const width = 64;
  const height = 20;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const points = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * width;
      const y = height - ((v - min) / range) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="inline-block align-middle">
      <polyline points={points} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}
