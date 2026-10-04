"use client";

import { useState } from "react";
import { ApiError, runStrategyBacktest } from "@/lib/api";
import type { StrategyBacktestResponse, StrategyMetrics, StrategyRuleInput } from "@/lib/types";

// STB-1..5: build entry and exit rules from dropdowns and test them on past prices.
// Nothing here places an order.

const FIELDS: { key: string; label: string; numeric: boolean }[] = [
  { key: "rsi_14", label: "RSI (14)", numeric: true },
  { key: "close_vs_sma_50_pct", label: "Price vs 50-day average (%)", numeric: true },
  { key: "close_vs_sma_200_pct", label: "Price vs 200-day average (%)", numeric: true },
  { key: "sma_20_vs_50_pct", label: "20-day vs 50-day average (%)", numeric: true },
  { key: "regime", label: "Market regime label", numeric: false },
];
const NUMERIC_OPS = [
  { key: ">", label: "is above" },
  { key: "<", label: "is below" },
  { key: ">=", label: "is at or above" },
  { key: "<=", label: "is at or below" },
  { key: "crosses_above", label: "crosses above" },
  { key: "crosses_below", label: "crosses below" },
];
const CATEGORY_OPS = [
  { key: "is", label: "is" },
  { key: "is_not", label: "is not" },
];
const REGIMES = ["Risk-On", "Constructive", "Neutral", "Cautious", "Risk-Off"];

const DEFAULT_RULE: StrategyRuleInput = { field: "close_vs_sma_50_pct", op: ">", value: 0 };

function pct(v: number | null | undefined, digits = 2) {
  return v === null || v === undefined ? "n/a" : `${v > 0 ? "+" : ""}${v.toFixed(digits)}%`;
}
function num(v: number | null | undefined, digits = 2) {
  return v === null || v === undefined ? "n/a" : v.toFixed(digits);
}

function RuleRow({
  rule,
  onChange,
  onRemove,
}: {
  rule: StrategyRuleInput;
  onChange: (r: StrategyRuleInput) => void;
  onRemove: () => void;
}) {
  const field = FIELDS.find((f) => f.key === rule.field) ?? FIELDS[0];
  const ops = field.numeric ? NUMERIC_OPS : CATEGORY_OPS;
  return (
    <div className="flex flex-wrap items-center gap-2 text-sm">
      <select
        value={rule.field}
        onChange={(e) => {
          const next = FIELDS.find((f) => f.key === e.target.value) ?? FIELDS[0];
          onChange(
            next.numeric
              ? { field: next.key, op: ">", value: next.key === "rsi_14" ? 50 : 0 }
              : { field: next.key, op: "is", value: "Risk-On" },
          );
        }}
        className="input py-1 text-sm"
      >
        {FIELDS.map((f) => (
          <option key={f.key} value={f.key}>
            {f.label}
          </option>
        ))}
      </select>
      <select value={rule.op} onChange={(e) => onChange({ ...rule, op: e.target.value })} className="input py-1 text-sm">
        {ops.map((o) => (
          <option key={o.key} value={o.key}>
            {o.label}
          </option>
        ))}
      </select>
      {field.numeric ? (
        <input
          value={String(rule.value)}
          onChange={(e) => onChange({ ...rule, value: e.target.value })}
          className="input w-24 py-1 text-sm"
          aria-label="Value"
        />
      ) : (
        <select value={String(rule.value)} onChange={(e) => onChange({ ...rule, value: e.target.value })} className="input py-1 text-sm">
          {REGIMES.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
      )}
      <button type="button" onClick={onRemove} className="text-xs text-slate-500 hover:text-slate-800">
        Remove
      </button>
    </div>
  );
}

function MetricsTable({ rows }: { rows: { label: string; m: StrategyMetrics }[] }) {
  return (
    <table className="w-full text-left text-sm">
      <thead className="text-xs uppercase tracking-wide text-slate-400">
        <tr>
          <th className="py-1 font-medium">Result</th>
          <th className="py-1 font-medium">Total</th>
          <th className="py-1 font-medium">CAGR</th>
          <th className="py-1 font-medium">Volatility</th>
          <th className="py-1 font-medium">Max drawdown</th>
          <th className="py-1 font-medium">Sharpe</th>
          <th className="py-1 font-medium">Worst month</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-slate-100">
        {rows.map((r) => (
          <tr key={r.label}>
            <td className="py-1.5 text-slate-700">{r.label}</td>
            <td>{pct(r.m.total_return_pct)}</td>
            <td>{pct(r.m.cagr_pct)}</td>
            <td>{num(r.m.volatility_pct, 1)}%</td>
            <td>{pct(r.m.max_drawdown_pct, 1)}</td>
            <td>{num(r.m.sharpe)}</td>
            <td>{pct(r.m.worst_month_pct, 1)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function StrategyBuilderPage() {
  const [tickers, setTickers] = useState("AAPL, MSFT");
  const [entry, setEntry] = useState<StrategyRuleInput[]>([DEFAULT_RULE]);
  const [exit, setExit] = useState<StrategyRuleInput[]>([{ field: "rsi_14", op: ">", value: 60 }]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<StrategyBacktestResponse | null>(null);

  async function handleRun(e: React.FormEvent) {
    e.preventDefault();
    const list = tickers.split(/[\s,]+/).map((t) => t.trim().toUpperCase()).filter(Boolean);
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await runStrategyBacktest({
        tickers: Array.from(new Set(list)),
        entry: entry.map((r) => ({ ...r, value: typeof r.value === "string" && FIELDS.find((f) => f.key === r.field)?.numeric ? Number(r.value) : r.value })),
        exit: exit.map((r) => ({ ...r, value: typeof r.value === "string" && FIELDS.find((f) => f.key === r.field)?.numeric ? Number(r.value) : r.value })),
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "The backtest could not be run.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">Strategy builder</h1>
      <p className="mt-1 text-sm text-slate-500">
        Build rules from the dropdowns and test them on past prices. Costs and slippage are included. Nothing is placed.
      </p>

      <form onSubmit={handleRun} className="mt-6 flex flex-col gap-5 rounded-lg border border-slate-200 bg-white p-5">
        <label className="flex flex-col gap-1 text-sm text-slate-700">
          Tickers to test (up to 20, five years of history)
          <input value={tickers} onChange={(e) => setTickers(e.target.value)} className="input max-w-md py-1 text-sm" />
        </label>

        <div>
          <p className="text-sm font-medium text-slate-900">Entry: buy when all of these hold</p>
          <div className="mt-2 flex flex-col gap-2">
            {entry.map((r, i) => (
              <RuleRow
                key={i}
                rule={r}
                onChange={(next) => setEntry(entry.map((x, j) => (j === i ? next : x)))}
                onRemove={() => setEntry(entry.filter((_, j) => j !== i))}
              />
            ))}
          </div>
          {entry.length < 5 && (
            <button type="button" onClick={() => setEntry([...entry, DEFAULT_RULE])} className="mt-2 text-xs font-medium text-slate-700 hover:underline">
              + Add entry rule
            </button>
          )}
        </div>

        <div>
          <p className="text-sm font-medium text-slate-900">Exit: sell when any of these hold</p>
          <div className="mt-2 flex flex-col gap-2">
            {exit.map((r, i) => (
              <RuleRow
                key={i}
                rule={r}
                onChange={(next) => setExit(exit.map((x, j) => (j === i ? next : x)))}
                onRemove={() => setExit(exit.filter((_, j) => j !== i))}
              />
            ))}
          </div>
          {exit.length < 5 && (
            <button type="button" onClick={() => setExit([...exit, { field: "rsi_14", op: ">", value: 60 }])} className="mt-2 text-xs font-medium text-slate-700 hover:underline">
              + Add exit rule
            </button>
          )}
        </div>

        <button type="submit" disabled={loading} className="btn-primary self-start disabled:opacity-50">
          {loading ? "Running backtest…" : "Run backtest"}
        </button>
      </form>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {result && (
        <div className="mt-6 flex flex-col gap-5 rounded-lg border border-slate-200 bg-white p-5">
          <p className="text-xs text-slate-500">
            {result.period.start.slice(0, 10)} to {result.period.end.slice(0, 10)} · {result.period.sessions} sessions · {result.trades}{" "}
            trades ({num(result.trades_per_year, 1)} a year) · costs {result.costs.cost_bps_per_side} bps + {result.costs.slippage_bps_per_side} bps slippage per side ·{" "}
            {result.variants_tried} rule variant{result.variants_tried === 1 ? "" : "s"} tried in the last 90 days
          </p>

          {result.warnings.length > 0 && (
            <div className="rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-800">
              <p className="font-medium">Overfitting check</p>
              <ul className="mt-1 list-disc pl-5">
                {result.warnings.map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            </div>
          )}

          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Full period against SPY</p>
            <div className="mt-2">
              <MetricsTable rows={[{ label: "Strategy", m: result.strategy }, { label: "SPY", m: result.benchmark_spy }]} />
            </div>
          </div>

          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-slate-400">First 70% and last 30% of dates</p>
            <div className="mt-2">
              <MetricsTable rows={[{ label: "In-sample", m: result.in_sample }, { label: "Out-of-sample", m: result.out_of_sample }]} />
            </div>
          </div>

          <ul className="flex flex-col gap-1 text-xs text-slate-500">
            {result.caveats.map((c) => (
              <li key={c}>• {c}</li>
            ))}
          </ul>
          <p className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">{result.disclaimer}</p>
        </div>
      )}
    </div>
  );
}
