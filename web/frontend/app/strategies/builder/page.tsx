"use client";

import { useMemo, useState } from "react";
import PlotlyChart from "@/components/PlotlyChart";
import { ApiError, runStrategyBacktest } from "@/lib/api";
import type { StrategyBacktestResponse, StrategyCheck, StrategyMetrics, StrategyRuleInput } from "@/lib/types";

// Strategy Builder v2 (STB-1..5, SB-R4, SB-X2, SB-S1..S3). Builds rules from dropdowns and tests
// them on past prices. Nothing here places an order.

const FIELDS: { key: string; label: string; numeric: boolean }[] = [
  { key: "rsi_14", label: "RSI (14)", numeric: true },
  { key: "close_vs_sma_50_pct", label: "Price vs 50-day average (%)", numeric: true },
  { key: "close_vs_sma_200_pct", label: "Price vs 200-day average (%)", numeric: true },
  { key: "sma_20_vs_50_pct", label: "20-day vs 50-day average (%)", numeric: true },
  { key: "dist_52w_high_pct", label: "Distance from 52-week high (%)", numeric: true },
  { key: "volume_vs_20d_pct", label: "Volume vs 20-day average (%)", numeric: true },
  { key: "atr_14_pct", label: "ATR (14) as % of price", numeric: true },
  { key: "regime", label: "Market regime label", numeric: false },
];
const NUMERIC_OPS = [
  { key: "crosses_above", label: "crosses above" },
  { key: "crosses_below", label: "crosses below" },
  { key: ">", label: "is above" },
  { key: "<", label: "is below" },
  { key: ">=", label: "is at or above" },
  { key: "<=", label: "is at or below" },
];
const CATEGORY_OPS = [
  { key: "is", label: "is" },
  { key: "is_not", label: "is not" },
];
const REGIMES = ["Risk-On", "Constructive", "Neutral", "Cautious", "Risk-Off"];

const DEFAULT_ENTRY: StrategyRuleInput = { field: "rsi_14", op: "crosses_above", value: 50 };
const DEFAULT_EXIT: StrategyRuleInput = { field: "rsi_14", op: ">", value: 60 };
const CHECK_STYLE: Record<StrategyCheck["status"], { badge: string; text: string }> = {
  pass: { badge: "bg-emerald-100 text-emerald-800", text: "PASS" },
  caution: { badge: "bg-amber-100 text-amber-800", text: "CAUTION" },
  fail: { badge: "bg-red-100 text-red-800", text: "FAIL" },
};

function pct(v: number | null | undefined, digits = 2) {
  return v === null || v === undefined ? "n/a" : `${v > 0 ? "+" : ""}${v.toFixed(digits)}%`;
}
function num(v: number | null | undefined, digits = 2) {
  return v === null || v === undefined ? "n/a" : (v > 0 && digits > 0 ? "+" : "") + v.toFixed(digits);
}
function plain(v: number | null | undefined, digits = 2, suffix = "") {
  return v === null || v === undefined ? "n/a" : `${v.toFixed(digits)}${suffix}`;
}

function fieldLabel(key: string) {
  return FIELDS.find((f) => f.key === key)?.label ?? key;
}
function opLabel(op: string) {
  return [...NUMERIC_OPS, ...CATEGORY_OPS].find((o) => o.key === op)?.label ?? op;
}
function ruleText(r: StrategyRuleInput) {
  return `${fieldLabel(r.field)} ${opLabel(r.op)} ${r.value}`;
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
              ? { field: next.key, op: "crosses_above", value: next.key === "rsi_14" ? 50 : 0 }
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
    <div className="overflow-x-auto">
      <table className="w-full min-w-[36rem] text-left text-sm">
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
              <td className="text-right">{pct(r.m.total_return_pct)}</td>
              <td className="text-right">{pct(r.m.cagr_pct)}</td>
              <td className="text-right">{plain(r.m.volatility_pct, 1, "%")}</td>
              <td className="text-right">{pct(r.m.max_drawdown_pct, 1)}</td>
              <td className="text-right">{plain(r.m.sharpe)}</td>
              <td className="text-right">{pct(r.m.worst_month_pct, 1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function verdictSentence(r: StrategyBacktestResponse) {
  const v = r.verdict;
  const bench = v.benchmark === "basket" ? "the same-basket buy-and-hold" : "SPY";
  if (v.beats_benchmark_after_costs && (v.sharpe_vs_basket ?? 0) > 0 && v.benchmark === "basket") {
    return `The rules beat ${bench} after costs, and their risk-adjusted return was higher too.`;
  }
  if (v.beats_benchmark_after_costs) {
    return `The rules beat ${bench} on return after costs, but not on risk-adjusted return.`;
  }
  return `The rules did not beat ${bench} after costs.`;
}

export default function StrategyBuilderPage() {
  const [tickers, setTickers] = useState("AAPL, MSFT");
  const [entry, setEntry] = useState<StrategyRuleInput[]>([DEFAULT_ENTRY]);
  const [exit, setExit] = useState<StrategyRuleInput[]>([DEFAULT_EXIT]);
  const [stopLoss, setStopLoss] = useState("");
  const [trailing, setTrailing] = useState("10");
  const [timeStop, setTimeStop] = useState("");
  const [waive, setWaive] = useState(false);
  const [cooldown, setCooldown] = useState("5");
  const [verdictBenchmark, setVerdictBenchmark] = useState<"basket" | "spy">("basket");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<StrategyBacktestResponse | null>(null);

  const hasProtective = Boolean(stopLoss || trailing || timeStop) || exit.length > 0;
  const sentence = useMemo(() => {
    const exitParts = exit.map(ruleText);
    if (trailing) exitParts.push(`a ${trailing}% trailing stop is hit`);
    if (stopLoss) exitParts.push(`a ${stopLoss}% stop-loss is hit`);
    if (timeStop) exitParts.push(`${timeStop} sessions have passed`);
    const entryText = entry.length ? entry.map(ruleText).join(" and ") : "(no entry rule)";
    return `Buy when ${entryText}. Sell when ${exitParts.length ? exitParts.join(", or ") : "(no exit set)"}.`;
  }, [entry, exit, trailing, stopLoss, timeStop]);

  function numeric(r: StrategyRuleInput): StrategyRuleInput {
    const isNumeric = FIELDS.find((f) => f.key === r.field)?.numeric ?? false;
    return isNumeric ? { ...r, value: Number(r.value) } : r;
  }

  async function handleRun(e: React.FormEvent) {
    e.preventDefault();
    const list = tickers.split(/[\s,]+/).map((t) => t.trim().toUpperCase()).filter(Boolean);
    const exits: { stop_loss_pct?: number; trailing_stop_pct?: number; time_stop_sessions?: number } = {};
    if (stopLoss) exits.stop_loss_pct = Number(stopLoss);
    if (trailing) exits.trailing_stop_pct = Number(trailing);
    if (timeStop) exits.time_stop_sessions = Number(timeStop);
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await runStrategyBacktest({
        tickers: Array.from(new Set(list)),
        entry: entry.map(numeric),
        exit: exit.map(numeric),
        exits,
        waive_protective_exit: waive,
        cooldown_sessions: Number(cooldown) || 0,
        verdict_benchmark: verdictBenchmark,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "The backtest could not be run.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">Strategy builder</h1>
      <p className="mt-1 text-sm text-slate-500">
        Test rules on past prices against buying the same stocks and holding them. Costs and slippage are included. Nothing is placed.
      </p>

      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-[400px_1fr]">
        <form onSubmit={handleRun} className="flex flex-col gap-5 rounded-lg border border-slate-200 bg-white p-5 lg:sticky lg:top-4 lg:self-start">
          <label className="flex flex-col gap-1 text-sm text-slate-700">
            Tickers (up to 20, five years of history)
            <input value={tickers} onChange={(e) => setTickers(e.target.value)} className="input py-1 text-sm" />
            <span className="text-xs text-slate-400">Chosen by hand, so results partly reflect the stocks picked.</span>
          </label>

          <p className="rounded-md bg-slate-50 px-3 py-2 text-sm text-slate-700">{sentence}</p>

          <div>
            <p className="text-sm font-medium text-slate-900">Entry: all of these</p>
            <div className="mt-2 flex flex-col gap-2">
              {entry.map((r, i) => (
                <RuleRow key={i} rule={r} onChange={(next) => setEntry(entry.map((x, j) => (j === i ? next : x)))} onRemove={() => setEntry(entry.filter((_, j) => j !== i))} />
              ))}
            </div>
            {entry.length < 5 && (
              <button type="button" onClick={() => setEntry([...entry, DEFAULT_ENTRY])} className="mt-2 text-xs font-medium text-slate-700 hover:underline">
                + Add entry rule
              </button>
            )}
          </div>

          <div>
            <p className="text-sm font-medium text-slate-900">Exit: any of these</p>
            <div className="mt-2 flex flex-col gap-2">
              {exit.map((r, i) => (
                <RuleRow key={i} rule={r} onChange={(next) => setExit(exit.map((x, j) => (j === i ? next : x)))} onRemove={() => setExit(exit.filter((_, j) => j !== i))} />
              ))}
            </div>
            {exit.length < 5 && (
              <button type="button" onClick={() => setExit([...exit, DEFAULT_EXIT])} className="mt-2 text-xs font-medium text-slate-700 hover:underline">
                + Add exit rule
              </button>
            )}
          </div>

          <div className="rounded-md border border-slate-200 p-3">
            <p className="text-sm font-medium text-slate-900">Protective exit</p>
            <div className="mt-2 grid grid-cols-3 gap-2 text-xs text-slate-600">
              <label className="flex flex-col gap-1">
                Trailing stop %
                <input value={trailing} onChange={(e) => setTrailing(e.target.value)} className="input py-1 text-sm" />
              </label>
              <label className="flex flex-col gap-1">
                Stop-loss %
                <input value={stopLoss} onChange={(e) => setStopLoss(e.target.value)} className="input py-1 text-sm" />
              </label>
              <label className="flex flex-col gap-1">
                Time stop (sessions)
                <input value={timeStop} onChange={(e) => setTimeStop(e.target.value)} className="input py-1 text-sm" />
              </label>
            </div>
            <label className="mt-3 flex items-start gap-2 text-xs text-slate-600">
              <input type="checkbox" checked={waive} onChange={(e) => setWaive(e.target.checked)} className="mt-0.5" />
              Run without a protective exit (the result will say so)
            </label>
            {!hasProtective && !waive && (
              <p className="mt-2 text-xs text-amber-700">Add a stop, trailing stop, time stop or exit rule, or tick the box above.</p>
            )}
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs text-slate-600">
            <label className="flex flex-col gap-1">
              Re-entry cooldown (sessions)
              <input value={cooldown} onChange={(e) => setCooldown(e.target.value)} className="input py-1 text-sm" />
            </label>
            <label className="flex flex-col gap-1">
              Verdict compares against
              <select value={verdictBenchmark} onChange={(e) => setVerdictBenchmark(e.target.value as "basket" | "spy")} className="input py-1 text-sm">
                <option value="basket">Same stocks, bought and held</option>
                <option value="spy">SPY</option>
              </select>
            </label>
          </div>

          <button type="submit" disabled={loading || (!hasProtective && !waive)} className="btn-primary self-start disabled:opacity-50">
            {loading ? "Running backtest…" : "Run backtest"}
          </button>
        </form>

        <div className="flex min-w-0 flex-col gap-5">
          {error && <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
          {!result && !error && !loading && (
            <p className="rounded-lg border border-dashed border-slate-300 p-6 text-sm text-slate-500">Run a backtest to see the verdict.</p>
          )}

          {result && (
            <>
              <div className="rounded-lg border border-slate-200 bg-white p-5">
                <p className="text-xs uppercase tracking-wide text-slate-400">Verdict</p>
                <p className="mt-1 text-base font-medium text-slate-900">{verdictSentence(result)}</p>
                {result.protective_exit.waived && (
                  <p className="mt-2 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">No protective exit: losing positions are held until an exit rule fires.</p>
                )}
                <dl className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
                  <div>
                    <dt className="text-xs text-slate-400">Excess CAGR vs {result.verdict.benchmark === "basket" ? "same stocks" : "SPY"}</dt>
                    <dd className="font-mono text-lg font-semibold text-slate-900">{pct(result.verdict.excess_cagr_pct)}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-slate-400">Sharpe vs same stocks</dt>
                    <dd className="font-mono text-lg font-semibold text-slate-900">{num(result.verdict.sharpe_vs_basket)}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-slate-400">Sharpe vs SPY</dt>
                    <dd className="font-mono text-lg font-semibold text-slate-900">{num(result.verdict.sharpe_vs_spy)}</dd>
                  </div>
                </dl>
                <ul className="mt-4 flex flex-col gap-2">
                  {result.checks.map((c) => (
                    <li key={c.label} className="flex items-start gap-3 text-sm">
                      <span className={`mt-0.5 shrink-0 rounded px-2 py-0.5 text-xs font-semibold ${CHECK_STYLE[c.status].badge}`}>
                        {CHECK_STYLE[c.status].text}
                      </span>
                      <span>
                        <span className="font-medium text-slate-800">{c.label}.</span>{" "}
                        <span className="text-slate-600">{c.detail}</span>
                      </span>
                    </li>
                  ))}
                </ul>
                {result.state_warnings.map((w) => (
                  <p key={w} className="mt-3 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">{w}</p>
                ))}
              </div>

              <div className="rounded-lg border border-slate-200 bg-white p-5">
                <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Growth of 100</p>
                <PlotlyChart
                  data={[
                    { x: result.equity_curve.dates, y: result.equity_curve.strategy, type: "scatter", mode: "lines", name: "Strategy", line: { color: "#1F4FD1", width: 2 } },
                    { x: result.equity_curve.dates, y: result.equity_curve.basket, type: "scatter", mode: "lines", name: "Same stocks, bought and held", line: { color: "#64748B", width: 1.5, dash: "dash" } },
                  ]}
                  layout={{ height: 300, margin: { t: 10, r: 16, b: 30, l: 50 }, paper_bgcolor: "#ffffff", plot_bgcolor: "#ffffff", legend: { orientation: "h", y: -0.2 } }}
                  style={{ width: "100%" }}
                  useResizeHandler
                  config={{ displayModeBar: false }}
                />
              </div>

              <div className="rounded-lg border border-slate-200 bg-white p-5">
                <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Comparison</p>
                <div className="mt-2">
                  <MetricsTable
                    rows={[
                      { label: "Strategy", m: result.strategy },
                      { label: "Same stocks, bought and held", m: result.basket },
                      { label: "SPY", m: result.benchmark_spy },
                    ]}
                  />
                </div>
                <dl className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
                  <div>
                    <dt className="text-xs text-slate-400">Cost drag</dt>
                    <dd>{plain(result.cost_drag.cagr_points, 2, " CAGR points")}</dd>
                    <dd className="text-xs text-slate-400">{plain(result.cost_drag.total_costs_pct_of_equity, 2, "% of equity")}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-slate-400">Trades</dt>
                    <dd>{result.trades} ({plain(result.trades_per_year, 1)} a year)</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-slate-400">Churn</dt>
                    <dd>{plain(result.churn_pct, 1, "%")} re-entered within 2 sessions</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-slate-400">Time invested</dt>
                    <dd>{plain(result.exposure_pct, 1, "%")}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-slate-400">Win rate</dt>
                    <dd>{plain(result.win_rate_pct, 1, "%")}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-slate-400">Average win / loss</dt>
                    <dd>{pct(result.avg_win_pct)} / {pct(result.avg_loss_pct)}</dd>
                  </div>
                </dl>
              </div>

              <div className="rounded-lg border border-slate-200 bg-white p-5">
                <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Out-of-sample check (first 70% and last 30% of dates)</p>
                <div className="mt-2">
                  <MetricsTable rows={[{ label: "In-sample", m: result.in_sample }, { label: "Out-of-sample", m: result.out_of_sample }]} />
                </div>
              </div>

              <div className="rounded-lg border border-slate-200 bg-white p-5">
                <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Contribution by stock</p>
                <ul className="mt-2 flex flex-col gap-1 text-sm">
                  {Object.entries(result.per_ticker_contribution).map(([t, c]) => (
                    <li key={t} className="flex justify-between">
                      <span className="text-slate-700">{t}</span>
                      <span className="font-mono text-slate-600">{pct(c)}</span>
                    </li>
                  ))}
                </ul>
                {result.top_ticker?.flag && (
                  <p className="mt-2 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
                    {result.top_ticker.ticker} supplies {plain(result.top_ticker.share_pct, 0, "%")} of the total gain, so the result depends heavily on one stock.
                  </p>
                )}
              </div>

              <div className="rounded-lg border border-slate-200 bg-white p-5">
                <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Trades (first 50)</p>
                <div className="mt-2 overflow-x-auto">
                  <table className="w-full min-w-[36rem] text-left text-xs">
                    <thead className="text-slate-400">
                      <tr>
                        <th className="py-1 font-medium">Stock</th>
                        <th className="py-1 font-medium">Entry</th>
                        <th className="py-1 font-medium">Exit</th>
                        <th className="py-1 font-medium">Exit reason</th>
                        <th className="py-1 text-right font-medium">Days</th>
                        <th className="py-1 text-right font-medium">Return</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {result.trade_log.slice(0, 50).map((t, i) => (
                        <tr key={i}>
                          <td className="py-1">{t.ticker}</td>
                          <td>{t.entry_date} @ {t.entry_price}</td>
                          <td>{t.exit_date} @ {t.exit_price}</td>
                          <td>{t.exit_reason.replace("_", " ")}</td>
                          <td className="text-right">{t.holding_days}</td>
                          <td className="text-right">{pct(t.return_pct)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="rounded-lg border border-slate-200 bg-white p-5">
                <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Notes</p>
                <p className="mt-2 text-xs text-slate-500">
                  {result.period.start} to {result.period.end} · {result.period.sessions} sessions · costs {result.costs.cost_bps_per_side} bps + {result.costs.slippage_bps_per_side} bps slippage per side · cooldown {result.cooldown_sessions} sessions · {result.variants_tried} rule variant{result.variants_tried === 1 ? "" : "s"} tried in the last 90 days
                  {result.chance_sharpe_bar !== null ? ` · chance bar ${result.chance_sharpe_bar.toFixed(2)} Sharpe (approximation)` : ""}
                </p>
                <ul className="mt-2 flex flex-col gap-1 text-xs text-slate-500">
                  {result.caveats.map((c) => (
                    <li key={c}>• {c}</li>
                  ))}
                </ul>
                <p className="mt-3 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">{result.disclaimer}</p>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
