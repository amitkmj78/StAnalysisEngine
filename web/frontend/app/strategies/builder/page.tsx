"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import PlotlyChart from "@/components/PlotlyChart";
import { ApiError, getStrategyPresets, runStrategyBacktest, saveStrategy } from "@/lib/api";
import type { StrategyBacktestResponse, StrategyCheck, StrategyRuleInput } from "@/lib/types";

// Strategy Builder (v2 layout). Builder on the left, results on the right, verdict first.
// Nothing here places an order. Controls for features not built yet are shown disabled.

const FIELDS: { key: string; label: string; numeric: boolean }[] = [
  { key: "rsi_14", label: "RSI (14)", numeric: true },
  { key: "close_vs_sma_50_pct", label: "Price vs 50-day avg (%)", numeric: true },
  { key: "close_vs_sma_200_pct", label: "Price vs 200-day avg (%)", numeric: true },
  { key: "sma_20_vs_50_pct", label: "20-day vs 50-day avg (%)", numeric: true },
  { key: "dist_52w_high_pct", label: "Distance from 52-week high (%)", numeric: true },
  { key: "volume_vs_20d_pct", label: "Volume vs 20-day avg (%)", numeric: true },
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
const DEFAULT_TICKERS = ["AAPL", "MSFT", "DELL", "VEEV", "HPE"];
const SECTORS = [
  "Information Technology", "Health Care", "Financials", "Consumer Discretionary", "Communication Services",
  "Industrials", "Consumer Staples", "Energy", "Utilities", "Real Estate", "Materials",
];

const BADGE: Record<StrategyCheck["status"], string> = {
  pass: "bg-emerald-100 text-emerald-800",
  caution: "bg-amber-100 text-amber-800",
  fail: "bg-red-100 text-red-800",
};
const BADGE_TEXT: Record<StrategyCheck["status"], string> = { pass: "PASS", caution: "CAUTION", fail: "FAIL" };

function pct(v: number | null | undefined, digits = 1) {
  return v === null || v === undefined ? "–" : `${v > 0 ? "+" : ""}${v.toFixed(digits)}%`;
}
function plain(v: number | null | undefined, digits = 2) {
  return v === null || v === undefined ? "–" : v.toFixed(digits);
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
function numeric(r: StrategyRuleInput): StrategyRuleInput {
  const isNumeric = FIELDS.find((f) => f.key === r.field)?.numeric ?? false;
  return isNumeric ? { ...r, value: Number(r.value) } : r;
}

// Entry state that an exit threshold already satisfies (same field, same direction).
function conflictIndex(entry: StrategyRuleInput[], exit: StrategyRuleInput[]) {
  for (let i = 0; i < entry.length; i++) {
    const e = entry[i];
    if (!["<", ">", "<=", ">="].includes(e.op)) continue;
    for (const x of exit) {
      if (x.field !== e.field || x.op !== e.op) continue;
      const ev = Number(e.value);
      const xv = Number(x.value);
      if ((e.op === ">" || e.op === ">=") && xv >= ev) return i;
      if ((e.op === "<" || e.op === "<=") && xv <= ev) return i;
    }
  }
  return -1;
}

function RuleRow({ rule, onChange, onRemove }: { rule: StrategyRuleInput; onChange: (r: StrategyRuleInput) => void; onRemove: () => void }) {
  const field = FIELDS.find((f) => f.key === rule.field) ?? FIELDS[0];
  const ops = field.numeric ? NUMERIC_OPS : CATEGORY_OPS;
  return (
    <div className="grid grid-cols-[1fr_1fr_4.5rem_auto] items-center gap-1.5 text-xs">
      <select
        value={rule.field}
        onChange={(e) => {
          const next = FIELDS.find((f) => f.key === e.target.value) ?? FIELDS[0];
          onChange(next.numeric ? { field: next.key, op: "crosses_above", value: next.key === "rsi_14" ? 50 : 0 } : { field: next.key, op: "is", value: "Risk-On" });
        }}
        className="input w-full py-1 text-xs"
        aria-label="Indicator"
      >
        {FIELDS.map((f) => (
          <option key={f.key} value={f.key}>
            {f.label}
          </option>
        ))}
      </select>
      <select value={rule.op} onChange={(e) => onChange({ ...rule, op: e.target.value })} className="input w-full py-1 text-xs" aria-label="Condition">
        {ops.map((o) => (
          <option key={o.key} value={o.key}>
            {o.label}
          </option>
        ))}
      </select>
      {field.numeric ? (
        <input value={String(rule.value)} onChange={(e) => onChange({ ...rule, value: e.target.value })} className="input w-full py-1 text-xs" aria-label="Value" />
      ) : (
        <select value={String(rule.value)} onChange={(e) => onChange({ ...rule, value: e.target.value })} className="input w-full py-1 text-xs" aria-label="Regime">
          {REGIMES.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
      )}
      <button type="button" onClick={onRemove} className="px-1 text-slate-400 hover:text-slate-700" aria-label="Remove rule">
        ×
      </button>
    </div>
  );
}

function Panel({ title, children, right }: { title: string; children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500">{title}</h2>
        {right}
      </div>
      <div className="mt-3">{children}</div>
    </section>
  );
}

function Stat({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: string }) {
  return (
    <div className="rounded-md border border-slate-200 p-3">
      <p className="text-xs text-slate-500">{label}</p>
      <p className={`mt-1 font-mono text-xl font-semibold ${tone ?? "text-slate-900"}`}>{value}</p>
      {sub && <p className="mt-0.5 text-xs text-slate-400">{sub}</p>}
    </div>
  );
}

export default function StrategyBuilderPage() {
  const [name, setName] = useState("RSI momentum");
  const [tickers, setTickers] = useState<string[]>(DEFAULT_TICKERS);
  const [tickerInput, setTickerInput] = useState("");
  const [entry, setEntry] = useState<StrategyRuleInput[]>([DEFAULT_ENTRY]);
  const [exit, setExit] = useState<StrategyRuleInput[]>([DEFAULT_EXIT]);
  const [trailing, setTrailing] = useState("10");
  const [stopLoss, setStopLoss] = useState("");
  const [timeStop, setTimeStop] = useState("");
  const [waive, setWaive] = useState(false);
  const [cooldown, setCooldown] = useState("5");
  const [verdictBenchmark, setVerdictBenchmark] = useState<"basket" | "spy">("basket");
  const [chartBenchmark, setChartBenchmark] = useState<"basket" | "spy" | "cash">("basket");
  const [tab, setTab] = useState<"robustness" | "trades" | "tickers">("robustness");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<StrategyBacktestResponse | null>(null);
  const [lastPayload, setLastPayload] = useState<Record<string, unknown> | null>(null);
  const [sector, setSector] = useState("");
  const [presetError, setPresetError] = useState<string | null>(null);
  const [saveState, setSaveState] = useState<string | null>(null);

  const hasProtective = Boolean(trailing || stopLoss || timeStop) || exit.length > 0;
  const conflict = conflictIndex(entry, exit);

  const plainWords = useMemo(() => {
    const exitParts = exit.map(ruleText);
    if (trailing) exitParts.push(`a ${trailing}% trailing stop is hit`);
    if (stopLoss) exitParts.push(`a ${stopLoss}% stop-loss is hit`);
    if (timeStop) exitParts.push(`${timeStop} sessions have passed`);
    return {
      entry: entry.length ? entry.map(ruleText).join(" and ") : "(no entry rule)",
      exit: exitParts.length ? exitParts.join(", or ") : "(no exit set)",
    };
  }, [entry, exit, trailing, stopLoss, timeStop]);

  function addTicker() {
    const t = tickerInput.trim().toUpperCase();
    if (t && !tickers.includes(t) && tickers.length < 20) setTickers([...tickers, t]);
    setTickerInput("");
  }

  async function runBacktest(sampling?: { sample_seed?: number; sample_size?: number }) {
    const exits: { stop_loss_pct?: number; trailing_stop_pct?: number; time_stop_sessions?: number } = {};
    if (stopLoss) exits.stop_loss_pct = Number(stopLoss);
    if (trailing) exits.trailing_stop_pct = Number(trailing);
    if (timeStop) exits.time_stop_sessions = Number(timeStop);
    const payload = {
      tickers,
      entry: entry.map(numeric),
      exit: exit.map(numeric),
      exits,
      waive_protective_exit: waive,
      cooldown_sessions: Number(cooldown) || 0,
      verdict_benchmark: verdictBenchmark,
      ...(sampling ? { source: "random_sample" as const, ...sampling } : { source: "hand_picked" as const }),
    };
    setLoading(true);
    setError(null);
    setResult(null);
    setSaveState(null);
    try {
      const res = await runStrategyBacktest(payload);
      setResult(res);
      setLastPayload(payload);
      if (res.selection?.seed != null) setTickers(res.selection.tickers ?? tickers);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "The backtest could not be run.");
    } finally {
      setLoading(false);
    }
  }

  async function handleRun(e: React.FormEvent) {
    e.preventDefault();
    await runBacktest();
  }

  async function loadSp500Sample() {
    setPresetError(null);
    try {
      const res = await getStrategyPresets({ kind: "sp500_sample", size: 10 });
      setTickers(res.tickers);
    } catch (err) {
      setPresetError(err instanceof ApiError ? err.message : "The preset could not be loaded.");
    }
  }

  async function loadSector(name: string) {
    setSector(name);
    if (!name) return;
    setPresetError(null);
    try {
      const res = await getStrategyPresets({ kind: "sector", size: 10, sector: name });
      setTickers(res.tickers);
    } catch (err) {
      setPresetError(err instanceof ApiError ? err.message : "The sector basket could not be loaded.");
    }
  }

  async function handleSave() {
    if (!result || !lastPayload) return;
    setSaveState("Saving…");
    try {
      await saveStrategy({ name: name.trim() || "Untitled strategy", definition: lastPayload, result });
      setSaveState("Saved. See Saved strategies to compare or share it.");
    } catch (err) {
      setSaveState(err instanceof ApiError ? err.message : "Could not save this run.");
    }
  }

  const r = result;
  const cagrVsBasket = r && r.strategy.cagr_pct != null && r.basket.cagr_pct != null ? r.strategy.cagr_pct - r.basket.cagr_pct : null;
  const cagrVsSpy = r && r.strategy.cagr_pct != null && r.benchmark_spy.cagr_pct != null ? r.strategy.cagr_pct - r.benchmark_spy.cagr_pct : null;

  const verdictText = !r
    ? "Run a backtest to see whether the rules added value."
    : r.verdict.beats_benchmark_after_costs && (r.verdict.sharpe_vs_basket ?? 0) > 0
      ? "The rules beat holding the same stocks after costs, and on risk-adjusted return too."
      : r.verdict.beats_benchmark_after_costs
        ? "The rules beat the benchmark on return after costs, but not after adjusting for risk."
        : "The rules did not beat the benchmark after costs.";

  const chart = useMemo(() => {
    if (!r) return null;
    const dates = r.equity_curve.dates;
    const strategy = r.equity_curve.strategy.map((v) => (v / 100) * 10000);
    const bench =
      chartBenchmark === "cash"
        ? dates.map(() => 10000)
        : chartBenchmark === "spy"
          ? r.equity_curve.strategy.map(() => null)
          : r.equity_curve.basket.map((v) => (v / 100) * 10000);
    let peak = -Infinity;
    const drawdown = strategy.map((v) => {
      peak = Math.max(peak, v);
      return ((v / peak - 1) * 100);
    });
    const splitIndex = Math.floor(dates.length * 0.7);
    return { dates, strategy, bench, drawdown, split: dates[splitIndex] };
  }, [r, chartBenchmark]);

  return (
    <div className="mx-auto max-w-[1320px] px-4 py-6">
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[320px_1fr]">
        {/* Builder */}
        <form onSubmit={handleRun} className="flex flex-col gap-4 lg:sticky lg:top-4 lg:self-start">
          <Panel title="Strategy">
            <div className="flex gap-2">
              <input value={name} onChange={(e) => setName(e.target.value)} className="input min-w-0 flex-1 py-1 text-sm" aria-label="Strategy name" />
              <button type="button" disabled className="rounded-md border border-slate-200 px-2 py-1 text-xs text-slate-400" title="Saved variants: coming next">
                Variants
              </button>
            </div>
          </Panel>

          <Panel title={`Universe · ${tickers.length} of 20`} right={<button type="button" onClick={loadSp500Sample} className="text-xs font-medium text-slate-700 hover:underline">Random S&amp;P 500 ×10</button>}>
            <div className="flex flex-wrap gap-1.5">
              {tickers.map((t) => (
                <span key={t} className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-700">
                  {t}
                  <button type="button" onClick={() => setTickers(tickers.filter((x) => x !== t))} className="text-slate-400 hover:text-slate-700" aria-label={`Remove ${t}`}>
                    ×
                  </button>
                </span>
              ))}
            </div>
            <input
              value={tickerInput}
              onChange={(e) => setTickerInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  addTicker();
                }
              }}
              placeholder="Add ticker"
              className="input mt-2 w-full py-1 text-xs"
              aria-label="Add ticker"
            />
            <select value={sector} onChange={(e) => loadSector(e.target.value)} className="input mt-2 w-full py-1 text-xs" aria-label="Sector basket">
              <option value="">Sector basket (largest 10)…</option>
              {SECTORS.map((x) => (
                <option key={x} value={x}>
                  {x}
                </option>
              ))}
            </select>
            {presetError && <p className="mt-2 text-xs text-red-700">{presetError}</p>}
            <p className="mt-3 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
              Hand-picked list. Results may reflect which stocks you chose rather than the rules.{" "}
              <button type="button" onClick={() => runBacktest({ sample_size: Math.max(tickers.length, 5) })} disabled={loading} className="font-semibold underline underline-offset-2 disabled:opacity-50">
                Re-run on {Math.max(tickers.length, 5)} random S&amp;P 500 stocks
              </button>
            </p>
          </Panel>

          <Panel title="In plain words">
            <p className="text-sm text-slate-700">
              Buy when <strong>{plainWords.entry}</strong>. Sell when <strong>{plainWords.exit}</strong>. Wait <strong>{cooldown || 0} sessions</strong> before buying the same stock again.
            </p>
          </Panel>

          <Panel title="Entry · buy when all hold">
            <div className="flex flex-col gap-2">
              {entry.map((rule, i) => (
                <div key={i} className="flex flex-col gap-2">
                  <RuleRow rule={rule} onChange={(next) => setEntry(entry.map((x, j) => (j === i ? next : x)))} onRemove={() => setEntry(entry.filter((_, j) => j !== i))} />
                  {conflict === i && (
                    <div className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
                      <p>
                        <strong>Caution: re-buys right after selling.</strong> An exit at this level also meets this entry, so the stock is bought back the next session.
                      </p>
                      <button
                        type="button"
                        onClick={() => setEntry(entry.map((x, j) => (j === i ? { ...x, op: x.op.startsWith(">") || x.op === ">=" ? "crosses_above" : "crosses_below" } : x)))}
                        className="mt-2 rounded border border-amber-300 bg-white px-2 py-1 font-medium"
                      >
                        Switch to “crosses above”
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
            {entry.length < 5 && (
              <button type="button" onClick={() => setEntry([...entry, DEFAULT_ENTRY])} className="mt-3 text-xs font-medium text-slate-700 hover:underline">
                + Add entry rule
              </button>
            )}
          </Panel>

          <Panel title="Exit · sell when any holds">
            <div className="flex flex-col gap-2">
              {exit.map((rule, i) => (
                <RuleRow key={i} rule={rule} onChange={(next) => setExit(exit.map((x, j) => (j === i ? next : x)))} onRemove={() => setExit(exit.filter((_, j) => j !== i))} />
              ))}
            </div>
            <div className="mt-3 rounded-md border border-dashed border-slate-300 p-3">
              <p className="text-xs font-semibold text-slate-700">Protective exit <span className="font-normal text-slate-400">· limits losers</span></p>
              <div className="mt-2 grid grid-cols-3 gap-2 text-xs text-slate-600">
                <label className="flex flex-col gap-1">
                  Trailing stop %
                  <input value={trailing} onChange={(e) => setTrailing(e.target.value)} className="input py-1 text-xs" />
                </label>
                <label className="flex flex-col gap-1">
                  Stop-loss %
                  <input value={stopLoss} onChange={(e) => setStopLoss(e.target.value)} className="input py-1 text-xs" />
                </label>
                <label className="flex flex-col gap-1">
                  Time stop (sessions)
                  <input value={timeStop} onChange={(e) => setTimeStop(e.target.value)} className="input py-1 text-xs" />
                </label>
              </div>
              <label className="mt-3 flex items-center gap-2 text-xs text-slate-600">
                <input type="checkbox" checked={waive} onChange={(e) => setWaive(e.target.checked)} />
                Run without a protective exit
              </label>
              {!hasProtective && !waive && <p className="mt-2 text-xs text-amber-700">Add a stop or exit rule, or tick the box above.</p>}
            </div>
            {exit.length < 5 && (
              <button type="button" onClick={() => setExit([...exit, DEFAULT_EXIT])} className="mt-3 text-xs font-medium text-slate-700 hover:underline">
                + Add exit rule
              </button>
            )}
          </Panel>

          <Panel title="Re-entry and execution">
            <div className="grid grid-cols-2 gap-3 text-xs text-slate-600">
              <label className="flex flex-col gap-1">
                Cooldown after exit
                <input value={cooldown} onChange={(e) => setCooldown(e.target.value)} className="input py-1 text-xs" />
              </label>
              <label className="flex flex-col gap-1">
                Sizing
                <select disabled className="input py-1 text-xs" title="Only equal weight per stock for now">
                  <option>Equal weight per stock</option>
                </select>
              </label>
              <label className="flex flex-col gap-1">
                Costs + slippage
                <span className="rounded border border-slate-200 px-2 py-1 text-slate-500">10 + 5 bps per side</span>
              </label>
              <label className="flex flex-col gap-1">
                Fills
                <span className="rounded border border-slate-200 px-2 py-1 text-slate-500">Next session’s open</span>
              </label>
              <label className="col-span-2 flex flex-col gap-1">
                Verdict compares against
                <select value={verdictBenchmark} onChange={(e) => setVerdictBenchmark(e.target.value as "basket" | "spy")} className="input py-1 text-xs">
                  <option value="basket">The same stocks, held</option>
                  <option value="spy">SPY</option>
                </select>
              </label>
            </div>
          </Panel>

          {error && <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
          <button type="submit" disabled={loading || tickers.length === 0 || (!hasProtective && !waive)} className="rounded-md bg-slate-900 px-4 py-2.5 text-sm font-semibold text-white hover:bg-slate-800 disabled:opacity-50">
            {loading ? "Running backtest…" : "Run backtest"}
          </button>
          <p className="text-center text-xs text-slate-400">Running this counts as one rule variant in the last 90 days. More variants make the best result less trustworthy.</p>
        </form>

        {/* Results */}
        <div className="flex min-w-0 flex-col gap-5">
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
            <Link href="/strategies/saved" className="font-medium text-slate-700 hover:underline">Saved strategies →</Link>
            {r && (
              <div className="flex items-center gap-2">
                <button type="button" onClick={handleSave} className="rounded-md border border-slate-300 px-3 py-1.5 font-medium text-slate-700 hover:bg-slate-50">
                  Save this run
                </button>
                {saveState && <span className="text-slate-500">{saveState}</span>}
              </div>
            )}
          </div>
          {r && (
            <p className="font-mono text-xs text-slate-500">
              {r.period.start} → {r.period.end} · {r.period.sessions.toLocaleString()} sessions · {r.trades} trades ({plain(r.trades_per_year, 0)} a year) · {r.costs.cost_bps_per_side} + {r.costs.slippage_bps_per_side} bps per side
            </p>
          )}

          <section className="rounded-lg border border-slate-200 bg-white p-5">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Did the rules add value?</p>
            <p className="mt-2 text-lg font-medium text-slate-900">{verdictText}</p>
            {r?.protective_exit.waived && (
              <p className="mt-2 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">No protective exit: losing positions are held until an exit rule fires.</p>
            )}
            {r && (
              <>
                <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
                  <Stat label="Excess CAGR vs same stocks held" value={pct(cagrVsBasket)} sub={`${pct(r.strategy.cagr_pct)} vs ${pct(r.basket.cagr_pct)}`} />
                  <Stat label="Excess CAGR vs SPY" value={pct(cagrVsSpy)} sub={`${pct(r.strategy.cagr_pct)} vs ${pct(r.benchmark_spy.cagr_pct)}`} />
                  <Stat
                    label="Sharpe vs SPY"
                    value={`${plain(r.strategy.sharpe)} vs ${plain(r.benchmark_spy.sharpe)}`}
                    tone={(r.verdict.sharpe_vs_spy ?? 0) > 0 ? "text-emerald-700" : "text-red-700"}
                    sub={(r.verdict.sharpe_vs_spy ?? 0) > 0 ? "Higher return per unit of risk" : "More risk for each unit of return"}
                  />
                </div>
                <ul className="mt-4 grid grid-cols-1 gap-2 md:grid-cols-2">
                  {r.checks.map((c) => (
                    <li key={c.label} className="flex items-start gap-2 rounded-md border border-slate-200 px-3 py-2 text-xs">
                      <span className={`mt-0.5 shrink-0 rounded px-1.5 py-0.5 font-semibold ${BADGE[c.status]}`}>{BADGE_TEXT[c.status]}</span>
                      <span>
                        <span className="font-medium text-slate-800">{c.label}.</span> <span className="text-slate-600">{c.detail}</span>
                      </span>
                    </li>
                  ))}
                </ul>
                {r.state_warnings.map((w) => (
                  <p key={w} className="mt-3 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">{w}</p>
                ))}
              </>
            )}
          </section>

          <section className="rounded-lg border border-slate-200 bg-white p-5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Growth of $10,000 and drawdown</p>
              <div className="flex rounded-md border border-slate-200 p-0.5 text-xs">
                {(["basket", "spy", "cash"] as const).map((b) => (
                  <button
                    key={b}
                    type="button"
                    onClick={() => setChartBenchmark(b)}
                    className={`rounded px-2.5 py-1 font-medium ${chartBenchmark === b ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-100"}`}
                  >
                    {b === "basket" ? "Same stocks" : b === "spy" ? "SPY" : "Cash"}
                  </button>
                ))}
                <button type="button" disabled className="rounded px-2.5 py-1 text-slate-300" title="QQQ: coming next">QQQ</button>
              </div>
            </div>
            {chart ? (
              <PlotlyChart
                data={[
                  { x: chart.dates, y: chart.strategy, type: "scatter", mode: "lines", name: "Strategy", line: { color: "#1F4FD1", width: 2 } },
                  ...(chartBenchmark === "spy"
                    ? []
                    : [{ x: chart.dates, y: chart.bench, type: "scatter" as const, mode: "lines" as const, name: chartBenchmark === "cash" ? "Cash" : "Same stocks, held", line: { color: "#64748B", width: 1.5, dash: "dash" as const } }]),
                  { x: chart.dates, y: chart.drawdown, type: "scatter", mode: "lines", name: "Strategy drawdown (%)", yaxis: "y2", fill: "tozeroy", fillcolor: "rgba(220,38,38,0.12)", line: { color: "rgba(220,38,38,0.4)", width: 1 } },
                ]}
                layout={{
                  height: 320,
                  margin: { t: 10, r: 50, b: 30, l: 60 },
                  paper_bgcolor: "#ffffff",
                  plot_bgcolor: "#ffffff",
                  legend: { orientation: "h", y: -0.18 },
                  yaxis: { title: { text: "$" } },
                  yaxis2: { overlaying: "y", side: "right", title: { text: "drawdown %" }, range: [-60, 0], showgrid: false },
                  shapes: chart.split ? [{ type: "line", x0: chart.split, x1: chart.split, y0: 0, y1: 1, xref: "x", yref: "paper", line: { color: "#94A3B8", dash: "dot" } }] : [],
                  annotations: chart.split ? [{ x: chart.split, y: 1, yref: "paper", text: "Out-of-sample →", showarrow: false, xanchor: "left", font: { size: 10, color: "#64748B" } }] : [],
                }}
                style={{ width: "100%" }}
                useResizeHandler
                config={{ displayModeBar: false }}
              />
            ) : (
              <p className="mt-6 rounded-md border border-dashed border-slate-300 p-10 text-center text-sm text-slate-500">The chart appears after a run.</p>
            )}
          </section>

          {r && (
            <section className="rounded-lg border border-slate-200 bg-white p-5">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Full period, after costs</p>
              <div className="mt-3 overflow-x-auto">
                <table className="w-full min-w-[40rem] text-right text-sm">
                  <thead className="text-xs text-slate-400">
                    <tr>
                      <th className="py-1 text-left font-medium"></th>
                      <th className="font-medium">Total</th>
                      <th className="font-medium">CAGR</th>
                      <th className="font-medium">Volatility</th>
                      <th className="font-medium">Max DD</th>
                      <th className="font-medium">Sharpe</th>
                      <th className="font-medium">Worst month</th>
                      <th className="font-medium">Cost drag</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-mono text-xs">
                    <tr>
                      <td className="py-2 text-left font-sans text-sm font-medium text-slate-800">Strategy</td>
                      <td>{pct(r.strategy.total_return_pct)}</td>
                      <td>{pct(r.strategy.cagr_pct)}</td>
                      <td>{plain(r.strategy.volatility_pct, 1)}%</td>
                      <td>{pct(r.strategy.max_drawdown_pct)}</td>
                      <td>{plain(r.strategy.sharpe)}</td>
                      <td>{pct(r.strategy.worst_month_pct)}</td>
                      <td>{r.cost_drag.cagr_points != null ? `${plain(r.cost_drag.cagr_points)} pts/yr` : "–"}</td>
                    </tr>
                    <tr>
                      <td className="py-2 text-left font-sans text-sm text-slate-700">Same stocks, held</td>
                      <td>{pct(r.basket.total_return_pct)}</td>
                      <td>{pct(r.basket.cagr_pct)}</td>
                      <td>{plain(r.basket.volatility_pct, 1)}%</td>
                      <td>{pct(r.basket.max_drawdown_pct)}</td>
                      <td>{plain(r.basket.sharpe)}</td>
                      <td>{pct(r.basket.worst_month_pct)}</td>
                      <td>–</td>
                    </tr>
                    <tr>
                      <td className="py-2 text-left font-sans text-sm text-slate-700">SPY</td>
                      <td>{pct(r.benchmark_spy.total_return_pct)}</td>
                      <td>{pct(r.benchmark_spy.cagr_pct)}</td>
                      <td>{plain(r.benchmark_spy.volatility_pct, 1)}%</td>
                      <td>{pct(r.benchmark_spy.max_drawdown_pct)}</td>
                      <td>{plain(r.benchmark_spy.sharpe)}</td>
                      <td>{pct(r.benchmark_spy.worst_month_pct)}</td>
                      <td>–</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {r && (
            <section className="rounded-lg border border-slate-200 bg-white p-5">
              <div className="flex gap-1 border-b border-slate-200 text-sm">
                {(
                  [
                    ["robustness", "Robustness"],
                    ["trades", "Trades"],
                    ["tickers", "Per ticker"],
                  ] as const
                ).map(([key, label]) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => setTab(key)}
                    className={`-mb-px border-b-2 px-3 py-2 font-medium ${tab === key ? "border-slate-900 text-slate-900" : "border-transparent text-slate-500 hover:text-slate-800"}`}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {tab === "robustness" && (
                <div className="mt-4 flex flex-col gap-4">
                  <div className="flex items-baseline justify-between">
                    <p className="text-sm font-medium text-slate-900">In-sample vs out-of-sample</p>
                    <p className="text-xs text-slate-400">Split at 70%</p>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full min-w-[32rem] text-right text-sm">
                      <thead className="text-xs text-slate-400">
                        <tr>
                          <th className="py-1 text-left font-medium">Period</th>
                          <th className="font-medium">CAGR</th>
                          <th className="font-medium">Volatility</th>
                          <th className="font-medium">Max DD</th>
                          <th className="font-medium">Sharpe</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 font-mono text-xs">
                        <tr>
                          <td className="py-2 text-left font-sans text-sm text-slate-700">In-sample (first 70%)</td>
                          <td>{pct(r.in_sample.cagr_pct)}</td>
                          <td>{plain(r.in_sample.volatility_pct, 1)}%</td>
                          <td>{pct(r.in_sample.max_drawdown_pct)}</td>
                          <td>{plain(r.in_sample.sharpe)}</td>
                        </tr>
                        <tr>
                          <td className="py-2 text-left font-sans text-sm text-slate-700">Out-of-sample (last 30%)</td>
                          <td>{pct(r.out_of_sample.cagr_pct)}</td>
                          <td>{plain(r.out_of_sample.volatility_pct, 1)}%</td>
                          <td>{pct(r.out_of_sample.max_drawdown_pct)}</td>
                          <td>{plain(r.out_of_sample.sharpe)}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                  <p className="text-xs text-slate-500">
                    A rule that works usually looks better in-sample than out. A large gap in the other direction points to the recent market rather than the rules.
                  </p>
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <Stat
                      label="Walk-forward windows beating the basket"
                      value={r.walk_forward ? `${r.walk_forward.beat_basket_windows} of ${r.walk_forward.test_windows}` : "–"}
                      sub="Six-month test windows, rules not refitted"
                    />
                    <Stat
                      label={`Probability Sharpe is real, after ${r.deflated_sharpe?.variants ?? 1} variant(s)`}
                      value={r.deflated_sharpe?.probability != null ? `${Math.round(r.deflated_sharpe.probability * 100)}%` : "–"}
                      sub="Deflated Sharpe, from your own runs in the last 90 days"
                    />
                  </div>
                  {r.sensitivity && (
                    <div>
                      <p className="text-sm font-medium text-slate-900">Sensitivity: each threshold moved ±{r.sensitivity.step_pct}%</p>
                      <div className="mt-2 overflow-x-auto">
                        <table className="w-full min-w-[32rem] text-right text-xs">
                          <thead className="text-slate-400">
                            <tr>
                              <th className="py-1 text-left font-medium">Threshold</th>
                              <th className="font-medium">−{r.sensitivity.step_pct}%</th>
                              <th className="font-medium">Base</th>
                              <th className="font-medium">+{r.sensitivity.step_pct}%</th>
                              <th className="font-medium">Sharpe swing</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-100 font-mono">
                            {r.sensitivity.rows.map((row) => (
                              <tr key={row.parameter}>
                                <td className="py-1.5 text-left font-sans text-slate-700">{row.parameter}</td>
                                {row.cells.map((c) => (
                                  <td key={c.factor}>{plain(c.sharpe)}</td>
                                ))}
                                <td>{plain(row.swing)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {tab === "trades" && (
                <div className="mt-4 overflow-x-auto">
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
                      {r.trade_log.slice(0, 100).map((t, i) => (
                        <tr key={i}>
                          <td className="py-1">{t.ticker}</td>
                          <td>{t.entry_date} @ {t.entry_price}</td>
                          <td>{t.exit_date} @ {t.exit_price}</td>
                          <td>{t.exit_reason.replace(/_/g, " ")}</td>
                          <td className="text-right">{t.holding_days}</td>
                          <td className="text-right font-mono">{pct(t.return_pct)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {r.trade_log.length > 100 && <p className="mt-2 text-xs text-slate-400">Showing the first 100 of {r.trade_log.length} trades.</p>}
                </div>
              )}

              {tab === "tickers" && (
                <ul className="mt-4 flex flex-col gap-2 text-sm">
                  {Object.entries(r.per_ticker_contribution).map(([t, c]) => (
                    <li key={t} className="flex justify-between">
                      <span className="text-slate-700">{t}</span>
                      <span className="font-mono text-slate-600">{pct(c)}</span>
                    </li>
                  ))}
                  {r.top_ticker?.flag && (
                    <li className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
                      {r.top_ticker.ticker} supplies {plain(r.top_ticker.share_pct, 0)}% of the total gain, so the result depends heavily on one stock.
                    </li>
                  )}
                </ul>
              )}
            </section>
          )}

          {r && (
            <details className="rounded-lg border border-slate-200 bg-white p-5">
              <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wide text-slate-500">Data notes ({r.caveats.length})</summary>
              <ul className="mt-3 flex flex-col gap-1 text-xs text-slate-500">
                {r.caveats.map((c) => (
                  <li key={c}>• {c}</li>
                ))}
              </ul>
              <p className="mt-3 text-xs text-slate-400">
                {r.variants_tried} rule variant{r.variants_tried === 1 ? "" : "s"} tried in the last 90 days · cooldown {r.cooldown_sessions} sessions
              </p>
            </details>
          )}

          <p className="text-center text-xs text-slate-400">Backtest of past prices with the rules and costs shown. Not a forecast, not a recommendation, and not an order: nothing is placed.</p>
        </div>
      </div>
    </div>
  );
}
