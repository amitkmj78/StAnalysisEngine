"use client";

import { useEffect, useState } from "react";

import CurrentPriceBadge from "@/components/CurrentPriceBadge";
import InfoModal, { type ColumnInfo } from "@/components/InfoModal";
import BacktestChart from "@/components/prediction/BacktestChart";
import ForecastChart from "@/components/prediction/ForecastChart";
import {
  ApiError,
  getChatProviders,
  getPredictionHistory,
  getPredictionNarrative,
  getPredictionSummary,
  savePrediction,
} from "@/lib/api";
import type { PredictionNarrative, PredictionSummary, SavedPrediction } from "@/lib/types";

const METRIC_INFO: Record<string, ColumnInfo> = {
  signal: {
    title: "10-Day Signal — what it means",
    body: [
      "BUY, HOLD, or SELL, derived directly from the model's own 10-day forecast versus today's close — nothing else feeds into it.",
      "BUY: the forecast implies at least +5% expected return over 10 days.",
      "SELL: the forecast implies -5% or worse.",
      "HOLD: the forecast falls between -5% and +5% — not enough expected movement either way to call it.",
      "It's a simple threshold read on the model's own point forecast, not a separate signal-generation model — so it's only as reliable as the forecast itself (see the backtest accuracy below).",
    ],
  },
  rmse: {
    title: "RMSE — Root Mean Squared Error",
    body: [
      "The typical size of the model's miss, in dollars, across the walk-forward backtest window.",
      "Squares each day's error before averaging, then square-roots the result — which weights big misses more heavily than small ones. A model that's usually close but occasionally way off will show a higher RMSE than its MAE.",
      "Shown next to the naive baseline's own RMSE below, so you can see whether the model's typical miss is bigger or smaller than just assuming no price change.",
    ],
  },
  mae: {
    title: "MAE — Mean Absolute Error",
    body: [
      "The average size of the model's miss, in dollars, treating every day's error equally regardless of whether it was a small miss or a large one (unlike RMSE, which penalizes large misses more).",
      "If RMSE is noticeably higher than MAE, that's a sign the model has a few bad days that are much worse than its typical miss, rather than being uniformly a little off.",
    ],
  },
  mape: {
    title: "MAPE — Mean Absolute Percentage Error",
    body: [
      "The average miss expressed as a percentage of the actual price, rather than in dollars — this is what makes it comparable across tickers at very different price levels (a $5 miss means very different things for a $20 stock vs. a $500 stock).",
    ],
  },
};

const PERIODS = [
  { label: "1 Week", value: "5d" },
  { label: "30 Days", value: "1mo" },
  { label: "6 Months", value: "6mo" },
  { label: "1 Year", value: "1y" },
  { label: "5 Years", value: "5y" },
];

export default function PredictPage() {
  const [ticker, setTicker] = useState("AAPL");
  const [period, setPeriod] = useState("1y");
  const [data, setData] = useState<PredictionSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeInfo, setActiveInfo] = useState<string | null>(null);

  const [providers, setProviders] = useState<string[]>([]);
  const [provider, setProvider] = useState("");
  const [narrative, setNarrative] = useState<PredictionNarrative | null>(null);
  const [narrativeLoading, setNarrativeLoading] = useState(false);
  const [narrativeError, setNarrativeError] = useState<string | null>(null);

  const [history, setHistory] = useState<SavedPrediction[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);

  useEffect(() => {
    getChatProviders()
      .then((res) => {
        setProviders(res.providers);
        setProvider(res.providers[0] ?? "");
      })
      .catch(() => {});
  }, []);

  async function runPrediction(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setNarrative(null);
    setNarrativeError(null);
    setSaveMessage(null);
    try {
      const summary = await getPredictionSummary(ticker.trim().toUpperCase(), period);
      setData(summary);
      loadHistory(summary.ticker);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong.");
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  async function loadHistory(forTicker: string) {
    setHistoryLoading(true);
    try {
      const res = await getPredictionHistory(forTicker);
      setHistory(res.predictions);
    } catch {
      // Non-fatal — history is supplementary, don't block the main page on it.
    } finally {
      setHistoryLoading(false);
    }
  }

  async function handleSave() {
    if (!data) return;
    setSaving(true);
    setSaveMessage(null);
    try {
      await savePrediction(data.ticker, data.period);
      setSaveMessage("Saved — check back after the forecast date to see how it did.");
      await loadHistory(data.ticker);
    } catch (err) {
      setSaveMessage(err instanceof ApiError ? err.message : "Could not save this prediction.");
    } finally {
      setSaving(false);
    }
  }

  async function runNarrative() {
    if (!data) return;
    setNarrativeLoading(true);
    setNarrativeError(null);
    try {
      const res = await getPredictionNarrative(data.ticker, data.period, provider || undefined);
      setNarrative(res);
    } catch (err) {
      setNarrativeError(err instanceof ApiError ? err.message : "Something went wrong.");
    } finally {
      setNarrativeLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">AI Price Prediction</h1>
      <p className="mt-1 text-sm text-slate-500">
        A backtested quant forecast for one ticker, shown next to how often it has actually beaten doing
        nothing.
      </p>

      <form onSubmit={runPrediction} className="mt-6 flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1">
          <label htmlFor="ticker" className="text-xs font-medium text-slate-500">
            Ticker
          </label>
          <input
            id="ticker"
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            className="w-32 rounded-md border border-slate-300 px-3 py-2 text-sm"
          />
        </div>
        <CurrentPriceBadge ticker={ticker} />
        <div className="flex flex-col gap-1">
          <label htmlFor="period" className="text-xs font-medium text-slate-500">
            Historical window
          </label>
          <select
            id="period"
            value={period}
            onChange={(e) => setPeriod(e.target.value)}
            className="rounded-md border border-slate-300 px-3 py-2 text-sm"
          >
            {PERIODS.map((p) => (
              <option key={p.value} value={p.value}>
                {p.label}
              </option>
            ))}
          </select>
        </div>
        <button
          type="submit"
          disabled={loading || !ticker.trim()}
          className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
        >
          {loading ? "Analyzing…" : "Analyze"}
        </button>
      </form>

      {loading && (
        <p className="mt-4 text-sm text-slate-500">
          Training the model and running a walk-forward backtest — this can take up to 20 seconds.
        </p>
      )}

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {data && !loading && (
        <div className="mt-6 flex flex-col gap-6">
          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <h2 className="text-lg font-semibold text-slate-900">{data.ticker} — Next-Day Forecast</h2>
            {data.warnings.includes("no_price_data") ? (
              <p className="mt-2 text-sm text-slate-500">No price data was available for that ticker.</p>
            ) : (
              <p className="mt-2 text-sm text-slate-600">
                Last close: <strong>${data.last_close?.toFixed(2)}</strong>
                {data.next_price !== null && (
                  <>
                    {" "}
                    &middot; Predicted next close: <strong>${data.next_price.toFixed(2)}</strong>
                  </>
                )}
              </p>
            )}
          </div>

          {data.signal && (
            <div>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                <MetricTile label="10-Day Signal" value={data.signal.signal} onInfoClick={() => setActiveInfo("signal")} />
                <MetricTile label="Expected Return (10d)" value={`${data.signal.expected_return_pct.toFixed(2)}%`} />
                <MetricTile label="Target Price" value={`$${data.signal.target_price.toFixed(2)}`} />
              </div>
              <p className="mt-2 text-xs text-slate-500">
                This is one data-driven signal, not a guarantee — see backtest accuracy below for how it has
                historically performed against a simple no-change baseline.
              </p>
              <div className="mt-3 flex items-center gap-3">
                <button
                  onClick={handleSave}
                  disabled={saving}
                  className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                >
                  {saving ? "Saving…" : "Save this prediction"}
                </button>
                {saveMessage && <span className="text-xs text-slate-500">{saveMessage}</span>}
              </div>
            </div>
          )}

          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <h3 className="font-semibold text-slate-900">Prediction History for {data.ticker}</h3>
            <p className="mt-1 text-sm text-slate-600">
              Real, forward-looking predictions you&apos;ve saved — checked automatically against what
              actually happened once each forecast date arrives. Different from the backtest below, which is
              a historical simulation, not a live record.
            </p>
            {historyLoading ? (
              <p className="mt-3 text-sm text-slate-500">Loading history…</p>
            ) : history.length === 0 ? (
              <p className="mt-3 text-sm text-slate-500">No saved predictions for {data.ticker} yet.</p>
            ) : (
              <div className="mt-3 overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                      <th className="px-2 py-2">Saved</th>
                      <th className="px-2 py-2">Last Close</th>
                      <th className="px-2 py-2">Predicted Next Close</th>
                      <th className="px-2 py-2">Actual Next Close</th>
                      <th className="px-2 py-2">Next-Day Error %</th>
                      <th className="px-2 py-2">Target Date</th>
                      <th className="px-2 py-2">Predicted Target</th>
                      <th className="px-2 py-2">Open Price</th>
                      <th className="px-2 py-2">Actual</th>
                      <th className="px-2 py-2">Error %</th>
                      <th className="px-2 py-2">Signal</th>
                      <th className="px-2 py-2">Correct?</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((p) => (
                      <tr key={p.id} className="border-b border-slate-100 last:border-0">
                        <td className="px-2 py-2 text-slate-700">{new Date(p.predicted_at).toLocaleDateString()}</td>
                        <td className="px-2 py-2 text-slate-700">
                          {p.last_close !== null ? `$${p.last_close.toFixed(2)}` : "—"}
                        </td>
                        <td className="px-2 py-2 text-slate-700">
                          {p.next_price !== null ? `$${p.next_price.toFixed(2)}` : "—"}
                        </td>
                        <td className="px-2 py-2 text-slate-700">
                          {p.actual_next_price !== null ? `$${p.actual_next_price.toFixed(2)}` : "pending"}
                        </td>
                        <td className="px-2 py-2 text-slate-700">
                          {p.next_price_error_pct !== null ? `${p.next_price_error_pct.toFixed(2)}%` : "—"}
                        </td>
                        <td className="px-2 py-2 text-slate-700">
                          {p.target_date ? new Date(p.target_date).toLocaleDateString() : "—"}
                        </td>
                        <td className="px-2 py-2 text-slate-700">
                          {p.target_price !== null ? `$${p.target_price.toFixed(2)}` : "—"}
                        </td>
                        <td className="px-2 py-2 text-slate-700">
                          {p.actual_target_open !== null ? `$${p.actual_target_open.toFixed(2)}` : "pending"}
                        </td>
                        <td className="px-2 py-2 text-slate-700">
                          {p.actual_target_price !== null ? `$${p.actual_target_price.toFixed(2)}` : "pending"}
                        </td>
                        <td className="px-2 py-2 text-slate-700">
                          {p.target_price_error_pct !== null ? `${p.target_price_error_pct.toFixed(2)}%` : "—"}
                        </td>
                        <td className="px-2 py-2 text-slate-700">{p.signal ?? "—"}</td>
                        <td className="px-2 py-2">
                          {p.signal_correct === null ? (
                            <span className="text-slate-400">pending</span>
                          ) : p.signal_correct ? (
                            <span className="text-emerald-600">✓ correct</span>
                          ) : (
                            <span className="text-red-600">✗ wrong</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div className="rounded-lg border border-indigo-200 bg-indigo-50/40 p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h3 className="font-semibold text-slate-900">AI Context (optional)</h3>
              {providers.length > 0 && (
                <select
                  value={provider}
                  onChange={(e) => setProvider(e.target.value)}
                  className="rounded-md border border-slate-300 bg-white px-2 py-1 text-xs"
                >
                  {providers.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              )}
            </div>
            <p className="mt-1 text-sm text-slate-600">
              Ask an LLM to summarize this forecast in plain English alongside recent news and earnings context —
              including calling out what might explain a big pre/post-market move — and flag whether they agree.
              This does not change the numbers above or below — it&apos;s a separate, on-demand read generated
              fresh each time you ask.
            </p>
            <button
              onClick={runNarrative}
              disabled={narrativeLoading}
              className="mt-3 rounded-md border border-indigo-300 bg-white px-3 py-1.5 text-sm font-medium text-indigo-700 hover:bg-indigo-50 disabled:opacity-50"
            >
              {narrativeLoading ? "Thinking…" : narrative ? "Regenerate" : "Get AI Context"}
            </button>

            {narrativeError && (
              <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{narrativeError}</p>
            )}

            {narrative && !narrativeLoading && (
              <div className="mt-4 flex flex-col gap-3">
                <p className="text-sm leading-relaxed whitespace-pre-wrap text-slate-800">{narrative.narrative}</p>
                <details className="text-xs text-slate-600" open>
                  <summary className="cursor-pointer font-medium text-slate-700">
                    Recent News &amp; Earnings Context ({narrative.provider})
                  </summary>
                  <p className="mt-2 max-h-64 overflow-y-auto whitespace-pre-wrap rounded-md bg-white p-3">
                    {narrative.sentiment_context}
                  </p>
                </details>
              </div>
            )}
          </div>

          {data.forecast ? (
            <div className="rounded-lg border border-slate-200 bg-white p-4">
              <ForecastChart ticker={data.ticker} forecast={data.forecast} />
            </div>
          ) : (
            <p className="text-sm text-slate-500">Not enough price history for a 10-day forecast.</p>
          )}

          <hr className="border-slate-200" />
          <h2 className="text-lg font-semibold text-slate-900">🎯 Backtest Accuracy</h2>

          {data.backtest && data.metrics ? (
            <>
              <div className="rounded-lg border border-slate-200 bg-white p-4">
                <BacktestChart ticker={data.ticker} backtest={data.backtest} />
              </div>

              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                <MetricTile label="RMSE" value={data.metrics.rmse.toFixed(2)} onInfoClick={() => setActiveInfo("rmse")} />
                <MetricTile label="MAE" value={data.metrics.mae.toFixed(2)} onInfoClick={() => setActiveInfo("mae")} />
                <MetricTile label="MAPE" value={`${data.metrics.mape.toFixed(2)}%`} onInfoClick={() => setActiveInfo("mape")} />
              </div>

              {data.metrics.naive_rmse !== null && data.metrics.naive_rmse !== undefined && (
                <div>
                  <p className="mb-2 text-xs text-slate-500">
                    Naive baseline = predicting no price change (tomorrow&apos;s price = today&apos;s close).
                    Delta shown as model minus naive, so negative is better.
                  </p>
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
                    <DeltaTile
                      label="RMSE vs Naive"
                      value={data.metrics.naive_rmse.toFixed(2)}
                      delta={data.metrics.rmse - data.metrics.naive_rmse}
                    />
                    <DeltaTile
                      label="MAE vs Naive"
                      value={data.metrics.naive_mae!.toFixed(2)}
                      delta={data.metrics.mae - data.metrics.naive_mae!}
                    />
                    <DeltaTile
                      label="MAPE vs Naive"
                      value={`${data.metrics.naive_mape!.toFixed(2)}%`}
                      delta={data.metrics.mape - data.metrics.naive_mape!}
                    />
                    <div
                      className={`flex items-center justify-center rounded-lg border p-3 text-sm font-medium ${
                        data.metrics.beats_naive
                          ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                          : "border-red-200 bg-red-50 text-red-700"
                      }`}
                    >
                      {data.metrics.beats_naive ? "Beats naive baseline" : "Does not beat naive"}
                    </div>
                  </div>
                </div>
              )}
            </>
          ) : (
            <p className="text-sm text-slate-500">Not enough price history for a walk-forward backtest.</p>
          )}

          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <h3 className="font-semibold text-slate-900">How To Read This Page</h3>
            <p className="mt-2 text-sm text-slate-600">
              The forecast comes from a gradient-boosted model trained on this ticker&apos;s own recent RSI,
              MACD, Bollinger position, and lagged returns — not on news, filings, or sentiment.
            </p>
            <p className="mt-2 text-sm text-slate-600">
              The naive baseline is simply &quot;assume tomorrow&apos;s price equals today&apos;s close.&quot;
              Backtesting across several tickers showed the model&apos;s raw predicted move is noisy enough
              that it loses to that baseline most of the time — so the forecast shown here is deliberately
              dampened toward &quot;no change&quot; before display, which measurably improves accuracy and
              cuts down on overconfident BUY/SELL calls. The metrics above are the actual, current test for
              this ticker, not a marketing claim.
            </p>
            <p className="mt-2 text-sm text-slate-600">
              The optional &quot;AI Context&quot; section above is separate from all of this: it reads today&apos;s
              news/sentiment and restates the forecast in plain English, but it does not feed back into the
              model, the forecast, or the backtest numbers on this page.
            </p>
          </div>
        </div>
      )}

      {activeInfo && METRIC_INFO[activeInfo] && (
        <InfoModal info={METRIC_INFO[activeInfo]} onClose={() => setActiveInfo(null)} />
      )}
    </div>
  );
}

function MetricTile({ label, value, onInfoClick }: { label: string; value: string; onInfoClick?: () => void }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <p className="flex items-center gap-1 text-xs text-slate-500">
        {label}
        {onInfoClick && (
          <button
            type="button"
            onClick={onInfoClick}
            title={`What is ${label}?`}
            className="flex h-3.5 w-3.5 items-center justify-center rounded-full border border-slate-300 text-[9px] font-normal text-slate-400 hover:border-slate-500 hover:text-slate-700"
          >
            i
          </button>
        )}
      </p>
      <p className="mt-1 text-xl font-semibold text-slate-900">{value}</p>
    </div>
  );
}

function DeltaTile({ label, value, delta }: { label: string; value: string; delta: number }) {
  const worse = delta > 0; // model minus naive; positive means model's error is larger
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="mt-1 text-xl font-semibold text-slate-900">{value}</p>
      <p className={`mt-0.5 text-xs font-medium ${worse ? "text-red-600" : "text-emerald-600"}`}>
        {worse ? "↑" : "↓"} {Math.abs(delta).toFixed(2)}
      </p>
    </div>
  );
}
