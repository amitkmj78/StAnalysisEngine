"use client";

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";

import CurrentPriceBadge from "@/components/CurrentPriceBadge";
import InfoModal, { type ColumnInfo } from "@/components/InfoModal";
import SafeBaselineBand from "@/components/SafeBaselineBand";
import TickerSearchInput from "@/components/TickerSearchInput";
import BacktestChart from "@/components/prediction/BacktestChart";
import ForecastChart from "@/components/prediction/ForecastChart";
import {
  ApiError,
  deleteNarrative,
  deletePrediction,
  getBaselineBand,
  getChatProviders,
  getNarrativeHistory,
  getPredictionActivity,
  getPredictionHistory,
  getPredictionNarrative,
  getPredictionSummary,
  saveNarrative,
  savePrediction,
} from "@/lib/api";
import type {
  BaselineBand,
  PredictionActivity,
  PredictionNarrative,
  PredictionSummary,
  SavedNarrative,
  SavedPrediction,
} from "@/lib/types";

function getMetricInfo(daysAhead: number): Record<string, ColumnInfo> {
  return {
  signal: {
    title: `${daysAhead}-Day Signal — what it means`,
    body: [
      `BUY, HOLD, or SELL, derived directly from the model's own ${daysAhead}-day forecast versus today's close — nothing else feeds into it.`,
      `BUY: the forecast implies at least +5% expected return over ${daysAhead} days.`,
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
}

const PERIODS = [
  { label: "1 Week", value: "5d" },
  { label: "30 Days", value: "1mo" },
  { label: "6 Months", value: "6mo" },
  { label: "1 Year", value: "1y" },
  { label: "5 Years", value: "5y" },
];

const FORECAST_HORIZONS = [5, 10, 20, 30, 60];

export default function PredictPage() {
  const searchParams = useSearchParams();
  const [ticker, setTicker] = useState(searchParams.get("ticker")?.trim().toUpperCase() || "AAPL");
  const [period, setPeriod] = useState("1y");
  const [daysAhead, setDaysAhead] = useState(10);
  const [shownDaysAhead, setShownDaysAhead] = useState(10);
  const [data, setData] = useState<PredictionSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeInfo, setActiveInfo] = useState<string | null>(null);

  const [providers, setProviders] = useState<string[]>([]);
  const [provider, setProvider] = useState("");
  const [narrative, setNarrative] = useState<PredictionNarrative | null>(null);
  const [narrativeLoading, setNarrativeLoading] = useState(false);
  const [narrativeError, setNarrativeError] = useState<string | null>(null);

  const [narrativeHistory, setNarrativeHistory] = useState<SavedNarrative[]>([]);
  const [narrativeHistoryLoading, setNarrativeHistoryLoading] = useState(false);
  const [savingNarrative, setSavingNarrative] = useState(false);
  const [saveNarrativeMessage, setSaveNarrativeMessage] = useState<string | null>(null);
  const [deletingNarrativeId, setDeletingNarrativeId] = useState<number | null>(null);
  const [compareNarrativeId, setCompareNarrativeId] = useState<number | null>(null);

  const [activity, setActivity] = useState<PredictionActivity | null>(null);
  const [activityLoading, setActivityLoading] = useState(false);
  const [activityError, setActivityError] = useState<string | null>(null);

  const [history, setHistory] = useState<SavedPrediction[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [priceRefreshKey, setPriceRefreshKey] = useState(0);

  const [compareInput, setCompareInput] = useState("");
  const [compareTickers, setCompareTickers] = useState<string[]>([]);
  const [compareData, setCompareData] = useState<
    Record<string, { summary: PredictionSummary | null; band: BaselineBand | null; loading: boolean; error: string | null }>
  >({});
  const [primaryBand, setPrimaryBand] = useState<BaselineBand | null>(null);

  useEffect(() => {
    getChatProviders()
      .then((res) => {
        setProviders(res.providers);
        setProvider(res.providers[0] ?? "");
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    const urlTicker = searchParams.get("ticker")?.trim().toUpperCase();
    if (urlTicker) {
      runAnalysis(urlTicker, period, daysAhead);
    }
    // Only meant to fire once, from the initial URL — not on every period/daysAhead change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function runAnalysis(forTicker: string, forPeriod: string, forDaysAhead: number) {
    setLoading(true);
    setError(null);
    setNarrative(null);
    setNarrativeError(null);
    setNarrativeHistory([]);
    setCompareNarrativeId(null);
    setSaveNarrativeMessage(null);
    setActivity(null);
    setActivityError(null);
    setSaveMessage(null);
    setCompareTickers([]);
    setCompareData({});
    setCompareInput("");
    setPrimaryBand(null);
    try {
      const summary = await getPredictionSummary(forTicker.trim().toUpperCase(), forPeriod, forDaysAhead);
      setData(summary);
      setShownDaysAhead(forDaysAhead);
      setPriceRefreshKey((k) => k + 1);
      // Auto-save every viewed forecast (not just ones a user remembers
      // to click Save on) so the accuracy leaderboard reflects real,
      // unbiased usage rather than a self-selected subset — the backend
      // dedupes to one row per ticker/period/day, so repeat views of
      // the same ticker today don't flood the table. Awaited (but its
      // own errors swallowed) so loadHistory below sees the fresh row.
      await autoSavePrediction(summary.ticker, forPeriod, forDaysAhead);
      loadHistory(summary.ticker);
      loadNarrativeHistory(summary.ticker);
      getBaselineBand(summary.ticker, { horizon: 30, confidence: 0.9 })
        .then(setPrimaryBand)
        .catch(() => setPrimaryBand(null));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong.");
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  function runPrediction(e: React.FormEvent) {
    e.preventDefault();
    runAnalysis(ticker, period, daysAhead);
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

  async function handleDeletePrediction(id: number) {
    setDeletingId(id);
    try {
      await deletePrediction(id);
      setHistory((prev) => prev.filter((p) => p.id !== id));
    } catch {
      // Non-fatal — leave the row in place if delete failed.
    } finally {
      setDeletingId(null);
    }
  }

  async function autoSavePrediction(forTicker: string, forPeriod: string, forDaysAhead: number) {
    try {
      const res = await savePrediction(forTicker, forPeriod, forDaysAhead);
      setSaveMessage(
        res.already_saved_today
          ? "Already tracking today's forecast for this ticker — checking back after the target date."
          : "Auto-saved for accuracy tracking — check back after the forecast date to see how it did."
      );
    } catch {
      // Silent — this runs on every page view, not a user-initiated
      // action, so a failure here shouldn't interrupt viewing the
      // forecast. The manual Save button below still works as a retry.
    }
  }

  async function handleSave() {
    if (!data) return;
    setSaving(true);
    setSaveMessage(null);
    try {
      const res = await savePrediction(data.ticker, data.period, shownDaysAhead);
      setSaveMessage(
        res.already_saved_today
          ? "Already tracking today's forecast for this ticker."
          : "Saved — check back after the forecast date to see how it did."
      );
      await loadHistory(data.ticker);
    } catch (err) {
      setSaveMessage(err instanceof ApiError ? err.message : "Could not save this prediction.");
    } finally {
      setSaving(false);
    }
  }

  async function runActivity() {
    if (!data) return;
    setActivityLoading(true);
    setActivityError(null);
    try {
      const res = await getPredictionActivity(data.ticker);
      setActivity(res);
    } catch (err) {
      setActivityError(err instanceof ApiError ? err.message : "Something went wrong.");
    } finally {
      setActivityLoading(false);
    }
  }

  async function runNarrative() {
    if (!data) return;
    setNarrativeLoading(true);
    setNarrativeError(null);
    try {
      const res = await getPredictionNarrative(data.ticker, data.period, provider || undefined, shownDaysAhead);
      setNarrative(res);
    } catch (err) {
      setNarrativeError(err instanceof ApiError ? err.message : "Something went wrong.");
    } finally {
      setNarrativeLoading(false);
    }
  }

  async function loadNarrativeHistory(forTicker: string) {
    setNarrativeHistoryLoading(true);
    try {
      const res = await getNarrativeHistory(forTicker);
      setNarrativeHistory(res.narratives);
    } catch {
      // Non-fatal — history is supplementary, don't block the main page on it.
    } finally {
      setNarrativeHistoryLoading(false);
    }
  }

  async function handleSaveNarrative() {
    if (!data || !narrative) return;
    setSavingNarrative(true);
    setSaveNarrativeMessage(null);
    try {
      await saveNarrative({
        ticker: narrative.ticker,
        provider: narrative.provider,
        period: data.period,
        days_ahead: shownDaysAhead,
        narrative: narrative.narrative,
        sentiment_context: narrative.sentiment_context,
      });
      setSaveNarrativeMessage("Saved — generate a new context later to compare it against this one.");
      await loadNarrativeHistory(data.ticker);
    } catch (err) {
      setSaveNarrativeMessage(err instanceof ApiError ? err.message : "Could not save this context.");
    } finally {
      setSavingNarrative(false);
    }
  }

  async function handleDeleteNarrative(id: number) {
    setDeletingNarrativeId(id);
    try {
      await deleteNarrative(id);
      setNarrativeHistory((prev) => prev.filter((n) => n.id !== id));
      if (compareNarrativeId === id) setCompareNarrativeId(null);
    } catch {
      // Non-fatal — leave the row in place if delete failed.
    } finally {
      setDeletingNarrativeId(null);
    }
  }

  async function fetchCompareTicker(t: string) {
    setCompareData((prev) => ({ ...prev, [t]: { summary: null, band: null, loading: true, error: null } }));
    const [summaryResult, bandResult] = await Promise.allSettled([
      getPredictionSummary(t, period, daysAhead),
      getBaselineBand(t, { horizon: 30, confidence: 0.9 }),
    ]);
    setCompareData((prev) => ({
      ...prev,
      [t]: {
        summary: summaryResult.status === "fulfilled" ? summaryResult.value : null,
        band: bandResult.status === "fulfilled" ? bandResult.value : null,
        loading: false,
        error:
          summaryResult.status === "rejected"
            ? summaryResult.reason instanceof ApiError
              ? summaryResult.reason.message
              : "Could not load this ticker."
            : null,
      },
    }));
  }

  function addCompareTicker(e: React.FormEvent) {
    e.preventDefault();
    const t = compareInput.trim().toUpperCase();
    if (!t || !data || t === data.ticker || compareTickers.includes(t)) {
      setCompareInput("");
      return;
    }
    setCompareTickers((prev) => [...prev, t]);
    setCompareInput("");
    fetchCompareTicker(t);
  }

  function removeCompareTicker(t: string) {
    setCompareTickers((prev) => prev.filter((x) => x !== t));
    setCompareData((prev) => {
      const next = { ...prev };
      delete next[t];
      return next;
    });
  }

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">AI Price Forecast</h1>
      <p className="mt-1 text-sm text-slate-500">
        A backtested quant forecast for one ticker, shown next to how often it has actually beaten doing
        nothing.
      </p>

      <form onSubmit={runPrediction} className="mt-6 flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1">
          <label htmlFor="ticker" className="text-xs font-medium text-slate-500">
            Ticker
          </label>
          <TickerSearchInput
            id="ticker"
            value={ticker}
            onChange={setTicker}
            className="w-40 rounded-md border border-slate-300 px-3 py-2 text-sm"
          />
        </div>
        <CurrentPriceBadge ticker={ticker} refreshKey={priceRefreshKey} />
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
        <div className="flex flex-col gap-1">
          <label htmlFor="days-ahead" className="text-xs font-medium text-slate-500">
            Forecast horizon
          </label>
          <select
            id="days-ahead"
            value={daysAhead}
            onChange={(e) => setDaysAhead(Number(e.target.value))}
            className="rounded-md border border-slate-300 px-3 py-2 text-sm"
          >
            {FORECAST_HORIZONS.map((d) => (
              <option key={d} value={d}>
                {d} days
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
                <MetricTile label={`${shownDaysAhead}-Day Signal`} value={data.signal.signal} onInfoClick={() => setActiveInfo("signal")} />
                <MetricTile label={`Expected Return (${shownDaysAhead}d)`} value={`${data.signal.expected_return_pct.toFixed(2)}%`} />
                <MetricTile label="Target Price" value={`$${data.signal.target_price.toFixed(2)}`} />
              </div>
              <p className="mt-2 text-xs text-slate-500">
                This is one data-driven signal, not a guarantee — see backtest accuracy below for how it has
                historically performed against a simple no-change baseline.
              </p>
              {data.signal.signal_unstable && (
                <p className="mt-2 inline-flex rounded-md bg-amber-50 px-2 py-1 text-xs font-medium text-amber-700">
                  Unstable — flipped {data.signal.signal_flip_count} times over its trailing {data.signal.signal_days_captured}-day history. Treat this call with less confidence.
                </p>
              )}
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
            <h3 className="font-semibold text-slate-900">Compare Tickers</h3>
            <p className="mt-1 text-sm text-slate-600">
              Add other tickers to see their signal, expected return, and Safe Baseline band next to{" "}
              {data.ticker}&apos;s — same {period} window and {shownDaysAhead}-day horizon, 30d/90% band.
            </p>
            <form onSubmit={addCompareTicker} className="mt-3 flex items-end gap-2">
              <div className="flex flex-col gap-1">
                <label htmlFor="compare-ticker" className="text-xs font-medium text-slate-500">
                  Add ticker
                </label>
                <TickerSearchInput
                  id="compare-ticker"
                  value={compareInput}
                  onChange={setCompareInput}
                  className="w-40 rounded-md border border-slate-300 px-3 py-2 text-sm"
                />
              </div>
              <button
                type="submit"
                disabled={!compareInput.trim()}
                className="rounded-md border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
              >
                Add
              </button>
            </form>

            {compareTickers.length > 0 && (
              <div className="mt-4 overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                      <th className="px-2 py-2">Ticker</th>
                      <th className="px-2 py-2">Last Close</th>
                      <th className="px-2 py-2">Predicted Next Close</th>
                      <th className="px-2 py-2">Signal</th>
                      <th className="px-2 py-2">Expected Return</th>
                      <th className="px-2 py-2">Target Price</th>
                      <th className="px-2 py-2">Baseline Floor</th>
                      <th className="px-2 py-2">Baseline Ceiling</th>
                      <th className="px-2 py-2"></th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-slate-100 bg-slate-50">
                      <td className="px-2 py-2 font-medium text-slate-900">{data.ticker}</td>
                      <td className="px-2 py-2 text-slate-700">
                        {data.last_close !== null ? `$${data.last_close.toFixed(2)}` : "—"}
                      </td>
                      <td className="px-2 py-2 text-slate-700">
                        {data.next_price !== null ? `$${data.next_price.toFixed(2)}` : "—"}
                      </td>
                      <td className="px-2 py-2 text-slate-700">{data.signal?.signal ?? "—"}</td>
                      <td className="px-2 py-2 text-slate-700">
                        {data.signal ? `${data.signal.expected_return_pct.toFixed(2)}%` : "—"}
                      </td>
                      <td className="px-2 py-2 text-slate-700">
                        {data.signal ? `$${data.signal.target_price.toFixed(2)}` : "—"}
                      </td>
                      <td className="px-2 py-2 text-slate-700">
                        {primaryBand ? `$${primaryBand.floor.toFixed(2)}` : "—"}
                      </td>
                      <td className="px-2 py-2 text-slate-700">
                        {primaryBand ? `$${primaryBand.ceiling.toFixed(2)}` : "—"}
                      </td>
                      <td className="px-2 py-2"></td>
                    </tr>
                    {compareTickers.map((t) => {
                      const row = compareData[t];
                      return (
                        <tr key={t} className="border-b border-slate-100 last:border-0">
                          <td className="px-2 py-2 font-medium text-slate-900">{t}</td>
                          {row?.loading ? (
                            <td className="px-2 py-2 text-slate-400" colSpan={7}>
                              Loading…
                            </td>
                          ) : row?.error ? (
                            <td className="px-2 py-2 text-red-600" colSpan={7}>
                              {row.error}
                            </td>
                          ) : (
                            <>
                              <td className="px-2 py-2 text-slate-700">
                                {row?.summary?.last_close != null ? `$${row.summary.last_close.toFixed(2)}` : "—"}
                              </td>
                              <td className="px-2 py-2 text-slate-700">
                                {row?.summary?.next_price != null ? `$${row.summary.next_price.toFixed(2)}` : "—"}
                              </td>
                              <td className="px-2 py-2 text-slate-700">{row?.summary?.signal?.signal ?? "—"}</td>
                              <td className="px-2 py-2 text-slate-700">
                                {row?.summary?.signal ? `${row.summary.signal.expected_return_pct.toFixed(2)}%` : "—"}
                              </td>
                              <td className="px-2 py-2 text-slate-700">
                                {row?.summary?.signal ? `$${row.summary.signal.target_price.toFixed(2)}` : "—"}
                              </td>
                              <td className="px-2 py-2 text-slate-700">
                                {row?.band ? `$${row.band.floor.toFixed(2)}` : "—"}
                              </td>
                              <td className="px-2 py-2 text-slate-700">
                                {row?.band ? `$${row.band.ceiling.toFixed(2)}` : "—"}
                              </td>
                            </>
                          )}
                          <td className="px-2 py-2 text-right">
                            <button
                              onClick={() => removeCompareTicker(t)}
                              className="rounded-md border border-red-200 px-2 py-1 text-xs font-medium text-red-700 hover:bg-red-50"
                            >
                              Remove
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <h3 className="font-semibold text-slate-900">Volume &amp; Ownership Activity</h3>
            <p className="mt-1 text-sm text-slate-600">
              Trading volume, plus who&apos;s actually been buying or selling: company insiders (officers and
              directors, from SEC filings) and institutional &quot;outsider&quot; holders (funds and firms with
              13F filings). Real counts, nothing modeled.
            </p>
            <button
              onClick={runActivity}
              disabled={activityLoading}
              className="mt-3 rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
            >
              {activityLoading ? "Loading…" : activity ? "Refresh" : "Load Volume & Ownership Activity"}
            </button>

            {activityError && (
              <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{activityError}</p>
            )}

            {activity && !activityLoading && (
              <div className="mt-4 flex flex-col gap-4">
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <MetricTile
                    label="Latest Volume"
                    value={activity.latest_volume !== null ? activity.latest_volume.toLocaleString() : "—"}
                  />
                  <MetricTile
                    label="Avg Volume (10d)"
                    value={activity.avg_volume_10d !== null ? activity.avg_volume_10d.toLocaleString() : "—"}
                  />
                </div>

                <div>
                  <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
                    Insiders — {activity.insider_period}
                  </p>
                  {activity.insider_buys === null && activity.insider_sells === null ? (
                    <p className="mt-1 text-sm text-slate-400">
                      No insider filing data available for this ticker (common for ETFs/funds — there&apos;s no
                      officer or director to file as an insider).
                    </p>
                  ) : (
                    <div className="mt-2 grid grid-cols-1 gap-3 sm:grid-cols-2">
                      <MetricTile label="Insider Buys" value={String(activity.insider_buys ?? "—")} />
                      <MetricTile label="Insider Sells" value={String(activity.insider_sells ?? "—")} />
                    </div>
                  )}
                </div>

                <div>
                  <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
                    Institutional (&quot;Outsider&quot;) Holders
                    {activity.institutional_as_of && ` — as of ${activity.institutional_as_of.slice(0, 10)}`}
                  </p>
                  {activity.institutional_increased === null && activity.institutional_decreased === null ? (
                    <p className="mt-1 text-sm text-slate-400">No institutional holder data available for this ticker.</p>
                  ) : (
                    <>
                      <div className="mt-2 grid grid-cols-1 gap-3 sm:grid-cols-3">
                        <MetricTile label="Increased Position" value={String(activity.institutional_increased ?? "—")} />
                        <MetricTile label="Decreased Position" value={String(activity.institutional_decreased ?? "—")} />
                        <MetricTile label="Unchanged" value={String(activity.institutional_unchanged ?? "—")} />
                      </div>
                      <p className="mt-2 text-xs text-slate-500">
                        Among the top {activity.institutional_holder_count} reported holders, by their most
                        recently filed position change — not the same as buy/sell transaction counts.
                      </p>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>

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
                      <th className="px-2 py-2"></th>
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
                        <td className="px-2 py-2 text-right">
                          <button
                            onClick={() => handleDeletePrediction(p.id)}
                            disabled={deletingId === p.id}
                            className="rounded-md border border-red-200 px-2 py-1 text-xs font-medium text-red-700 hover:bg-red-50 disabled:opacity-50"
                          >
                            {deletingId === p.id ? "…" : "Delete"}
                          </button>
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
                <div className="flex items-center gap-3">
                  <button
                    onClick={handleSaveNarrative}
                    disabled={savingNarrative}
                    className="rounded-md border border-indigo-300 bg-white px-3 py-1.5 text-xs font-medium text-indigo-700 hover:bg-indigo-50 disabled:opacity-50"
                  >
                    {savingNarrative ? "Saving…" : "Save this context"}
                  </button>
                  {saveNarrativeMessage && <span className="text-xs text-slate-500">{saveNarrativeMessage}</span>}
                </div>
              </div>
            )}

            {(narrativeHistoryLoading || narrativeHistory.length > 0) && (
              <div className="mt-4 border-t border-indigo-200 pt-4">
                <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Saved Contexts for {data.ticker}
                </h4>
                {narrativeHistoryLoading ? (
                  <p className="mt-2 text-sm text-slate-500">Loading saved contexts…</p>
                ) : (
                  <div className="mt-2 flex flex-col gap-2">
                    {narrativeHistory.map((n) => (
                      <div key={n.id} className="flex flex-wrap items-center justify-between gap-2 rounded-md bg-white px-3 py-2 text-sm">
                        <span className="text-slate-700">
                          {new Date(n.saved_at).toLocaleString()} &middot; {n.provider}
                        </span>
                        <span className="flex items-center gap-2">
                          <button
                            onClick={() => setCompareNarrativeId(compareNarrativeId === n.id ? null : n.id)}
                            className="rounded-md border border-slate-300 px-2 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
                          >
                            {compareNarrativeId === n.id ? "Hide Compare" : "Compare"}
                          </button>
                          <button
                            onClick={() => handleDeleteNarrative(n.id)}
                            disabled={deletingNarrativeId === n.id}
                            className="rounded-md border border-red-200 px-2 py-1 text-xs font-medium text-red-700 hover:bg-red-50 disabled:opacity-50"
                          >
                            {deletingNarrativeId === n.id ? "…" : "Delete"}
                          </button>
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                {compareNarrativeId !== null && (() => {
                  const saved = narrativeHistory.find((n) => n.id === compareNarrativeId);
                  if (!saved) return null;
                  return (
                    <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-2">
                      <div className="rounded-md border border-slate-200 bg-white p-3">
                        <p className="text-xs font-semibold text-slate-500">
                          Saved {new Date(saved.saved_at).toLocaleString()} &middot; {saved.provider}
                        </p>
                        <p className="mt-2 text-sm leading-relaxed whitespace-pre-wrap text-slate-800">
                          {saved.narrative}
                        </p>
                      </div>
                      <div className="rounded-md border border-indigo-200 bg-white p-3">
                        <p className="text-xs font-semibold text-slate-500">Current</p>
                        {narrative ? (
                          <p className="mt-2 text-sm leading-relaxed whitespace-pre-wrap text-slate-800">
                            {narrative.narrative}
                          </p>
                        ) : (
                          <p className="mt-2 text-sm text-slate-400">
                            Click &quot;Get AI Context&quot; above to generate a new one to compare.
                          </p>
                        )}
                      </div>
                    </div>
                  );
                })()}
              </div>
            )}
          </div>

          {data.forecast ? (
            <div className="rounded-lg border border-slate-200 bg-white p-4">
              <ForecastChart ticker={data.ticker} forecast={data.forecast} />
            </div>
          ) : (
            <p className="text-sm text-slate-500">Not enough price history for a {shownDaysAhead}-day forecast.</p>
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

          <SafeBaselineBand ticker={data.ticker} />

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

      {activeInfo && getMetricInfo(shownDaysAhead)[activeInfo] && (
        <InfoModal info={getMetricInfo(shownDaysAhead)[activeInfo]} onClose={() => setActiveInfo(null)} />
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
