"use client";

import { useEffect, useState } from "react";

import {
  ApiError,
  comparePredictionsToFund,
  deletePrediction,
  getPredictionAccuracyLeaderboard,
  getPredictionHistory,
} from "@/lib/api";
import type { PredictionAccuracyLeaderboard, PredictionCompareResponse, SavedPrediction } from "@/lib/types";

export default function PredictionsPage() {
  const [predictions, setPredictions] = useState<SavedPrediction[] | null>(null);
  const [compare, setCompare] = useState<PredictionCompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [compareLoading, setCompareLoading] = useState(false);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  const [leaderboard, setLeaderboard] = useState<PredictionAccuracyLeaderboard | null>(null);
  const [leaderboardLoading, setLeaderboardLoading] = useState(true);
  const [leaderboardError, setLeaderboardError] = useState<string | null>(null);

  useEffect(() => {
    load();
    loadLeaderboard();
  }, []);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const res = await getPredictionHistory();
      setPredictions(res.predictions);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load saved predictions.");
    } finally {
      setLoading(false);
    }
  }

  async function loadLeaderboard() {
    setLeaderboardLoading(true);
    setLeaderboardError(null);
    try {
      setLeaderboard(await getPredictionAccuracyLeaderboard());
    } catch (err) {
      setLeaderboardError(err instanceof ApiError ? err.message : "Failed to load the accuracy leaderboard.");
    } finally {
      setLeaderboardLoading(false);
    }
  }

  async function handleDelete(id: number) {
    setDeletingId(id);
    setError(null);
    try {
      await deletePrediction(id);
      setPredictions((prev) => (prev ? prev.filter((p) => p.id !== id) : prev));
      loadLeaderboard();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to delete this prediction.");
    } finally {
      setDeletingId(null);
    }
  }

  async function loadCompare() {
    setCompareLoading(true);
    setCompareError(null);
    try {
      setCompare(await comparePredictionsToFund());
    } catch (err) {
      setCompareError(err instanceof ApiError ? err.message : "Failed to load comparison.");
    } finally {
      setCompareLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h1 className="text-2xl font-semibold text-slate-900">Prediction History</h1>
        <button
          onClick={() => {
            load();
            loadLeaderboard();
          }}
          disabled={loading || leaderboardLoading}
          className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
        >
          {loading || leaderboardLoading ? "Refreshing…" : "Refresh all"}
        </button>
      </div>
      <p className="mt-1 text-sm text-slate-500">
        Every prediction you&apos;ve saved, across every ticker, in one place. Predictions are auto-verified
        against real prices as each one&apos;s target date arrives — this page doesn&apos;t poll continuously,
        so use Refresh to pull the latest state (e.g. after saving a new prediction elsewhere, or once a target
        date you&apos;re waiting on should have passed).
      </p>

      <div className="mt-6 rounded-lg border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="font-semibold text-slate-900">Accuracy Leaderboard</h2>
          <button
            onClick={loadLeaderboard}
            disabled={leaderboardLoading}
            className="rounded-md border border-slate-200 px-2.5 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50 disabled:opacity-50"
          >
            {leaderboardLoading ? "Refreshing…" : "Refresh"}
          </button>
        </div>
        <details className="mt-1">
          <summary className="cursor-pointer text-sm text-slate-600 hover:text-slate-800">
            How this works
          </summary>
          <p className="mt-1.5 text-sm text-slate-600">
            Every prediction you save has a target date (e.g. 10 trading days out). It sits as{" "}
            <strong>pending</strong> until that date actually arrives — only then does it become{" "}
            <strong>verified</strong> and count toward a ticker&apos;s win rate. This page doesn&apos;t
            auto-refresh on its own (verification happens as target dates pass, not continuously) — use{" "}
            <strong>Refresh</strong> above to pull the latest state, e.g. right after saving a new prediction
            or when a target date you&apos;ve been waiting on should have just passed. A ticker needs at least{" "}
            {leaderboard?.min_verified_for_recommendation ?? 3} verified predictions before it&apos;s eligible
            to be suggested below.
          </p>
        </details>

        {leaderboardLoading && <p className="mt-3 text-sm text-slate-500">Loading…</p>}
        {leaderboardError && (
          <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{leaderboardError}</p>
        )}

        {leaderboard && !leaderboardLoading && (
          <>
            {(() => {
              const totalPredictions = leaderboard.tickers.reduce((sum, r) => sum + r.total_predictions, 0);
              const totalVerified = leaderboard.tickers.reduce((sum, r) => sum + r.verified_count, 0);
              return totalPredictions > 0 ? (
                <p className="mt-3 text-xs text-slate-500">
                  <span className="font-medium text-slate-700">{totalVerified}</span> verified,{" "}
                  <span className="font-medium text-slate-700">{totalPredictions - totalVerified}</span> still
                  pending (target date hasn&apos;t arrived yet), across{" "}
                  <span className="font-medium text-slate-700">{leaderboard.tickers.length}</span> tickers.
                </p>
              ) : null;
            })()}

            {leaderboard.suggested_ticker ? (
              <div className="mt-3 rounded-md border border-emerald-200 bg-emerald-50 px-3 py-2.5">
                <p className="text-sm text-emerald-900">
                  <strong>{leaderboard.suggested_ticker}</strong> has the best track record — {leaderboard.suggested_reason}
                </p>
                <a
                  href={`/predict?ticker=${encodeURIComponent(leaderboard.suggested_ticker)}`}
                  className="mt-1 inline-block text-xs font-medium text-emerald-700 underline hover:text-emerald-900"
                >
                  View forecast for {leaderboard.suggested_ticker}
                </a>
              </div>
            ) : (
              <p className="mt-3 rounded-md bg-slate-50 px-3 py-2 text-sm text-slate-500">
                No ticker has enough verified predictions yet for a portfolio suggestion.
              </p>
            )}

            {leaderboard.tickers.length > 0 && (
              <div className="mt-3 overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                      <th className="px-2 py-1.5">Rank</th>
                      <th className="px-2 py-1.5">Ticker</th>
                      <th className="px-2 py-1.5">Verified / Total</th>
                      <th className="px-2 py-1.5">Win Rate</th>
                      <th className="px-2 py-1.5">Avg Next-Day Error</th>
                      <th className="px-2 py-1.5">Avg Target Error</th>
                    </tr>
                  </thead>
                  <tbody>
                    {leaderboard.tickers.map((row) => (
                      <tr
                        key={row.ticker}
                        className={`border-b border-slate-100 last:border-0 ${row.rank === null ? "text-slate-400" : ""}`}
                      >
                        <td className="px-2 py-1.5">{row.rank ?? "—"}</td>
                        <td className={`px-2 py-1.5 font-medium ${row.rank === null ? "text-slate-500" : "text-slate-900"}`}>
                          {row.ticker}
                          {row.ticker === leaderboard.suggested_ticker && " 🏆"}
                        </td>
                        <td className="px-2 py-1.5">
                          {row.verified_count} / {row.total_predictions}
                        </td>
                        <td className="px-2 py-1.5">
                          {row.win_rate !== null ? (
                            <span className="text-slate-700">{(row.win_rate * 100).toFixed(0)}%</span>
                          ) : (
                            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-400">pending</span>
                          )}
                        </td>
                        <td className="px-2 py-1.5">
                          {row.avg_next_price_error_pct !== null ? `${row.avg_next_price_error_pct.toFixed(2)}%` : "—"}
                        </td>
                        <td className="px-2 py-1.5">
                          {row.avg_target_price_error_pct !== null ? `${row.avg_target_price_error_pct.toFixed(2)}%` : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </div>

      {loading && <p className="mt-4 text-sm text-slate-500">Loading…</p>}
      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {predictions && !loading && (
        <div className="mt-6 overflow-x-auto rounded-lg border border-slate-200 bg-white">
          {predictions.length === 0 ? (
            <p className="px-4 py-6 text-sm text-slate-500">
              No saved predictions yet — save one from the{" "}
              <a href="/predict" className="text-slate-900 underline">
                Price Prediction
              </a>{" "}
              page.
            </p>
          ) : (
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                  <th className="px-3 py-2">Ticker</th>
                  <th className="px-3 py-2">Saved</th>
                  <th className="px-3 py-2">Last Close</th>
                  <th className="px-3 py-2">Target Date</th>
                  <th className="px-3 py-2">Predicted Target</th>
                  <th className="px-3 py-2">Actual</th>
                  <th className="px-3 py-2">Error %</th>
                  <th className="px-3 py-2">Signal</th>
                  <th className="px-3 py-2">Correct?</th>
                  <th className="px-3 py-2"></th>
                </tr>
              </thead>
              <tbody>
                {predictions.map((p) => (
                  <tr key={p.id} className="border-b border-slate-100 last:border-0">
                    <td className="px-3 py-2 font-medium text-slate-800">{p.ticker}</td>
                    <td className="px-3 py-2 text-slate-600">{new Date(p.predicted_at).toLocaleDateString()}</td>
                    <td className="px-3 py-2 text-slate-600">{p.last_close !== null ? `$${p.last_close.toFixed(2)}` : "—"}</td>
                    <td className="px-3 py-2 text-slate-600">{p.target_date ? new Date(p.target_date).toLocaleDateString() : "—"}</td>
                    <td className="px-3 py-2 text-slate-600">{p.target_price !== null ? `$${p.target_price.toFixed(2)}` : "—"}</td>
                    <td className="px-3 py-2 text-slate-600">
                      {p.actual_target_price !== null ? `$${p.actual_target_price.toFixed(2)}` : "pending"}
                    </td>
                    <td className="px-3 py-2 text-slate-600">
                      {p.target_price_error_pct !== null ? `${p.target_price_error_pct.toFixed(2)}%` : "—"}
                    </td>
                    <td className="px-3 py-2 text-slate-600">{p.signal ?? "—"}</td>
                    <td className="px-3 py-2">
                      {p.signal_correct === null ? (
                        <span className="text-slate-400">pending</span>
                      ) : p.signal_correct ? (
                        <span className="text-emerald-600">✓ correct</span>
                      ) : (
                        <span className="text-red-600">✗ wrong</span>
                      )}
                    </td>
                    <td className="px-3 py-2 text-right">
                      <button
                        onClick={() => handleDelete(p.id)}
                        disabled={deletingId === p.id}
                        className="rounded-md border border-red-200 px-2.5 py-1 text-xs font-medium text-red-700 hover:bg-red-50 disabled:opacity-50"
                      >
                        {deletingId === p.id ? "…" : "Delete"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      <div className="mt-8 rounded-lg border border-indigo-200 bg-indigo-50/40 p-5">
        <h2 className="font-semibold text-slate-900">Compare Against a Top Fund</h2>
        <p className="mt-1 text-sm text-slate-600">
          For each saved prediction: the stock&apos;s actual return since you saved it, next to what the
          current top-ranked fund (Balanced Core, All categories) returned over that same stretch — plus the
          model&apos;s originally predicted return, so you can see whether picking the stock over the fund
          would have paid off.
        </p>
        <button
          onClick={loadCompare}
          disabled={compareLoading}
          className="mt-3 rounded-md border border-indigo-300 bg-white px-3 py-1.5 text-sm font-medium text-indigo-700 hover:bg-indigo-50 disabled:opacity-50"
        >
          {compareLoading ? "Comparing…" : compare ? "Refresh Comparison" : "Compare"}
        </button>

        {compareError && (
          <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{compareError}</p>
        )}

        {compare && !compareLoading && (
          <div className="mt-4">
            {compare.top_fund && (
              <p className="text-sm text-slate-600">
                Benchmark: <strong>{compare.top_fund.ticker}</strong> — {compare.top_fund.name}
                {compare.fund_current_price !== null && ` ($${compare.fund_current_price.toFixed(2)} now)`}
              </p>
            )}
            {compare.comparisons.length === 0 ? (
              <p className="mt-2 text-sm text-slate-500">No saved predictions to compare yet.</p>
            ) : (
              <div className="mt-3 overflow-x-auto rounded-lg border border-slate-200 bg-white">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                      <th className="px-3 py-2">Ticker</th>
                      <th className="px-3 py-2">Held</th>
                      <th className="px-3 py-2 text-right">Return</th>
                      <th className="px-3 py-2 text-right">Benchmark</th>
                      <th className="px-3 py-2 text-right">vs. Bmk</th>
                      <th className="px-3 py-2 text-right">Predicted</th>
                      <th className="px-3 py-2 text-right">Error</th>
                    </tr>
                  </thead>
                  <tbody>
                    {compare.comparisons.map((c) => {
                      const heldDays = Math.max(
                        0,
                        Math.floor((Date.now() - new Date(c.predicted_at).getTime()) / 86_400_000)
                      );
                      const vsBmk =
                        c.stock_return_since_saved_pct !== null && c.fund_return_since_saved_pct !== null
                          ? c.stock_return_since_saved_pct - c.fund_return_since_saved_pct
                          : null;
                      const errorPct =
                        c.stock_return_since_saved_pct !== null && c.predicted_return_pct !== null
                          ? c.stock_return_since_saved_pct - c.predicted_return_pct
                          : null;
                      return (
                        <tr key={c.prediction_id} className="border-b border-slate-100 last:border-0">
                          <td className="px-3 py-2 font-medium text-slate-800">{c.ticker}</td>
                          <td className="px-3 py-2 text-slate-600">{heldDays}d</td>
                          <Signed value={c.stock_return_since_saved_pct} />
                          <Signed value={c.fund_return_since_saved_pct} />
                          <Signed value={vsBmk} bold />
                          <Signed value={c.predicted_return_pct} />
                          <Signed value={errorPct} invert />
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
            <p className="mt-3 text-xs text-slate-500">
              <strong>Return</strong> / <strong>Benchmark</strong> are measured from the date you saved the
              prediction to right now (the benchmark is the current top-ranked fund) — an apples-to-apples
              window regardless of whether the target date has arrived yet. <strong>vs. Bmk</strong> is Return
              minus Benchmark (positive = the stock beat the fund). <strong>Predicted</strong> is the model&apos;s
              original forecast return; <strong>Error</strong> is Return minus Predicted (negative = the model
              overshot; positive = it undershot).
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function Signed({ value, bold, invert }: { value: number | null; bold?: boolean; invert?: boolean }) {
  if (value === null) {
    return <td className="px-3 py-2 text-right text-slate-400">—</td>;
  }
  // `invert` flips the color read for Error, where negative (model
  // overshot) isn't necessarily "bad" the way a negative Return is —
  // still shown with sign, just without implying "red = worse" here.
  const positive = value >= 0;
  const colorClass = invert ? "text-slate-700" : positive ? "text-emerald-600" : "text-red-600";
  return (
    <td className={`px-3 py-2 text-right ${bold ? "font-semibold" : ""} ${colorClass}`}>
      {positive ? "+" : ""}
      {value.toFixed(1)}%
    </td>
  );
}
