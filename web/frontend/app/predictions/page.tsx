"use client";

import { useEffect, useState } from "react";

import { ApiError, comparePredictionsToFund, getPredictionHistory } from "@/lib/api";
import type { PredictionCompareResponse, SavedPrediction } from "@/lib/types";

export default function PredictionsPage() {
  const [predictions, setPredictions] = useState<SavedPrediction[] | null>(null);
  const [compare, setCompare] = useState<PredictionCompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [compareLoading, setCompareLoading] = useState(false);

  useEffect(() => {
    load();
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
      <h1 className="text-2xl font-semibold text-slate-900">My Saved Predictions</h1>
      <p className="mt-1 text-sm text-slate-500">
        Every prediction you&apos;ve saved, across every ticker, in one place — auto-verified in the background as
        target dates arrive.
      </p>

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
                      <th className="px-3 py-2">Saved</th>
                      <th className="px-3 py-2">Model Predicted</th>
                      <th className="px-3 py-2">Stock Actual (target)</th>
                      <th className="px-3 py-2">Stock Since Saved</th>
                      <th className="px-3 py-2">Fund Since Saved</th>
                      <th className="px-3 py-2">Stock Beat Fund?</th>
                    </tr>
                  </thead>
                  <tbody>
                    {compare.comparisons.map((c) => {
                      const beat =
                        c.stock_return_since_saved_pct !== null && c.fund_return_since_saved_pct !== null
                          ? c.stock_return_since_saved_pct > c.fund_return_since_saved_pct
                          : null;
                      return (
                        <tr key={c.prediction_id} className="border-b border-slate-100 last:border-0">
                          <td className="px-3 py-2 font-medium text-slate-800">{c.ticker}</td>
                          <td className="px-3 py-2 text-slate-600">{new Date(c.predicted_at).toLocaleDateString()}</td>
                          <td className="px-3 py-2 text-slate-600">
                            {c.predicted_return_pct !== null ? `${c.predicted_return_pct.toFixed(2)}%` : "—"}
                          </td>
                          <td className="px-3 py-2 text-slate-600">
                            {c.actual_return_pct !== null ? `${c.actual_return_pct.toFixed(2)}%` : "pending"}
                          </td>
                          <td className="px-3 py-2 text-slate-600">
                            {c.stock_return_since_saved_pct !== null ? `${c.stock_return_since_saved_pct.toFixed(2)}%` : "—"}
                          </td>
                          <td className="px-3 py-2 text-slate-600">
                            {c.fund_return_since_saved_pct !== null ? `${c.fund_return_since_saved_pct.toFixed(2)}%` : "—"}
                          </td>
                          <td className="px-3 py-2">
                            {beat === null ? (
                              <span className="text-slate-400">—</span>
                            ) : beat ? (
                              <span className="text-emerald-600">✓ beat fund</span>
                            ) : (
                              <span className="text-red-600">✗ lagged fund</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
            <p className="mt-3 text-xs text-slate-500">
              &quot;Stock Since Saved&quot; and &quot;Fund Since Saved&quot; are both measured from the date you
              saved the prediction to right now — an apples-to-apples window regardless of whether the
              target date has arrived yet. &quot;Model Predicted&quot; and &quot;Stock Actual (target)&quot;
              instead compare the original forecast horizon specifically, and stay &quot;pending&quot; until
              that target date passes.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
