"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";

import {
  ApiError,
  getStockDetail,
  getStockPeers,
  getStockPosition,
  getStockPriceHistory,
  getStockSignalHistory,
  getTwoScore,
  getTwoScoreHistory,
  getTwoScoreWeeklyChange,
} from "@/lib/api";
import type {
  StockDetailResponse,
  StockPeersResponse,
  StockPositionResponse,
  StockPriceHistoryRange,
  StockPriceHistoryResponse,
  StockSignalHistoryResponse,
  TwoScoreHistoryResponse,
  TwoScoreResponse,
  TwoScoreWeeklyChangeResponse,
} from "@/lib/types";
import Sparkline from "@/components/portfolio/Sparkline";
import PriceHistoryChart from "@/components/stock-detail/PriceHistoryChart";

function signalBadgeClass(signal: string): string {
  if (signal === "Buy") return "bg-emerald-50 text-emerald-700";
  if (signal === "Trim") return "bg-red-50 text-red-700";
  return "bg-slate-100 text-slate-600";
}

function confidenceClass(label: string): string {
  if (label === "high") return "text-emerald-700";
  if (label === "medium") return "text-amber-700";
  if (label === "low") return "text-red-700";
  return "text-slate-400";
}

function ScoreCard({
  title,
  horizon,
  score,
  signal,
  confidence,
  sectorPercentile,
  sectorKey,
  trend,
}: {
  title: string;
  horizon: string;
  score: number | null;
  signal: string;
  confidence: { score: number | null; label: string };
  sectorPercentile: number | null;
  sectorKey: string;
  trend?: { weekly_series: [string, number][]; flagged: boolean; change_pts: number | null };
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-slate-900">{title}</h2>
          <p className="text-xs text-slate-500">{horizon}</p>
        </div>
        <span className={`rounded-full px-3 py-1 text-sm font-semibold ${signalBadgeClass(signal)}`}>{signal}</span>
      </div>
      <div className="mt-3 flex items-end gap-2">
        <span className="text-4xl font-semibold text-slate-900">{score !== null ? score.toFixed(0) : "—"}</span>
        <span className="pb-1 text-sm text-slate-400">/ 100</span>
      </div>
      <p className="mt-1 text-xs text-slate-500">
        {sectorPercentile !== null
          ? `Top ${Math.max(1, Math.round(100 - sectorPercentile))}% of ${sectorKey}`
          : `No sector rank yet — ${sectorKey}`}
      </p>
      <p className={`mt-1 text-xs font-medium ${confidenceClass(confidence.label)}`}>
        Confidence: {confidence.label}
        {confidence.score !== null ? ` (${confidence.score.toFixed(0)})` : ""}
      </p>
      {trend && trend.weekly_series.length >= 2 && (
        <div className="mt-3 flex items-center gap-2">
          <Sparkline values={trend.weekly_series.map(([, v]) => v)} color="#1F4FD1" />
          {trend.flagged && trend.change_pts !== null && (
            <span className={`text-xs font-semibold ${trend.change_pts >= 0 ? "text-emerald-700" : "text-red-700"}`}>
              {trend.change_pts >= 0 ? "+" : ""}
              {trend.change_pts.toFixed(0)} pts this week
            </span>
          )}
        </div>
      )}
    </div>
  );
}

export default function StockScorePage() {
  const params = useParams<{ ticker: string }>();
  const router = useRouter();
  const ticker = (params.ticker as string)?.toUpperCase() ?? "";

  const [jumpTicker, setJumpTicker] = useState("");
  const [data, setData] = useState<TwoScoreResponse | null>(null);
  const [history, setHistory] = useState<TwoScoreHistoryResponse | null>(null);
  const [weeklyChange, setWeeklyChange] = useState<TwoScoreWeeklyChangeResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [detail, setDetail] = useState<StockDetailResponse | null>(null);
  const [position, setPosition] = useState<StockPositionResponse | null>(null);
  const [loggedIn, setLoggedIn] = useState<boolean | null>(null);
  const [signalHistory, setSignalHistory] = useState<StockSignalHistoryResponse | null>(null);
  const [peers, setPeers] = useState<StockPeersResponse | null>(null);

  const [priceRange, setPriceRange] = useState<StockPriceHistoryRange>("1Y");
  const [priceHistory, setPriceHistory] = useState<StockPriceHistoryResponse | null>(null);
  const [priceLoading, setPriceLoading] = useState(true);

  useEffect(() => {
    if (!ticker) return;
    setLoading(true);
    setError(null);
    setData(null);
    Promise.all([getTwoScore(ticker), getTwoScoreHistory(ticker).catch(() => null), getTwoScoreWeeklyChange(ticker).catch(() => null)])
      .then(([score, hist, weekly]) => {
        setData(score);
        setHistory(hist);
        setWeeklyChange(weekly);
      })
      .catch((err) => {
        setError(
          err instanceof ApiError && err.status === 404
            ? `No score on record yet for ${ticker}. Scores are computed nightly after market close — check back after the next trading day.`
            : err instanceof ApiError
            ? err.message
            : "Could not load this stock's score.",
        );
      })
      .finally(() => setLoading(false));
  }, [ticker]);

  useEffect(() => {
    if (!ticker) return;
    setDetail(null);
    setSignalHistory(null);
    setPeers(null);
    getStockDetail(ticker).then(setDetail).catch(() => setDetail(null));
    getStockSignalHistory(ticker).then(setSignalHistory).catch(() => setSignalHistory(null));
    getStockPeers(ticker).then(setPeers).catch(() => setPeers(null));
    getStockPosition(ticker)
      .then((pos) => {
        setPosition(pos);
        setLoggedIn(true);
      })
      .catch((err) => {
        if (err instanceof ApiError && err.status === 401) {
          setLoggedIn(false);
        }
        setPosition(null);
      });
  }, [ticker]);

  useEffect(() => {
    if (!ticker) return;
    setPriceLoading(true);
    getStockPriceHistory(ticker, priceRange)
      .then(setPriceHistory)
      .catch(() => setPriceHistory(null))
      .finally(() => setPriceLoading(false));
  }, [ticker, priceRange]);

  function handleJump(e: React.FormEvent) {
    e.preventDefault();
    const t = jumpTicker.trim().toUpperCase();
    if (t) router.push(`/stock/${t}`);
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Stock Detail</p>
          <h1 className="text-2xl font-semibold text-slate-900">
            {ticker}
            {detail?.current_price !== null && detail?.current_price !== undefined && (
              <span className="ml-3 text-lg font-normal text-slate-500">${detail.current_price.toFixed(2)}</span>
            )}
          </h1>
          {data && (
            <p className="text-xs text-slate-500">
              {data.sector_key} · as of {data.as_of_date}
            </p>
          )}
        </div>
        <form onSubmit={handleJump} className="flex gap-2">
          <input
            value={jumpTicker}
            onChange={(e) => setJumpTicker(e.target.value.toUpperCase())}
            placeholder="Jump to ticker…"
            className="w-44 rounded-md border border-slate-300 px-3 py-1.5 text-sm uppercase"
          />
          <button
            type="submit"
            className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-100"
          >
            Go
          </button>
        </form>
      </div>

      <div className="mt-6">
        <PriceHistoryChart
          ticker={ticker}
          data={priceHistory}
          range={priceRange}
          onRangeChange={setPriceRange}
          loading={priceLoading}
        />
      </div>

      {loading && <p className="mt-6 text-sm text-slate-500">Loading…</p>}
      {error && <p className="mt-6 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {data && (
        <>
          <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2">
            <ScoreCard
              title="Short-Term Score"
              horizon="10-90 day horizon"
              score={data.short_score}
              signal={data.short_signal}
              confidence={data.short_confidence}
              sectorPercentile={data.short_sector_percentile}
              sectorKey={data.sector_key}
              trend={history?.short_term}
            />
            <ScoreCard
              title="Long-Term Score"
              horizon="1-3 year horizon"
              score={data.long_score}
              signal={data.long_signal}
              confidence={data.long_confidence}
              sectorPercentile={data.long_sector_percentile}
              sectorKey={data.sector_key}
              trend={history?.long_term}
            />
          </div>

          <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="rounded-lg border border-slate-200 bg-white p-5">
              <h3 className="text-sm font-semibold text-slate-900">Top Drivers</h3>
              {data.explanations.drivers.length === 0 ? (
                <p className="mt-2 text-xs text-slate-400">No positive drivers today.</p>
              ) : (
                <ul className="mt-2 flex flex-col gap-1 text-sm text-emerald-700">
                  {data.explanations.drivers.map((d) => (
                    <li key={d.factor}>
                      +{d.contribution.toFixed(1)} pts — {d.factor}
                    </li>
                  ))}
                </ul>
              )}
            </div>
            <div className="rounded-lg border border-slate-200 bg-white p-5">
              <h3 className="text-sm font-semibold text-slate-900">Top Drags</h3>
              {data.explanations.drags.length === 0 ? (
                <p className="mt-2 text-xs text-slate-400">No negative drags today.</p>
              ) : (
                <ul className="mt-2 flex flex-col gap-1 text-sm text-red-700">
                  {data.explanations.drags.map((d) => (
                    <li key={d.factor}>
                      {d.contribution.toFixed(1)} pts — {d.factor}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>

          {weeklyChange?.change && (
            <div className="mt-4 rounded-md bg-blue-50 px-3 py-2 text-sm text-blue-800">
              Since {weeklyChange.compared_to}: {weeklyChange.change.factor} moved the most (
              {weeklyChange.change.delta_contribution >= 0 ? "+" : ""}
              {weeklyChange.change.delta_contribution.toFixed(1)} pts contribution).
            </div>
          )}

          <div className="mt-6 rounded-lg border border-slate-200 bg-white p-5">
            <h3 className="text-sm font-semibold text-slate-900">Why these scores</h3>
            <ul className="mt-2 flex flex-col gap-2 text-sm text-slate-700">
              <li>{data.sentences.momentum}</li>
              <li>{data.sentences.reversal}</li>
              <li>{data.sentences.value}</li>
              <li>{data.sentences.growth}</li>
              <li>{data.sentences.low_vol}</li>
            </ul>
            <p className="mt-3 text-xs text-slate-400">
              Rules-based composite scores from real, already-captured data — not a trained prediction model.
              Earnings-revisions/surprise and Quality factors aren&apos;t included yet (no data source for them exists).
            </p>
          </div>
        </>
      )}

      <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-semibold text-slate-900">Key Stats</h3>
          {detail ? (
            <dl className="mt-2 grid grid-cols-2 gap-y-2 text-sm">
              <dt className="text-slate-500">Sector</dt>
              <dd className="text-right text-slate-900">{detail.sector ?? "—"}</dd>
              <dt className="text-slate-500">Forward P/E</dt>
              <dd className="text-right text-slate-900">
                {detail.fundamentals.forward_pe !== null ? detail.fundamentals.forward_pe.toFixed(1) : "—"}
              </dd>
              <dt className="text-slate-500">Revenue Growth</dt>
              <dd className="text-right text-slate-900">
                {detail.fundamentals.revenue_growth_pct !== null
                  ? `${detail.fundamentals.revenue_growth_pct.toFixed(1)}%`
                  : "—"}
              </dd>
              <dt className="text-slate-500">Earnings Growth</dt>
              <dd className="text-right text-slate-900">
                {detail.fundamentals.earnings_growth_pct !== null
                  ? `${detail.fundamentals.earnings_growth_pct.toFixed(1)}%`
                  : "—"}
              </dd>
            </dl>
          ) : (
            <p className="mt-2 text-xs text-slate-400">No fundamentals on record for {ticker} yet.</p>
          )}
        </div>

        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-semibold text-slate-900">Earnings &amp; Dividends</h3>
          {detail ? (
            <div className="mt-2 text-sm text-slate-700">
              <p>
                Next earnings:{" "}
                {detail.next_earnings ? (
                  <>
                    {detail.next_earnings.date}
                    {detail.next_earnings.eps_estimate !== null
                      ? ` (est. EPS $${detail.next_earnings.eps_estimate.toFixed(2)})`
                      : ""}
                  </>
                ) : (
                  "none scheduled"
                )}
              </p>
              {detail.recent_dividends.length > 0 ? (
                <ul className="mt-2 flex flex-col gap-1 text-xs text-slate-500">
                  {detail.recent_dividends.map((d) => (
                    <li key={d.date}>
                      {d.date}: ${d.amount.toFixed(4)}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="mt-2 text-xs text-slate-400">No dividend history — {ticker} hasn&apos;t paid one.</p>
              )}
            </div>
          ) : (
            <p className="mt-2 text-xs text-slate-400">Loading…</p>
          )}
        </div>
      </div>

      {loggedIn && (
        <div className="mt-6 rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-semibold text-slate-900">Your Position</h3>
          {position?.owned ? (
            <dl className="mt-2 grid grid-cols-2 gap-y-2 text-sm sm:grid-cols-4">
              <dt className="text-slate-500">Shares</dt>
              <dd className="text-right text-slate-900 sm:text-left">{position.shares}</dd>
              <dt className="text-slate-500">Avg Cost</dt>
              <dd className="text-right text-slate-900 sm:text-left">${position.avg_cost?.toFixed(2)}</dd>
              <dt className="text-slate-500">Gain/Loss</dt>
              <dd
                className={`text-right sm:text-left ${
                  (position.gain_loss_pct ?? 0) >= 0 ? "text-emerald-700" : "text-red-700"
                }`}
              >
                {position.gain_loss_pct !== null && position.gain_loss_pct !== undefined
                  ? `${position.gain_loss_pct >= 0 ? "+" : ""}${position.gain_loss_pct.toFixed(1)}%`
                  : "—"}
              </dd>
              <dt className="text-slate-500">Portfolio Weight</dt>
              <dd className="text-right text-slate-900 sm:text-left">
                {position.weight_pct !== null && position.weight_pct !== undefined
                  ? `${position.weight_pct.toFixed(1)}%`
                  : "—"}
              </dd>
            </dl>
          ) : (
            <p className="mt-2 text-xs text-slate-400">You don&apos;t own {ticker} in this portfolio.</p>
          )}
        </div>
      )}

      <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-semibold text-slate-900">Signal History</h3>
          {signalHistory && signalHistory.history.length > 0 ? (
            <>
              <ul className="mt-2 flex flex-col gap-1 text-xs text-slate-600">
                {signalHistory.history.slice(0, 8).map((h) => (
                  <li key={h.as_of_date} className="flex justify-between">
                    <span>{h.as_of_date}</span>
                    <span>
                      {h.short_signal} / {h.long_signal}
                    </span>
                  </li>
                ))}
              </ul>
              <p className="mt-2 text-xs text-slate-400">{signalHistory.note}</p>
            </>
          ) : (
            <p className="mt-2 text-xs text-slate-400">No signal history on record for {ticker} yet.</p>
          )}
        </div>

        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-semibold text-slate-900">Similar Stocks</h3>
          {peers && peers.peers.length > 0 ? (
            <ul className="mt-2 flex flex-col gap-1 text-sm">
              {peers.peers.map((p) => (
                <li key={p.ticker} className="flex justify-between">
                  <a href={`/stock/${p.ticker}`} className="text-blue-700 hover:underline">
                    {p.ticker}
                  </a>
                  <span className="text-slate-500">${p.market_cap_b.toFixed(1)}B</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-2 text-xs text-slate-400">No same-sector peers found for {ticker}.</p>
          )}
        </div>
      </div>
    </div>
  );
}
