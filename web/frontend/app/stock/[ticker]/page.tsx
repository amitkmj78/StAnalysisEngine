"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";

import {
  ApiError,
  getPortfolioPositions,
  getStockDetail,
  getStockPeers,
  getStockPosition,
  getStockPriceHistory,
  getStockSentiment,
  getStockSignalHistory,
  getTwoScore,
  getTwoScoreHistory,
  getTwoScoreWeeklyChange,
} from "@/lib/api";
import type {
  PortfolioPosition,
  StockDetailResponse,
  StockPeersResponse,
  StockPositionResponse,
  StockPriceHistoryRange,
  StockPriceHistoryResponse,
  StockSentimentResponse,
  StockSignalHistoryResponse,
  TwoScoreHistoryResponse,
  TwoScoreResponse,
  TwoScoreWeeklyChangeResponse,
} from "@/lib/types";
import Sparkline from "@/components/portfolio/Sparkline";
import PriceHistoryChart from "@/components/stock-detail/PriceHistoryChart";
import TickerSearchInput from "@/components/TickerSearchInput";

function signalBadgeClass(signal: string): string {
  if (signal === "Buy") return "bg-emerald-50 text-emerald-700";
  if (signal === "Trim") return "bg-red-50 text-red-700";
  return "bg-slate-100 text-slate-600";
}

// Same three-way read as the badge, spent as a quiet left-edge stripe on
// each card instead of a second loud color — status is visible at a
// glance without competing with the badge itself.
function signalAccentClass(signal: string): string {
  if (signal === "Buy") return "border-l-emerald-400";
  if (signal === "Trim") return "border-l-red-400";
  return "border-l-slate-300";
}

function sentimentBadgeClass(label: string): string {
  if (label === "Bullish") return "bg-emerald-50 text-emerald-700";
  if (label === "Bearish") return "bg-red-50 text-red-700";
  return "bg-slate-100 text-slate-600";
}

function sentimentAccentClass(label: string): string {
  if (label === "Bullish") return "border-l-emerald-400";
  if (label === "Bearish") return "border-l-red-400";
  return "border-l-slate-300";
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
    <div className={`rounded-lg border border-l-4 border-slate-200 bg-white p-5 ${signalAccentClass(signal)}`}>
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-slate-900">{title}</h2>
          <p className="text-xs text-slate-500">{horizon}</p>
        </div>
        <span className={`rounded-full px-3 py-1 text-sm font-semibold ${signalBadgeClass(signal)}`}>{signal}</span>
      </div>
      <div className="mt-3 flex items-end gap-2">
        <span className="text-4xl font-semibold tracking-tight text-slate-900">
          {score !== null ? score.toFixed(0) : "—"}
        </span>
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
  const [holdings, setHoldings] = useState<PortfolioPosition[]>([]);
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

  // On-demand only (real web-search + LLM cost per call) -- never fetched
  // automatically with the rest of the page.
  const [sentiment, setSentiment] = useState<StockSentimentResponse | null>(null);
  const [sentimentLoading, setSentimentLoading] = useState(false);
  const [sentimentError, setSentimentError] = useState<string | null>(null);

  // The user's whole portfolio, not the current ticker's -- fetched once,
  // not re-fetched on every ticker change. 401 (logged out) just leaves
  // it empty; the picker below only renders when there's something to pick.
  useEffect(() => {
    getPortfolioPositions()
      .then((res) => setHoldings(res.positions))
      .catch(() => setHoldings([]));
  }, []);

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
    setSentiment(null);
    setSentimentError(null);
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

  function handleSelectTicker(t: string) {
    const clean = t.trim().toUpperCase();
    if (clean) router.push(`/stock/${clean}`);
  }

  function handleLoadSentiment() {
    setSentimentLoading(true);
    setSentimentError(null);
    getStockSentiment(ticker)
      .then(setSentiment)
      .catch((err) => {
        setSentimentError(
          err instanceof ApiError && err.status === 401
            ? "Sign in to load today's news and sentiment reading."
            : err instanceof ApiError
            ? err.message
            : "Could not load news and sentiment right now.",
        );
      })
      .finally(() => setSentimentLoading(false));
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Stock Detail</p>
          <div className="flex items-baseline gap-3">
            <h1 className="text-3xl font-semibold tracking-tight text-slate-900">{ticker}</h1>
            {detail?.current_price !== null && detail?.current_price !== undefined && (
              <span className="text-lg font-medium text-slate-500">${detail.current_price.toFixed(2)}</span>
            )}
          </div>
          {data && (
            <div className="mt-1.5 flex items-center gap-2">
              <span className="inline-block rounded-full bg-slate-100 px-2.5 py-0.5 text-xs font-medium text-slate-600">
                {data.sector_key}
              </span>
              <span className="text-xs text-slate-400">as of {data.as_of_date}</span>
            </div>
          )}
        </div>
        <form onSubmit={handleJump} className="flex flex-wrap items-center gap-2">
          {holdings.length > 0 && (
            <select
              value=""
              onChange={(e) => e.target.value && handleSelectTicker(e.target.value)}
              aria-label="Jump to a stock you hold"
              className="input"
            >
              <option value="">Your holdings…</option>
              {holdings.map((h) => (
                <option key={h.id} value={h.ticker}>
                  {h.ticker} — {h.name}
                </option>
              ))}
            </select>
          )}
          <TickerSearchInput
            value={jumpTicker}
            onChange={setJumpTicker}
            onSelect={handleSelectTicker}
            placeholder="Ticker or company name…"
            className="input w-56"
          />
          <button type="submit" className="btn-primary">
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
            <div className="rounded-lg border border-l-4 border-slate-200 border-l-emerald-400 bg-white p-5">
              <h3 className="text-sm font-semibold text-slate-900">Top Drivers</h3>
              {data.explanations.drivers.length === 0 ? (
                <p className="mt-2 text-xs text-slate-400">No positive drivers today.</p>
              ) : (
                <ul className="mt-2 flex flex-col gap-1.5">
                  {data.explanations.drivers.map((d) => (
                    <li key={d.factor} className="flex items-center justify-between gap-3 text-sm">
                      <span className="text-slate-700">{d.factor}</span>
                      <span className="shrink-0 font-mono text-xs font-semibold text-emerald-700">
                        +{d.contribution.toFixed(1)}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
            <div className="rounded-lg border border-l-4 border-slate-200 border-l-red-400 bg-white p-5">
              <h3 className="text-sm font-semibold text-slate-900">Top Drags</h3>
              {data.explanations.drags.length === 0 ? (
                <p className="mt-2 text-xs text-slate-400">No negative drags today.</p>
              ) : (
                <ul className="mt-2 flex flex-col gap-1.5">
                  {data.explanations.drags.map((d) => (
                    <li key={d.factor} className="flex items-center justify-between gap-3 text-sm">
                      <span className="text-slate-700">{d.factor}</span>
                      <span className="shrink-0 font-mono text-xs font-semibold text-red-700">
                        {d.contribution.toFixed(1)}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>

          {weeklyChange?.change && (
            <div className="mt-4 flex items-center gap-2 rounded-md border border-blue-100 bg-blue-50 px-3 py-2 text-sm text-blue-800">
              <span className="font-medium">Since {weeklyChange.compared_to}:</span>
              <span>
                {weeklyChange.change.factor} moved the most (
                {weeklyChange.change.delta_contribution >= 0 ? "+" : ""}
                {weeklyChange.change.delta_contribution.toFixed(1)} pts contribution)
              </span>
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
            <p className="mt-3 border-t border-slate-100 pt-3 text-xs text-slate-400">
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

      <div
        className={`mt-6 rounded-lg border border-l-4 border-slate-200 bg-white p-5 ${
          sentiment?.label ? sentimentAccentClass(sentiment.label) : "border-l-slate-200"
        }`}
      >
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-900">News &amp; Sentiment</h3>
          {sentiment?.label && (
            <span className={`rounded-full px-3 py-1 text-sm font-semibold ${sentimentBadgeClass(sentiment.label)}`}>
              {sentiment.label}
            </span>
          )}
        </div>
        {!sentiment && !sentimentLoading && (
          <>
            <p className="mt-2 text-xs text-slate-400">
              Today&apos;s real news and earnings coverage for {ticker}, read live and summarized on demand (not
              bundled into the page load, since it costs a real web search and AI call).
            </p>
            <button
              onClick={handleLoadSentiment}
              className="mt-3 rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-100"
            >
              Load news &amp; sentiment
            </button>
          </>
        )}
        {sentimentLoading && <p className="mt-2 text-sm text-slate-500">Reading today&apos;s news…</p>}
        {sentimentError && <p className="mt-2 text-sm text-red-700">{sentimentError}</p>}
        {sentiment && !sentiment.label && (
          <p className="mt-2 text-xs text-slate-400">
            Couldn&apos;t form a clear sentiment reading for {ticker} from today&apos;s coverage.
          </p>
        )}
        {sentiment?.label && sentiment.reasoning && (
          <p className="mt-2 text-sm text-slate-700">{sentiment.reasoning}</p>
        )}
      </div>

      {loggedIn && (
        <div className="mt-6 rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-semibold text-slate-900">Your Position</h3>
          {position?.owned ? (
            <div className="mt-3 grid grid-cols-2 gap-4 sm:grid-cols-4">
              <div>
                <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Shares</p>
                <p className="mt-0.5 text-lg font-semibold text-slate-900">{position.shares}</p>
              </div>
              <div>
                <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Avg Cost</p>
                <p className="mt-0.5 text-lg font-semibold text-slate-900">${position.avg_cost?.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Gain/Loss</p>
                <p
                  className={`mt-0.5 text-lg font-semibold ${
                    (position.gain_loss_pct ?? 0) >= 0 ? "text-emerald-700" : "text-red-700"
                  }`}
                >
                  {position.gain_loss_pct !== null && position.gain_loss_pct !== undefined
                    ? `${position.gain_loss_pct >= 0 ? "+" : ""}${position.gain_loss_pct.toFixed(1)}%`
                    : "—"}
                </p>
              </div>
              <div>
                <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Portfolio Weight</p>
                <p className="mt-0.5 text-lg font-semibold text-slate-900">
                  {position.weight_pct !== null && position.weight_pct !== undefined
                    ? `${position.weight_pct.toFixed(1)}%`
                    : "—"}
                </p>
              </div>
            </div>
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
              <ul className="mt-2 flex flex-col divide-y divide-slate-100">
                {signalHistory.history.slice(0, 8).map((h) => (
                  <li key={h.as_of_date} className="flex items-center justify-between gap-2 py-1.5 text-xs">
                    <span className="text-slate-500">{h.as_of_date}</span>
                    <span className="flex gap-1.5">
                      <span className={`rounded-full px-2 py-0.5 font-medium ${signalBadgeClass(h.short_signal)}`}>
                        {h.short_signal}
                      </span>
                      <span className={`rounded-full px-2 py-0.5 font-medium ${signalBadgeClass(h.long_signal)}`}>
                        {h.long_signal}
                      </span>
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
            <ul className="mt-2 flex flex-col divide-y divide-slate-100">
              {peers.peers.map((p) => (
                <li key={p.ticker}>
                  <Link
                    href={`/stock/${p.ticker}`}
                    className="flex items-center justify-between gap-2 py-1.5 text-sm hover:text-blue-700"
                  >
                    <span>
                      <span className="font-medium text-blue-700">{p.ticker}</span>
                      {p.name && <span className="ml-1.5 text-xs text-slate-400">{p.name}</span>}
                    </span>
                    <span className="shrink-0 font-mono text-xs text-slate-500">${p.market_cap_b.toFixed(1)}B</span>
                  </Link>
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
