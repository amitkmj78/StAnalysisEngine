"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";

import {
  ApiError,
  getFilingSummaries,
  getPortfolioPositions,
  getPortfolios,
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
  FilingSummariesResponse,
  Portfolio,
  PortfolioPosition,
  SignalOutcome,
  StockDetailResponse,
  StockPeersResponse,
  StockPositionResponse,
  StockPriceHistoryRange,
  StockPriceHistoryResponse,
  StockSentimentResponse,
  StockSignalHistoryResponse,
  TwoScoreFactorKey,
  TwoScoreHistoryResponse,
  TwoScoreResponse,
  TwoScoreWeeklyChangeResponse,
} from "@/lib/types";
import Sparkline from "@/components/portfolio/Sparkline";
import MetricLabel from "@/components/MetricLabel";
import FactorDrilldownModal from "@/components/stock-detail/FactorDrilldownModal";
import PriceHistoryChart from "@/components/stock-detail/PriceHistoryChart";
import TickerSearchInput from "@/components/TickerSearchInput";

function signalBadgeClass(signal: string): string {
  if (signal === "Buy") return "bg-emerald-50 text-emerald-700";
  if (signal === "Trim") return "bg-red-50 text-red-700";
  return "bg-slate-100 text-slate-600";
}

// SCR-3: "Top 8% of S&P 500" -- a friendly name for the universe_id the
// score was computed against, not the raw internal id.
function universeLabel(universeId: string): string {
  return universeId === "All" ? "S&P 500" : universeId;
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

// DET-3: hit/miss (or, for Hold, just the realized return -- it made no
// directional call to grade) next to a historical signal badge. null
// means the horizon hasn't elapsed yet, not that nothing happened.
function OutcomeMark({ outcome }: { outcome: SignalOutcome | null }) {
  if (!outcome) {
    return (
      <span className="text-slate-300" title="Horizon hasn't elapsed yet">
        …
      </span>
    );
  }
  const returnLabel = `${outcome.realized_return_pct >= 0 ? "+" : ""}${outcome.realized_return_pct.toFixed(1)}% by ${outcome.exit_date}`;
  if (outcome.outcome === "hit") {
    return (
      <span className="font-semibold text-emerald-600" title={returnLabel}>
        ✓
      </span>
    );
  }
  if (outcome.outcome === "miss") {
    return (
      <span className="font-semibold text-red-600" title={returnLabel}>
        ✗
      </span>
    );
  }
  return (
    <span className={outcome.realized_return_pct >= 0 ? "text-emerald-600" : "text-red-600"} title={`by ${outcome.exit_date}`}>
      {outcome.realized_return_pct >= 0 ? "+" : ""}
      {outcome.realized_return_pct.toFixed(1)}%
    </span>
  );
}

function confidenceClass(label: string): string {
  if (label === "high") return "text-emerald-700";
  if (label === "medium") return "text-amber-700";
  if (label === "low") return "text-red-700";
  return "text-slate-400";
}

// REG-3: a caution note, not a confidence number change -- see
// services/market_regime_service.py's module docstring for why this
// ships despite a failed validation gate, and derive_confidence's own
// docstring for why regime is attached alongside confidence rather than
// folded into it.
function regimeCautionCopy(regime: string | null): string | null {
  if (regime === "Risk-Off") {
    return "Market regime: Risk-Off — this signal's track record in risk-off conditions may differ from its usual calibration.";
  }
  if (regime === "Cautious") {
    return "Market regime: Cautious — treat this signal with extra awareness of broader market stress.";
  }
  return null;
}

function ScoreCard({
  title,
  horizon,
  score,
  signal,
  confidence,
  universePercentile,
  universeLabel,
  sectorRank,
  sectorKey,
  trend,
  regime,
}: {
  title: string;
  horizon: string;
  score: number | null;
  signal: string;
  confidence: { score: number | null; label: string };
  universePercentile: number | null;
  universeLabel: string;
  sectorRank: { rank: number; of: number } | null;
  sectorKey: string;
  trend?: { weekly_series: [string, number][]; flagged: boolean; change_pts: number | null };
  regime: string | null;
}) {
  const caution = regimeCautionCopy(regime);
  // SCR-3's exact target format: "Top 8% of S&P 500, top 3 of 22 in Semis".
  const rankParts: string[] = [];
  if (universePercentile !== null) {
    rankParts.push(`Top ${Math.max(1, Math.round(100 - universePercentile))}% of ${universeLabel}`);
  }
  if (sectorRank !== null) {
    rankParts.push(`top ${sectorRank.rank} of ${sectorRank.of} in ${sectorKey}`);
  }

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
        {rankParts.length > 0 ? rankParts.join(", ") : `No rank yet — ${sectorKey}`}
      </p>
      <p className={`mt-1 text-xs font-medium ${confidenceClass(confidence.label)}`}>
        Confidence: {confidence.label}
        {confidence.score !== null ? ` (${confidence.score.toFixed(0)})` : ""}
      </p>
      {caution && (
        <p className="mt-1.5 rounded-md bg-amber-50 px-2 py-1 text-[11px] font-medium text-amber-800">{caution}</p>
      )}
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
  const [openFactor, setOpenFactor] = useState<TwoScoreFactorKey | null>(null);
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [holdings, setHoldings] = useState<(PortfolioPosition & { portfolioName: string })[]>([]);
  const [data, setData] = useState<TwoScoreResponse | null>(null);
  const [history, setHistory] = useState<TwoScoreHistoryResponse | null>(null);
  const [weeklyChange, setWeeklyChange] = useState<TwoScoreWeeklyChangeResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [detail, setDetail] = useState<StockDetailResponse | null>(null);
  const [ownedPositions, setOwnedPositions] = useState<(StockPositionResponse & { portfolioName: string })[]>([]);
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

  // SUM-1: already computed by a daily background job -- a plain read, no
  // fresh LLM cost at request time, so (unlike sentiment above) this is
  // safe to auto-fetch with the rest of the page.
  const [filingSummaries, setFilingSummaries] = useState<FilingSummariesResponse | null>(null);

  // Every portfolio, fetched once -- both the holdings picker and "Your
  // Position" below need the full list, not just GET /portfolio/positions'
  // default (the oldest active portfolio only, when no portfolio_id is
  // given). 401 (logged out) just leaves everything empty.
  useEffect(() => {
    getPortfolios()
      .then((res) => {
        setPortfolios(res.portfolios);
        setLoggedIn(true);
      })
      .catch((err) => {
        if (err instanceof ApiError && err.status === 401) setLoggedIn(false);
        setPortfolios([]);
      });
  }, []);

  // One fetch per portfolio, merged -- runs once `portfolios` has loaded
  // (and again if it changes), not tied to the currently-viewed ticker.
  useEffect(() => {
    if (portfolios.length === 0) {
      setHoldings([]);
      return;
    }
    let cancelled = false;
    Promise.all(
      portfolios.map((p) =>
        getPortfolioPositions(p.id)
          .then((r) => r.positions.map((pos) => ({ ...pos, portfolioName: p.name })))
          .catch(() => []),
      ),
    ).then((perPortfolio) => {
      if (!cancelled) setHoldings(perPortfolio.flat());
    });
    return () => {
      cancelled = true;
    };
  }, [portfolios]);

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
    setFilingSummaries(null);
    getStockDetail(ticker).then(setDetail).catch(() => setDetail(null));
    getStockSignalHistory(ticker).then(setSignalHistory).catch(() => setSignalHistory(null));
    getStockPeers(ticker).then(setPeers).catch(() => setPeers(null));
    getFilingSummaries(ticker).then(setFilingSummaries).catch(() => setFilingSummaries(null));
  }, [ticker]);

  // Checked against every portfolio -- GET /stock/{ticker}/position with
  // no portfolio_id only checks the oldest active one, so a stock held
  // in a second or third portfolio would otherwise read as not owned.
  useEffect(() => {
    if (!ticker || portfolios.length === 0) {
      setOwnedPositions([]);
      return;
    }
    let cancelled = false;
    Promise.all(
      portfolios.map((p) =>
        getStockPosition(ticker, p.id)
          .then((pos) => (pos.owned ? [{ ...pos, portfolioName: p.name }] : []))
          .catch(() => []),
      ),
    ).then((perPortfolio) => {
      if (!cancelled) setOwnedPositions(perPortfolio.flat());
    });
    return () => {
      cancelled = true;
    };
  }, [ticker, portfolios]);

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

  // DET-4: a "change" is any day whose short/long signal differs from the
  // prior day on record -- the first day has nothing to compare against,
  // so it's never marked as one.
  const signalChanges = useMemo(() => {
    if (!signalHistory || signalHistory.history.length < 2) return [];
    const sorted = [...signalHistory.history].sort((a, b) => a.as_of_date.localeCompare(b.as_of_date));
    const changes: { date: string; label: string }[] = [];
    for (let i = 1; i < sorted.length; i++) {
      const prev = sorted[i - 1];
      const curr = sorted[i];
      const parts: string[] = [];
      if (curr.short_signal !== prev.short_signal) parts.push(`Short ${prev.short_signal}→${curr.short_signal}`);
      if (curr.long_signal !== prev.long_signal) parts.push(`Long ${prev.long_signal}→${curr.long_signal}`);
      if (parts.length > 0) changes.push({ date: curr.as_of_date, label: parts.join(", ") });
    }
    return changes;
  }, [signalHistory]);

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
                  {portfolios.length > 1 ? ` (${h.portfolioName})` : ""}
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
          pastEarnings={detail?.past_earnings ?? []}
          recentDividends={detail?.recent_dividends ?? []}
          signalChanges={signalChanges}
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
              universePercentile={data.short_universe_percentile}
              universeLabel={universeLabel(data.universe_id)}
              sectorRank={data.short_sector_rank}
              sectorKey={data.sector_key}
              trend={history?.short_term}
              regime={data.regime}
            />
            <ScoreCard
              title="Long-Term Score"
              horizon="1-3 year horizon"
              score={data.long_score}
              signal={data.long_signal}
              confidence={data.long_confidence}
              universePercentile={data.long_universe_percentile}
              universeLabel={universeLabel(data.universe_id)}
              sectorRank={data.long_sector_rank}
              sectorKey={data.sector_key}
              trend={history?.long_term}
              regime={data.regime}
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
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-900">Why these scores</h3>
              <span className="text-xs text-slate-400">Click a factor for its history</span>
            </div>
            <ul className="mt-2 flex flex-col gap-1 text-sm text-slate-700">
              {(
                [
                  "momentum",
                  "reversal",
                  "earnings_surprise",
                  "earnings_revisions",
                  "value",
                  "growth",
                  "low_vol",
                  "quality",
                ] as const
              ).map((f) => (
                <li key={f}>
                  <button
                    onClick={() => setOpenFactor(f)}
                    className="w-full rounded-md px-2 py-1 text-left hover:bg-slate-50 hover:text-blue-700"
                  >
                    {data.sentences[f]}
                  </button>
                </li>
              ))}
            </ul>
            <p className="mt-3 border-t border-slate-100 pt-3 text-xs text-slate-400">
              Rules-based composite scores from real, already-captured data — not a trained prediction model.
            </p>
          </div>
        </>
      )}

      <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-semibold text-slate-900">Key Stats</h3>
          {detail ? (
            <dl className="mt-2 grid grid-cols-2 gap-y-2 text-sm">
              <dt className="text-slate-500"><MetricLabel>Sector</MetricLabel></dt>
              <dd className="text-right text-slate-900">{detail.sector ?? "—"}</dd>
              <dt className="text-slate-500"><MetricLabel term="Forward PE">Forward P/E</MetricLabel></dt>
              <dd className="text-right text-slate-900">
                {detail.fundamentals.forward_pe !== null ? detail.fundamentals.forward_pe.toFixed(1) : "—"}
              </dd>
              <dt className="text-slate-500"><MetricLabel term="Revenue Growth %">Revenue Growth</MetricLabel></dt>
              <dd className="text-right text-slate-900">
                {detail.fundamentals.revenue_growth_pct !== null
                  ? `${detail.fundamentals.revenue_growth_pct.toFixed(1)}%`
                  : "—"}
              </dd>
              <dt className="text-slate-500"><MetricLabel term="Earnings Growth %">Earnings Growth</MetricLabel></dt>
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

      {detail && detail.past_earnings.length > 0 && (
        <div className="mt-6 rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-semibold text-slate-900">Earnings History</h3>
          {detail.typical_earnings_move && (
            <p className="mt-1 text-xs text-slate-500">
              Usually moves &plusmn;{detail.typical_earnings_move.avg_abs_move_pct}% the day after reporting (based
              on {detail.typical_earnings_move.quarters_counted} of the last 8 quarters).
            </p>
          )}
          <div className="mt-3 overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                  <th className="px-2 py-1.5">Date</th>
                  <th className="px-2 py-1.5"><MetricLabel>EPS Est.</MetricLabel></th>
                  <th className="px-2 py-1.5"><MetricLabel>EPS Actual</MetricLabel></th>
                  <th className="px-2 py-1.5"><MetricLabel>Beat / Miss</MetricLabel></th>
                  <th className="px-2 py-1.5"><MetricLabel term="Earnings Revenue">Revenue</MetricLabel></th>
                  <th className="px-2 py-1.5"><MetricLabel>Next-Day Move</MetricLabel></th>
                </tr>
              </thead>
              <tbody>
                {detail.past_earnings.map((row) => {
                  const move = detail.earnings_moves.find((m) => m.date === row.date);
                  return (
                    <tr key={row.date} className="border-b border-slate-100 last:border-0">
                      <td className="px-2 py-1.5 text-slate-700">{row.date}</td>
                      <td className="px-2 py-1.5 text-slate-700">
                        {row.eps_estimate !== null ? `$${row.eps_estimate.toFixed(2)}` : "—"}
                      </td>
                      <td className="px-2 py-1.5 text-slate-700">
                        {row.reported_eps !== null ? `$${row.reported_eps.toFixed(2)}` : "—"}
                      </td>
                      <td className="px-2 py-1.5">
                        {row.eps_beat === null ? (
                          <span className="text-slate-400">—</span>
                        ) : (
                          <span
                            className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                              row.eps_beat ? "bg-emerald-50 text-emerald-700" : "bg-red-50 text-red-700"
                            }`}
                          >
                            {row.eps_beat ? "Beat" : "Miss"}
                            {row.surprise_pct !== null
                              ? ` ${row.surprise_pct >= 0 ? "+" : ""}${row.surprise_pct.toFixed(1)}%`
                              : ""}
                          </span>
                        )}
                      </td>
                      <td className="px-2 py-1.5 text-xs text-slate-400">Not available</td>
                      <td className="px-2 py-1.5 text-slate-700">
                        {move && move.move_pct !== null ? (
                          <span className={move.move_pct >= 0 ? "text-emerald-700" : "text-red-700"}>
                            {move.move_pct >= 0 ? "+" : ""}
                            {move.move_pct.toFixed(2)}%
                          </span>
                        ) : (
                          "—"
                        )}
                        {move && (
                          <span className="ml-1 text-xs text-slate-400">
                            ({move.market_timing === "before_market" ? "BMO" : "AMC"})
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-slate-400">
            Before/after-market (BMO/AMC) is inferred from the report&apos;s timestamp, not a confirmed flag from
            the data provider. Revenue beat/miss isn&apos;t available for past quarters — no data source has a
            historical record of what was estimated at the time.
          </p>
        </div>
      )}

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

      {filingSummaries && filingSummaries.filings.length > 0 && (
        <div className="mt-6 rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-semibold text-slate-900">Filing Summary</h3>
          <div className="mt-3 flex flex-col gap-4">
            {filingSummaries.filings.map((f) => (
              <div key={f.form_type} className="border-t border-slate-100 pt-3 first:border-t-0 first:pt-0">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
                    {f.form_type} · filed {f.filing_date}
                    {f.compared_to_prior_filing ? "" : " (first on file)"}
                  </p>
                  <a
                    href={f.document_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs font-medium text-indigo-600 hover:underline"
                  >
                    View full filing on SEC.gov
                  </a>
                </div>
                <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-slate-800">{f.summary}</p>
                <p className="mt-2 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">{f.method}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {loggedIn && (
        <div className="mt-6 rounded-lg border border-slate-200 bg-white p-5">
          <h3 className="text-sm font-semibold text-slate-900">Your Position</h3>
          {ownedPositions.length > 0 ? (
            <div className="mt-3 flex flex-col divide-y divide-slate-100">
              {ownedPositions.map((pos) => (
                <div key={pos.portfolioName} className="grid grid-cols-2 gap-4 py-3 first:pt-0 last:pb-0 sm:grid-cols-5">
                  {portfolios.length > 1 && (
                    <div className="col-span-2 sm:col-span-1">
                      <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Portfolio</p>
                      <p className="mt-0.5 text-sm font-semibold text-slate-900">{pos.portfolioName}</p>
                    </div>
                  )}
                  <div>
                    <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
                      <MetricLabel>Shares</MetricLabel>
                    </p>
                    <p className="mt-0.5 text-lg font-semibold text-slate-900">
                      {pos.shares !== null && pos.shares !== undefined
                        ? Number(pos.shares.toFixed(4)).toString()
                        : "—"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
                      <MetricLabel term="Avg Cost Paid">Avg Cost</MetricLabel>
                    </p>
                    <p className="mt-0.5 text-lg font-semibold text-slate-900">${pos.avg_cost?.toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
                      <MetricLabel>Gain/Loss</MetricLabel>
                    </p>
                    <p
                      className={`mt-0.5 text-lg font-semibold ${
                        (pos.gain_loss_pct ?? 0) >= 0 ? "text-emerald-700" : "text-red-700"
                      }`}
                    >
                      {pos.gain_loss_pct !== null && pos.gain_loss_pct !== undefined
                        ? `${pos.gain_loss_pct >= 0 ? "+" : ""}${pos.gain_loss_pct.toFixed(1)}%`
                        : "—"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
                      <MetricLabel term="Weight">Portfolio Weight</MetricLabel>
                    </p>
                    <p className="mt-0.5 text-lg font-semibold text-slate-900">
                      {pos.weight_pct !== null && pos.weight_pct !== undefined ? `${pos.weight_pct.toFixed(1)}%` : "—"}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="mt-2 text-xs text-slate-400">
              You don&apos;t own {ticker} in {portfolios.length > 1 ? "any of your portfolios" : "this portfolio"}.
            </p>
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
                    <span className="flex gap-2">
                      <span className="flex items-center gap-1">
                        <span className={`rounded-full px-2 py-0.5 font-medium ${signalBadgeClass(h.short_signal)}`}>
                          {h.short_signal}
                        </span>
                        <OutcomeMark outcome={h.short_outcome} />
                      </span>
                      <span className="flex items-center gap-1">
                        <span className={`rounded-full px-2 py-0.5 font-medium ${signalBadgeClass(h.long_signal)}`}>
                          {h.long_signal}
                        </span>
                        <OutcomeMark outcome={h.long_outcome} />
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
                    <span className="min-w-0">
                      <span className="font-medium text-blue-700">{p.ticker}</span>
                      {p.name && <span className="ml-1.5 text-xs text-slate-400">{p.name}</span>}
                    </span>
                    <span className="flex shrink-0 items-center gap-1.5">
                      {p.short_signal && (
                        <span className={`rounded-full px-1.5 py-0.5 text-[10px] font-medium ${signalBadgeClass(p.short_signal)}`}>
                          S:{p.short_signal}
                        </span>
                      )}
                      {p.long_signal && (
                        <span className={`rounded-full px-1.5 py-0.5 text-[10px] font-medium ${signalBadgeClass(p.long_signal)}`}>
                          L:{p.long_signal}
                        </span>
                      )}
                      <span className="font-mono text-xs text-slate-500">${p.market_cap_b.toFixed(1)}B</span>
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-2 text-xs text-slate-400">No same-sector peers found for {ticker}.</p>
          )}
        </div>
      </div>

      {openFactor && <FactorDrilldownModal ticker={ticker} factor={openFactor} onClose={() => setOpenFactor(null)} />}
    </div>
  );
}
