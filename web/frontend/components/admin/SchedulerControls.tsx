"use client";

import { useEffect, useState } from "react";

import {
  ApiError,
  disableEarningsReleaseSummaries,
  disableEveningRecap,
  disableFilingSummaries,
  disableNews8k,
  disableMarketRegime,
  disableMorningBrief,
  disablePaperAccountEquityCapture,
  disableVerifyPredictions,
  enableEarningsReleaseSummaries,
  enableEveningRecap,
  enableFilingSummaries,
  enableNews8k,
  enableMarketRegime,
  enableMorningBrief,
  enablePaperAccountEquityCapture,
  enableChallengeNotifications,
  disableChallengeNotifications,
  enableVerifyPredictions,
  getAdminSettings,
} from "@/lib/api";

function JobToggleCard({
  title,
  description,
  enabled,
  busy,
  onToggle,
}: {
  title: string;
  description: string;
  enabled: boolean | null;
  busy: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="font-semibold text-slate-900">{title}</h2>
          <p className="mt-1 text-sm text-slate-600">{description}</p>
        </div>
        {enabled !== null && (
          <span
            className={`shrink-0 rounded-full px-3 py-1 text-xs font-medium ${
              enabled ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-500"
            }`}
          >
            {enabled ? "Enabled" : "Disabled"}
          </span>
        )}
      </div>

      <button
        onClick={onToggle}
        disabled={busy || enabled === null}
        className={`mt-4 rounded-md px-4 py-2 text-sm font-medium disabled:opacity-50 ${
          enabled
            ? "border border-red-200 text-red-700 hover:bg-red-50"
            : "bg-slate-900 text-white hover:bg-slate-800"
        }`}
      >
        {busy ? "Updating…" : enabled ? "Disable" : "Enable"}
      </button>
    </div>
  );
}

export default function SchedulerControls() {
  const [verifyEnabled, setVerifyEnabled] = useState<boolean | null>(null);
  const [regimeEnabled, setRegimeEnabled] = useState<boolean | null>(null);
  const [filingSummariesEnabled, setFilingSummariesEnabled] = useState<boolean | null>(null);
  const [news8kEnabled, setNews8kEnabled] = useState<boolean | null>(null);
  const [earningsReleaseSummariesEnabled, setEarningsReleaseSummariesEnabled] = useState<boolean | null>(null);
  const [eveningRecapEnabled, setEveningRecapEnabled] = useState<boolean | null>(null);
  const [morningBriefEnabled, setMorningBriefEnabled] = useState<boolean | null>(null);
  const [paperAccountEquityCaptureEnabled, setPaperAccountEquityCaptureEnabled] = useState<boolean | null>(null);
  const [challengeNotificationsEnabled, setChallengeNotificationsEnabled] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyKey, setBusyKey] = useState<string | null>(null);

  async function load() {
    setError(null);
    try {
      const settings = await getAdminSettings();
      setVerifyEnabled(settings.verify_predictions_enabled);
      setRegimeEnabled(settings.market_regime_enabled);
      setFilingSummariesEnabled(settings.filing_summaries_enabled);
      setNews8kEnabled(settings.news_8k_enabled);
      setEarningsReleaseSummariesEnabled(settings.earnings_release_summaries_enabled);
      setEveningRecapEnabled(settings.evening_recap_enabled);
      setMorningBriefEnabled(settings.morning_brief_enabled);
      setPaperAccountEquityCaptureEnabled(settings.paper_account_equity_capture_enabled);
      setChallengeNotificationsEnabled(settings.challenge_notifications_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load scheduler settings.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleToggleVerify() {
    setBusyKey("verify");
    setError(null);
    try {
      const result = verifyEnabled ? await disableVerifyPredictions() : await enableVerifyPredictions();
      setVerifyEnabled(result.verify_predictions_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update scheduler setting.");
    } finally {
      setBusyKey(null);
    }
  }

  async function handleToggleRegime() {
    setBusyKey("regime");
    setError(null);
    try {
      const result = regimeEnabled ? await disableMarketRegime() : await enableMarketRegime();
      setRegimeEnabled(result.market_regime_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update scheduler setting.");
    } finally {
      setBusyKey(null);
    }
  }

  async function handleToggleFilingSummaries() {
    setBusyKey("filing-summaries");
    setError(null);
    try {
      const result = filingSummariesEnabled ? await disableFilingSummaries() : await enableFilingSummaries();
      setFilingSummariesEnabled(result.filing_summaries_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update scheduler setting.");
    } finally {
      setBusyKey(null);
    }
  }

  async function handleToggleNews8k() {
    setBusyKey("news-8k");
    setError(null);
    try {
      const result = news8kEnabled ? await disableNews8k() : await enableNews8k();
      setNews8kEnabled(result.news_8k_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update scheduler setting.");
    } finally {
      setBusyKey(null);
    }
  }

  async function handleToggleEarningsReleaseSummaries() {
    setBusyKey("earnings-release-summaries");
    setError(null);
    try {
      const result = earningsReleaseSummariesEnabled
        ? await disableEarningsReleaseSummaries()
        : await enableEarningsReleaseSummaries();
      setEarningsReleaseSummariesEnabled(result.earnings_release_summaries_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update scheduler setting.");
    } finally {
      setBusyKey(null);
    }
  }

  async function handleToggleEveningRecap() {
    setBusyKey("evening-recap");
    setError(null);
    try {
      const result = eveningRecapEnabled ? await disableEveningRecap() : await enableEveningRecap();
      setEveningRecapEnabled(result.evening_recap_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update scheduler setting.");
    } finally {
      setBusyKey(null);
    }
  }

  async function handleToggleMorningBrief() {
    setBusyKey("morning-brief");
    setError(null);
    try {
      const result = morningBriefEnabled ? await disableMorningBrief() : await enableMorningBrief();
      setMorningBriefEnabled(result.morning_brief_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update scheduler setting.");
    } finally {
      setBusyKey(null);
    }
  }

  async function handleToggleChallengeNotifications() {
    setBusyKey("challenge-notifications");
    setError(null);
    try {
      const result = challengeNotificationsEnabled
        ? await disableChallengeNotifications()
        : await enableChallengeNotifications();
      setChallengeNotificationsEnabled(result.challenge_notifications_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update scheduler setting.");
    } finally {
      setBusyKey(null);
    }
  }

  async function handleTogglePaperAccountEquityCapture() {
    setBusyKey("paper-account-equity-capture");
    setError(null);
    try {
      const result = paperAccountEquityCaptureEnabled
        ? await disablePaperAccountEquityCapture()
        : await enablePaperAccountEquityCapture();
      setPaperAccountEquityCaptureEnabled(result.paper_account_equity_capture_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update scheduler setting.");
    } finally {
      setBusyKey(null);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      {error && <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <JobToggleCard
        title="Auto-Verify Saved Predictions"
        description="Background job that checks saved predictions against real prices every 15 minutes. Disabling it stops future runs — saved predictions already verified stay as they are, and new ones simply won't be checked until this is turned back on."
        enabled={verifyEnabled}
        busy={busyKey === "verify"}
        onToggle={handleToggleVerify}
      />

      <JobToggleCard
        title="Market Regime Banner"
        description="Daily job (weekdays 18:10 ET) that computes the site-wide regime reading shown in the banner on every page. This wires up a scoring engine that failed its own release-gate backtest three times — see the banner's own disclosure for the full history. Disabling it stops new daily readings; the banner falls back to showing nothing until re-enabled."
        enabled={regimeEnabled}
        busy={busyKey === "regime"}
        onToggle={handleToggleRegime}
      />

      <JobToggleCard
        title="SEC Filing Summaries"
        description="Daily job (weekdays 20:00 ET) that checks SEC EDGAR for new 10-K/10-Q filings for every ticker any user holds or watchlists, and summarizes what changed vs the prior filing. Hits a real external government API plus LLM cost on a schedule. Disabling it stops new filing checks; existing summaries stay visible."
        enabled={filingSummariesEnabled}
        busy={busyKey === "filing-summaries"}
        onToggle={handleToggleFilingSummaries}
      />

      <JobToggleCard
        title="News from SEC 8-K filings"
        description="Hourly job (at 15 minutes past the hour) that stores the last 30 days of SEC 8-K filings for every ticker any user holds or watchlists, so each stock page can list its recent company announcements. Calls SEC EDGAR on a schedule, so it stays off until enabled. Disabling it stops new fetches; stored filings stay visible."
        enabled={news8kEnabled}
        busy={busyKey === "news-8k"}
        onToggle={handleToggleNews8k}
      />

      <JobToggleCard
        title="Earnings Release Summaries"
        description="Daily job (weekdays 21:00 ET, after filing summaries) that checks SEC EDGAR for each ticker's new 8-K earnings press release and summarizes it — the press release only, not a transcript of the call (EDGAR doesn't have one, so analyst Q&A is never covered). Shares the same daily LLM provider quota as Filing Summaries, so enabling both increases the odds either one hits its daily limit. Disabling it stops new checks; existing summaries stay visible."
        enabled={earningsReleaseSummariesEnabled}
        busy={busyKey === "earnings-release-summaries"}
        onToggle={handleToggleEarningsReleaseSummaries}
      />

      <JobToggleCard
        title="Evening Recap"
        description="Daily job (weekdays 16:30 ET) that emails every user with at least one position a same-day recap: portfolio vs. SPY today, plus the day's top contributors/detractors by holding. Pure arithmetic, no LLM cost. Delivery honors each user's quiet hours and digest preference like any other alert. Disabling it stops future recaps."
        enabled={eveningRecapEnabled}
        busy={busyKey === "evening-recap"}
        onToggle={handleToggleEveningRecap}
      />

      <JobToggleCard
        title="Morning Brief"
        description="Daily job (weekdays 07:00 ET) that emails every user with at least one position a 5-section brief meant to read in about 2 minutes: overnight moves, signal changes, earnings today, market regime, and top news on their 3 most-moved/changed tickers. The top-news section makes real LLM calls (capped at 3/user/day), sharing the same daily provider quota as Filing/Earnings-Release Summaries. Disabling it stops future briefs."
        enabled={morningBriefEnabled}
        busy={busyKey === "morning-brief"}
        onToggle={handleToggleMorningBrief}
      />

      <JobToggleCard
        title="Paper-Account Equity Capture"
        description="Daily job (weekdays 16:20 ET) that records one equity snapshot per linked paper-trading account, using that user's own stored Alpaca credentials. This is what Challenge leaderboards use to compute return and risk over a challenge's date range -- without it, every leaderboard row shows 'not enough data yet.' Disabling it stops new snapshots; past ones stay usable."
        enabled={paperAccountEquityCaptureEnabled}
        busy={busyKey === "paper-account-equity-capture"}
        onToggle={handleTogglePaperAccountEquityCapture}
      />

      <JobToggleCard
        title="Challenge Notifications"
        description="Daily job (weekdays 16:40 ET) that emails challenge members their rank, alerts them when someone passes them, reminds them two days before a challenge ends, and posts the final result the day after. Each message goes out once per member per day and respects their quiet hours and alert preferences. Disabling it stops new messages."
        enabled={challengeNotificationsEnabled}
        busy={busyKey === "challenge-notifications"}
        onToggle={handleToggleChallengeNotifications}
      />
    </div>
  );
}
