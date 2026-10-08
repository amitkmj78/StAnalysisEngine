"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import {
  ApiError,
  createCommunityIdea,
  getCommunityIdeas,
  reportCommunityIdea,
  setDisplayName,
} from "@/lib/api";
import type { CommunityIdea } from "@/lib/types";

function fmtPct(v: number | null): string {
  if (v === null) return "—";
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
}

export default function CommunityFeedPage() {
  const [ideas, setIdeas] = useState<CommunityIdea[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [ticker, setTicker] = useState("");
  const [direction, setDirection] = useState<"LONG" | "SHORT">("LONG");
  const [horizonDays, setHorizonDays] = useState("10");
  const [target, setTarget] = useState("");
  const [stop, setStop] = useState("");
  const [hasPosition, setHasPosition] = useState(false);
  const [disclosureNote, setDisclosureNote] = useState("");
  const [attested, setAttested] = useState(false);
  const [saving, setSaving] = useState(false);

  const [displayNameDraft, setDisplayNameDraft] = useState("");
  const [savingDisplayName, setSavingDisplayName] = useState(false);
  const [needsDisplayName, setNeedsDisplayName] = useState(false);

  async function load() {
    try {
      const res = await getCommunityIdeas();
      setIdeas(res.ideas);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load the idea feed.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleSetDisplayName(e: React.FormEvent) {
    e.preventDefault();
    setSavingDisplayName(true);
    setError(null);
    try {
      await setDisplayName(displayNameDraft);
      setNeedsDisplayName(false);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not set that display name.");
    } finally {
      setSavingDisplayName(false);
    }
  }

  async function handlePublish(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    setError(null);
    try {
      await createCommunityIdea({
        ticker: ticker.trim().toUpperCase(),
        direction,
        horizon_days: Number(horizonDays),
        target: target ? Number(target) : null,
        stop: stop ? Number(stop) : null,
        has_position: hasPosition,
        disclosure_note: disclosureNote || null,
        attested_no_promotion: attested,
      });
      setTicker("");
      setTarget("");
      setStop("");
      setHasPosition(false);
      setDisclosureNote("");
      setAttested(false);
      await load();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "Could not publish this idea.";
      setError(msg);
      if (msg.toLowerCase().includes("display name")) setNeedsDisplayName(true);
    } finally {
      setSaving(false);
    }
  }

  async function handleReport(ideaId: number) {
    const reason = window.prompt("Why are you reporting this idea? (e.g. pump-and-dump, paid promotion, spam)");
    if (!reason || !reason.trim()) return;
    try {
      await reportCommunityIdea(ideaId, reason.trim());
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not submit the report.");
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <div className="flex items-center justify-between">
        <h1 className="font-display text-2xl font-semibold text-slate-900">Community Ideas</h1>
        <Link href="/community/leaderboard" className="text-sm font-medium text-slate-600 hover:underline">
          Leaderboard
        </Link>
      </div>
      <p className="mt-1 text-sm text-slate-500">
        Publish a timestamped, LOCKED idea (no edits, ever) — it&apos;s auto-scored against SPY once its horizon
        elapses. The app&apos;s own model publishes its own ideas too, scored by the same rules — see the{" "}
        <Link href="/community/authors/model" className="underline">
          model&apos;s own profile
        </Link>
        .
      </p>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {needsDisplayName && (
        <form onSubmit={handleSetDisplayName} className="mt-4 rounded-md border border-amber-200 bg-amber-50 p-4">
          <label htmlFor="display-name" className="text-sm font-medium text-amber-900">
            Set a public display name before publishing (2-30 characters, must be unique)
          </label>
          <div className="mt-2 flex gap-2">
            <input
              id="display-name"
              type="text"
              value={displayNameDraft}
              onChange={(e) => setDisplayNameDraft(e.target.value)}
              className="flex-1 rounded-md border border-slate-300 px-2 py-1.5 text-sm"
            />
            <button
              type="submit"
              disabled={savingDisplayName}
              className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
            >
              Save
            </button>
          </div>
        </form>
      )}

      <form onSubmit={handlePublish} className="mt-6 rounded-xl border border-slate-200 bg-white p-5">
        <h2 className="font-semibold text-slate-900">Publish an idea</h2>
        <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div>
            <label htmlFor="idea-ticker" className="text-xs font-medium text-slate-500">
              Ticker
            </label>
            <input
              id="idea-ticker"
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              required
              className="mt-1 block w-full rounded-md border border-slate-300 px-2 py-1.5 text-sm"
            />
          </div>
          <div>
            <label htmlFor="idea-direction" className="text-xs font-medium text-slate-500">
              Direction
            </label>
            <select
              id="idea-direction"
              value={direction}
              onChange={(e) => setDirection(e.target.value as "LONG" | "SHORT")}
              className="mt-1 block w-full rounded-md border border-slate-300 px-2 py-1.5 text-sm"
            >
              <option value="LONG">LONG</option>
              <option value="SHORT">SHORT</option>
            </select>
          </div>
          <div>
            <label htmlFor="idea-horizon" className="text-xs font-medium text-slate-500">
              Horizon (days)
            </label>
            <input
              id="idea-horizon"
              type="number"
              min={1}
              max={252}
              value={horizonDays}
              onChange={(e) => setHorizonDays(e.target.value)}
              required
              className="mt-1 block w-full rounded-md border border-slate-300 px-2 py-1.5 text-sm"
            />
          </div>
          <div>
            <label htmlFor="idea-target" className="text-xs font-medium text-slate-500">
              Target (optional)
            </label>
            <input
              id="idea-target"
              type="number"
              step="any"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              className="mt-1 block w-full rounded-md border border-slate-300 px-2 py-1.5 text-sm"
            />
          </div>
          <div>
            <label htmlFor="idea-stop" className="text-xs font-medium text-slate-500">
              Stop (optional)
            </label>
            <input
              id="idea-stop"
              type="number"
              step="any"
              value={stop}
              onChange={(e) => setStop(e.target.value)}
              className="mt-1 block w-full rounded-md border border-slate-300 px-2 py-1.5 text-sm"
            />
          </div>
        </div>

        <div className="mt-4 rounded-md border border-slate-200 bg-slate-50 p-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Required disclosure</p>
          <label className="mt-2 flex items-center gap-2 text-sm text-slate-700">
            <input type="checkbox" checked={hasPosition} onChange={(e) => setHasPosition(e.target.checked)} />
            I currently hold a position in this ticker
          </label>
          <label htmlFor="idea-disclosure-note" className="mt-2 block text-xs font-medium text-slate-500">
            Disclosure note (optional)
          </label>
          <input
            id="idea-disclosure-note"
            value={disclosureNote}
            onChange={(e) => setDisclosureNote(e.target.value)}
            placeholder="e.g. I hold 50 shares bought last week"
            className="mt-1 block w-full rounded-md border border-slate-300 px-2 py-1.5 text-sm"
          />
          <label className="mt-3 flex items-start gap-2 text-sm text-slate-700">
            <input
              type="checkbox"
              checked={attested}
              onChange={(e) => setAttested(e.target.checked)}
              className="mt-0.5"
              required
            />
            I am not paid to promote this ticker, and I understand pump-and-dump schemes are prohibited and
            reportable.
          </label>
        </div>

        <button
          type="submit"
          disabled={saving}
          className="mt-4 rounded-md bg-indigo-700 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-800 disabled:opacity-50"
        >
          {saving ? "Publishing…" : "Publish (locked, no edits)"}
        </button>
      </form>

      <div className="mt-8 flex flex-col gap-3">
        {(ideas ?? []).map((idea) => (
          <div key={idea.id} className="rounded-md border border-slate-200 bg-white p-4 text-sm">
            <div className="flex items-start justify-between gap-3">
              <div>
                <Link href={`/community/authors/${idea.author_user_id ?? "model"}`} className="font-semibold text-slate-900 hover:underline">
                  {idea.display_name}
                </Link>{" "}
                <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${idea.direction === "LONG" ? "bg-emerald-50 text-emerald-700" : "bg-red-50 text-red-700"}`}>
                  {idea.direction}
                </span>{" "}
                <Link href={`/stock/${idea.ticker}`} className="font-medium text-slate-800 hover:underline">
                  {idea.ticker}
                </Link>
                <span className="ml-2 text-xs text-slate-400">{idea.horizon_days}d horizon · entry ${idea.entry_price.toFixed(2)}</span>
              </div>
              <button onClick={() => handleReport(idea.id)} className="shrink-0 text-xs text-slate-400 hover:text-red-700">
                Report
              </button>
            </div>
            <div className="mt-1 text-xs text-slate-500">
              {idea.has_position ? "Discloses holding a position" : "Discloses no position"}
              {idea.disclosure_note ? ` — ${idea.disclosure_note}` : ""}
            </div>
            <div className="mt-2 text-xs">
              {idea.scored_at ? (
                <span className={idea.outcome === "hit" ? "text-emerald-700" : "text-red-700"}>
                  Scored: {idea.outcome} ({fmtPct(idea.realized_return_pct)}, excess vs SPY {fmtPct(idea.excess_vs_spy_pct)})
                </span>
              ) : (
                <span className="text-slate-400">Not yet scored — matures {idea.horizon_days} trading days after {new Date(idea.created_at).toLocaleDateString()}</span>
              )}
            </div>
          </div>
        ))}
        {ideas !== null && ideas.length === 0 && <p className="text-sm text-slate-500">No ideas published yet.</p>}
      </div>
    </div>
  );
}
