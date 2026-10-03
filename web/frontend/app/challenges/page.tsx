"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";

import {
  acceptChallengeInvite,
  ApiError,
  createChallenge,
  declineChallengeInvite,
  getDiscoverability,
  joinChallenge,
  listChallenges,
  listMyChallengeInvites,
  setDiscoverability,
} from "@/lib/api";
import type { Challenge, ChallengeInvite } from "@/lib/types";

export default function ChallengesPage() {
  const searchParams = useSearchParams();

  const [challenges, setChallenges] = useState<Challenge[] | undefined>(undefined); // undefined = loading
  const [invites, setInvites] = useState<ChallengeInvite[]>([]);
  const [discoverable, setDiscoverableState] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);

  const [name, setName] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [scoring, setScoring] = useState("return");
  const [includeModel, setIncludeModel] = useState(false);
  const [creating, setCreating] = useState(false);

  const [joinCode, setJoinCode] = useState("");
  const [joining, setJoining] = useState(false);
  const [respondingTo, setRespondingTo] = useState<number | null>(null);
  const [togglingDiscoverable, setTogglingDiscoverable] = useState(false);

  function load() {
    listChallenges()
      .then((res) => setChallenges(res.challenges))
      .catch((err) => setError(err instanceof ApiError ? err.message : "Could not load your challenges."));
    listMyChallengeInvites()
      .then((res) => setInvites(res.invites))
      .catch(() => {});
    getDiscoverability()
      .then((res) => setDiscoverableState(res.discoverable))
      .catch(() => {});
  }

  useEffect(() => {
    load();
    const joinParam = searchParams.get("join");
    if (joinParam) setJoinCode(joinParam.toUpperCase());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    setCreating(true);
    setError(null);
    setNote(null);
    try {
      const res = await createChallenge({
        name: name.trim(),
        start_date: startDate || undefined,
        end_date: endDate || undefined,
        scoring,
        include_quant_model: includeModel,
      });
      setNote(`Created "${res.name}". Share join code ${res.join_code} with friends.`);
      setName("");
      setStartDate("");
      setEndDate("");
      setScoring("return");
      setIncludeModel(false);
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not create this challenge.");
    } finally {
      setCreating(false);
    }
  }

  async function handleJoin(e: React.FormEvent) {
    e.preventDefault();
    setJoining(true);
    setError(null);
    setNote(null);
    try {
      const res = await joinChallenge(joinCode.trim());
      setNote(`Joined "${res.name}".`);
      setJoinCode("");
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not join with that code.");
    } finally {
      setJoining(false);
    }
  }

  async function handleAcceptInvite(invite: ChallengeInvite) {
    setRespondingTo(invite.id);
    setError(null);
    setNote(null);
    try {
      await acceptChallengeInvite(invite.id);
      setNote(`Joined "${invite.challenge_name}".`);
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not accept this invite.");
    } finally {
      setRespondingTo(null);
    }
  }

  async function handleDeclineInvite(invite: ChallengeInvite) {
    setRespondingTo(invite.id);
    setError(null);
    try {
      await declineChallengeInvite(invite.id);
      setInvites((prev) => prev.filter((i) => i.id !== invite.id));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not decline this invite.");
    } finally {
      setRespondingTo(null);
    }
  }

  async function handleToggleDiscoverable() {
    if (discoverable === null) return;
    setTogglingDiscoverable(true);
    setError(null);
    try {
      const res = await setDiscoverability(!discoverable);
      setDiscoverableState(res.discoverable);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not update this setting.");
    } finally {
      setTogglingDiscoverable(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">Challenges</h1>
          <p className="mt-1 text-sm text-slate-500">
            Compete with friends using your paper-trading account. Leaderboards show both return and risk, not just
            who&apos;s up the most.{" "}
            <Link href="/portfolio/paper-trading" className="text-indigo-600 hover:underline">
              Link a paper account
            </Link>{" "}
            first if you haven&apos;t.
          </p>
        </div>
        <Link href="/portfolio" className="text-sm font-medium text-slate-600 hover:underline">
          ← Back to Portfolio
        </Link>
      </div>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {note && <p className="mt-4 rounded-md bg-emerald-50 px-3 py-2 text-sm text-emerald-700">{note}</p>}

      <div className="mt-4 flex items-center justify-between gap-3 rounded-md bg-slate-50 px-4 py-3">
        <div>
          <p className="text-sm font-medium text-slate-900">Discoverable to connect with the community</p>
          <p className="text-xs text-slate-500">
            When on, other members can find and invite you to their challenges.
          </p>
        </div>
        <button
          onClick={handleToggleDiscoverable}
          disabled={discoverable === null || togglingDiscoverable}
          className={`shrink-0 rounded-full px-3 py-1 text-xs font-medium disabled:opacity-50 ${
            discoverable ? "bg-emerald-50 text-emerald-700" : "bg-slate-200 text-slate-600"
          }`}
        >
          {togglingDiscoverable ? "Updating…" : discoverable ? "On" : "Off"}
        </button>
      </div>

      {invites.length > 0 && (
        <div className="mt-4 flex flex-col gap-2">
          <h2 className="text-sm font-semibold text-slate-900">Invitations</h2>
          {invites.map((invite) => (
            <div
              key={invite.id}
              className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-indigo-200 bg-indigo-50 p-4"
            >
              <p className="text-sm text-slate-900">
                <span className="font-medium">{invite.invited_by_label}</span> invited you to{" "}
                <span className="font-medium">{invite.challenge_name}</span>
              </p>
              <div className="flex gap-2">
                <button
                  onClick={() => handleAcceptInvite(invite)}
                  disabled={respondingTo === invite.id}
                  className="rounded-md bg-slate-900 px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50"
                >
                  Accept
                </button>
                <button
                  onClick={() => handleDeclineInvite(invite)}
                  disabled={respondingTo === invite.id}
                  className="rounded-md border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-700 disabled:opacity-50"
                >
                  Decline
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2">
        <form onSubmit={handleCreate} className="rounded-lg border border-slate-200 bg-white p-5">
          <h2 className="text-sm font-semibold text-slate-900">Create a challenge</h2>
          <p className="mt-1 text-xs text-slate-500">Defaults to the current calendar month if left blank.</p>
          <div className="mt-3 flex flex-col gap-3">
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Challenge name"
              required
              className="rounded-md border border-slate-300 px-3 py-1.5 text-sm"
            />
            <div className="flex gap-2">
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="flex-1 rounded-md border border-slate-300 px-3 py-1.5 text-sm"
              />
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="flex-1 rounded-md border border-slate-300 px-3 py-1.5 text-sm"
              />
            </div>
            <label className="flex flex-col gap-1 text-xs text-slate-500">
              Ranked by
              <select
                value={scoring}
                onChange={(e) => setScoring(e.target.value)}
                className="rounded-md border border-slate-300 px-3 py-1.5 text-sm text-slate-900"
              >
                <option value="return">Raw return</option>
                <option value="excess_spy">Excess return vs S&amp;P 500</option>
                <option value="sharpe">Sharpe ratio (return vs volatility)</option>
                <option value="sortino">Sortino ratio (return vs downside)</option>
                <option value="calmar">Calmar (return / max drawdown)</option>
              </select>
            </label>
            <label className="flex items-start gap-2 text-xs text-slate-600">
              <input
                type="checkbox"
                checked={includeModel}
                onChange={(e) => setIncludeModel(e.target.checked)}
                className="mt-0.5 h-4 w-4 rounded border-slate-300"
              />
              <span>
                Include the quant model as a member. It appears as a hypothetical daily portfolio of its published picks, not a live account.
              </span>
            </label>
            <button
              type="submit"
              disabled={creating}
              className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
            >
              {creating ? "Creating…" : "Create"}
            </button>
          </div>
        </form>

        <form onSubmit={handleJoin} className="rounded-lg border border-slate-200 bg-white p-5">
          <h2 className="text-sm font-semibold text-slate-900">Join with a code</h2>
          <p className="mt-1 text-xs text-slate-500">Ask whoever created the challenge for their 6-character code.</p>
          <div className="mt-3 flex flex-col gap-3">
            <input
              value={joinCode}
              onChange={(e) => setJoinCode(e.target.value.toUpperCase())}
              placeholder="Join code"
              required
              maxLength={6}
              className="rounded-md border border-slate-300 px-3 py-1.5 text-sm uppercase tracking-wide"
            />
            <button
              type="submit"
              disabled={joining}
              className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
            >
              {joining ? "Joining…" : "Join"}
            </button>
          </div>
        </form>
      </div>

      <div className="mt-6">
        <h2 className="text-sm font-semibold text-slate-900">Your challenges</h2>
        {challenges === undefined ? (
          <p className="mt-2 text-sm text-slate-500">Loading…</p>
        ) : challenges.length === 0 ? (
          <p className="mt-2 text-sm text-slate-500">You haven&apos;t created or joined a challenge yet.</p>
        ) : (
          <div className="mt-3 flex flex-col gap-2">
            {challenges.map((c) => (
              <Link
                key={c.id}
                href={`/challenges/${c.id}`}
                className="flex items-center justify-between rounded-lg border border-slate-200 bg-white p-4 hover:border-slate-300 hover:bg-slate-50"
              >
                <div>
                  <p className="font-medium text-slate-900">{c.name}</p>
                  <p className="text-xs text-slate-500">
                    {c.start_date} – {c.end_date} · {c.member_count} member{c.member_count === 1 ? "" : "s"}
                  </p>
                </div>
                <span className="text-sm text-indigo-600">View →</span>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
