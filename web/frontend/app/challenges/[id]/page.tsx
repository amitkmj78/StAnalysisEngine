"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";

import {
  ApiError,
  getChallenge,
  getChallengeLeaderboard,
  inviteEmailToChallenge,
  inviteUserToChallenge,
  leaveChallenge,
  listDiscoverableUsers,
} from "@/lib/api";
import type { ChallengeDetail, ChallengeLeaderboardEntry, DiscoverableUser } from "@/lib/types";

function fmtPct(v: number | null): string {
  if (v === null) return "—";
  const sign = v >= 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}%`;
}

const MIN_DAYS_FOR_CONFIDENT_READING = 5;

export default function ChallengeDetailPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const challengeId = Number(params.id);

  const [challenge, setChallenge] = useState<ChallengeDetail | null | undefined>(undefined);
  const [entries, setEntries] = useState<ChallengeLeaderboardEntry[]>([]);
  const [discoverableUsers, setDiscoverableUsers] = useState<DiscoverableUser[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [leaving, setLeaving] = useState(false);
  const [copied, setCopied] = useState(false);
  const [invitingUserId, setInvitingUserId] = useState<string | null>(null);
  const [inviteEmail, setInviteEmail] = useState("");
  const [sendingEmailInvite, setSendingEmailInvite] = useState(false);

  function load() {
    getChallenge(challengeId)
      .then(setChallenge)
      .catch((err) => {
        if (err instanceof ApiError && (err.status === 403 || err.status === 404)) {
          setChallenge(null);
        } else {
          setError(err instanceof ApiError ? err.message : "Could not load this challenge.");
        }
      });
    getChallengeLeaderboard(challengeId)
      .then((res) => setEntries(res.entries))
      .catch(() => {});
    listDiscoverableUsers(challengeId)
      .then((res) => setDiscoverableUsers(res.users))
      .catch(() => {});
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [challengeId]);

  async function handleCopyCode() {
    if (!challenge) return;
    try {
      await navigator.clipboard.writeText(challenge.join_code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // clipboard access denied -- the code is still visible on screen to copy by hand
    }
  }

  async function handleInviteUser(user: DiscoverableUser) {
    setInvitingUserId(user.id);
    setError(null);
    setNote(null);
    try {
      await inviteUserToChallenge(challengeId, user.id);
      setNote(`Invited ${user.email}.`);
      setDiscoverableUsers((prev) => prev.filter((u) => u.id !== user.id));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not send that invite.");
    } finally {
      setInvitingUserId(null);
    }
  }

  async function handleInviteEmail(e: React.FormEvent) {
    e.preventDefault();
    setSendingEmailInvite(true);
    setError(null);
    setNote(null);
    try {
      await inviteEmailToChallenge(challengeId, inviteEmail.trim());
      setNote(`Sent an invite to ${inviteEmail.trim()}.`);
      setInviteEmail("");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not send that invite.");
    } finally {
      setSendingEmailInvite(false);
    }
  }

  async function handleLeave() {
    if (!window.confirm("Leave this challenge? You can rejoin later with the join code.")) return;
    setLeaving(true);
    setError(null);
    try {
      await leaveChallenge(challengeId);
      router.push("/challenges");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not leave this challenge.");
      setLeaving(false);
    }
  }

  if (challenge === undefined) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-8">
        <p className="text-sm text-slate-500">Loading…</p>
      </div>
    );
  }

  if (challenge === null) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-8">
        <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">
          Couldn&apos;t load this challenge — you may not be a member of it.
        </p>
        <Link href="/challenges" className="mt-4 inline-block text-sm text-indigo-600 hover:underline">
          ← All challenges
        </Link>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <Link href="/challenges" className="text-sm text-indigo-600 hover:underline">
        ← All challenges
      </Link>
      <div className="mt-2 flex flex-wrap items-start justify-between gap-2">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">{challenge.name}</h1>
          <p className="mt-1 text-sm text-slate-500">
            {challenge.start_date} – {challenge.end_date} · {challenge.members.length} member
            {challenge.members.length === 1 ? "" : "s"}
          </p>
        </div>
        <button
          onClick={handleLeave}
          disabled={leaving}
          className="rounded-md border border-red-200 px-3 py-1.5 text-sm text-red-700 hover:bg-red-50 disabled:opacity-50"
        >
          {leaving ? "Leaving…" : "Leave challenge"}
        </button>
      </div>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {note && <p className="mt-4 rounded-md bg-emerald-50 px-3 py-2 text-sm text-emerald-700">{note}</p>}

      <div className="mt-4 flex items-center gap-2 rounded-md bg-slate-50 px-3 py-2 text-sm">
        <span className="text-slate-500">Join code:</span>
        <span className="font-mono font-semibold tracking-wide text-slate-900">{challenge.join_code}</span>
        <button onClick={handleCopyCode} className="ml-auto text-xs font-medium text-indigo-600 hover:underline">
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>

      <div className="mt-4 rounded-lg border border-slate-200 bg-white p-5">
        <h2 className="text-sm font-semibold text-slate-900">Invite someone</h2>

        <form onSubmit={handleInviteEmail} className="mt-2 flex gap-2">
          <input
            type="email"
            value={inviteEmail}
            onChange={(e) => setInviteEmail(e.target.value)}
            placeholder="Friend's email"
            required
            className="flex-1 rounded-md border border-slate-300 px-3 py-1.5 text-sm"
          />
          <button
            type="submit"
            disabled={sendingEmailInvite}
            className="shrink-0 rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
          >
            {sendingEmailInvite ? "Sending…" : "Email invite"}
          </button>
        </form>

        {discoverableUsers.length > 0 && (
          <div className="mt-4">
            <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
              Connect with the community
            </p>
            <div className="mt-2 flex flex-col gap-1.5">
              {discoverableUsers.map((u) => (
                <div key={u.id} className="flex items-center justify-between rounded-md bg-slate-50 px-3 py-1.5">
                  <span className="text-sm text-slate-700">{u.email}</span>
                  <button
                    onClick={() => handleInviteUser(u)}
                    disabled={invitingUserId === u.id}
                    className="text-xs font-medium text-indigo-600 hover:underline disabled:opacity-50"
                  >
                    {invitingUserId === u.id ? "Inviting…" : "Invite"}
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="mt-6 overflow-x-auto rounded-lg border border-slate-200 bg-white">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
              <th className="px-4 py-3">Rank</th>
              <th className="px-4 py-3">Member</th>
              <th className="px-4 py-3 text-right">Return</th>
              <th className="px-4 py-3 text-right">Max Drawdown</th>
              <th className="px-4 py-3 text-right">Volatility</th>
              <th className="px-4 py-3 text-right">Days of Data</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((e, i) => (
              <tr key={e.email} className="border-b border-slate-100 last:border-0">
                <td className="px-4 py-3 text-slate-500">{i + 1}</td>
                <td className="px-4 py-3 text-slate-900">{e.email}</td>
                <td
                  className={`px-4 py-3 text-right font-medium ${
                    e.return_pct === null ? "text-slate-400" : e.return_pct >= 0 ? "text-emerald-600" : "text-red-600"
                  }`}
                >
                  {fmtPct(e.return_pct)}
                </td>
                <td className="px-4 py-3 text-right text-slate-700">{fmtPct(e.max_drawdown_pct)}</td>
                <td className="px-4 py-3 text-right text-slate-700">{fmtPct(e.annualized_volatility_pct)}</td>
                <td className="px-4 py-3 text-right text-slate-500">
                  {e.days_of_data}
                  {e.days_of_data > 0 && e.days_of_data < MIN_DAYS_FOR_CONFIDENT_READING && (
                    <span title="Early data -- interpret with caution" className="ml-1 text-amber-500">
                      ⚠
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-slate-400">
        Return and max drawdown are computed from each member&apos;s linked paper-trading account, from the first
        snapshot on/after this challenge started to the latest on/before today (or when it ended). A member with no
        linked account, or too little data yet, shows &quot;—&quot; rather than a 0% that isn&apos;t real.
      </p>
    </div>
  );
}
