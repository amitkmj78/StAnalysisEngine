"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { ApiError, createSocialGroup, getSocialGroups, joinSocialGroup } from "@/lib/api";
import type { SocialGroup } from "@/lib/types";

export default function SocialGroupsPage() {
  const [groups, setGroups] = useState<SocialGroup[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [topic, setTopic] = useState("");
  const [ticker, setTicker] = useState("");
  const [isPrivate, setIsPrivate] = useState(false);
  const [creating, setCreating] = useState(false);

  async function load() {
    try {
      const res = await getSocialGroups();
      setGroups(res.groups);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load groups.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleCreate() {
    if (!name.trim()) return;
    setCreating(true);
    try {
      await createSocialGroup({
        name: name.trim(), topic: topic.trim() || undefined, ticker: ticker.trim() || undefined, is_private: isPrivate,
      });
      setName("");
      setTopic("");
      setTicker("");
      setIsPrivate(false);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not create group.");
    } finally {
      setCreating(false);
    }
  }

  async function handleJoin(groupId: number) {
    try {
      await joinSocialGroup(groupId);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not join group.");
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <h1 className="font-display text-2xl font-semibold text-slate-900">Groups</h1>
      <p className="mt-1 text-sm text-slate-500">
        Public or private, by topic, ticker or strategy. A private group&apos;s posts are only visible to its members.
      </p>

      <div className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
        <div className="flex flex-wrap gap-2">
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Group name" className="flex-1 rounded-md border border-slate-300 px-3 py-1.5 text-sm" />
          <input value={topic} onChange={(e) => setTopic(e.target.value)} placeholder="Topic (optional)" className="w-36 rounded-md border border-slate-300 px-2 py-1.5 text-sm" />
          <input value={ticker} onChange={(e) => setTicker(e.target.value)} placeholder="Ticker (optional)" className="w-28 rounded-md border border-slate-300 px-2 py-1.5 text-sm" />
          <label className="flex items-center gap-1.5 text-sm text-slate-600">
            <input type="checkbox" checked={isPrivate} onChange={(e) => setIsPrivate(e.target.checked)} /> Private
          </label>
          <button onClick={handleCreate} disabled={creating || !name.trim()} className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50">
            Create
          </button>
        </div>
      </div>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {groups === null && !error && <p className="mt-6 text-sm text-slate-500">Loading…</p>}

      <div className="mt-6 flex flex-col gap-3">
        {groups?.map((g) => (
          <div key={g.id} className="flex items-center justify-between rounded-xl border border-slate-200 bg-white p-4 text-sm">
            <div>
              <Link href={`/social/groups/${g.id}`} className="font-medium text-slate-800 hover:underline">
                {g.name}
              </Link>
              {g.is_private && <span className="ml-2 rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-semibold text-slate-600">PRIVATE</span>}
              <p className="text-xs text-slate-400">
                {g.topic && `#${g.topic} `}{g.ticker && `$${g.ticker} `}· {g.member_count} member{g.member_count === 1 ? "" : "s"}
              </p>
            </div>
            {!g.is_member && !g.is_private && (
              <button onClick={() => handleJoin(g.id)} className="rounded-md border border-slate-300 px-3 py-1 text-xs hover:bg-slate-50">
                Join
              </button>
            )}
            {g.is_member && <span className="text-xs text-emerald-700">Joined</span>}
          </div>
        ))}
        {groups?.length === 0 && <p className="text-sm text-slate-500">No groups yet -- create the first one.</p>}
      </div>
    </div>
  );
}
