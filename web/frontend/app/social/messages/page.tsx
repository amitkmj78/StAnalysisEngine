"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { ApiError, getDmConversations, getDmThread, getCurrentUser, sendDm } from "@/lib/api";
import type { DirectMessage, DmConversation } from "@/lib/types";

export default function SocialMessagesPage() {
  const [conversations, setConversations] = useState<DmConversation[] | null>(null);
  const [myUserId, setMyUserId] = useState<string | null>(null);
  const [activeUserId, setActiveUserId] = useState<string | null>(null);
  const [thread, setThread] = useState<DirectMessage[] | null>(null);
  const [draft, setDraft] = useState("");
  const [newRecipient, setNewRecipient] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function loadConversations() {
    try {
      const [res, me] = await Promise.all([getDmConversations(), getCurrentUser()]);
      setConversations(res.conversations);
      setMyUserId(me.id);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load messages.");
    }
  }

  useEffect(() => {
    loadConversations();
  }, []);

  async function openThread(userId: string) {
    setActiveUserId(userId);
    try {
      const res = await getDmThread(userId);
      setThread(res.messages);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load this conversation.");
    }
  }

  async function handleSend(toUserId: string) {
    const text = draft.trim();
    if (!text) return;
    try {
      await sendDm(toUserId, text);
      setDraft("");
      await openThread(toUserId);
      await loadConversations();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not send -- they may only accept messages from people they follow or allow.");
    }
  }

  async function handleStartNew() {
    const id = newRecipient.trim();
    if (!id) return;
    setNewRecipient("");
    await openThread(id);
  }

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <div className="flex items-center justify-between">
        <h1 className="font-display text-2xl font-semibold text-slate-900">Messages</h1>
        <Link href="/social/feed" className="text-sm font-medium text-slate-600 hover:underline">Feed</Link>
      </div>
      <p className="mt-1 text-sm text-slate-500">
        Only people you follow, or have explicitly allowed, can message you -- same the other way around.
      </p>
      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div className="sm:col-span-1">
          <div className="flex gap-2">
            <input value={newRecipient} onChange={(e) => setNewRecipient(e.target.value)} placeholder="User ID to message" className="flex-1 rounded-md border border-slate-300 px-2 py-1 text-xs" />
            <button onClick={handleStartNew} className="rounded-md border border-slate-300 px-2 py-1 text-xs hover:bg-slate-50">Go</button>
          </div>
          <div className="mt-3 flex flex-col gap-1">
            {conversations?.map((c) => (
              <button
                key={c.other_user_id}
                onClick={() => openThread(c.other_user_id)}
                className={`rounded-md px-2 py-2 text-left text-sm ${activeUserId === c.other_user_id ? "bg-slate-100" : "hover:bg-slate-50"}`}
              >
                <strong>{c.display_name ?? "User"}</strong>
                <p className="truncate text-xs text-slate-500">{c.last_message}</p>
              </button>
            ))}
            {conversations?.length === 0 && <p className="text-sm text-slate-500">No conversations yet.</p>}
          </div>
        </div>

        <div className="sm:col-span-2">
          {activeUserId ? (
            <div className="rounded-xl border border-slate-200 bg-white p-4">
              <div className="flex h-72 flex-col gap-2 overflow-y-auto">
                {thread?.map((m) => (
                  <div key={m.id} className={`max-w-[80%] rounded-md px-3 py-1.5 text-sm ${m.sender_user_id === myUserId ? "ml-auto bg-slate-900 text-white" : "bg-slate-100 text-slate-800"}`}>
                    {m.body}
                  </div>
                ))}
                {thread?.length === 0 && <p className="text-sm text-slate-500">Say hello.</p>}
              </div>
              <div className="mt-3 flex gap-2">
                <input
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSend(activeUserId)}
                  placeholder="Message..."
                  className="flex-1 rounded-md border border-slate-300 px-3 py-1.5 text-sm"
                />
                <button onClick={() => handleSend(activeUserId)} disabled={!draft.trim()} className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50">
                  Send
                </button>
              </div>
            </div>
          ) : (
            <p className="text-sm text-slate-500">Select a conversation, or enter a user ID to start one.</p>
          )}
        </div>
      </div>
    </div>
  );
}
