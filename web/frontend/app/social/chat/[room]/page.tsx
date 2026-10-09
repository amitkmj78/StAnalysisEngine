"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { ApiError, getChatMessages, postChatMessage } from "@/lib/api";
import type { ChatMessage } from "@/lib/types";

const POLL_MS = 4000;

export default function SocialChatRoomPage() {
  const params = useParams<{ room: string }>();
  const room = decodeURIComponent(params.room);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [marketOpen, setMarketOpen] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  async function poll() {
    try {
      const res = await getChatMessages(room, 0);
      setMessages(res.messages);
      setMarketOpen(res.market_open);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load chat.");
    }
  }

  useEffect(() => {
    poll();
    const interval = setInterval(poll, POLL_MS);
    return () => clearInterval(interval);
  }, [room]);

  async function handleSend() {
    const text = draft.trim();
    if (!text) return;
    try {
      await postChatMessage(room, text);
      setDraft("");
      await poll();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not send.");
    }
  }

  return (
    <div className="mx-auto max-w-2xl px-4 py-8">
      <div className="flex items-center justify-between">
        <h1 className="font-display text-2xl font-semibold text-slate-900">
          #{room === "general" ? "general" : room} chat
        </h1>
        <Link href="/social/feed" className="text-sm font-medium text-slate-600 hover:underline">Feed</Link>
      </div>
      <p className="mt-1 text-sm text-slate-500">
        Open during market hours (Mon-Fri, 9:30-16:00 ET) -- not a market-holiday-aware calendar yet. Refreshes every
        few seconds; not real-time (this app has no WebSocket layer).
      </p>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <div className="mt-4 flex h-96 flex-col gap-2 overflow-y-auto rounded-xl border border-slate-200 bg-white p-4">
        {messages.map((m) => (
          <div key={m.id} className="text-sm">
            <strong className="text-slate-800">{m.display_name ?? "User"}</strong>{" "}
            <span className="text-xs text-slate-400">{new Date(m.created_at).toLocaleTimeString()}</span>
            <p className="text-slate-700">{m.body}</p>
          </div>
        ))}
        {messages.length === 0 && <p className="text-sm text-slate-500">No messages yet.</p>}
      </div>

      <div className="mt-3 flex gap-2">
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder={marketOpen === false ? "Chat is closed outside market hours" : "Message..."}
          disabled={marketOpen === false}
          className="flex-1 rounded-md border border-slate-300 px-3 py-1.5 text-sm disabled:bg-slate-50"
        />
        <button onClick={handleSend} disabled={marketOpen === false || !draft.trim()} className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50">
          Send
        </button>
      </div>
    </div>
  );
}
