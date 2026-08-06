"use client";

import { useEffect, useState } from "react";

import CurrentPriceBadge from "@/components/CurrentPriceBadge";
import TickerSearchInput from "@/components/TickerSearchInput";
import { ApiError, askMetaAgent, getChatProviders } from "@/lib/api";
import type { ChatAskResponse } from "@/lib/types";

export default function ChatPage() {
  const [providers, setProviders] = useState<string[]>([]);
  const [provider, setProvider] = useState("");
  const [ticker, setTicker] = useState("AAPL");
  const [question, setQuestion] = useState("");

  const [result, setResult] = useState<ChatAskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getChatProviders()
      .then((res) => {
        setProviders(res.providers);
        setProvider(res.providers[0] ?? "");
      })
      .catch(() => {});
  }, []);

  async function ask(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await askMetaAgent(ticker.trim().toUpperCase(), question.trim(), provider || undefined);
      setResult(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">Assistant</h1>
      <p className="mt-1 text-sm text-slate-500">
        Ask a free-form question about a ticker. The agent pulls live data and tools to answer — each answer is a
        single response, not a running conversation.
      </p>

      <form onSubmit={ask} className="mt-6 flex flex-col gap-3">
        <div className="flex flex-wrap items-end gap-3">
          <Field label="Ticker">
            <TickerSearchInput value={ticker} onChange={setTicker} className="input w-36" />
          </Field>
          <CurrentPriceBadge ticker={ticker} />
          <Field label="Provider">
            <select value={provider} onChange={(e) => setProvider(e.target.value)} className="input">
              {providers.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </Field>
        </div>

        <Field label="Question">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            className="input min-h-24 resize-y"
            placeholder="e.g. What's the near-term outlook and what would change your mind?"
          />
        </Field>

        <button type="submit" disabled={loading || !question.trim()} className="btn-primary self-start">
          {loading ? "Thinking…" : "Ask"}
        </button>
      </form>

      {loading && <p className="mt-4 text-sm text-slate-500">The agent is researching {ticker.toUpperCase()}, this can take a few seconds…</p>}
      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {result && !loading && (
        <div className="mt-6 rounded-lg border border-slate-200 bg-white p-5">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
            {result.ticker} · {result.provider}
          </p>
          <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-slate-800">{result.answer}</p>
        </div>
      )}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs font-medium text-slate-500">{label}</label>
      {children}
    </div>
  );
}
