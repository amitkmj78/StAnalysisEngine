"use client";

import { useEffect, useState } from "react";

import CurrentPriceBadge from "@/components/CurrentPriceBadge";
import PortfolioSwitcher from "@/components/PortfolioSwitcher";
import TickerSearchInput from "@/components/TickerSearchInput";
import { ApiError, askMetaAgent, getChatProviders } from "@/lib/api";
import type { ChatAskResponse } from "@/lib/types";

type Scope = "ticker" | "portfolio" | "general";

export default function ChatPage() {
  const [scope, setScope] = useState<Scope>("ticker");
  const [providers, setProviders] = useState<string[]>([]);
  const [provider, setProvider] = useState("");
  const [ticker, setTicker] = useState("AAPL");
  const [portfolioId, setPortfolioId] = useState<number | null>(null);
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
      const res = await askMetaAgent(
        scope === "portfolio"
          ? { scope, portfolio_id: portfolioId ?? undefined, question: question.trim(), provider: provider || undefined }
          : scope === "general"
            ? { scope, question: question.trim(), provider: provider || undefined }
            : { scope, ticker: ticker.trim().toUpperCase(), question: question.trim(), provider: provider || undefined }
      );
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
        Ask a free-form question about a ticker, your whole portfolio, or the market in general. Answers are built
        from the research stored in this app, with the sources it used. Each answer is a single response.
      </p>

      <div className="mt-4 flex gap-1 rounded-md border border-slate-300 bg-white p-1" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={scope === "ticker"}
          onClick={() => setScope("ticker")}
          className={`flex-1 rounded px-3 py-1.5 text-sm font-medium ${
            scope === "ticker" ? "bg-emerald-600 text-white" : "text-slate-600 hover:bg-slate-50"
          }`}
        >
          Ask about a ticker
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={scope === "portfolio"}
          onClick={() => setScope("portfolio")}
          className={`flex-1 rounded px-3 py-1.5 text-sm font-medium ${
            scope === "portfolio" ? "bg-emerald-600 text-white" : "text-slate-600 hover:bg-slate-50"
          }`}
        >
          Ask about my portfolio
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={scope === "general"}
          onClick={() => setScope("general")}
          className={`flex-1 rounded px-3 py-1.5 text-sm font-medium ${
            scope === "general" ? "bg-emerald-600 text-white" : "text-slate-600 hover:bg-slate-50"
          }`}
        >
          General question
        </button>
      </div>

      <form onSubmit={ask} className="mt-4 flex flex-col gap-3">
        <div className="flex flex-wrap items-end gap-3">
          {scope === "ticker" ? (
            <>
              <Field label="Ticker">
                <TickerSearchInput value={ticker} onChange={setTicker} className="input w-36" />
              </Field>
              <CurrentPriceBadge ticker={ticker} />
            </>
          ) : scope === "portfolio" ? (
            <PortfolioSwitcher selectedPortfolioId={portfolioId} onChange={setPortfolioId} />
          ) : (
            <p className="text-sm text-slate-500">No ticker needed. Answers use stored filings, earnings releases, and market data.</p>
          )}
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
            placeholder={
              scope === "ticker"
                ? "e.g. What's the near-term outlook and what would change your mind?"
                : "e.g. What's my biggest sector exposure, and how risky is my portfolio?"
            }
          />
        </Field>

        <button type="submit" disabled={loading || !question.trim()} className="btn-primary self-start">
          {loading ? "Thinking…" : "Ask"}
        </button>
      </form>

      {loading && (
        <p className="mt-4 text-sm text-slate-500">
          {scope === "ticker"
            ? `The agent is researching ${ticker.toUpperCase()}, this can take a few seconds…`
            : scope === "portfolio"
              ? "The agent is reviewing your portfolio, this can take a few seconds…"
              : "Searching the stored research for your question…"}
        </p>
      )}
      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {result && !loading && (
        <div className="mt-6 rounded-lg border border-slate-200 bg-white p-5">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
            {result.ticker === "PORTFOLIO" ? "Your Portfolio" : result.ticker === "GENERAL" ? "General question" : result.ticker} · {result.provider}
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
