"use client";

import { useEffect, useState } from "react";

import { ApiError, getAdminIntegrations, testAdminIntegration } from "@/lib/api";
import type { AdminIntegration, AdminIntegrationTestResult } from "@/lib/types";

const CATEGORY_ORDER = ["LLM", "Market Data", "Search"];

type TestState = { status: "idle" | "running" | "done"; result?: AdminIntegrationTestResult; error?: string };

export default function IntegrationsPanel() {
  const [integrations, setIntegrations] = useState<AdminIntegration[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [tests, setTests] = useState<Record<string, TestState>>({});
  const [testingAll, setTestingAll] = useState(false);

  useEffect(() => {
    load();
  }, []);

  function load() {
    setLoadError(null);
    getAdminIntegrations()
      .then((res) => setIntegrations(res.integrations))
      .catch((err) => setLoadError(err instanceof ApiError ? err.message : "Failed to load integrations."));
  }

  async function runTest(key: string) {
    setTests((prev) => ({ ...prev, [key]: { status: "running" } }));
    try {
      const result = await testAdminIntegration(key);
      setTests((prev) => ({ ...prev, [key]: { status: "done", result } }));
    } catch (err) {
      setTests((prev) => ({
        ...prev,
        [key]: { status: "done", error: err instanceof ApiError ? err.message : "Test failed." },
      }));
    }
  }

  async function runAllConfigured() {
    if (!integrations) return;
    setTestingAll(true);
    for (const integration of integrations.filter((i) => i.configured)) {
      await runTest(integration.key);
    }
    setTestingAll(false);
  }

  if (loadError) {
    return <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{loadError}</p>;
  }
  if (!integrations) {
    return <p className="text-sm text-slate-500">Loading…</p>;
  }

  const byCategory = new Map<string, AdminIntegration[]>();
  for (const integration of integrations) {
    const list = byCategory.get(integration.category) ?? [];
    list.push(integration);
    byCategory.set(integration.category, list);
  }
  const categories = [
    ...CATEGORY_ORDER.filter((c) => byCategory.has(c)),
    ...[...byCategory.keys()].filter((c) => !CATEGORY_ORDER.includes(c)),
  ];

  return (
    <div className="flex flex-col gap-6">
      <div className="flex justify-end">
        <button
          onClick={runAllConfigured}
          disabled={testingAll}
          className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
        >
          {testingAll ? "Testing all…" : "Test all configured"}
        </button>
      </div>

      {categories.map((category) => (
        <div key={category}>
          <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">{category}</h3>
          <div className="mt-2 flex flex-col gap-2">
            {byCategory.get(category)!.map((integration) => (
              <IntegrationRow
                key={integration.key}
                integration={integration}
                state={tests[integration.key] ?? { status: "idle" }}
                onTest={() => runTest(integration.key)}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function IntegrationRow({
  integration,
  state,
  onTest,
}: {
  integration: AdminIntegration;
  state: TestState;
  onTest: () => void;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="font-medium text-slate-900">{integration.name}</span>
          <span
            className={`rounded-full px-2 py-0.5 text-xs font-medium ${
              integration.configured ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-500"
            }`}
          >
            {integration.configured ? "Configured" : "Not configured"}
          </span>
        </div>
        <button
          onClick={onTest}
          disabled={state.status === "running" || !integration.configured}
          className="rounded-md border border-slate-300 px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
        >
          {state.status === "running" ? "Testing…" : "Test"}
        </button>
      </div>
      <p className="mt-1.5 text-xs text-slate-500">{integration.note}</p>

      {state.status === "done" && (
        <div
          className={`mt-2 rounded-md px-2.5 py-2 text-xs ${
            state.error || state.result?.ok === false
              ? "bg-red-50 text-red-700"
              : "bg-emerald-50 text-emerald-800"
          }`}
        >
          {state.error ? (
            state.error
          ) : (
            <>
              {state.result!.ok ? "✓ " : "✗ "}
              {state.result!.detail}
              {state.result!.latency_ms !== null && (
                <span className="ml-2 text-slate-400">({state.result!.latency_ms}ms)</span>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
