"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import {
  ApiError,
  createConditionAlert,
  deleteConditionAlert,
  getConditionAlertFields,
  getConditionAlerts,
} from "@/lib/api";
import type { Condition, ConditionAlert, ConditionAlertFields, ConditionCombinator } from "@/lib/types";

const EMPTY_CONDITION: Condition = { field: "", op: "", value: "" };

function describeCondition(c: Condition): string {
  return `${c.field} ${c.op} ${c.value}`;
}

export default function ConditionAlertsPage() {
  const [fields, setFields] = useState<ConditionAlertFields | null>(null);
  const [alerts, setAlerts] = useState<ConditionAlert[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [ticker, setTicker] = useState("");
  const [combinator, setCombinator] = useState<ConditionCombinator>("AND");
  const [conditions, setConditions] = useState<Condition[]>([{ ...EMPTY_CONDITION }]);
  const [saving, setSaving] = useState(false);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  async function load() {
    try {
      setAlerts(await getConditionAlerts());
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load alerts.");
    }
  }

  useEffect(() => {
    getConditionAlertFields()
      .then(setFields)
      .catch(() => setError("Failed to load the available conditions."));
    load();
  }, []);

  function updateCondition(i: number, patch: Partial<Condition>) {
    setConditions((prev) => prev.map((c, idx) => (idx === i ? { ...c, ...patch } : c)));
  }

  function addCondition() {
    if (!fields || conditions.length >= fields.max_conditions) return;
    setConditions((prev) => [...prev, { ...EMPTY_CONDITION }]);
  }

  function removeCondition(i: number) {
    setConditions((prev) => (prev.length > 1 ? prev.filter((_, idx) => idx !== i) : prev));
  }

  function isCategoryField(field: string): boolean {
    return !!fields && field in fields.category_fields;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSaving(true);
    try {
      await createConditionAlert(ticker.trim().toUpperCase(), conditions, combinator);
      setTicker("");
      setConditions([{ ...EMPTY_CONDITION }]);
      setCombinator("AND");
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not save this alert.");
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete(id: number) {
    setDeletingId(id);
    try {
      await deleteConditionAlert(id);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not delete this alert.");
    } finally {
      setDeletingId(null);
    }
  }

  const triggered = (alerts ?? []).filter((a) => a.triggered_at);
  const pending = (alerts ?? []).filter((a) => !a.triggered_at);

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <div className="flex items-center justify-between">
        <h1 className="font-display text-2xl font-semibold text-slate-900">Condition Alerts</h1>
        <Link href="/alerts" className="text-sm font-medium text-slate-600 hover:underline">
          Back to Alerts
        </Link>
      </div>
      <p className="mt-1 text-sm text-slate-500">
        Combine price, indicator, score, signal, regime and earnings conditions with AND/OR — fires once, when the
        whole combination is true. Simple price/score targets still live on the{" "}
        <Link href="/stock-finder" className="underline">
          Stock Finder page
        </Link>
        .
      </p>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <form onSubmit={handleSubmit} className="mt-6 rounded-xl border border-slate-200 bg-white p-5">
        <h2 className="font-semibold text-slate-900">New alert</h2>
        <div className="mt-3">
          <label htmlFor="condition-alert-ticker" className="text-xs font-medium text-slate-500">
            Ticker
          </label>
          <input
            id="condition-alert-ticker"
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            placeholder="e.g. AAPL"
            required
            className="mt-1 block w-32 rounded-md border border-slate-300 px-2 py-1.5 text-sm"
          />
        </div>

        <div className="mt-4 flex flex-col gap-2">
          {conditions.map((c, i) => (
            <div key={i} className="flex flex-wrap items-center gap-2">
              <select
                value={c.field}
                onChange={(e) => updateCondition(i, { field: e.target.value, op: "", value: "" })}
                required
                className="rounded-md border border-slate-300 px-2 py-1.5 text-sm"
              >
                <option value="" disabled>
                  Field…
                </option>
                {fields &&
                  Object.entries(fields.numeric_fields).map(([key, label]) => (
                    <option key={key} value={key}>
                      {label}
                    </option>
                  ))}
                {fields &&
                  Object.keys(fields.category_fields).map((key) => (
                    <option key={key} value={key}>
                      {key}
                    </option>
                  ))}
              </select>

              <select
                value={c.op}
                onChange={(e) => updateCondition(i, { op: e.target.value })}
                required
                disabled={!c.field}
                className="rounded-md border border-slate-300 px-2 py-1.5 text-sm"
              >
                <option value="" disabled>
                  Op…
                </option>
                {fields &&
                  (isCategoryField(c.field) ? fields.category_ops : fields.numeric_ops).map((op) => (
                    <option key={op} value={op}>
                      {op}
                    </option>
                  ))}
              </select>

              {isCategoryField(c.field) ? (
                <select
                  value={String(c.value)}
                  onChange={(e) => updateCondition(i, { value: e.target.value })}
                  required
                  className="rounded-md border border-slate-300 px-2 py-1.5 text-sm"
                >
                  <option value="" disabled>
                    Value…
                  </option>
                  {fields?.category_fields[c.field]?.map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="number"
                  step="any"
                  value={c.value}
                  onChange={(e) => updateCondition(i, { value: e.target.value })}
                  required
                  placeholder="value"
                  className="w-28 rounded-md border border-slate-300 px-2 py-1.5 text-sm"
                />
              )}

              {conditions.length > 1 && (
                <button
                  type="button"
                  onClick={() => removeCondition(i)}
                  className="text-xs font-medium text-slate-400 hover:text-red-700"
                >
                  Remove
                </button>
              )}
            </div>
          ))}
        </div>

        <div className="mt-3 flex items-center gap-3">
          <button
            type="button"
            onClick={addCondition}
            disabled={!fields || conditions.length >= fields.max_conditions}
            className="rounded-md border border-slate-300 px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-40"
          >
            + Add condition
          </button>
          {conditions.length > 1 && (
            <label className="flex items-center gap-1.5 text-xs text-slate-600">
              Combine with
              <select
                value={combinator}
                onChange={(e) => setCombinator(e.target.value as ConditionCombinator)}
                className="rounded-md border border-slate-300 px-2 py-1 text-xs"
              >
                {(fields?.combinators ?? ["AND", "OR"]).map((op) => (
                  <option key={op} value={op}>
                    {op}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>

        <button
          type="submit"
          disabled={saving || !fields}
          className="mt-4 rounded-md bg-indigo-700 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-800 disabled:opacity-50"
        >
          {saving ? "Saving…" : "Save alert"}
        </button>
      </form>

      <div className="mt-8">
        <h2 className="font-semibold text-slate-900">Pending ({pending.length})</h2>
        {pending.length === 0 ? (
          <p className="mt-1 text-sm text-slate-500">No pending condition alerts.</p>
        ) : (
          <div className="mt-2 flex flex-col gap-2">
            {pending.map((a) => (
              <div key={a.id} className="flex items-center justify-between gap-3 rounded-md border border-slate-200 bg-white px-3 py-2 text-sm">
                <div>
                  <strong>{a.ticker}</strong> — {a.combinator} of: {a.conditions.map(describeCondition).join(a.combinator === "AND" ? " and " : " or ")}
                </div>
                <button
                  onClick={() => handleDelete(a.id)}
                  disabled={deletingId === a.id}
                  className="shrink-0 rounded-md border border-slate-300 bg-white px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                >
                  Delete
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {triggered.length > 0 && (
        <div className="mt-6">
          <h2 className="font-semibold text-slate-900">Triggered ({triggered.length})</h2>
          <div className="mt-2 flex flex-col gap-2">
            {triggered.map((a) => (
              <div key={a.id} className="flex items-center justify-between gap-3 rounded-md border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm">
                <div>
                  <strong>{a.ticker}</strong> — {a.triggered_detail}
                  <div className="text-xs text-slate-400">{a.triggered_at && new Date(a.triggered_at).toLocaleString()}</div>
                </div>
                <button
                  onClick={() => handleDelete(a.id)}
                  disabled={deletingId === a.id}
                  className="shrink-0 rounded-md border border-slate-300 bg-white px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                >
                  Delete
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
