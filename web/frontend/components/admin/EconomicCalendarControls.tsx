"use client";

import { useEffect, useState } from "react";

import {
  ApiError,
  createEconomicCalendarOverride,
  deleteEconomicCalendarOverride,
  disableEconomicCalendar,
  enableEconomicCalendar,
  getAdminSettings,
  getEconomicCalendarOverrides,
} from "@/lib/api";
import type { EconomicEventOverride, EconomicEventType } from "@/lib/types";

const EVENT_TYPES: { value: EconomicEventType; label: string }[] = [
  { value: "CPI", label: "CPI" },
  { value: "PPI", label: "PPI" },
  { value: "GDP", label: "GDP" },
  { value: "RETAIL_SALES", label: "Retail Sales" },
  { value: "JOBS_REPORT", label: "Jobs Report" },
  { value: "FOMC", label: "FOMC" },
];

const BULK_PLACEHOLDER = `FOMC | 2026-12-16 | FOMC Decision
FOMC | 2027-01-27 | FOMC Decision
FOMC | 2027-03-17 | FOMC Decision | with new economic projections`;

type ParsedBulkLine =
  | { event_type: EconomicEventType; event_date: string; title: string; note?: string }
  | { error: string; raw: string };

// The Fed (and BLS/BEA/Census, for a manual correction) publish a whole
// year of dates at once -- typing each one into the single-row form above
// is a real chore at that volume, but a full CSV-upload UI is overkill for
// ~8-12 rows a year. This splits the difference: paste several `TYPE |
// DATE | Title | optional note` lines and submit them as a batch, reusing
// the same single-row create endpoint/validation per line rather than a
// new backend code path.
function parseBulkLine(line: string): ParsedBulkLine {
  const parts = line.split("|").map((p) => p.trim());
  const [typeRaw, dateRaw, titleRaw, noteRaw] = parts;
  const type = (typeRaw ?? "").toUpperCase();
  if (!EVENT_TYPES.some((t) => t.value === type)) {
    return { error: `Unknown type "${typeRaw ?? ""}" -- must be one of ${EVENT_TYPES.map((t) => t.value).join(", ")}`, raw: line };
  }
  if (!dateRaw || !/^\d{4}-\d{2}-\d{2}$/.test(dateRaw)) {
    return { error: `Date must be YYYY-MM-DD, got "${dateRaw ?? ""}"`, raw: line };
  }
  if (!titleRaw) {
    return { error: "Title is required (third field)", raw: line };
  }
  return { event_type: type as EconomicEventType, event_date: dateRaw, title: titleRaw, note: noteRaw || undefined };
}

export default function EconomicCalendarControls() {
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [overrides, setOverrides] = useState<EconomicEventOverride[]>([]);
  const [overridesError, setOverridesError] = useState<string | null>(null);

  const [eventType, setEventType] = useState<EconomicEventType>("FOMC");
  const [eventDate, setEventDate] = useState("");
  const [title, setTitle] = useState("");
  const [note, setNote] = useState("");
  const [adding, setAdding] = useState(false);
  const [addError, setAddError] = useState<string | null>(null);

  const [bulkText, setBulkText] = useState("");
  const [bulkAdding, setBulkAdding] = useState(false);
  const [bulkResult, setBulkResult] = useState<{ added: number; failures: { raw: string; error: string }[] } | null>(
    null
  );

  async function load() {
    setError(null);
    try {
      const settings = await getAdminSettings();
      setEnabled(settings.economic_calendar_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load economic calendar status.");
    }
    try {
      const res = await getEconomicCalendarOverrides();
      setOverrides(res.overrides);
    } catch (err) {
      setOverridesError(err instanceof ApiError ? err.message : "Failed to load override dates.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleToggle() {
    setBusy(true);
    setError(null);
    try {
      const result = enabled ? await disableEconomicCalendar() : await enableEconomicCalendar();
      setEnabled(result.economic_calendar_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update setting.");
    } finally {
      setBusy(false);
    }
  }

  async function handleAddOverride() {
    if (!eventDate || !title.trim()) {
      setAddError("Date and title are required.");
      return;
    }
    setAdding(true);
    setAddError(null);
    try {
      const created = await createEconomicCalendarOverride({
        event_type: eventType,
        event_date: eventDate,
        title: title.trim(),
        note: note.trim() || undefined,
      });
      setOverrides((prev) => [...prev, created].sort((a, b) => a.event_date.localeCompare(b.event_date)));
      setEventDate("");
      setTitle("");
      setNote("");
    } catch (err) {
      setAddError(err instanceof ApiError ? err.message : "Failed to add override.");
    } finally {
      setAdding(false);
    }
  }

  async function handleBulkAdd() {
    const lines = bulkText.split("\n").map((l) => l.trim()).filter(Boolean);
    if (lines.length === 0) return;

    setBulkAdding(true);
    setBulkResult(null);
    const added: EconomicEventOverride[] = [];
    const failures: { raw: string; error: string }[] = [];

    for (const line of lines) {
      const parsed = parseBulkLine(line);
      if ("error" in parsed) {
        failures.push(parsed);
        continue;
      }
      try {
        added.push(await createEconomicCalendarOverride(parsed));
      } catch (err) {
        failures.push({ raw: line, error: err instanceof ApiError ? err.message : "Failed to add." });
      }
    }

    if (added.length > 0) {
      setOverrides((prev) => [...prev, ...added].sort((a, b) => a.event_date.localeCompare(b.event_date)));
    }
    // Leave only the failed lines in the textarea so fixing and resubmitting doesn't re-create the ones that worked.
    setBulkText(failures.map((f) => f.raw).join("\n"));
    setBulkResult({ added: added.length, failures });
    setBulkAdding(false);
  }

  async function handleDeleteOverride(id: string) {
    try {
      await deleteEconomicCalendarOverride(id);
      setOverrides((prev) => prev.filter((o) => o.id !== id));
    } catch (err) {
      setOverridesError(err instanceof ApiError ? err.message : "Failed to delete override.");
    }
  }

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="font-semibold text-slate-900 dark:text-slate-50">Economic Calendar Banner</h2>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
            Shows every logged-in user a heads-up when CPI, PPI, GDP, Retail Sales or the Jobs Report is scheduled
            within the next week — fetched live from the free FRED API. FOMC meeting dates aren&apos;t reliably
            available from FRED, so they (and any corrections to the other types) come only from the override dates
            below. Awareness only — never a buy/sell signal.
          </p>
          <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
            Requires FRED_API_KEY to be configured on the server for CPI/PPI/GDP/Retail Sales/Jobs Report. With no
            key and no override dates, the banner just has nothing to show.
          </p>
        </div>
        {enabled !== null && (
          <span
            className={`shrink-0 rounded-full px-3 py-1 text-xs font-medium ${
              enabled ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400"
            }`}
          >
            {enabled ? "On" : "Off"}
          </span>
        )}
      </div>

      {error && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <div className="mt-4">
        <button
          onClick={handleToggle}
          disabled={busy || enabled === null}
          className={`rounded-md px-4 py-2 text-sm font-medium disabled:opacity-50 ${
            enabled ? "border border-red-200 text-red-700 hover:bg-red-50" : "bg-slate-900 text-white hover:bg-slate-800"
          }`}
        >
          {busy ? "Updating…" : enabled ? "Turn off" : "Turn on"}
        </button>
      </div>

      <div className="mt-5 border-t border-slate-200 dark:border-slate-700 pt-4">
        <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-50">Override dates (required for FOMC)</h3>

        {overridesError && <p className="mt-2 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{overridesError}</p>}

        {overrides.length > 0 && (
          <ul className="mt-3 flex flex-col gap-1.5">
            {overrides.map((o) => (
              <li
                key={o.id}
                className="flex items-center justify-between gap-2 rounded-md border border-slate-200 dark:border-slate-700 px-3 py-2 text-sm"
              >
                <span>
                  <span className="font-medium">{o.event_type}</span> — {o.title} ({o.event_date})
                  {o.note ? <span className="text-slate-500 dark:text-slate-400"> · {o.note}</span> : null}
                </span>
                <button
                  onClick={() => handleDeleteOverride(o.id)}
                  className="shrink-0 text-xs font-medium text-red-600 hover:underline"
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        )}

        <div className="mt-3 flex flex-wrap items-end gap-2 rounded-md border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 p-3">
          <div className="flex flex-col gap-1">
            <label htmlFor="eco-cal-event-type" className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Type
            </label>
            <select
              id="eco-cal-event-type"
              value={eventType}
              onChange={(e) => setEventType(e.target.value as EconomicEventType)}
              className="rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 px-2 py-1.5 text-sm"
            >
              {EVENT_TYPES.map((t) => (
                <option key={t.value} value={t.value}>
                  {t.label}
                </option>
              ))}
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <label htmlFor="eco-cal-event-date" className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Date
            </label>
            <input
              id="eco-cal-event-date"
              type="date"
              value={eventDate}
              onChange={(e) => setEventDate(e.target.value)}
              className="rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 px-2 py-1.5 text-sm"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label htmlFor="eco-cal-title" className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Title
            </label>
            <input
              id="eco-cal-title"
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g. FOMC Decision"
              className="w-44 rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 px-2 py-1.5 text-sm"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label htmlFor="eco-cal-note" className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Note (optional)
            </label>
            <input
              id="eco-cal-note"
              type="text"
              value={note}
              onChange={(e) => setNote(e.target.value)}
              className="w-44 rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 px-2 py-1.5 text-sm"
            />
          </div>
          <button
            onClick={handleAddOverride}
            disabled={adding}
            className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
          >
            {adding ? "Adding…" : "Add"}
          </button>
          {addError && <p className="w-full text-xs text-red-600">{addError}</p>}
        </div>

        <div className="mt-4 rounded-md border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 p-3">
          <label htmlFor="eco-cal-bulk" className="text-xs font-medium text-slate-500 dark:text-slate-400">
            Paste multiple (one per line: <code>TYPE | YYYY-MM-DD | Title | note</code>, note optional) — for
            entering a year of FOMC dates at once instead of one at a time
          </label>
          <textarea
            id="eco-cal-bulk"
            value={bulkText}
            onChange={(e) => setBulkText(e.target.value)}
            placeholder={BULK_PLACEHOLDER}
            rows={4}
            className="mt-1 w-full rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 px-2 py-1.5 font-mono text-xs"
          />
          <button
            onClick={handleBulkAdd}
            disabled={bulkAdding || !bulkText.trim()}
            className="mt-2 rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
          >
            {bulkAdding ? "Adding…" : "Add All"}
          </button>
          {bulkResult && (
            <div className="mt-2 text-xs">
              <p className={bulkResult.failures.length > 0 ? "text-amber-700" : "text-emerald-700"}>
                Added {bulkResult.added}
                {bulkResult.failures.length > 0 ? `, ${bulkResult.failures.length} failed (left in the box above)` : "."}
              </p>
              {bulkResult.failures.length > 0 && (
                <ul className="mt-1 flex flex-col gap-0.5 text-red-600">
                  {bulkResult.failures.map((f, i) => (
                    <li key={i}>
                      <code>{f.raw}</code> — {f.error}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
