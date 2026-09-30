"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import {
  ApiError,
  deleteAlertPreference,
  getAlertPreferences,
  getAlertSettings,
  upsertAlertPreference,
  upsertAlertSettings,
} from "@/lib/api";
import type { AlertNotificationSettings, AlertPreferenceOverride, AlertPreferenceType, AlertPreferencesResponse } from "@/lib/types";

const ALERT_TYPES: { value: AlertPreferenceType; label: string }[] = [
  { value: "signal_change", label: "Signal change" },
  { value: "earnings", label: "Earnings in 2 days" },
  { value: "cost_drop", label: "Holding down from cost" },
];

export default function AlertSettingsPage() {
  const [prefs, setPrefs] = useState<AlertPreferencesResponse | null>(null);
  const [settings, setSettings] = useState<AlertNotificationSettings | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [savingKey, setSavingKey] = useState<string | null>(null);

  // Draft state for the settings form -- only written back on Save.
  const [quietStart, setQuietStart] = useState("");
  const [quietEnd, setQuietEnd] = useState("");
  const [digestEnabled, setDigestEnabled] = useState(false);
  const [digestTime, setDigestTime] = useState("08:00");
  const [webhookEnabled, setWebhookEnabled] = useState(false);
  const [webhookUrl, setWebhookUrl] = useState("");

  const [overrideTicker, setOverrideTicker] = useState("");
  const [overrideType, setOverrideType] = useState<AlertPreferenceType>("signal_change");

  async function load() {
    setError(null);
    try {
      const [p, s] = await Promise.all([getAlertPreferences(), getAlertSettings()]);
      setPrefs(p);
      setSettings(s);
      setQuietStart(s.quiet_hours_start ?? "");
      setQuietEnd(s.quiet_hours_end ?? "");
      setDigestEnabled(s.digest_enabled);
      setDigestTime(s.digest_time);
      setWebhookEnabled(s.webhook_enabled);
      setWebhookUrl(s.webhook_url ?? "");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load alert settings.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  function globalPrefFor(alertType: AlertPreferenceType): AlertPreferenceOverride | undefined {
    return prefs?.overrides.find((o) => o.ticker === null && o.alert_type === alertType);
  }

  async function toggleGlobalChannel(alertType: AlertPreferenceType, field: "enabled" | "channel_email" | "channel_inapp") {
    const key = `global-${alertType}`;
    setSavingKey(key);
    setError(null);
    try {
      const current = globalPrefFor(alertType);
      const base = current ?? { enabled: true, channel_email: true, channel_inapp: true };
      await upsertAlertPreference({
        alert_type: alertType,
        enabled: field === "enabled" ? !base.enabled : base.enabled,
        channel_email: field === "channel_email" ? !base.channel_email : base.channel_email,
        channel_inapp: field === "channel_inapp" ? !base.channel_inapp : base.channel_inapp,
      });
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to save preference.");
    } finally {
      setSavingKey(null);
    }
  }

  async function handleDeleteOverride(id: number) {
    setSavingKey(`delete-${id}`);
    try {
      await deleteAlertPreference(id);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to remove override.");
    } finally {
      setSavingKey(null);
    }
  }

  async function handleSaveSettings(e: React.FormEvent) {
    e.preventDefault();
    setSavingKey("settings");
    setError(null);
    try {
      const saved = await upsertAlertSettings({
        quiet_hours_start: quietStart || null,
        quiet_hours_end: quietEnd || null,
        digest_enabled: digestEnabled,
        digest_time: digestTime,
        webhook_enabled: webhookEnabled,
        webhook_url: webhookUrl || null,
      });
      setSettings(saved);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to save settings.");
    } finally {
      setSavingKey(null);
    }
  }

  const tickerOverrides = (prefs?.overrides ?? []).filter((o) => o.ticker !== null);

  async function handleAddOverride(e: React.FormEvent) {
    e.preventDefault();
    const ticker = overrideTicker.trim().toUpperCase();
    if (!ticker) return;
    setSavingKey("add-override");
    setError(null);
    try {
      await upsertAlertPreference({
        alert_type: overrideType,
        ticker,
        enabled: false, // muting one ticker's alert type is the common case; toggle back on isn't exposed yet, remove+re-add instead
        channel_email: true,
        channel_inapp: true,
      });
      setOverrideTicker("");
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to add override.");
    } finally {
      setSavingKey(null);
    }
  }

  return (
    <div className="mx-auto max-w-2xl px-4 py-8">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold text-slate-900">Alert Settings</h1>
        <Link href="/alerts" className="text-sm font-medium text-slate-600 hover:underline">
          Inbox
        </Link>
      </div>
      <p className="mt-1 text-sm text-slate-500">
        Choose which alert types email you (in-app delivery always happens regardless), set quiet hours, and
        optionally receive one daily digest instead of individual emails.
      </p>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
      {(!prefs || !settings) && !error && <p className="mt-6 text-sm text-slate-500">Loading…</p>}

      {prefs && settings && (
        <>
          <div className="mt-6 overflow-x-auto rounded-lg border border-slate-200 bg-white">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                  <th className="px-3 py-2">Alert type</th>
                  <th className="px-3 py-2 text-center">Enabled</th>
                  <th className="px-3 py-2 text-center">Email</th>
                  <th className="px-3 py-2 text-center">In-app</th>
                </tr>
              </thead>
              <tbody>
                {ALERT_TYPES.map(({ value, label }) => {
                  const pref = globalPrefFor(value);
                  const enabled = pref?.enabled ?? true;
                  const email = pref?.channel_email ?? true;
                  const inapp = pref?.channel_inapp ?? true;
                  const busy = savingKey === `global-${value}`;
                  return (
                    <tr key={value} className="border-b border-slate-100 last:border-0">
                      <td className="px-3 py-2 font-medium text-slate-800">{label}</td>
                      <td className="px-3 py-2 text-center">
                        <input
                          type="checkbox"
                          checked={enabled}
                          disabled={busy}
                          onChange={() => toggleGlobalChannel(value, "enabled")}
                        />
                      </td>
                      <td className="px-3 py-2 text-center">
                        <input
                          type="checkbox"
                          checked={email}
                          disabled={busy || !enabled}
                          onChange={() => toggleGlobalChannel(value, "channel_email")}
                        />
                      </td>
                      <td className="px-3 py-2 text-center">
                        <input
                          type="checkbox"
                          checked={inapp}
                          disabled={busy || !enabled}
                          onChange={() => toggleGlobalChannel(value, "channel_inapp")}
                        />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="mt-6">
            <h2 className="text-sm font-semibold text-slate-800">Per-stock overrides</h2>
            <p className="mt-1 text-xs text-slate-500">
              Mute one alert type for a specific ticker — overrides the global setting above for that stock only.
            </p>
            <form onSubmit={handleAddOverride} className="mt-2 flex flex-wrap items-end gap-2">
              <input
                value={overrideTicker}
                onChange={(e) => setOverrideTicker(e.target.value)}
                placeholder="Ticker"
                className="w-28 rounded-md border border-slate-300 px-3 py-2 text-sm uppercase"
              />
              <select
                value={overrideType}
                onChange={(e) => setOverrideType(e.target.value as AlertPreferenceType)}
                className="rounded-md border border-slate-300 px-3 py-2 text-sm"
              >
                {ALERT_TYPES.map((t) => (
                  <option key={t.value} value={t.value}>
                    Mute {t.label.toLowerCase()}
                  </option>
                ))}
              </select>
              <button
                type="submit"
                disabled={savingKey === "add-override" || !overrideTicker.trim()}
                className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
              >
                Add
              </button>
            </form>

            {tickerOverrides.length > 0 && (
              <div className="mt-3 flex flex-col gap-1">
                {tickerOverrides.map((o) => (
                  <div
                    key={o.id}
                    className="flex items-center justify-between rounded-md border border-slate-200 bg-white px-3 py-2 text-sm"
                  >
                    <span>
                      <strong>{o.ticker}</strong> — {ALERT_TYPES.find((t) => t.value === o.alert_type)?.label ?? o.alert_type}:{" "}
                      {o.enabled ? (o.channel_email ? "email on" : "email off, in-app only") : "disabled"}
                    </span>
                    <button
                      onClick={() => handleDeleteOverride(o.id)}
                      disabled={savingKey === `delete-${o.id}`}
                      className="rounded-md border border-slate-300 bg-white px-2 py-0.5 text-xs text-slate-600 hover:bg-slate-50 disabled:opacity-50"
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <form onSubmit={handleSaveSettings} className="mt-8 rounded-lg border border-slate-200 bg-white p-5">
            <h2 className="text-sm font-semibold text-slate-800">Quiet hours &amp; digest</h2>
            <p className="mt-1 text-xs text-slate-500">
              During quiet hours (or any time digest mode is on), alerts queue and arrive together instead of
              immediately.
            </p>

            <div className="mt-4 flex flex-wrap items-end gap-4">
              <label className="flex flex-col gap-1 text-xs font-medium text-slate-500">
                Quiet hours start
                <input
                  type="time"
                  value={quietStart}
                  onChange={(e) => setQuietStart(e.target.value)}
                  className="rounded-md border border-slate-300 px-3 py-2 text-sm"
                />
              </label>
              <label className="flex flex-col gap-1 text-xs font-medium text-slate-500">
                Quiet hours end
                <input
                  type="time"
                  value={quietEnd}
                  onChange={(e) => setQuietEnd(e.target.value)}
                  className="rounded-md border border-slate-300 px-3 py-2 text-sm"
                />
              </label>
            </div>

            <div className="mt-4 flex flex-wrap items-end gap-4">
              <label className="flex items-center gap-2 text-sm text-slate-700">
                <input type="checkbox" checked={digestEnabled} onChange={(e) => setDigestEnabled(e.target.checked)} />
                Send one daily digest instead of individual emails
              </label>
              {digestEnabled && (
                <label className="flex flex-col gap-1 text-xs font-medium text-slate-500">
                  Digest time
                  <input
                    type="time"
                    value={digestTime}
                    onChange={(e) => setDigestTime(e.target.value)}
                    className="rounded-md border border-slate-300 px-3 py-2 text-sm"
                  />
                </label>
              )}
            </div>

            <h2 className="mt-6 text-sm font-semibold text-slate-800">Webhook (for power users)</h2>
            <p className="mt-1 text-xs text-slate-500">
              {settings.has_webhook_secret
                ? "A signing secret has been generated for your webhook."
                : "A signing secret is generated the first time you enable this."}
            </p>
            <div className="mt-3 flex flex-wrap items-end gap-4">
              <label className="flex items-center gap-2 text-sm text-slate-700">
                <input type="checkbox" checked={webhookEnabled} onChange={(e) => setWebhookEnabled(e.target.checked)} />
                Enabled
              </label>
              <label className="flex flex-1 min-w-[240px] flex-col gap-1 text-xs font-medium text-slate-500">
                URL
                <input
                  type="url"
                  value={webhookUrl}
                  onChange={(e) => setWebhookUrl(e.target.value)}
                  placeholder="https://example.com/webhook"
                  className="rounded-md border border-slate-300 px-3 py-2 text-sm"
                />
              </label>
            </div>

            <button
              type="submit"
              disabled={savingKey === "settings"}
              className="mt-5 rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
            >
              {savingKey === "settings" ? "Saving…" : "Save"}
            </button>
          </form>
        </>
      )}
    </div>
  );
}
