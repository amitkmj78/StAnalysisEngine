"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import {
  ApiError, getBlockedUsers, getCurrentUser, getMutes, muteContent, setMyThemeColor, setPortfolioSharing,
  unblockUser, unmute, updateSocialProfile,
} from "@/lib/api";
import type { ExperienceLevel } from "@/lib/types";
import { applyThemeAccent, applyThemeMode, getStoredThemeMode, THEME_PALETTE, type ThemeMode } from "@/lib/theme";

const MODES: { value: ThemeMode; label: string; description: string }[] = [
  { value: "light", label: "Light", description: "Always light, regardless of your device." },
  { value: "dark", label: "Dark", description: "Always dark, regardless of your device." },
  { value: "system", label: "System", description: "Follows your device's light/dark setting." },
];

const LEVELS: { value: ExperienceLevel; label: string; description: string }[] = [
  {
    value: "beginner",
    label: "Beginner",
    description:
      "Simplifies the app: a few advanced tools are tucked out of the main menu until you turn this off, " +
      "and connecting a real brokerage account requires finishing some paper trading and a short risk quiz first.",
  },
  {
    value: "intermediate",
    label: "Intermediate",
    description: "The full app, no gates — for someone comfortable with the basics already.",
  },
  {
    value: "experienced",
    label: "Experienced",
    description: "Same as Intermediate. Shown on your community profile.",
  },
];

export default function SettingsPage() {
  const [userId, setUserId] = useState<string | null>(null);
  const [level, setLevel] = useState<ExperienceLevel | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [themeOverride, setThemeOverride] = useState<string | null>(null);
  const [themeSaving, setThemeSaving] = useState<string | "default" | null>(null);
  const [themeError, setThemeError] = useState<string | null>(null);
  const [mode, setMode] = useState<ThemeMode>("system");

  const [portfolioSharing, setPortfolioSharingState] = useState(false);
  const [portfolioSharingSaving, setPortfolioSharingSaving] = useState(false);
  const [portfolioSharingError, setPortfolioSharingError] = useState<string | null>(null);

  const [blocks, setBlocks] = useState<{ user_id: string; display_name: string | null }[] | null>(null);
  const [mutes, setMutes] = useState<{ id: number; mute_type: string; mute_value: string; created_at: string }[] | null>(null);
  const [newMuteType, setNewMuteType] = useState<"user" | "ticker" | "keyword">("ticker");
  const [newMuteValue, setNewMuteValue] = useState("");
  const [blockMuteError, setBlockMuteError] = useState<string | null>(null);

  async function loadBlocksAndMutes() {
    try {
      const [b, m] = await Promise.all([getBlockedUsers(), getMutes()]);
      setBlocks(b.blocks);
      setMutes(m.mutes);
    } catch (err) {
      setBlockMuteError(err instanceof ApiError ? err.message : "Could not load blocks/mutes.");
    }
  }

  useEffect(() => {
    getCurrentUser()
      .then((user) => {
        setUserId(user.id);
        setLevel(user.experience_level);
        setThemeOverride(user.theme_accent_color);
        setPortfolioSharingState(user.portfolio_sharing_enabled);
      })
      .catch(() => setError("Could not load your settings."))
      .finally(() => setLoading(false));
    setMode(getStoredThemeMode());
    loadBlocksAndMutes();
  }, []);

  async function handlePortfolioSharingToggle(enabled: boolean) {
    setPortfolioSharingSaving(true);
    setPortfolioSharingError(null);
    try {
      const res = await setPortfolioSharing(enabled);
      setPortfolioSharingState(res.portfolio_sharing_enabled);
    } catch (err) {
      setPortfolioSharingError(err instanceof ApiError ? err.message : "Could not save this setting.");
    } finally {
      setPortfolioSharingSaving(false);
    }
  }

  async function handleUnblock(userId: string) {
    try {
      await unblockUser(userId);
      await loadBlocksAndMutes();
    } catch (err) {
      setBlockMuteError(err instanceof ApiError ? err.message : "Could not unblock.");
    }
  }

  async function handleUnmute(muteId: number) {
    try {
      await unmute(muteId);
      await loadBlocksAndMutes();
    } catch (err) {
      setBlockMuteError(err instanceof ApiError ? err.message : "Could not unmute.");
    }
  }

  async function handleAddMute() {
    const value = newMuteValue.trim();
    if (!value) return;
    try {
      await muteContent(newMuteType, value);
      setNewMuteValue("");
      await loadBlocksAndMutes();
    } catch (err) {
      setBlockMuteError(err instanceof ApiError ? err.message : "Could not add mute.");
    }
  }

  function handleModeChange(next: ThemeMode) {
    setMode(next);
    applyThemeMode(next);
  }

  async function saveTheme(color: string | null) {
    setThemeSaving(color ?? "default");
    setThemeError(null);
    try {
      const res = await setMyThemeColor(color);
      setThemeOverride(res.theme_accent_color);
      applyThemeAccent(res.effective_theme_accent_color, res.effective_theme_accent_soft);
    } catch (err) {
      setThemeError(err instanceof ApiError ? err.message : "Could not save your theme color.");
    } finally {
      setThemeSaving(null);
    }
  }

  async function save(next: ExperienceLevel) {
    setSaving(true);
    setError(null);
    setSaved(false);
    try {
      await updateSocialProfile({ experience_level: next });
      setLevel(next);
      setSaved(true);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not save this setting.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="mx-auto max-w-2xl px-4 py-8">
      <h1 className="font-display text-2xl font-semibold text-slate-900 dark:text-slate-50">Settings</h1>
      {userId && (
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Also shown on your{" "}
          <Link href={`/community/authors/${userId}`} className="underline decoration-dotted">
            community profile
          </Link>
          .
        </p>
      )}

      <div className="mt-6 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5">
        <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-50">Security</h2>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Two-factor authentication and your account&apos;s recent activity.
        </p>
        <Link
          href="/settings/security"
          className="mt-3 inline-block rounded-md border border-slate-300 dark:border-slate-600 px-3 py-1.5 text-sm font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800"
        >
          Manage security settings
        </Link>
      </div>

      <div className="mt-6 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5">
        <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-50">Experience level</h2>
        {loading ? (
          <p className="mt-3 text-sm text-slate-500 dark:text-slate-400">Loading…</p>
        ) : (
          <div className="mt-3 flex flex-col gap-3">
            {LEVELS.map((opt) => (
              <label
                key={opt.value}
                className={`flex cursor-pointer flex-col gap-1 rounded-lg border px-4 py-3 text-sm ${
                  level === opt.value ? "border-slate-900 bg-slate-50 dark:bg-slate-800" : "border-slate-200 dark:border-slate-700"
                }`}
              >
                <span className="flex items-center gap-2 font-medium text-slate-900 dark:text-slate-50">
                  <input
                    type="radio"
                    name="experience_level"
                    checked={level === opt.value}
                    disabled={saving}
                    onChange={() => save(opt.value)}
                  />
                  {opt.label}
                </span>
                <span className="text-slate-500 dark:text-slate-400">{opt.description}</span>
              </label>
            ))}
          </div>
        )}
        {saving && <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">Saving…</p>}
        {saved && !saving && <p className="mt-3 text-xs font-medium text-emerald-700">Saved.</p>}
        {error && <p className="mt-3 text-xs font-medium text-red-700">{error}</p>}
      </div>

      <div className="mt-6 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5">
        <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-50">Appearance</h2>
        <div className="mt-3 grid grid-cols-3 gap-2">
          {MODES.map((m) => (
            <button
              key={m.value}
              type="button"
              onClick={() => handleModeChange(m.value)}
              aria-pressed={mode === m.value}
              className={`flex flex-col gap-0.5 rounded-lg border px-3 py-2 text-left text-sm ${
                mode === m.value ? "border-slate-900 bg-slate-50 dark:bg-slate-800" : "border-slate-200 dark:border-slate-700"
              }`}
            >
              <span className="font-medium text-slate-900 dark:text-slate-50">{m.label}</span>
              <span className="text-xs text-slate-500 dark:text-slate-400">{m.description}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="mt-6 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5">
        <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-50">Theme color</h2>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Overrides the site&apos;s default accent color for just your account. A curated palette, not a freeform
          color.
        </p>
        {loading ? (
          <p className="mt-3 text-sm text-slate-500 dark:text-slate-400">Loading…</p>
        ) : (
          <div className="mt-3 flex flex-wrap items-center gap-2">
            {THEME_PALETTE.map((c) => (
              <button
                key={c.accent}
                type="button"
                onClick={() => saveTheme(c.accent)}
                disabled={themeSaving !== null}
                aria-pressed={themeOverride === c.accent}
                title={c.name}
                className={`flex h-9 w-9 items-center justify-center rounded-full border-2 disabled:opacity-50 ${
                  themeOverride === c.accent ? "border-slate-900" : "border-transparent"
                }`}
                style={{ backgroundColor: c.accent }}
              >
                {themeSaving === c.accent && <span className="h-2 w-2 animate-pulse rounded-full bg-white dark:bg-slate-900" />}
              </button>
            ))}
            <button
              type="button"
              onClick={() => saveTheme(null)}
              disabled={themeSaving !== null || themeOverride === null}
              className="ml-1 rounded-md border border-slate-300 dark:border-slate-600 px-2.5 py-1 text-xs font-medium text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 disabled:opacity-50"
            >
              {themeSaving === "default" ? "Resetting…" : "Use site default"}
            </button>
          </div>
        )}
        {themeError && <p className="mt-3 text-xs font-medium text-red-700">{themeError}</p>}
      </div>

      <div className="mt-6 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5">
        <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-50">Portfolio sharing</h2>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Off by default. When on, your community profile shows a percentage-only summary of your portfolio --
          top holdings by weight and total return % -- never dollar amounts.
        </p>
        <label className="mt-3 flex items-center gap-2 text-sm text-slate-700 dark:text-slate-200">
          <input
            type="checkbox"
            checked={portfolioSharing}
            disabled={portfolioSharingSaving}
            onChange={(e) => handlePortfolioSharingToggle(e.target.checked)}
          />
          Show a percentage-only portfolio summary on my community profile
        </label>
        {portfolioSharingError && <p className="mt-3 text-xs font-medium text-red-700">{portfolioSharingError}</p>}
      </div>

      <div className="mt-6 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5">
        <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-50">Blocked &amp; muted</h2>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Blocking stops someone from seeing or messaging you. Muting just hides matching posts from your own feed --
          it doesn&apos;t notify anyone.
        </p>

        <div className="mt-3 flex flex-wrap items-end gap-2">
          <select
            value={newMuteType}
            onChange={(e) => setNewMuteType(e.target.value as "user" | "ticker" | "keyword")}
            className="rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 px-2 py-1.5 text-sm text-slate-900 dark:text-slate-50"
          >
            <option value="ticker">Ticker</option>
            <option value="keyword">Keyword</option>
            <option value="user">User (display name or id)</option>
          </select>
          <input
            value={newMuteValue}
            onChange={(e) => setNewMuteValue(e.target.value)}
            placeholder={newMuteType === "ticker" ? "e.g. GME" : newMuteType === "keyword" ? "e.g. moon" : "user id"}
            className="w-48 rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 px-3 py-1.5 text-sm text-slate-900 dark:text-slate-50"
          />
          <button
            type="button"
            onClick={handleAddMute}
            disabled={!newMuteValue.trim()}
            className="rounded-md bg-slate-900 dark:bg-slate-100 px-3 py-1.5 text-sm font-medium text-white dark:text-slate-900 hover:bg-slate-800 dark:hover:bg-slate-200 disabled:opacity-50"
          >
            Mute
          </button>
        </div>

        {mutes && mutes.length > 0 && (
          <ul className="mt-3 flex flex-col gap-1.5 text-sm">
            {mutes.map((m) => (
              <li key={m.id} className="flex items-center justify-between gap-2">
                <span className="text-slate-700 dark:text-slate-200">
                  <span className="text-xs uppercase text-slate-400 dark:text-slate-500">{m.mute_type}</span> {m.mute_value}
                </span>
                <button onClick={() => handleUnmute(m.id)} className="text-xs font-medium text-slate-500 dark:text-slate-400 hover:underline">
                  Unmute
                </button>
              </li>
            ))}
          </ul>
        )}

        {blocks && blocks.length > 0 && (
          <>
            <h3 className="mt-4 text-xs font-semibold uppercase text-slate-500 dark:text-slate-400">Blocked users</h3>
            <ul className="mt-1.5 flex flex-col gap-1.5 text-sm">
              {blocks.map((b) => (
                <li key={b.user_id} className="flex items-center justify-between gap-2">
                  <span className="text-slate-700 dark:text-slate-200">{b.display_name ?? b.user_id}</span>
                  <button onClick={() => handleUnblock(b.user_id)} className="text-xs font-medium text-slate-500 dark:text-slate-400 hover:underline">
                    Unblock
                  </button>
                </li>
              ))}
            </ul>
          </>
        )}
        {blockMuteError && <p className="mt-3 text-xs font-medium text-red-700">{blockMuteError}</p>}
      </div>
    </div>
  );
}
