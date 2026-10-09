"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { ApiError, getCurrentUser, updateSocialProfile } from "@/lib/api";
import type { ExperienceLevel } from "@/lib/types";

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

  useEffect(() => {
    getCurrentUser()
      .then((user) => {
        setUserId(user.id);
        setLevel(user.experience_level);
      })
      .catch(() => setError("Could not load your settings."))
      .finally(() => setLoading(false));
  }, []);

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
      <h1 className="font-display text-2xl font-semibold text-slate-900">Settings</h1>
      {userId && (
        <p className="mt-1 text-sm text-slate-500">
          Also shown on your{" "}
          <Link href={`/community/authors/${userId}`} className="underline decoration-dotted">
            community profile
          </Link>
          .
        </p>
      )}

      <div className="mt-6 rounded-xl border border-slate-200 bg-white p-5">
        <h2 className="text-sm font-semibold text-slate-900">Experience level</h2>
        {loading ? (
          <p className="mt-3 text-sm text-slate-500">Loading…</p>
        ) : (
          <div className="mt-3 flex flex-col gap-3">
            {LEVELS.map((opt) => (
              <label
                key={opt.value}
                className={`flex cursor-pointer flex-col gap-1 rounded-lg border px-4 py-3 text-sm ${
                  level === opt.value ? "border-slate-900 bg-slate-50" : "border-slate-200"
                }`}
              >
                <span className="flex items-center gap-2 font-medium text-slate-900">
                  <input
                    type="radio"
                    name="experience_level"
                    checked={level === opt.value}
                    disabled={saving}
                    onChange={() => save(opt.value)}
                  />
                  {opt.label}
                </span>
                <span className="text-slate-500">{opt.description}</span>
              </label>
            ))}
          </div>
        )}
        {saving && <p className="mt-3 text-xs text-slate-500">Saving…</p>}
        {saved && !saving && <p className="mt-3 text-xs font-medium text-emerald-700">Saved.</p>}
        {error && <p className="mt-3 text-xs font-medium text-red-700">{error}</p>}
      </div>
    </div>
  );
}
