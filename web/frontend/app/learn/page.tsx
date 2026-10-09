"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { getLearningProgress } from "@/lib/api";
import { LESSONS } from "@/lib/lessons";
import type { LessonProgressEntry } from "@/lib/types";

export default function LearnPage() {
  const [progress, setProgress] = useState<Record<string, LessonProgressEntry>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getLearningProgress()
      .then((res) => {
        const byId: Record<string, LessonProgressEntry> = {};
        for (const entry of res.lessons) byId[entry.lesson_id] = entry;
        setProgress(byId);
      })
      .catch(() => setProgress({}))
      .finally(() => setLoading(false));
  }, []);

  const doneCount = Object.keys(progress).length;

  return (
    <div className="mx-auto max-w-2xl px-4 py-8">
      <h1 className="font-display text-2xl font-semibold text-slate-900">Learning Paths</h1>
      <p className="mt-1 text-sm text-slate-500">
        Short lessons with a quiz at the end, each linking to a live example in the app. General education, not
        personalized financial advice.
        {!loading && ` ${doneCount} of ${LESSONS.length} completed.`}
      </p>

      <div className="mt-6 flex flex-col gap-3">
        {LESSONS.map((lesson) => {
          const done = progress[lesson.id];
          return (
            <Link
              key={lesson.id}
              href={`/learn/${lesson.id}`}
              className="flex items-start justify-between gap-3 rounded-xl border border-slate-200 bg-white p-4 hover:border-slate-300"
            >
              <div>
                <p className="text-sm font-semibold text-slate-900">{lesson.title}</p>
                <p className="mt-0.5 text-xs text-slate-500">{lesson.summary}</p>
              </div>
              {done ? (
                <span className="flex-none rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-medium text-emerald-700">
                  ✓ {done.quiz_score}/{done.quiz_total}
                </span>
              ) : (
                <span className="flex-none rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-500">
                  Not started
                </span>
              )}
            </Link>
          );
        })}
      </div>
    </div>
  );
}
