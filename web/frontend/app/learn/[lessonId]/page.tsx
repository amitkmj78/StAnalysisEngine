"use client";

import Link from "next/link";
import { notFound, useParams } from "next/navigation";
import { useState } from "react";

import { ApiError, recordLearningProgress } from "@/lib/api";
import { getLesson } from "@/lib/lessons";

export default function LessonPage() {
  const params = useParams<{ lessonId: string }>();
  const lesson = getLesson(params.lessonId);

  const [answers, setAnswers] = useState<number[]>([]);
  const [submitted, setSubmitted] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!lesson) {
    notFound();
    return null;
  }

  const allAnswered = answers.length === lesson.quiz.length && answers.every((a) => a !== undefined);
  const score = submitted ? answers.filter((a, i) => a === lesson.quiz[i].correctIndex).length : 0;

  function pick(questionIndex: number, choiceIndex: number) {
    if (submitted) return;
    setAnswers((prev) => {
      const next = [...prev];
      next[questionIndex] = choiceIndex;
      return next;
    });
  }

  async function submitQuiz() {
    setSubmitted(true);
    const finalScore = answers.filter((a, i) => a === lesson!.quiz[i].correctIndex).length;
    setSaving(true);
    setError(null);
    try {
      await recordLearningProgress(lesson!.id, finalScore, lesson!.quiz.length);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not save your progress.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="mx-auto max-w-2xl px-4 py-8">
      <Link href="/learn" className="text-sm font-medium text-slate-600 hover:underline">
        ← Learning Paths
      </Link>
      <h1 className="mt-2 font-display text-2xl font-semibold text-slate-900">{lesson.title}</h1>

      <div className="mt-4 flex flex-col gap-3 text-sm text-slate-700">
        {lesson.body.map((para, i) => (
          <p key={i}>{para}</p>
        ))}
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        <Link
          href={lesson.liveExample.href}
          className="inline-block rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
        >
          Try it: {lesson.liveExample.label} →
        </Link>
        {lesson.guideHref && (
          <Link
            href={lesson.guideHref}
            className="inline-block rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
          >
            Read the deeper guide →
          </Link>
        )}
      </div>

      <div className="mt-8 rounded-xl border border-slate-200 bg-white p-5">
        <h2 className="text-sm font-semibold text-slate-900">Quick quiz</h2>
        <div className="mt-4 flex flex-col gap-5">
          {lesson.quiz.map((q, qi) => (
            <div key={qi}>
              <p className="text-sm font-medium text-slate-900">{qi + 1}. {q.question}</p>
              <div className="mt-2 flex flex-col gap-1.5">
                {q.choices.map((choice, ci) => {
                  const isChosen = answers[qi] === ci;
                  const isCorrect = ci === q.correctIndex;
                  let style = "border-slate-200 text-slate-700 hover:bg-slate-50";
                  if (submitted) {
                    if (isCorrect) style = "border-emerald-300 bg-emerald-50 text-emerald-800";
                    else if (isChosen) style = "border-red-300 bg-red-50 text-red-700";
                    else style = "border-slate-200 text-slate-400";
                  } else if (isChosen) {
                    style = "border-slate-900 bg-slate-50 text-slate-900";
                  }
                  return (
                    <button
                      key={ci}
                      type="button"
                      disabled={submitted}
                      onClick={() => pick(qi, ci)}
                      className={`rounded-md border px-3 py-2 text-left text-sm ${style}`}
                    >
                      {choice}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        {!submitted ? (
          <button
            type="button"
            disabled={!allAnswered}
            onClick={submitQuiz}
            className="btn-primary mt-5 disabled:opacity-50"
          >
            Check answers
          </button>
        ) : (
          <div className="mt-5">
            <p className="text-sm font-semibold text-slate-900">
              Score: {score}/{lesson.quiz.length}
            </p>
            {saving && <p className="mt-1 text-xs text-slate-500">Saving…</p>}
            {error && <p className="mt-1 text-xs font-medium text-red-700">{error}</p>}
            <Link href="/learn" className="mt-3 inline-block text-sm font-medium text-slate-700 hover:underline">
              Back to Learning Paths
            </Link>
          </div>
        )}
      </div>
    </div>
  );
}
