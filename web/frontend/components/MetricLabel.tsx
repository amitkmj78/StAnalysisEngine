"use client";

import { useState } from "react";

import { GLOSSARY } from "@/lib/glossary";
import InfoModal, { type ColumnInfo } from "./InfoModal";

/**
 * Wraps a metric/factor label with a "what is this?" tooltip, looked up from
 * the shared glossary (lib/glossary.ts) by `term` (or by the label text
 * itself when `term` is omitted and `children` is a plain string) -- this is
 * the whole point of LRN-1's shared-glossary fix: adding a tooltip anywhere
 * in the app is just wrapping a label here, no per-page state or modal
 * wiring needed. Silently renders the plain label with no trigger when no
 * glossary entry matches -- never a dead button, never an error.
 */
export default function MetricLabel({
  term,
  children,
  className,
  extraBody,
  info: infoOverride,
}: {
  term?: string;
  /** Optional -- omit when the label text is already rendered elsewhere
   * (e.g. a sortable column header that renders its own button) and only
   * the info trigger itself is needed. */
  children?: React.ReactNode;
  className?: string;
  /** Extra paragraphs appended after the glossary entry's own body --
   * for the rare case where a tooltip needs request-specific data (e.g.
   * "BUY calls have been right 62% of the time so far") that can't live
   * in the static glossary. Most callers don't need this. */
  extraBody?: string[];
  /** Full override, bypassing the glossary lookup entirely -- for the
   * rarer case where even the title/structure varies at render time (e.g.
   * "10-Day Signal" vs "30-Day Signal" depending on a selected horizon),
   * not just extra appended text. Most callers don't need this either. */
  info?: ColumnInfo;
}) {
  const [open, setOpen] = useState(false);
  const key = term ?? (typeof children === "string" ? children : undefined);
  const baseInfo = infoOverride ?? (key ? GLOSSARY[key] : undefined);
  const info = baseInfo && extraBody?.length ? { ...baseInfo, body: [...baseInfo.body, ...extraBody] } : baseInfo;

  return (
    <span className={`inline-flex items-center gap-1 ${className ?? ""}`}>
      {children}
      {info && (
        <button
          type="button"
          onClick={() => setOpen(true)}
          title={`What is ${info.title}?`}
          className="flex h-4 w-4 items-center justify-center rounded-full border border-slate-300 text-[10px] font-normal normal-case text-slate-400 hover:border-slate-500 hover:text-slate-700"
        >
          i
        </button>
      )}
      {open && info && <InfoModal info={info} onClose={() => setOpen(false)} />}
    </span>
  );
}
