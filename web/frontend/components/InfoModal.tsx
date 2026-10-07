"use client";

import { useEffect } from "react";

export interface ColumnInfo {
  title: string;
  body: string[];
}

export default function InfoModal({ info, onClose }: { info: ColumnInfo; onClose: () => void }) {
  // NFR-7: Escape closes the modal too, so the backdrop's click-to-close
  // isn't the only way out for someone who can't (or doesn't want to) click.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  return (
    <div
      role="button"
      tabIndex={-1}
      aria-label="Close"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4"
      onClick={(e) => {
        // Only the backdrop itself closes on click -- a click that bubbled
        // up from the panel below doesn't, so no separate stopPropagation
        // handler (and no onClick on a non-interactive "dialog"-role div)
        // is needed.
        if (e.target === e.currentTarget) onClose();
      }}
      onKeyDown={(e) => {
        if (e.key === "Escape") onClose();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={info.title}
        className="max-h-[80vh] w-full max-w-md overflow-y-auto rounded-lg bg-white p-5 shadow-xl"
      >
        <div className="flex items-start justify-between gap-4">
          <h3 className="text-base font-semibold text-slate-900">{info.title}</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-700" aria-label="Close">
            ✕
          </button>
        </div>
        <div className="mt-3 flex flex-col gap-2">
          {info.body.map((p, i) => (
            <p key={i} className="text-sm leading-relaxed text-slate-600">
              {p}
            </p>
          ))}
        </div>
      </div>
    </div>
  );
}
