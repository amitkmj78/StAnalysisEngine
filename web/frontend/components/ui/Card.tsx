import type { ReactNode } from "react";

/** `rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5` -- the single most
 * common of the 30+ card variants found across the app (see the UI-
 * consistency audit). `padding` lets a denser list-item use case (e.g.
 * a feed row) drop to `p-4` without inventing a second component. */
export default function Card({
  padding = "p-5",
  className = "",
  children,
}: {
  padding?: string;
  className?: string;
  children: ReactNode;
}) {
  return <div className={`rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 ${padding} ${className}`}>{children}</div>;
}
