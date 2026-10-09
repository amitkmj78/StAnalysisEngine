import type { ReactNode } from "react";

/** `rounded-xl border border-slate-200 bg-white p-5` -- the single most
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
  return <div className={`rounded-xl border border-slate-200 bg-white ${padding} ${className}`}>{children}</div>;
}
