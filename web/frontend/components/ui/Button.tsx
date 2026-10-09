import type { ButtonHTMLAttributes } from "react";

/** The three button "families" found across the app, picked to match
 * the single most common look of each rather than inventing a new one
 * -- `primary` uses the shared "Price outlook" accent token (globals.css)
 * instead of slate-900/indigo-700, the two different "primary" colors
 * different pages had used for the same role. */
type ButtonVariant = "primary" | "secondary" | "danger";

const VARIANT_CLASSES: Record<ButtonVariant, string> = {
  primary: "bg-[var(--pf-accent)] text-white hover:opacity-90",
  secondary: "border border-slate-300 bg-white text-slate-700 hover:bg-slate-50",
  danger: "border border-red-300 bg-white text-red-700 hover:bg-red-50",
};

export default function Button({
  variant = "primary",
  className = "",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant }) {
  return (
    <button
      {...props}
      className={`rounded-md px-4 py-2 text-sm font-medium disabled:opacity-50 ${VARIANT_CLASSES[variant]} ${className}`}
    />
  );
}
