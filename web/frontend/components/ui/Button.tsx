import type { ButtonHTMLAttributes } from "react";

/** The three button "families" found across the app, picked to match
 * the single most common look of each rather than inventing a new one
 * -- `primary` standardizes on `bg-slate-900` (the most common primary
 * color; a few pages used `indigo-700` instead for the same role, an
 * inconsistency this fixes where applied). */
type ButtonVariant = "primary" | "secondary" | "danger";

const VARIANT_CLASSES: Record<ButtonVariant, string> = {
  primary: "bg-slate-900 text-white hover:bg-slate-800",
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
