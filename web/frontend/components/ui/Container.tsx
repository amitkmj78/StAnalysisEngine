import type { ReactNode } from "react";

/** The page-width wrapper every page currently hand-rolls as its own
 * `mx-auto max-w-Nxl px-4 py-8` div, with a different N picked ad hoc
 * per page. `size` covers the widths actually in use across the app
 * (2xl/3xl/5xl/7xl) -- not a new scale invented for this component. */
type ContainerSize = "sm" | "md" | "lg" | "xl";

const SIZE_CLASSES: Record<ContainerSize, string> = {
  sm: "max-w-2xl",
  md: "max-w-3xl",
  lg: "max-w-5xl",
  xl: "max-w-7xl",
};

export default function Container({
  size = "md",
  className = "",
  children,
}: {
  size?: ContainerSize;
  className?: string;
  children: ReactNode;
}) {
  return <div className={`mx-auto ${SIZE_CLASSES[size]} px-4 py-8 ${className}`}>{children}</div>;
}
