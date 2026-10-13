const ADMIN_EMAIL = "amitkmj78@gmail.com";

// ADMIN_EMAIL is kept as a permanent fallback alongside the role column
// (not replaced by it) -- mirrors web/backend/admin.py::require_admin, so a
// bad or stale role value can never lock out the original account.
export function isAdmin(email: string | null | undefined, role?: string | null): boolean {
  if (role === "admin") return true;
  return !!email && email.toLowerCase() === ADMIN_EMAIL.toLowerCase();
}
