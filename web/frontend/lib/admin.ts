const ADMIN_EMAIL = "amitkmj78@gmail.com";

export function isAdmin(email: string | null | undefined): boolean {
  return !!email && email.toLowerCase() === ADMIN_EMAIL.toLowerCase();
}
