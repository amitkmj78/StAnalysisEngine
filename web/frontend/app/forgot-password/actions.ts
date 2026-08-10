"use server";

import { redirect } from "next/navigation";

// See app/login/actions.ts for why this differs from NEXT_PUBLIC_API_BASE_URL.
const BACKEND_URL = process.env.BACKEND_INTERNAL_URL || process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8010";

export async function requestPasswordReset(formData: FormData) {
  const email = formData.get("email") as string;

  const res = await fetch(`${BACKEND_URL}/api/v1/auth/forgot-password`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
  });

  // The backend always returns the same generic message regardless of
  // whether the account exists — that's the point (no enumeration via this
  // form). A non-OK response here means something actually broke (rate
  // limit, network), not "email not found."
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    redirect(`/forgot-password?error=${encodeURIComponent(body.detail || "Could not process that request.")}`);
  }

  const data = await res.json();
  redirect(`/forgot-password?info=${encodeURIComponent(data.message)}`);
}
