"use server";

import { cookies } from "next/headers";
import { redirect } from "next/navigation";

import { SESSION_COOKIE_NAME } from "@/lib/session";

// See app/login/actions.ts for why this differs from NEXT_PUBLIC_API_BASE_URL.
const BACKEND_URL = process.env.BACKEND_INTERNAL_URL || process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8010";
const COOKIE_SECURE = process.env.COOKIE_SECURE === "true";
const SESSION_MAX_AGE = 60 * 60 * 24 * 7;

export async function signup(formData: FormData) {
  const email = formData.get("email") as string;
  const password = formData.get("password") as string;

  const res = await fetch(`${BACKEND_URL}/api/v1/auth/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    redirect(`/signup?error=${encodeURIComponent(body.detail || "Sign up failed")}`);
  }

  // No email-confirmation loop in v1 (no email-sending infra without
  // Supabase) — the account is active immediately, so log the user straight
  // in rather than showing a "check your email" message that would be a lie.
  const { token } = await res.json();
  const cookieStore = await cookies();
  cookieStore.set(SESSION_COOKIE_NAME, token, {
    httpOnly: true,
    secure: COOKIE_SECURE,
    sameSite: "lax",
    maxAge: SESSION_MAX_AGE,
    path: "/",
  });

  redirect("/predict");
}
