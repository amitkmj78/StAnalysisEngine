"use server";

import { cookies } from "next/headers";
import { redirect } from "next/navigation";

import { SESSION_COOKIE_NAME } from "@/lib/session";

// Server actions run in Node, not the browser — they have no page origin to
// resolve a relative path against, so they need an absolute URL to the
// backend. NEXT_PUBLIC_API_BASE_URL is for client-side fetches (resolved
// against window.location.origin) and is "/api" in production, which is
// not a valid fetch() target here.
const BACKEND_URL = process.env.BACKEND_INTERNAL_URL || process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8010";
const COOKIE_SECURE = process.env.COOKIE_SECURE === "true";
const SESSION_MAX_AGE = 60 * 60 * 24 * 7; // 7 days, matches backend SESSION_TTL

export async function login(formData: FormData) {
  const email = formData.get("email") as string;
  const password = formData.get("password") as string;

  const res = await fetch(`${BACKEND_URL}/api/v1/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    redirect(`/login?error=${encodeURIComponent(body.detail || "Login failed")}`);
  }

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
