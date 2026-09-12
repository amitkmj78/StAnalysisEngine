"use server";

import { cookies, headers } from "next/headers";
import { redirect } from "next/navigation";

import { SESSION_COOKIE_NAME } from "@/lib/session";

// See app/login/actions.ts for why this differs from NEXT_PUBLIC_API_BASE_URL.
const BACKEND_URL = process.env.BACKEND_INTERNAL_URL || process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8010";
const COOKIE_SECURE = process.env.COOKIE_SECURE === "true";
const SESSION_MAX_AGE = 60 * 60 * 24 * 7;

export async function signup(formData: FormData) {
  const email = formData.get("email") as string;
  const password = formData.get("password") as string;

  // See app/login/actions.ts — this fetch is server-to-server, so the
  // real visitor IP has to be forwarded explicitly from the incoming
  // request's own X-Forwarded-For (set by nginx on the browser's hop).
  const incomingHeaders = await headers();
  const forwardedFor = incomingHeaders.get("x-forwarded-for");

  const res = await fetch(`${BACKEND_URL}/api/v1/auth/signup`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(forwardedFor ? { "X-Forwarded-For": forwardedFor } : {}),
    },
    body: JSON.stringify({ email, password }),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    redirect(`/signup?error=${encodeURIComponent(body.detail || "Sign up failed")}`);
  }

  const data = await res.json();

  // Every signup except the admin's own account starts pending — no session
  // is issued until an admin approves it from /admin/users, so there is no
  // token to set a cookie with here.
  if (data.pending) {
    redirect(
      "/login?info=" +
        encodeURIComponent("Account created — it needs admin approval before you can sign in.")
    );
  }

  const cookieStore = await cookies();
  cookieStore.set(SESSION_COOKIE_NAME, data.token, {
    httpOnly: true,
    secure: COOKIE_SECURE,
    sameSite: "lax",
    maxAge: SESSION_MAX_AGE,
    path: "/",
  });

  redirect("/predict");
}
