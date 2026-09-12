"use server";

import { cookies, headers } from "next/headers";
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

  // This fetch is server-to-server (Next.js -> FastAPI, both on the same
  // box) — nginx never sees it, so it carries no X-Forwarded-For of its
  // own. The browser's actual request DID go through nginx to reach this
  // Server Action, though, so its X-Forwarded-For (the real visitor IP)
  // is on the incoming request here; forward it explicitly so the
  // backend's last-login-IP recording (web/backend/auth.get_client_ip)
  // sees the visitor, not this server calling itself.
  const incomingHeaders = await headers();
  const forwardedFor = incomingHeaders.get("x-forwarded-for");

  const res = await fetch(`${BACKEND_URL}/api/v1/auth/login`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(forwardedFor ? { "X-Forwarded-For": forwardedFor } : {}),
    },
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
