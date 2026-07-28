"use server";

import { cookies } from "next/headers";
import { redirect } from "next/navigation";

import { SESSION_COOKIE_NAME } from "@/lib/session";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "";
const COOKIE_SECURE = process.env.COOKIE_SECURE === "true";
const SESSION_MAX_AGE = 60 * 60 * 24 * 7; // 7 days, matches backend SESSION_TTL

export async function login(formData: FormData) {
  const email = formData.get("email") as string;
  const password = formData.get("password") as string;

  const res = await fetch(`${API_BASE}/api/v1/auth/login`, {
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
