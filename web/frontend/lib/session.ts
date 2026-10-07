import { jwtVerify } from "jose";
import { cookies } from "next/headers";

// jose (not jsonwebtoken) specifically because this needs to run in both
// the Edge runtime (proxy.ts) and Node (Server Components) — Node's
// jsonwebtoken/crypto aren't available on the Edge runtime.
const secretKey = new TextEncoder().encode(process.env.SESSION_SECRET!);

export const SESSION_COOKIE_NAME = "session";

export interface SessionUser {
  id: string;
  email: string;
}

export async function verifySessionToken(token: string): Promise<SessionUser | null> {
  try {
    const { payload } = await jwtVerify(token, secretKey, { algorithms: ["HS256"] });
    if (typeof payload.sub !== "string" || typeof payload.email !== "string") return null;
    return { id: payload.sub, email: payload.email };
  } catch {
    return null;
  }
}

// Server Component helper (layout.tsx, page.tsx) — reads the httpOnly
// cookie directly via next/headers. proxy.ts runs on the Edge runtime and
// reads the cookie from the NextRequest instead, calling verifySessionToken
// directly rather than this wrapper.
export async function getSession(): Promise<SessionUser | null> {
  const cookieStore = await cookies();
  const token = cookieStore.get(SESSION_COOKIE_NAME)?.value;
  if (!token) return null;
  return verifySessionToken(token);
}

// Same reasoning as app/login/actions.ts's BACKEND_URL: this runs server-to-
// server (Node/Edge, no page origin), so it needs an absolute backend URL.
const BACKEND_URL = process.env.BACKEND_INTERNAL_URL || process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8010";

// Stock Detail is the home screen; this is which ticker it opens on for a
// given signed-in user. A user sets this via "Set as home" on the Stock
// Detail page itself (PUT /api/v1/auth/me/default-ticker) -- falls back to
// SPY if they've never set one, or if the lookup fails for any reason (an
// unreachable backend should never block login/signup/home from redirecting).
export async function getHomeTicker(token: string): Promise<string> {
  try {
    const res = await fetch(`${BACKEND_URL}/api/v1/auth/me`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    });
    if (!res.ok) return "SPY";
    const data = await res.json();
    return (data.default_ticker as string | null) || "SPY";
  } catch {
    return "SPY";
  }
}
