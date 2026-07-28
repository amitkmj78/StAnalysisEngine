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
