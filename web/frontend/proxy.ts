import { NextResponse, type NextRequest } from "next/server";

import { isAdmin } from "@/lib/admin";
import { SESSION_COOKIE_NAME, verifySessionToken } from "@/lib/session";

const PROTECTED_PREFIXES = [
  "/predict",
  "/stock-finder",
  "/index-fund",
  "/entry",
  "/monthly-plan",
  "/strategies",
  "/trade-journal",
  "/portfolio",
  "/chat",
  "/admin",
];
const ADMIN_PREFIXES = ["/admin"];
const AUTH_PREFIXES = ["/login", "/signup"];

// Renamed from `middleware` per Next.js 16 (middleware.ts is deprecated,
// see node_modules/next/dist/docs/.../proxy.md — same behavior, new name).
export async function proxy(request: NextRequest) {
  const token = request.cookies.get(SESSION_COOKIE_NAME)?.value;
  const user = token ? await verifySessionToken(token) : null;

  const path = request.nextUrl.pathname;
  const isProtected = PROTECTED_PREFIXES.some((p) => path.startsWith(p));
  const isAuthRoute = AUTH_PREFIXES.some((p) => path.startsWith(p));

  if (isProtected && !user) {
    const url = request.nextUrl.clone();
    url.pathname = "/login";
    return NextResponse.redirect(url);
  }

  const isAdminRoute = ADMIN_PREFIXES.some((p) => path.startsWith(p));
  if (isAdminRoute && user && !isAdmin(user.email)) {
    const url = request.nextUrl.clone();
    url.pathname = "/predict";
    return NextResponse.redirect(url);
  }

  if (isAuthRoute && user) {
    const url = request.nextUrl.clone();
    url.pathname = "/predict";
    return NextResponse.redirect(url);
  }

  return NextResponse.next({ request });
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)"],
};
