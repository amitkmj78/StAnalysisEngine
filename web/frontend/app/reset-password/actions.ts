"use server";

import { redirect } from "next/navigation";

// See app/login/actions.ts for why this differs from NEXT_PUBLIC_API_BASE_URL.
const BACKEND_URL = process.env.BACKEND_INTERNAL_URL || process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8010";

export async function resetPassword(formData: FormData) {
  const token = formData.get("token") as string;
  const password = formData.get("password") as string;
  const confirmPassword = formData.get("confirm_password") as string;

  if (password !== confirmPassword) {
    redirect(`/reset-password?token=${encodeURIComponent(token)}&error=${encodeURIComponent("Passwords don't match.")}`);
  }

  const res = await fetch(`${BACKEND_URL}/api/v1/auth/reset-password`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token, password }),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    redirect(
      `/reset-password?token=${encodeURIComponent(token)}&error=${encodeURIComponent(body.detail || "Could not reset your password.")}`
    );
  }

  redirect("/login?info=" + encodeURIComponent("Password updated — log in with your new password."));
}
