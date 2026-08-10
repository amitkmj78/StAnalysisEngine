import Link from "next/link";

import { resetPassword } from "./actions";

export default async function ResetPasswordPage({
  searchParams,
}: {
  searchParams: Promise<{ token?: string; error?: string }>;
}) {
  const params = await searchParams;
  const token = params.token || "";

  return (
    <div className="mx-auto mt-24 max-w-sm px-4">
      <h1 className="mb-1 text-2xl font-semibold text-slate-900">Set a new password</h1>

      {!token ? (
        <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">
          This link is missing its reset token — copy the full link from the email again, or{" "}
          <Link href="/forgot-password" className="underline">
            request a new one
          </Link>
          .
        </p>
      ) : (
        <>
          <p className="mb-6 text-sm text-slate-500">Choose a new password for your account.</p>

          {params.error && (
            <p className="mb-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{params.error}</p>
          )}

          <form action={resetPassword} className="flex flex-col gap-3">
            <input type="hidden" name="token" value={token} />
            <input
              name="password"
              type="password"
              placeholder="New password"
              required
              minLength={8}
              autoComplete="new-password"
              className="rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none"
            />
            <input
              name="confirm_password"
              type="password"
              placeholder="Confirm new password"
              required
              minLength={8}
              autoComplete="new-password"
              className="rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none"
            />
            <button
              type="submit"
              className="mt-1 rounded-md bg-slate-900 px-3 py-2 text-sm font-medium text-white hover:bg-slate-800"
            >
              Reset password
            </button>
          </form>
        </>
      )}

      <p className="mt-4 text-sm text-slate-500">
        <Link href="/login" className="font-medium text-slate-900 underline">
          Back to log in
        </Link>
      </p>
    </div>
  );
}
