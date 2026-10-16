import { redirect } from "next/navigation";

import { verifyTwoFactorLogin } from "../actions";

export default async function TwoFactorLoginPage({
  searchParams,
}: {
  searchParams: Promise<{ token?: string; error?: string }>;
}) {
  const params = await searchParams;
  if (!params.token) {
    redirect("/login");
  }

  return (
    <div className="mx-auto mt-24 max-w-sm px-4">
      <h1 className="font-display mb-1 text-2xl font-semibold text-slate-900 dark:text-slate-50">Enter your code</h1>
      <p className="mb-6 text-sm text-slate-500 dark:text-slate-400">
        Open your authenticator app and enter the 6-digit code for this account.
      </p>

      {params.error && (
        <p className="mb-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{params.error}</p>
      )}

      <form action={verifyTwoFactorLogin} className="flex flex-col gap-3">
        <input type="hidden" name="partial_token" value={params.token} />
        <input
          name="code"
          type="text"
          inputMode="numeric"
          pattern="[0-9]{6}"
          maxLength={6}
          placeholder="123456"
          required
          autoComplete="one-time-code"
          className="rounded-md border border-slate-300 dark:border-slate-600 px-3 py-2 text-center text-lg tracking-[0.3em] focus:border-slate-500 dark:focus:border-slate-400 focus:outline-none"
        />
        <button
          type="submit"
          className="mt-1 rounded-md bg-slate-900 px-3 py-2 text-sm font-medium text-white hover:bg-slate-800"
        >
          Verify
        </button>
      </form>

      <p className="mt-4 text-sm">
        <a href="/login" className="text-slate-500 dark:text-slate-400 underline hover:text-slate-700 dark:hover:text-slate-200">
          Back to log in
        </a>
      </p>
    </div>
  );
}
