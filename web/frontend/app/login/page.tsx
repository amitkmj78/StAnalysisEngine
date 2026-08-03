import Link from "next/link";

import { login } from "./actions";

export default async function LoginPage({
  searchParams,
}: {
  searchParams: Promise<{ error?: string; info?: string }>;
}) {
  const params = await searchParams;

  return (
    <div className="mx-auto mt-24 max-w-sm px-4">
      <h1 className="mb-1 text-2xl font-semibold text-slate-900">Log in</h1>
      <p className="mb-1 text-sm text-slate-500">AI Price Prediction &amp; Best Stock Finder</p>
      <p className="mb-6 flex flex-wrap gap-x-4 gap-y-1 text-sm">
        <a href="/marketing.html" target="_blank" rel="noopener noreferrer" className="text-emerald-700 underline hover:text-emerald-800">
          How it works &amp; what&apos;s inside
        </a>
        <Link href="/track-record" className="text-emerald-700 underline hover:text-emerald-800">
          Live track record
        </Link>
      </p>

      {params.info && (
        <p className="mb-4 rounded-md bg-blue-50 px-3 py-2 text-sm text-blue-700">{params.info}</p>
      )}
      {params.error && (
        <p className="mb-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{params.error}</p>
      )}

      <form action={login} className="flex flex-col gap-3">
        <input
          name="email"
          type="email"
          placeholder="Email"
          required
          autoComplete="email"
          className="rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none"
        />
        <input
          name="password"
          type="password"
          placeholder="Password"
          required
          autoComplete="current-password"
          className="rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none"
        />
        <button
          type="submit"
          className="mt-1 rounded-md bg-slate-900 px-3 py-2 text-sm font-medium text-white hover:bg-slate-800"
        >
          Log in
        </button>
      </form>

      <p className="mt-4 text-sm text-slate-500">
        No account?{" "}
        <Link href="/signup" className="font-medium text-slate-900 underline">
          Sign up
        </Link>
      </p>
    </div>
  );
}
