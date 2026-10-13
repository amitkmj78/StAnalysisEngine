import Link from "next/link";

export default function CommunityGuidelinesPage() {
  return (
    <div className="mx-auto max-w-2xl px-4 py-10">
      <Link href="/community" className="text-sm text-[var(--pf-accent)] hover:underline">
        ← Community
      </Link>
      <h1 className="font-display mt-2 text-2xl font-semibold text-slate-900 dark:text-slate-50">Community Guidelines</h1>

      <div className="mt-3 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
        <strong>Draft wording, not reviewed by counsel (SAF-1).</strong> The rules below are real and enforced
        (accepting them is required to sign up), but the exact legal language has not been reviewed by a lawyer.
        This exists to get the mechanism right -- required acceptance, a real warn-then-ban enforcement path --
        ahead of that review, not to finalize the words.
      </div>

      <p className="mt-4 text-sm leading-relaxed text-slate-600 dark:text-slate-300">
        Every member agrees to these rules at sign-up. Violations lead to a warning first; repeated or serious
        violations lead to the account being deactivated.
      </p>

      <div className="mt-6 flex flex-col gap-5">
        <section>
          <h2 className="text-base font-semibold text-slate-900 dark:text-slate-50">No pump-and-dump or coordinated trading</h2>
          <p className="mt-1 text-sm leading-relaxed text-slate-700 dark:text-slate-200">
            Don&apos;t post to artificially inflate or deflate interest in a stock, and don&apos;t coordinate with
            others (here or elsewhere) to time posts or trades together for that purpose.
          </p>
        </section>
        <section>
          <h2 className="text-base font-semibold text-slate-900 dark:text-slate-50">Disclose paid promotion</h2>
          <p className="mt-1 text-sm leading-relaxed text-slate-700 dark:text-slate-200">
            If you were paid or compensated to post about a security, label the post as a paid promotion. Undisclosed
            paid promotion is a guidelines violation regardless of whether the underlying information is accurate.
          </p>
        </section>
        <section>
          <h2 className="text-base font-semibold text-slate-900 dark:text-slate-50">No impersonation</h2>
          <p className="mt-1 text-sm leading-relaxed text-slate-700 dark:text-slate-200">
            Don&apos;t represent yourself as someone else, as a company or fund you&apos;re not affiliated with, or
            as having credentials or a track record that isn&apos;t yours.
          </p>
        </section>
        <section>
          <h2 className="text-base font-semibold text-slate-900 dark:text-slate-50">No harassment</h2>
          <p className="mt-1 text-sm leading-relaxed text-slate-700 dark:text-slate-200">
            Don&apos;t target another member with repeated unwanted contact, threats, or abuse. Use block and mute
            (in Settings) if another member&apos;s content isn&apos;t for you -- you don&apos;t need their
            cooperation to stop seeing them.
          </p>
        </section>
        <section>
          <h2 className="text-base font-semibold text-slate-900 dark:text-slate-50">Disclose your position</h2>
          <p className="mt-1 text-sm leading-relaxed text-slate-700 dark:text-slate-200">
            A post about a specific ticker must say whether you hold it. This is self-declared unless you&apos;ve
            linked a brokerage account showing a real position in that ticker, in which case it&apos;s shown as
            verified.
          </p>
        </section>
        <section>
          <h2 className="text-base font-semibold text-slate-900 dark:text-slate-50">Not investment advice</h2>
          <p className="mt-1 text-sm leading-relaxed text-slate-700 dark:text-slate-200">
            Nothing posted here, by another member or by this app&apos;s own model, is individualized investment
            advice or a recommendation to buy or sell any security. Past performance does not indicate future
            results.
          </p>
        </section>
      </div>

      <p className="mt-6 text-xs text-slate-500 dark:text-slate-400">
        Report a post, comment, group or profile that violates these guidelines using the Report action wherever you
        see it. Reports go to a moderation queue for review.
      </p>
    </div>
  );
}
