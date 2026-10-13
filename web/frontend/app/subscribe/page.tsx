"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { DisclosureBanner } from "@/components/DisclosureBanner";
import { ApiError, getMySubscription, openBillingPortal, startCheckout, submitEnquiry } from "@/lib/api";
import type { MySubscription } from "@/lib/types";

export default function SubscribePage() {
  const [subscription, setSubscription] = useState<MySubscription | null>(null);
  const [loggedIn, setLoggedIn] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [enquiryType, setEnquiryType] = useState("licensing");
  const [enquiryEmail, setEnquiryEmail] = useState("");
  const [enquiryMessage, setEnquiryMessage] = useState("");
  const [enquirySubmitted, setEnquirySubmitted] = useState(false);
  const [enquiryError, setEnquiryError] = useState<string | null>(null);

  useEffect(() => {
    getMySubscription()
      .then((sub) => {
        setSubscription(sub);
        setLoggedIn(true);
      })
      .catch((err) => {
        if (err instanceof ApiError && err.status === 401) {
          setLoggedIn(false);
        } else {
          setError(err instanceof ApiError ? err.message : "Failed to load subscription status.");
        }
      });
  }, []);

  async function handleSubscribe() {
    setBusy(true);
    setError(null);
    try {
      const { url } = await startCheckout();
      window.location.href = url;
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to start checkout.");
      setBusy(false);
    }
  }

  async function handleManage() {
    setBusy(true);
    setError(null);
    try {
      const { url } = await openBillingPortal();
      window.location.href = url;
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to open the billing portal.");
      setBusy(false);
    }
  }

  async function handleEnquirySubmit(e: React.FormEvent) {
    e.preventDefault();
    setEnquiryError(null);
    try {
      await submitEnquiry({ enquiry_type: enquiryType, contact_email: enquiryEmail, message: enquiryMessage });
      setEnquirySubmitted(true);
    } catch (err) {
      setEnquiryError(err instanceof ApiError ? err.message : "Failed to submit — try again.");
    }
  }

  const isPaid = subscription?.tier === "paid" && subscription?.status === "active";

  return (
    <div className="mx-auto max-w-2xl px-4 py-10">
      <h1 className="font-display text-2xl font-semibold text-slate-900 dark:text-slate-50">Subscribe</h1>
      <p className="mt-2 text-sm leading-relaxed text-slate-600 dark:text-slate-300">
        Get current-day rankings instead of the free tier&apos;s delayed view, plus full CSV export
        of the published history.
      </p>
      <div className="mt-4">
        <DisclosureBanner />
      </div>

      {error && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <div className="mt-6 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5">
        {loggedIn === null && <p className="text-sm text-slate-500 dark:text-slate-400">Loading…</p>}

        {loggedIn === false && (
          <p className="text-sm text-slate-600 dark:text-slate-300">
            <Link href="/login" className="font-medium text-slate-900 dark:text-slate-50 underline">
              Sign in
            </Link>{" "}
            to subscribe.
          </p>
        )}

        {loggedIn && subscription && (
          <>
            <p className="text-sm text-slate-600 dark:text-slate-300">
              Current plan: <span className="font-medium text-slate-900 dark:text-slate-50">{isPaid ? "Paid" : "Free"}</span>
              {subscription.current_period_end && isPaid && (
                <> — renews {new Date(subscription.current_period_end).toLocaleDateString()}</>
              )}
            </p>
            <button
              onClick={isPaid ? handleManage : handleSubscribe}
              disabled={busy}
              className="mt-4 rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
            >
              {busy ? "Loading…" : isPaid ? "Manage subscription" : "Subscribe"}
            </button>
          </>
        )}
      </div>

      <div className="mt-10">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-50">Licensing, API, or institutional access</h2>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">Not what you&apos;re looking for? Tell us what you need.</p>
        {enquirySubmitted ? (
          <p className="mt-4 rounded-md bg-emerald-50 px-3 py-2 text-sm text-emerald-700">
            Thanks — we&apos;ll be in touch.
          </p>
        ) : (
          <form
            onSubmit={handleEnquirySubmit}
            className="mt-4 flex flex-col gap-3 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5"
          >
            <select
              value={enquiryType}
              onChange={(e) => setEnquiryType(e.target.value)}
              className="rounded-md border border-slate-300 dark:border-slate-600 px-3 py-2 text-sm"
            >
              <option value="licensing">Signal licensing</option>
              <option value="api">API access</option>
              <option value="institutional">Institutional</option>
              <option value="other">Other</option>
            </select>
            <input
              type="email"
              required
              placeholder="Your email"
              value={enquiryEmail}
              onChange={(e) => setEnquiryEmail(e.target.value)}
              className="rounded-md border border-slate-300 dark:border-slate-600 px-3 py-2 text-sm"
            />
            <textarea
              placeholder="What are you looking for?"
              value={enquiryMessage}
              onChange={(e) => setEnquiryMessage(e.target.value)}
              rows={3}
              className="rounded-md border border-slate-300 dark:border-slate-600 px-3 py-2 text-sm"
            />
            {enquiryError && <p className="text-sm text-red-700">{enquiryError}</p>}
            <button
              type="submit"
              className="self-start rounded-md border border-slate-300 dark:border-slate-600 px-4 py-2 text-sm font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700"
            >
              Submit
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
