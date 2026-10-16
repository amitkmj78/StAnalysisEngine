"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { ApiError, confirm2FA, disable2FA, enroll2FA, getAccountAuditLog, getCurrentUser } from "@/lib/api";
import type { AccountAuditLogEntry, TwoFactorEnrollResponse } from "@/lib/types";

export default function SecuritySettingsPage() {
  const [loading, setLoading] = useState(true);
  const [totpEnabled, setTotpEnabled] = useState(false);

  const [enrollment, setEnrollment] = useState<TwoFactorEnrollResponse | null>(null);
  const [enrollError, setEnrollError] = useState<string | null>(null);
  const [enrolling, setEnrolling] = useState(false);

  const [code, setCode] = useState("");
  const [confirming, setConfirming] = useState(false);
  const [confirmError, setConfirmError] = useState<string | null>(null);

  const [disablePassword, setDisablePassword] = useState("");
  const [disabling, setDisabling] = useState(false);
  const [disableError, setDisableError] = useState<string | null>(null);
  const [showDisableForm, setShowDisableForm] = useState(false);

  const [auditLog, setAuditLog] = useState<AccountAuditLogEntry[] | null>(null);
  const [auditLogError, setAuditLogError] = useState<string | null>(null);

  useEffect(() => {
    getCurrentUser()
      .then((user) => setTotpEnabled(user.totp_enabled))
      .catch(() => setEnrollError("Could not load your security settings."))
      .finally(() => setLoading(false));
    getAccountAuditLog()
      .then((res) => setAuditLog(res.events))
      .catch((err) => setAuditLogError(err instanceof ApiError ? err.message : "Could not load your account activity."));
  }, []);

  async function handleStartEnroll() {
    setEnrolling(true);
    setEnrollError(null);
    try {
      const res = await enroll2FA();
      setEnrollment(res);
    } catch (err) {
      setEnrollError(err instanceof ApiError ? err.message : "Could not start enrollment.");
    } finally {
      setEnrolling(false);
    }
  }

  async function handleConfirm() {
    if (!code.trim()) return;
    setConfirming(true);
    setConfirmError(null);
    try {
      await confirm2FA(code.trim());
      setTotpEnabled(true);
      setEnrollment(null);
      setCode("");
    } catch (err) {
      setConfirmError(err instanceof ApiError ? err.message : "That code didn't work.");
    } finally {
      setConfirming(false);
    }
  }

  async function handleDisable() {
    if (!disablePassword) return;
    setDisabling(true);
    setDisableError(null);
    try {
      await disable2FA(disablePassword);
      setTotpEnabled(false);
      setShowDisableForm(false);
      setDisablePassword("");
    } catch (err) {
      setDisableError(err instanceof ApiError ? err.message : "Could not disable two-factor authentication.");
    } finally {
      setDisabling(false);
    }
  }

  return (
    <div className="mx-auto max-w-2xl px-4 py-8">
      <h1 className="font-display text-2xl font-semibold text-slate-900 dark:text-slate-50">Security</h1>
      <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
        <Link href="/settings" className="underline decoration-dotted">
          Back to settings
        </Link>
      </p>

      <div className="mt-6 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5">
        <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-50">Two-factor authentication</h2>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Require a 6-digit code from an authenticator app (like Google Authenticator or Authy) in addition to your
          password when logging in.
        </p>

        {loading ? (
          <p className="mt-3 text-sm text-slate-500 dark:text-slate-400">Loading…</p>
        ) : totpEnabled ? (
          <div className="mt-4">
            <p className="text-sm font-medium text-emerald-700">Two-factor authentication is on.</p>
            {!showDisableForm ? (
              <button
                type="button"
                onClick={() => setShowDisableForm(true)}
                className="mt-3 rounded-md border border-red-300 px-3 py-1.5 text-sm font-medium text-red-700 hover:bg-red-50"
              >
                Turn off
              </button>
            ) : (
              <div className="mt-3 flex flex-col gap-2">
                <label htmlFor="disable-2fa-password" className="text-sm text-slate-700 dark:text-slate-200">
                  Confirm your password to turn off two-factor authentication.
                </label>
                <div className="flex gap-2">
                  <input
                    id="disable-2fa-password"
                    type="password"
                    value={disablePassword}
                    onChange={(e) => setDisablePassword(e.target.value)}
                    placeholder="Current password"
                    autoComplete="current-password"
                    className="w-56 rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 px-3 py-1.5 text-sm text-slate-900 dark:text-slate-50"
                  />
                  <button
                    type="button"
                    onClick={handleDisable}
                    disabled={disabling || !disablePassword}
                    className="rounded-md bg-red-700 px-3 py-1.5 text-sm font-medium text-white hover:bg-red-800 disabled:opacity-50"
                  >
                    {disabling ? "Turning off…" : "Confirm"}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setShowDisableForm(false);
                      setDisablePassword("");
                      setDisableError(null);
                    }}
                    className="rounded-md border border-slate-300 dark:border-slate-600 px-3 py-1.5 text-sm text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800"
                  >
                    Cancel
                  </button>
                </div>
                {disableError && <p className="text-xs font-medium text-red-700">{disableError}</p>}
              </div>
            )}
          </div>
        ) : enrollment ? (
          <div className="mt-4 flex flex-col gap-3">
            <p className="text-sm text-slate-700 dark:text-slate-200">
              Scan this QR code with your authenticator app, or enter the key manually.
            </p>
            <img src={enrollment.qr_code_data_uri} alt="2FA QR code" className="h-40 w-40 rounded-md border border-slate-200 dark:border-slate-700" />
            <p className="font-mono text-xs text-slate-500 dark:text-slate-400 break-all">{enrollment.secret}</p>
            <label htmlFor="2fa-confirm-code" className="text-sm text-slate-700 dark:text-slate-200">Then enter the 6-digit code it shows:</label>
            <div className="flex gap-2">
              <input
                id="2fa-confirm-code"
                value={code}
                onChange={(e) => setCode(e.target.value)}
                type="text"
                inputMode="numeric"
                pattern="[0-9]{6}"
                maxLength={6}
                placeholder="123456"
                autoComplete="one-time-code"
                className="w-32 rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 px-3 py-1.5 text-center text-sm tracking-widest text-slate-900 dark:text-slate-50"
              />
              <button
                type="button"
                onClick={handleConfirm}
                disabled={confirming || code.trim().length !== 6}
                className="rounded-md bg-slate-900 dark:bg-slate-100 px-3 py-1.5 text-sm font-medium text-white dark:text-slate-900 hover:bg-slate-800 dark:hover:bg-slate-200 disabled:opacity-50"
              >
                {confirming ? "Checking…" : "Confirm"}
              </button>
              <button
                type="button"
                onClick={() => {
                  setEnrollment(null);
                  setCode("");
                  setConfirmError(null);
                }}
                className="rounded-md border border-slate-300 dark:border-slate-600 px-3 py-1.5 text-sm text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800"
              >
                Cancel
              </button>
            </div>
            {confirmError && <p className="text-xs font-medium text-red-700">{confirmError}</p>}
          </div>
        ) : (
          <div className="mt-3">
            <button
              type="button"
              onClick={handleStartEnroll}
              disabled={enrolling}
              className="rounded-md bg-slate-900 dark:bg-slate-100 px-3 py-1.5 text-sm font-medium text-white dark:text-slate-900 hover:bg-slate-800 dark:hover:bg-slate-200 disabled:opacity-50"
            >
              {enrolling ? "Starting…" : "Set up two-factor authentication"}
            </button>
            {enrollError && <p className="mt-2 text-xs font-medium text-red-700">{enrollError}</p>}
          </div>
        )}
      </div>

      <div className="mt-6 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5">
        <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-50">Account activity</h2>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Logins, password resets, and security changes on your account. Most recent first.
        </p>
        {auditLogError && <p className="mt-3 text-xs font-medium text-red-700">{auditLogError}</p>}
        {auditLog && auditLog.length === 0 && (
          <p className="mt-3 text-sm text-slate-500 dark:text-slate-400">No activity recorded yet.</p>
        )}
        {auditLog && auditLog.length > 0 && (
          <ul className="mt-3 flex flex-col divide-y divide-slate-100 dark:divide-slate-800 text-sm">
            {auditLog.map((entry, i) => (
              <li key={i} className="flex items-center justify-between gap-2 py-2">
                <span className="text-slate-700 dark:text-slate-200">{describeAuditEvent(entry)}</span>
                <span className="shrink-0 text-xs text-slate-400 dark:text-slate-500">
                  {new Date(entry.created_at).toLocaleString()}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function describeAuditEvent(entry: AccountAuditLogEntry): string {
  const detail = entry.detail ?? {};
  switch (entry.event_type) {
    case "login_success":
      return detail.via === "2fa" ? "Logged in (with two-factor code)" : "Logged in";
    case "login_failed":
      return detail.reason === "bad_2fa_code" ? "Failed login attempt (wrong two-factor code)" : "Failed login attempt (wrong password)";
    case "login_password_ok_2fa_pending":
      return "Password accepted, waiting on two-factor code";
    case "2fa_enabled":
      return "Two-factor authentication turned on";
    case "2fa_disabled":
      return "Two-factor authentication turned off";
    case "password_reset":
      return "Password reset";
    case "role_changed":
      return `Role changed to ${detail.new_role ?? "?"}`;
    case "account_deactivated":
      return "Account deactivated";
    case "account_reactivated":
      return "Account reactivated";
    default:
      return entry.event_type;
  }
}
