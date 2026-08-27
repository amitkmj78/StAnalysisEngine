"use client";

import { useEffect, useState } from "react";

import {
  ApiError,
  approveUser,
  deactivateUser,
  deleteUser,
  getAdminUsers,
  reactivateUser,
  rejectUser,
  sendWelcomeEmail,
} from "@/lib/api";
import type { AdminUser } from "@/lib/types";

export default function UserApprovalPanel({ currentUserEmail }: { currentUserEmail: string }) {
  const [users, setUsers] = useState<AdminUser[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [emailStatus, setEmailStatus] = useState<Record<string, "sent" | "failed">>({});

  async function load() {
    setError(null);
    try {
      setUsers(await getAdminUsers());
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load users.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleApprove(id: string) {
    setBusyId(id);
    try {
      await approveUser(id);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Approve failed.");
    } finally {
      setBusyId(null);
    }
  }

  async function handleReject(id: string) {
    setBusyId(id);
    try {
      await rejectUser(id);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Reject failed.");
    } finally {
      setBusyId(null);
    }
  }

  async function handleSendWelcomeEmail(id: string) {
    setBusyId(id);
    setEmailStatus((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
    try {
      await sendWelcomeEmail(id);
      setEmailStatus((prev) => ({ ...prev, [id]: "sent" }));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to send welcome email.");
      setEmailStatus((prev) => ({ ...prev, [id]: "failed" }));
    } finally {
      setBusyId(null);
    }
  }

  async function handleDeactivate(id: string, email: string) {
    if (!window.confirm(`Deactivate ${email}? They won't be able to log in again until reactivated. Their data is kept.`)) {
      return;
    }
    setBusyId(id);
    try {
      await deactivateUser(id);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Deactivate failed.");
    } finally {
      setBusyId(null);
    }
  }

  async function handleReactivate(id: string) {
    setBusyId(id);
    try {
      await reactivateUser(id);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Reactivate failed.");
    } finally {
      setBusyId(null);
    }
  }

  async function handleDelete(id: string, email: string) {
    if (!window.confirm(`Delete ${email}? This removes their account and all their saved data — this can't be undone.`)) {
      return;
    }
    setBusyId(id);
    try {
      await deleteUser(id);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Delete failed.");
    } finally {
      setBusyId(null);
    }
  }

  if (users === null && !error) {
    return <p className="text-sm text-slate-500">Loading users…</p>;
  }

  const pending = (users ?? []).filter((u) => !u.approved);
  const approved = (users ?? []).filter((u) => u.approved);

  return (
    <div className="flex flex-col gap-6">
      {error && <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <div>
        <h2 className="text-lg font-semibold text-slate-900">
          Pending Approval {pending.length > 0 && <span className="text-amber-600">({pending.length})</span>}
        </h2>
        {pending.length === 0 ? (
          <p className="mt-1 text-sm text-slate-500">No signups waiting on approval.</p>
        ) : (
          <div className="mt-3 overflow-x-auto rounded-lg border border-slate-200 bg-white">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                  <th className="px-3 py-2">Email</th>
                  <th className="px-3 py-2">Signed Up</th>
                  <th className="px-3 py-2"></th>
                </tr>
              </thead>
              <tbody>
                {pending.map((u) => (
                  <tr key={u.id} className="border-b border-slate-100 last:border-0">
                    <td className="px-3 py-2 text-slate-700">{u.email}</td>
                    <td className="px-3 py-2 text-slate-500">{new Date(u.created_at).toLocaleString()}</td>
                    <td className="px-3 py-2 text-right">
                      <div className="flex justify-end gap-2">
                        <button
                          onClick={() => handleApprove(u.id)}
                          disabled={busyId === u.id}
                          className="rounded-md bg-emerald-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
                        >
                          Approve
                        </button>
                        <button
                          onClick={() => handleReject(u.id)}
                          disabled={busyId === u.id}
                          className="rounded-md border border-slate-300 px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
                        >
                          Reject
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div>
        <h2 className="text-lg font-semibold text-slate-900">Approved Users ({approved.length})</h2>
        <div className="mt-3 overflow-x-auto rounded-lg border border-slate-200 bg-white">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                <th className="px-3 py-2">Email</th>
                <th className="px-3 py-2">Status</th>
                <th className="px-3 py-2">Portfolios</th>
                <th className="px-3 py-2">Signed Up</th>
                <th className="px-3 py-2"></th>
              </tr>
            </thead>
            <tbody>
              {approved.map((u) => (
                <tr key={u.id} className="border-b border-slate-100 last:border-0">
                  <td className="px-3 py-2 text-slate-700">{u.email}</td>
                  <td className="px-3 py-2">
                    <span
                      className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                        u.is_active ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-500"
                      }`}
                    >
                      {u.is_active ? "Active" : "Inactive"}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-slate-600">
                    {u.portfolio_count > 0 ? (
                      <>
                        {u.portfolio_count} portfolio{u.portfolio_count === 1 ? "" : "s"}
                        <span className="text-slate-400"> · {u.position_count} position{u.position_count === 1 ? "" : "s"}</span>
                      </>
                    ) : (
                      <span className="text-slate-400">None</span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-slate-500">{new Date(u.created_at).toLocaleString()}</td>
                  <td className="px-3 py-2 text-right">
                    <div className="flex items-center justify-end gap-2">
                      {emailStatus[u.id] === "sent" && (
                        <span className="text-xs font-medium text-emerald-700">Sent</span>
                      )}
                      {emailStatus[u.id] === "failed" && (
                        <span className="text-xs font-medium text-red-700">Failed</span>
                      )}
                      <button
                        onClick={() => handleSendWelcomeEmail(u.id)}
                        disabled={busyId === u.id}
                        className="rounded-md border border-slate-300 px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
                      >
                        {busyId === u.id ? "Sending…" : "Send Welcome Email"}
                      </button>
                      {u.email.toLowerCase() !== currentUserEmail.toLowerCase() && (
                        <>
                          {u.is_active ? (
                            <button
                              onClick={() => handleDeactivate(u.id, u.email)}
                              disabled={busyId === u.id}
                              className="rounded-md border border-amber-200 px-2.5 py-1 text-xs font-medium text-amber-700 hover:bg-amber-50 disabled:opacity-50"
                            >
                              Deactivate
                            </button>
                          ) : (
                            <button
                              onClick={() => handleReactivate(u.id)}
                              disabled={busyId === u.id}
                              className="rounded-md border border-emerald-200 px-2.5 py-1 text-xs font-medium text-emerald-700 hover:bg-emerald-50 disabled:opacity-50"
                            >
                              Reactivate
                            </button>
                          )}
                          <button
                            onClick={() => handleDelete(u.id, u.email)}
                            disabled={busyId === u.id}
                            className="rounded-md border border-red-200 px-2.5 py-1 text-xs font-medium text-red-700 hover:bg-red-50 disabled:opacity-50"
                          >
                            Delete
                          </button>
                        </>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
