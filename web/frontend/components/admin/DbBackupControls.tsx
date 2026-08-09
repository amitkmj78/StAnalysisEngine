"use client";

import { useEffect, useState } from "react";

import {
  ApiError,
  backupNow,
  disableDbBackup,
  enableDbBackup,
  getAdminSettings,
  getBackupStatus,
  restoreTestNow,
} from "@/lib/api";
import type { BackupRun } from "@/lib/types";

function formatBytes(bytes: number | null): string {
  if (bytes === null) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function DbBackupControls() {
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [runs, setRuns] = useState<BackupRun[] | null>(null);
  const [backupTables, setBackupTables] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [backingUp, setBackingUp] = useState(false);
  const [backupResult, setBackupResult] = useState<string | null>(null);
  const [backupError, setBackupError] = useState<string | null>(null);

  const [restoring, setRestoring] = useState(false);
  const [restoreResult, setRestoreResult] = useState<string | null>(null);
  const [restoreError, setRestoreError] = useState<string | null>(null);

  async function load() {
    setError(null);
    try {
      const [settings, status] = await Promise.all([getAdminSettings(), getBackupStatus()]);
      setEnabled(settings.db_backup_enabled);
      setRuns(status.recent_runs);
      setBackupTables(status.backup_tables);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load backup status.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleToggle() {
    setBusy(true);
    setError(null);
    try {
      const result = enabled ? await disableDbBackup() : await enableDbBackup();
      setEnabled(result.db_backup_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update backup setting.");
    } finally {
      setBusy(false);
    }
  }

  async function handleBackupNow() {
    setBackingUp(true);
    setBackupError(null);
    setBackupResult(null);
    try {
      const res = await backupNow();
      setBackupResult(
        res.error
          ? `Backup failed: ${res.error}`
          : `Backed up — structural check ${res.structural_check_passed ? "passed" : "FAILED"}.`
      );
      await load();
    } catch (err) {
      setBackupError(err instanceof ApiError ? err.message : "Failed to back up.");
    } finally {
      setBackingUp(false);
    }
  }

  async function handleRestoreTest() {
    setRestoring(true);
    setRestoreError(null);
    setRestoreResult(null);
    try {
      const res = await restoreTestNow();
      if (!res.restore_succeeded) {
        setRestoreResult(`Restore failed: ${res.error ?? "unknown error"}`);
      } else {
        const mismatches = Object.entries(res.row_counts).filter(([, v]) => !v.match);
        setRestoreResult(
          res.all_match
            ? "Restore test passed — every table's row count matched exactly."
            : `Restore succeeded but ${mismatches.length} table(s) had a row-count mismatch.`
        );
      }
      await load();
    } catch (err) {
      setRestoreError(err instanceof ApiError ? err.message : "Failed to run restore test.");
    } finally {
      setRestoring(false);
    }
  }

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="font-semibold text-slate-900">Database Backups (NFR-03)</h2>
          <p className="mt-1 text-sm text-slate-600">
            Nightly backup of the published record and PIT store ({backupTables.join(", ") || "…"}) to S3, with a
            free structural-integrity check on every run. A real restore-into-a-throwaway-database test — the
            actual "restore-tested" requirement — runs automatically every quarter, or on demand below.
          </p>
        </div>
        {enabled !== null && (
          <span
            className={`shrink-0 rounded-full px-3 py-1 text-xs font-medium ${
              enabled ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-500"
            }`}
          >
            {enabled ? "Backing up" : "Paused"}
          </span>
        )}
      </div>

      {error && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <button
          onClick={handleToggle}
          disabled={busy || enabled === null}
          className={`rounded-md px-4 py-2 text-sm font-medium disabled:opacity-50 ${
            enabled
              ? "border border-red-200 text-red-700 hover:bg-red-50"
              : "bg-slate-900 text-white hover:bg-slate-800"
          }`}
        >
          {busy ? "Updating…" : enabled ? "Pause" : "Resume"}
        </button>
        <button
          onClick={handleBackupNow}
          disabled={backingUp}
          className="rounded-md border border-emerald-300 bg-white px-4 py-2 text-sm font-medium text-emerald-700 hover:bg-emerald-50 disabled:opacity-50"
        >
          {backingUp ? "Backing up…" : "Backup Now"}
        </button>
        <button
          onClick={handleRestoreTest}
          disabled={restoring}
          className="rounded-md border border-indigo-300 bg-white px-4 py-2 text-sm font-medium text-indigo-700 hover:bg-indigo-50 disabled:opacity-50"
        >
          {restoring ? "Restoring…" : "Run Restore Test"}
        </button>
      </div>

      {backupResult && <p className="mt-3 text-sm text-emerald-700">{backupResult}</p>}
      {backupError && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{backupError}</p>}
      {restoreResult && <p className="mt-2 text-sm text-emerald-700">{restoreResult}</p>}
      {restoreError && <p className="mt-2 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{restoreError}</p>}

      {runs && runs.length > 0 && (
        <div className="mt-4 overflow-x-auto rounded-md border border-slate-200">
          <table className="min-w-full text-xs">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-left uppercase tracking-wide text-slate-500">
                <th className="px-2 py-1.5">Started</th>
                <th className="px-2 py-1.5">Size</th>
                <th className="px-2 py-1.5">Structural check</th>
                <th className="px-2 py-1.5">Restore test</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.id} className="border-b border-slate-100 last:border-0">
                  <td className="px-2 py-1.5 text-slate-600">{new Date(run.started_at_utc).toLocaleString()}</td>
                  <td className="px-2 py-1.5 text-slate-600">{formatBytes(run.size_bytes)}</td>
                  <td className="px-2 py-1.5">
                    {run.error ? (
                      <span className="text-red-600" title={run.error}>Failed</span>
                    ) : run.structural_check_passed ? (
                      <span className="text-emerald-600">Passed</span>
                    ) : (
                      <span className="text-amber-600">—</span>
                    )}
                  </td>
                  <td className="px-2 py-1.5">
                    {!run.restore_test_run ? (
                      <span className="text-slate-400">Not tested</span>
                    ) : run.restore_test_passed ? (
                      <span className="text-emerald-600">Passed</span>
                    ) : (
                      <span className="text-red-600">Failed</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
