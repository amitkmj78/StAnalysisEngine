"use client";

import { useEffect, useState } from "react";

import { ApiError, disablePublishSignals, enablePublishSignals, getAdminSettings, publishSignalsNow } from "@/lib/api";

export default function PublishSignalsControls() {
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [confirming, setConfirming] = useState(false);

  const [publishing, setPublishing] = useState(false);
  const [publishResult, setPublishResult] = useState<string | null>(null);
  const [publishError, setPublishError] = useState<string | null>(null);

  async function load() {
    setError(null);
    try {
      const settings = await getAdminSettings();
      setEnabled(settings.publish_signals_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load publication settings.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleToggle() {
    if (!enabled && !confirming) {
      setConfirming(true);
      return;
    }
    setBusy(true);
    setError(null);
    setConfirming(false);
    try {
      const result = enabled ? await disablePublishSignals() : await enablePublishSignals();
      setEnabled(result.publish_signals_enabled);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to update publication setting.");
    } finally {
      setBusy(false);
    }
  }

  async function handlePublishNow() {
    setPublishing(true);
    setPublishError(null);
    setPublishResult(null);
    try {
      const res = await publishSignalsNow();
      setPublishResult(
        res.published > 0
          ? `Published ${res.published} signals just now.`
          : "Already published for today — nothing new to publish."
      );
    } catch (err) {
      setPublishError(err instanceof ApiError ? err.message : "Failed to publish.");
    } finally {
      setPublishing(false);
    }
  }

  return (
    <div className="rounded-lg border border-amber-200 bg-amber-50/40 p-5">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="font-semibold text-slate-900">Public Signal Publication</h2>
          <p className="mt-1 text-sm text-slate-600">
            Daily scheduled job (weekdays, 4:10pm ET) that commits the top-25 momentum ranking to the public,
            append-only track record. This is irreversible once turned on — enabling it starts the real
            publication clock, not a preview. Do not enable before counsel has confirmed unpaid, impersonal
            publication carries no registration requirement.
          </p>
          <p className="mt-2 text-xs text-slate-500">
            Note: enabling this only affects <em>future</em> scheduled runs — if today&apos;s 4:10pm ET run
            already passed, use &quot;Publish Now&quot; below to catch up today instead of waiting until
            tomorrow.
          </p>
        </div>
        {enabled !== null && (
          <span
            className={`shrink-0 rounded-full px-3 py-1 text-xs font-medium ${
              enabled ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-500"
            }`}
          >
            {enabled ? "Publishing" : "Disabled"}
          </span>
        )}
      </div>

      {error && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {confirming && !enabled && (
        <p className="mt-3 rounded-md bg-amber-100 px-3 py-2 text-sm text-amber-800">
          Click again to confirm — this starts the live public track record.
        </p>
      )}

      <div className="mt-4 flex items-center gap-2">
        <button
          onClick={handleToggle}
          disabled={busy || enabled === null}
          className={`rounded-md px-4 py-2 text-sm font-medium disabled:opacity-50 ${
            enabled
              ? "border border-red-200 text-red-700 hover:bg-red-50"
              : confirming
              ? "bg-amber-600 text-white hover:bg-amber-700"
              : "bg-slate-900 text-white hover:bg-slate-800"
          }`}
        >
          {busy ? "Updating…" : enabled ? "Disable" : confirming ? "Confirm Enable" : "Enable Publishing"}
        </button>
        {confirming && !enabled && (
          <button
            onClick={() => setConfirming(false)}
            disabled={busy}
            className="rounded-md border border-slate-300 px-3 py-2 text-sm font-medium text-slate-600 hover:bg-slate-100"
          >
            Cancel
          </button>
        )}
        {enabled && (
          <button
            onClick={handlePublishNow}
            disabled={publishing}
            className="rounded-md border border-emerald-300 bg-white px-4 py-2 text-sm font-medium text-emerald-700 hover:bg-emerald-50 disabled:opacity-50"
          >
            {publishing ? "Publishing…" : "Publish Now"}
          </button>
        )}
      </div>

      {publishResult && <p className="mt-3 text-sm text-emerald-700">{publishResult}</p>}
      {publishError && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{publishError}</p>}
    </div>
  );
}
