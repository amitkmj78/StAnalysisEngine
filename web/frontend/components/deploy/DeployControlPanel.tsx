"use client";

import { useEffect, useRef, useState } from "react";

import { ApiError } from "@/lib/api";
import { awsDeployApi, type AwsConfig, type AwsStatus, type AwsVerifyResult, type JobPollResult } from "@/lib/aws-deploy-api";

const REGIONS = ["us-east-1", "us-east-2", "us-west-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-southeast-1"];
const INSTANCE_TYPES = ["t3.micro", "t3.small", "t3.medium", "t3.large"];

interface JobState {
  jobId: string;
  action: string;
  status: JobPollResult["status"];
  lines: string[];
  startedAt: string;
  finishedAt: string | null;
}

function lineColor(l: string): string {
  if (l.includes("✓")) return "text-emerald-400";
  if (l.includes("✗")) return "text-red-400";
  if (l.includes("⚠")) return "text-amber-400";
  return "text-slate-300";
}

function elapsed(start: string, end: string | null) {
  const ms = (end ? new Date(end) : new Date()).getTime() - new Date(start).getTime();
  return ms < 60000 ? `${(ms / 1000).toFixed(0)}s` : `${(ms / 60000).toFixed(1)}m`;
}

function JobConsole({ job, onClose, onCancel }: { job: JobState; onClose: () => void; onCancel: () => void }) {
  const running = job.status === "running";
  const [cancelling, setCancelling] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [job.lines.length]);

  async function handleCancel() {
    setCancelling(true);
    try {
      await awsDeployApi.cancelJob(job.jobId);
    } catch {
      // best-effort
    }
    onCancel();
    setCancelling(false);
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/55">
      <div className="flex max-h-[80vh] w-[min(92vw,800px)] flex-col rounded-xl border border-slate-700 bg-slate-900 shadow-2xl">
        <div className="flex items-center justify-between border-b border-slate-800 px-5 py-3">
          <div className="flex items-center gap-3">
            <span className="text-sm font-semibold text-slate-100">{job.action}</span>
            <span
              className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
                running
                  ? "bg-amber-900/50 text-amber-300"
                  : job.status === "success"
                    ? "bg-emerald-900/50 text-emerald-300"
                    : "bg-red-900/50 text-red-300"
              }`}
            >
              {running ? "Running…" : job.status === "success" ? "Done" : "Failed"}
              {!running && ` · ${elapsed(job.startedAt, job.finishedAt)}`}
            </span>
          </div>
          <div className="flex gap-2">
            {running && (
              <button
                onClick={handleCancel}
                disabled={cancelling}
                className="rounded-md border border-amber-800 px-2.5 py-1 text-xs text-amber-300 hover:bg-amber-950 disabled:opacity-50"
              >
                {cancelling ? "…" : "Cancel"}
              </button>
            )}
            <button onClick={onClose} className="rounded-md border border-red-900 px-2.5 py-1 text-xs text-red-300 hover:bg-red-950">
              Close
            </button>
          </div>
        </div>
        <div ref={logRef} className="flex-1 overflow-y-auto px-5 py-4 font-mono text-xs leading-relaxed">
          {job.lines.length === 0 ? (
            <span className="text-slate-500">No output yet…</span>
          ) : (
            job.lines.map((l, i) => (
              <div key={i} className={lineColor(l)}>
                {l || " "}
              </div>
            ))
          )}
          {running && <div className="mt-1 text-slate-500">▌ running…</div>}
        </div>
      </div>
    </div>
  );
}

export default function DeployControlPanel() {
  const [config, setConfig] = useState<AwsConfig | null>(null);
  const [verify, setVerify] = useState<AwsVerifyResult | null>(null);
  const [status, setStatus] = useState<AwsStatus | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [configOpen, setConfigOpen] = useState(false);
  const [form, setForm] = useState({ access_key_id: "", secret_access_key: "", region: "us-east-1", key_name: "stanalysisengine-key" });
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [showSecret, setShowSecret] = useState(false);

  const [creatingKey, setCreatingKey] = useState(false);
  const [launchType, setLaunchType] = useState("t3.small");
  const [launchVolume, setLaunchVolume] = useState(20);
  const [launching, setLaunching] = useState(false);
  const [deployingIp, setDeployingIp] = useState<string | null>(null);

  const [activeJob, setActiveJob] = useState<JobState | null>(null);
  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  async function refresh() {
    try {
      const [cfg, ver, st] = await Promise.all([awsDeployApi.getConfig(), awsDeployApi.verifyConfig(), awsDeployApi.getStatus()]);
      setConfig(cfg);
      setForm((f) => ({ ...f, ...cfg }));
      setVerify(ver);
      setStatus(st);
      setLoadError(null);
    } catch (err) {
      setLoadError(err instanceof ApiError ? err.message : "Could not load AWS status.");
    }
  }

  useEffect(() => {
    refresh();
    return () => {
      if (pollRef.current) clearTimeout(pollRef.current);
    };
  }, []);

  function startPolling(jobId: string) {
    const poll = async (cursor: number) => {
      try {
        const r = await awsDeployApi.pollJob(jobId, cursor);
        setActiveJob((prev) => (prev ? { ...prev, status: r.status, lines: [...prev.lines, ...r.lines] } : prev));
        if (r.status === "running") {
          pollRef.current = setTimeout(() => poll(r.cursor), 2000);
        } else {
          setLaunching(false);
          setDeployingIp(null);
          refresh();
        }
      } catch {
        pollRef.current = setTimeout(() => poll(cursor), 3000);
      }
    };
    poll(0);
  }

  async function handleSaveConfig() {
    setSaving(true);
    try {
      await awsDeployApi.saveConfig(form);
      await refresh();
      setConfigOpen(false);
    } catch (err) {
      setLoadError(err instanceof ApiError ? err.message : "Save failed.");
    } finally {
      setSaving(false);
    }
  }

  async function handleTest() {
    setTesting(true);
    try {
      const r = await awsDeployApi.verifyConfig();
      setVerify(r);
    } finally {
      setTesting(false);
    }
  }

  async function handleCreateKeyPair() {
    setCreatingKey(true);
    try {
      await awsDeployApi.createKeyPair();
      await refresh();
    } catch (err) {
      setLoadError(err instanceof ApiError ? err.message : "Key pair creation failed.");
    } finally {
      setCreatingKey(false);
    }
  }

  async function handleLaunch() {
    setLaunching(true);
    try {
      const { job_id } = await awsDeployApi.launchInstance(launchType, launchVolume);
      setActiveJob({ jobId: job_id, action: `Launch EC2 ${launchType}`, status: "running", lines: [], startedAt: new Date().toISOString(), finishedAt: null });
      startPolling(job_id);
    } catch (err) {
      setLaunching(false);
      setLoadError(err instanceof ApiError ? err.message : "Launch failed.");
    }
  }

  async function handleDeploy(ip: string) {
    setDeployingIp(ip);
    try {
      const { job_id } = await awsDeployApi.deploy(ip);
      setActiveJob({ jobId: job_id, action: `Deploy → ${ip}`, status: "running", lines: [], startedAt: new Date().toISOString(), finishedAt: null });
      startPolling(job_id);
    } catch (err) {
      setDeployingIp(null);
      setLoadError(err instanceof ApiError ? err.message : "Deploy failed.");
    }
  }

  function closeJob() {
    if (pollRef.current) clearTimeout(pollRef.current);
    setActiveJob(null);
  }

  function cancelJob() {
    if (pollRef.current) clearTimeout(pollRef.current);
    setActiveJob((prev) => (prev ? { ...prev, status: "error" } : prev));
    setLaunching(false);
    setDeployingIp(null);
  }

  const connected = verify?.ok;
  const anyJobRunning = activeJob?.status === "running";
  const runningInstance = status?.instances.find((i) => i.state === "running" && i.public_ip);

  return (
    <div className="mb-8 rounded-xl border border-indigo-200 bg-indigo-50/30 p-5">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-900">Live Deploy Control Panel</h2>
        <span className="text-xs text-slate-500">Runs locally against your AWS account — never shipped to the deployed app</span>
      </div>

      {loadError && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{loadError}</p>}

      {/* Account config bar */}
      <div className="mt-4 overflow-hidden rounded-lg border border-slate-200 bg-white">
        <button onClick={() => setConfigOpen((o) => !o)} className="flex w-full items-center gap-3 px-4 py-3 text-left">
          <span className={`h-2 w-2 rounded-full ${connected ? "bg-emerald-500" : "bg-slate-300"}`} />
          <div className="flex-1">
            <div className="text-sm font-medium text-slate-900">AWS credentials</div>
            <div className="text-xs text-slate-500">
              {connected ? `Connected · account ${verify?.account} · ${config?.region}` : verify?.error || "Not verified"}
            </div>
          </div>
          <span className="text-xs text-slate-400">{configOpen ? "▲" : "▼"}</span>
        </button>

        {configOpen && (
          <div className="border-t border-slate-100 bg-slate-50 px-4 py-4">
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <Field label="Access Key ID">
                <input
                  value={form.access_key_id}
                  onChange={(e) => setForm((f) => ({ ...f, access_key_id: e.target.value }))}
                  className="input"
                  placeholder="AKIA..."
                />
              </Field>
              <Field label="Secret Access Key">
                <div className="flex gap-2">
                  <input
                    type={showSecret ? "text" : "password"}
                    value={form.secret_access_key}
                    onChange={(e) => setForm((f) => ({ ...f, secret_access_key: e.target.value }))}
                    className="input flex-1"
                  />
                  <button type="button" onClick={() => setShowSecret((s) => !s)} className="text-xs text-slate-500">
                    {showSecret ? "hide" : "show"}
                  </button>
                </div>
              </Field>
              <Field label="Region">
                <select value={form.region} onChange={(e) => setForm((f) => ({ ...f, region: e.target.value }))} className="input">
                  {REGIONS.map((r) => (
                    <option key={r} value={r}>
                      {r}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Key Pair Name">
                <input
                  value={form.key_name}
                  onChange={(e) => setForm((f) => ({ ...f, key_name: e.target.value }))}
                  className="input"
                />
              </Field>
            </div>
            <div className="mt-3 flex gap-2">
              <button onClick={handleTest} disabled={testing} className="rounded-md border border-slate-300 px-3 py-1.5 text-sm text-slate-700 hover:bg-white disabled:opacity-50">
                {testing ? "Testing…" : "Test connection"}
              </button>
              <button onClick={handleSaveConfig} disabled={saving} className="btn-primary">
                {saving ? "Saving…" : "Save & apply"}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Instance status */}
      <div className="mt-4 rounded-lg border border-slate-200 bg-white p-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-900">EC2 Instance</h3>
          {!status?.key_pair.exists && (
            <button onClick={handleCreateKeyPair} disabled={creatingKey || !connected} className="rounded-md border border-slate-300 px-3 py-1.5 text-xs text-slate-700 hover:bg-slate-50 disabled:opacity-50">
              {creatingKey ? "Creating…" : "Create key pair"}
            </button>
          )}
        </div>

        {status && status.instances.length > 0 ? (
          <div className="mt-3 flex flex-col gap-2">
            {status.instances.map((inst) => (
              <div key={inst.id} className="flex items-center justify-between rounded-md border border-slate-100 px-3 py-2 text-sm">
                <span>
                  <span className={`mr-2 inline-block h-2 w-2 rounded-full ${inst.state === "running" ? "bg-emerald-500" : "bg-slate-300"}`} />
                  {inst.id} · {inst.type} · {inst.state} {inst.public_ip && `· ${inst.public_ip}`}
                </span>
                {inst.state === "running" && inst.public_ip && (
                  <button
                    onClick={() => handleDeploy(inst.public_ip)}
                    disabled={anyJobRunning || deployingIp === inst.public_ip}
                    className="rounded-md bg-indigo-600 px-3 py-1 text-xs font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
                  >
                    {deployingIp === inst.public_ip ? "Deploying…" : `Deploy app to ${inst.public_ip}`}
                  </button>
                )}
              </div>
            ))}
          </div>
        ) : (
          <p className="mt-2 text-sm text-slate-500">No instances yet.</p>
        )}

        <div className="mt-3 flex flex-wrap items-end gap-3 border-t border-slate-100 pt-3">
          <Field label="Instance type">
            <select value={launchType} onChange={(e) => setLaunchType(e.target.value)} className="input">
              {INSTANCE_TYPES.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Volume (GB)">
            <input type="number" min={8} max={100} value={launchVolume} onChange={(e) => setLaunchVolume(Number(e.target.value))} className="input w-20" />
          </Field>
          <button
            onClick={handleLaunch}
            disabled={anyJobRunning || launching || !status?.key_pair.exists || !connected}
            className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
          >
            {launching ? "Launching…" : "Launch Instance"}
          </button>
          {!status?.key_pair.exists && <span className="text-xs text-slate-500">Create a key pair first.</span>}
        </div>
      </div>

      {activeJob && <JobConsole job={activeJob} onClose={closeJob} onCancel={cancelJob} />}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs font-medium text-slate-500">{label}</label>
      {children}
    </div>
  );
}
