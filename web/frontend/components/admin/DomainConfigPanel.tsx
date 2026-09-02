"use client";

import { Fragment, useEffect, useState } from "react";

import {
  ApiError,
  getCrawlSearchDomains,
  getCrawlSearchStats,
  resetCrawlSearchDomain,
  saveCrawlSearchDomainOverrides,
} from "@/lib/api";
import type { CrawlSearchDomainConfig, CrawlSearchStats } from "@/lib/types";

interface FormState {
  enabled: boolean;
  max_pages: string;
  max_depth: string;
  min_delay_seconds: string;
  boost: string;
  use_sitemap: boolean;
  include: string;
  exclude: string;
  tags: string;
}

function toFormState(cfg: CrawlSearchDomainConfig): FormState {
  return {
    enabled: cfg.enabled,
    max_pages: String(cfg.max_pages),
    max_depth: String(cfg.max_depth),
    min_delay_seconds: String(cfg.min_delay_seconds),
    boost: String(cfg.boost),
    use_sitemap: cfg.use_sitemap,
    include: cfg.include.join(", "),
    exclude: cfg.exclude.join(", "),
    tags: cfg.tags.join(", "),
  };
}

// Saving always pins every field the form manages as an explicit DB
// override — simpler and more predictable than reconciling a partial diff
// against domains.yaml. "Reset to file" is the escape hatch back to the
// checked-in baseline.
function toOverrides(form: FormState): Record<string, unknown> {
  return {
    enabled: form.enabled,
    max_pages: form.max_pages,
    max_depth: form.max_depth,
    min_delay_seconds: form.min_delay_seconds,
    boost: form.boost,
    use_sitemap: form.use_sitemap,
    include: form.include,
    exclude: form.exclude,
    tags: form.tags,
  };
}

function fmtDate(iso: string | null): string {
  if (!iso) return "never";
  return new Date(iso).toLocaleString();
}

export default function DomainConfigPanel() {
  const [domains, setDomains] = useState<CrawlSearchDomainConfig[] | null>(null);
  const [stats, setStats] = useState<CrawlSearchStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [form, setForm] = useState<FormState | null>(null);
  const [saving, setSaving] = useState(false);
  const [resetting, setResetting] = useState<string | null>(null);

  async function load() {
    setError(null);
    try {
      const [domainsRes, statsRes] = await Promise.all([getCrawlSearchDomains(), getCrawlSearchStats()]);
      setDomains(domainsRes.domains);
      setStats(statsRes);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not reach CrawlSearch.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  function startEdit(cfg: CrawlSearchDomainConfig) {
    setExpanded(cfg.domain);
    setForm(toFormState(cfg));
  }

  function cancelEdit() {
    setExpanded(null);
    setForm(null);
  }

  async function handleSave(domain: string) {
    if (!form) return;
    setSaving(true);
    setError(null);
    try {
      await saveCrawlSearchDomainOverrides(domain, toOverrides(form));
      setExpanded(null);
      setForm(null);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not save domain config.");
    } finally {
      setSaving(false);
    }
  }

  async function handleReset(domain: string) {
    setResetting(domain);
    setError(null);
    try {
      await resetCrawlSearchDomain(domain);
      if (expanded === domain) {
        setExpanded(null);
        setForm(null);
      }
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not reset domain config.");
    } finally {
      setResetting(null);
    }
  }

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5">
      <div>
        <h2 className="font-semibold text-slate-900">CrawlSearch — Domain Config</h2>
        <p className="mt-1 text-sm text-slate-600">
          Per-domain crawl settings, merged from domains.yaml and any live overrides below. A saved edit here
          takes effect on the next crawl (on-demand or scheduled) — no redeploy needed.
        </p>
      </div>

      {stats && (
        <p className="mt-3 text-xs text-slate-500">
          {stats.indexed_pages.toLocaleString()} pages indexed across {stats.domains_with_pages} domains ·{" "}
          {stats.enabled_domains}/{stats.configured_domains} domains enabled · {stats.retention_days}-day
          retention · {stats.pages_last_24h} page{stats.pages_last_24h === 1 ? "" : "s"} crawled in the last 24h
        </p>
      )}

      {error && <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {loading && <p className="mt-4 text-sm text-slate-500">Loading…</p>}

      {domains && (
        <div className="mt-4 overflow-x-auto rounded-md border border-slate-200">
          <table className="min-w-full text-xs">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-left font-medium uppercase tracking-wide text-slate-500">
                <th className="px-2 py-1.5">Domain</th>
                <th className="px-2 py-1.5">Enabled</th>
                <th className="px-2 py-1.5 text-right">Max Pages</th>
                <th className="px-2 py-1.5 text-right">Depth</th>
                <th className="px-2 py-1.5 text-right">Delay (s)</th>
                <th className="px-2 py-1.5 text-right">Boost</th>
                <th className="px-2 py-1.5 text-right">Pages Indexed</th>
                <th className="px-2 py-1.5">Last Crawled</th>
                <th className="px-2 py-1.5">Actions</th>
              </tr>
            </thead>
            <tbody>
              {domains.map((cfg) => (
                <Fragment key={cfg.domain}>
                  <tr className="border-b border-slate-100 last:border-0">
                    <td className="px-2 py-1.5 font-medium text-slate-800">
                      {cfg.domain}
                      {!cfg.in_yaml && (
                        <span className="ml-1.5 rounded-full bg-amber-50 px-1.5 py-0.5 text-[10px] font-semibold text-amber-700">
                          DB-only
                        </span>
                      )}
                      {cfg.overridden_keys.length > 0 && (
                        <span className="ml-1.5 rounded-full bg-blue-50 px-1.5 py-0.5 text-[10px] font-semibold text-blue-700">
                          overridden
                        </span>
                      )}
                    </td>
                    <td className="px-2 py-1.5">
                      <span
                        className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                          cfg.enabled ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-500"
                        }`}
                      >
                        {cfg.enabled ? "Yes" : "No"}
                      </span>
                    </td>
                    <td className="px-2 py-1.5 text-right text-slate-600">{cfg.max_pages}</td>
                    <td className="px-2 py-1.5 text-right text-slate-600">{cfg.max_depth}</td>
                    <td className="px-2 py-1.5 text-right text-slate-600">{cfg.min_delay_seconds}</td>
                    <td className="px-2 py-1.5 text-right text-slate-600">{cfg.boost}</td>
                    <td className="px-2 py-1.5 text-right text-slate-600">{cfg.pages}</td>
                    <td className="px-2 py-1.5 text-slate-500">{fmtDate(cfg.last_crawled)}</td>
                    <td className="px-2 py-1.5">
                      <div className="flex gap-2">
                        <button
                          onClick={() => (expanded === cfg.domain ? cancelEdit() : startEdit(cfg))}
                          className="rounded-md border border-slate-200 px-2 py-1 text-[11px] font-medium text-slate-700 hover:bg-slate-50"
                        >
                          {expanded === cfg.domain ? "Close" : "Edit"}
                        </button>
                        {cfg.overridden_keys.length > 0 && (
                          <button
                            onClick={() => handleReset(cfg.domain)}
                            disabled={resetting === cfg.domain}
                            className="rounded-md border border-amber-200 px-2 py-1 text-[11px] font-medium text-amber-700 hover:bg-amber-50 disabled:opacity-50"
                          >
                            {resetting === cfg.domain ? "Resetting…" : "Reset to file"}
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                  {expanded === cfg.domain && form && (
                    <tr className="border-b border-slate-100 bg-slate-50">
                      <td colSpan={9} className="px-3 py-3">
                        <div className="flex flex-wrap items-end gap-3">
                          <label className="flex items-center gap-1.5 text-xs text-slate-700">
                            <input
                              type="checkbox"
                              checked={form.enabled}
                              onChange={(e) => setForm({ ...form, enabled: e.target.checked })}
                            />
                            Enabled
                          </label>
                          <label className="flex items-center gap-1.5 text-xs text-slate-700">
                            <input
                              type="checkbox"
                              checked={form.use_sitemap}
                              onChange={(e) => setForm({ ...form, use_sitemap: e.target.checked })}
                            />
                            Use sitemap
                          </label>
                          <label className="flex flex-col gap-1 text-xs text-slate-600">
                            Max pages
                            <input
                              type="number"
                              min={1}
                              value={form.max_pages}
                              onChange={(e) => setForm({ ...form, max_pages: e.target.value })}
                              className="w-20 rounded-md border border-slate-300 px-2 py-1 text-xs"
                            />
                          </label>
                          <label className="flex flex-col gap-1 text-xs text-slate-600">
                            Max depth
                            <input
                              type="number"
                              min={0}
                              value={form.max_depth}
                              onChange={(e) => setForm({ ...form, max_depth: e.target.value })}
                              className="w-16 rounded-md border border-slate-300 px-2 py-1 text-xs"
                            />
                          </label>
                          <label className="flex flex-col gap-1 text-xs text-slate-600">
                            Delay (s)
                            <input
                              type="number"
                              min={0}
                              step="0.5"
                              value={form.min_delay_seconds}
                              onChange={(e) => setForm({ ...form, min_delay_seconds: e.target.value })}
                              className="w-16 rounded-md border border-slate-300 px-2 py-1 text-xs"
                            />
                          </label>
                          <label className="flex flex-col gap-1 text-xs text-slate-600">
                            Boost
                            <input
                              type="number"
                              min={0}
                              step="0.1"
                              value={form.boost}
                              onChange={(e) => setForm({ ...form, boost: e.target.value })}
                              className="w-16 rounded-md border border-slate-300 px-2 py-1 text-xs"
                            />
                          </label>
                          <label className="flex flex-col gap-1 text-xs text-slate-600">
                            Include (comma-separated path substrings)
                            <input
                              type="text"
                              value={form.include}
                              onChange={(e) => setForm({ ...form, include: e.target.value })}
                              placeholder="/markets/, /article/"
                              className="w-56 rounded-md border border-slate-300 px-2 py-1 text-xs"
                            />
                          </label>
                          <label className="flex flex-col gap-1 text-xs text-slate-600">
                            Exclude (comma-separated path substrings)
                            <input
                              type="text"
                              value={form.exclude}
                              onChange={(e) => setForm({ ...form, exclude: e.target.value })}
                              placeholder="/video/, /sponsored/"
                              className="w-56 rounded-md border border-slate-300 px-2 py-1 text-xs"
                            />
                          </label>
                          <label className="flex flex-col gap-1 text-xs text-slate-600">
                            Tags (comma-separated)
                            <input
                              type="text"
                              value={form.tags}
                              onChange={(e) => setForm({ ...form, tags: e.target.value })}
                              placeholder="finance, news"
                              className="w-40 rounded-md border border-slate-300 px-2 py-1 text-xs"
                            />
                          </label>
                          <div className="flex gap-2">
                            <button
                              onClick={() => handleSave(cfg.domain)}
                              disabled={saving}
                              className="rounded-md bg-slate-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-800 disabled:opacity-50"
                            >
                              {saving ? "Saving…" : "Save"}
                            </button>
                            <button
                              onClick={cancelEdit}
                              disabled={saving}
                              className="rounded-md border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                            >
                              Cancel
                            </button>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
