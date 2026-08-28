"use client";

import { useEffect, useState } from "react";

import { ApiError, createPortfolio, deletePortfolio, getPortfolios } from "@/lib/api";
import type { Portfolio } from "@/lib/types";

const STORAGE_KEY = "stanalysisengine.selectedPortfolioId";

export default function PortfolioSwitcher({
  selectedPortfolioId,
  onChange,
  onPortfoliosChange,
  reloadSignal,
}: {
  selectedPortfolioId: number | null;
  onChange: (portfolioId: number) => void;
  /** Fires whenever the portfolio list (re)loads — lets the parent page
   * mirror the list (e.g. to populate a "move to" destination picker)
   * without owning the fetch itself. */
  onPortfoliosChange?: (portfolios: Portfolio[]) => void;
  /** Bump this (e.g. after a move/delete changes position counts) to make
   * the switcher re-fetch without resetting the current selection. */
  reloadSignal?: number;
}) {
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [createError, setCreateError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  async function load(preferId?: number) {
    setLoading(true);
    setError(null);
    try {
      const res = await getPortfolios();
      setPortfolios(res.portfolios);
      onPortfoliosChange?.(res.portfolios);
      if (res.portfolios.length === 0) return;

      const stored = preferId ?? Number(localStorage.getItem(STORAGE_KEY));
      const match = res.portfolios.find((p) => p.id === stored);
      onChange((match ?? res.portfolios[0]).id);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load your portfolios.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (reloadSignal !== undefined) {
      load(selectedPortfolioId ?? undefined);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reloadSignal]);

  function handleSelect(id: number) {
    localStorage.setItem(STORAGE_KEY, String(id));
    onChange(id);
  }

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    const name = newName.trim();
    if (!name) {
      setCreateError("Enter a name for the new portfolio.");
      return;
    }
    setSaving(true);
    setCreateError(null);
    try {
      const created = await createPortfolio(name);
      setNewName("");
      setCreating(false);
      localStorage.setItem(STORAGE_KEY, String(created.id));
      await load(created.id);
    } catch (err) {
      setCreateError(err instanceof ApiError ? err.message : "Could not create that portfolio.");
    } finally {
      setSaving(false);
    }
  }

  const selectedPortfolio = portfolios.find((p) => p.id === selectedPortfolioId) ?? null;

  async function handleDelete() {
    if (!selectedPortfolio) return;
    const positionNote =
      selectedPortfolio.position_count > 0
        ? ` It has ${selectedPortfolio.position_count} position${selectedPortfolio.position_count === 1 ? "" : "s"} — they won't be deleted, just no longer reachable from any portfolio you can see.`
        : "";
    if (!window.confirm(`Delete "${selectedPortfolio.name}"?${positionNote} This can't be undone from here.`)) {
      return;
    }
    setDeleting(true);
    setDeleteError(null);
    try {
      await deletePortfolio(selectedPortfolio.id);
      await load();
    } catch (err) {
      setDeleteError(err instanceof ApiError ? err.message : "Could not delete this portfolio.");
    } finally {
      setDeleting(false);
    }
  }

  if (loading) return <p className="text-sm text-slate-500">Loading portfolios…</p>;
  if (error) return <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>;

  return (
    <div className="flex flex-wrap items-end gap-3">
      <div className="flex flex-col gap-1">
        <label className="text-xs font-medium text-slate-500">Portfolio</label>
        <select
          value={selectedPortfolioId ?? ""}
          onChange={(e) => handleSelect(Number(e.target.value))}
          className="input"
        >
          {portfolios.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name} ({p.position_count})
            </option>
          ))}
        </select>
      </div>

      <button
        type="button"
        onClick={handleDelete}
        disabled={deleting || !selectedPortfolio || portfolios.length <= 1}
        title={portfolios.length <= 1 ? "You need at least one portfolio — create another before deleting this one." : "Delete this portfolio"}
        className="rounded-md border border-red-200 px-3 py-2 text-sm font-medium text-red-600 hover:bg-red-50 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {deleting ? "Deleting…" : "Delete Portfolio"}
      </button>

      {creating ? (
        <form onSubmit={handleCreate} className="flex items-end gap-2">
          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-slate-500">New portfolio name</label>
            <input
              autoFocus
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="e.g. Retirement"
              className="input w-48"
            />
          </div>
          <button type="submit" disabled={saving} className="btn-primary">
            {saving ? "Creating…" : "Create"}
          </button>
          <button
            type="button"
            onClick={() => {
              setCreating(false);
              setCreateError(null);
              setNewName("");
            }}
            className="rounded-md border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
          >
            Cancel
          </button>
          {createError && <p className="w-full text-xs text-red-600">{createError}</p>}
        </form>
      ) : (
        <button
          type="button"
          onClick={() => setCreating(true)}
          className="rounded-md border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
        >
          + New Portfolio
        </button>
      )}
      {deleteError && <p className="w-full text-xs text-red-600">{deleteError}</p>}
    </div>
  );
}
