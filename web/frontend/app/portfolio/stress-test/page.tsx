"use client";

import { useEffect, useState } from "react";

import {
  ApiError,
  deleteStressScenario,
  getStressTestPresets,
  listStressScenarios,
  runCustomScenario,
  runStressTest,
  saveStressScenario,
} from "@/lib/api";
import type {
  CustomScenarioResult,
  ScenarioComponent,
  ScenarioComponentKind,
  SavedStressScenario,
  StressTestPresetResult,
  StressTestPresetsResponse,
  StressTestReplayResult,
} from "@/lib/types";
import PortfolioSwitcher from "@/components/PortfolioSwitcher";
import { useUrlState } from "@/lib/useUrlState";

// Mirrors services/market_data_service.py::SECTOR_ETFS' 11 GICS sectors --
// labels only, the backend's compute_sectors already classifies each
// holding, no ETF ticker needed client-side.
const SECTORS = [
  "Technology", "Financials", "Health Care", "Consumer Discretionary", "Consumer Staples",
  "Energy", "Industrials", "Materials", "Real Estate", "Utilities", "Communication Services",
];

const FACTOR_BENCHMARKS = [
  { ticker: "SPY", label: "Market (SPY)" },
  { ticker: "XLK", label: "Tech (XLK)" },
  { ticker: "USO", label: "Oil (USO)" },
  { ticker: "TLT", label: "Rates (TLT)" },
];

function SkeletonBlock({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded-md bg-slate-200 ${className}`} />;
}

function fmtPct(v: number | null, decimals = 1): string {
  return v === null || v === undefined ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(decimals)}%`;
}

function fmtDollars(v: number | null): string {
  return v === null || v === undefined ? "—" : `${v >= 0 ? "+" : ""}$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

function MethodBox({ method }: { method: string }) {
  return <p className="mt-2 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">{method}</p>;
}

type RunState<T> = { data: T | null; loading: boolean; error: string | null };

export default function StressTestPage() {
  const [{ p: portfolioParam }, setUrlState] = useUrlState({ p: "" });
  const [selectedPortfolioId, setSelectedPortfolioId] = useState<number | null>(
    portfolioParam ? Number(portfolioParam) : null,
  );

  const [catalog, setCatalog] = useState<StressTestPresetsResponse | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);

  const [presetResults, setPresetResults] = useState<Record<string, RunState<StressTestPresetResult>>>({});
  const [replayResults, setReplayResults] = useState<Record<string, RunState<StressTestReplayResult>>>({});

  const [customComponents, setCustomComponents] = useState<ScenarioComponent[]>([]);
  const [customResult, setCustomResult] = useState<RunState<CustomScenarioResult>>({
    data: null, loading: false, error: null,
  });
  const [newSectorChoice, setNewSectorChoice] = useState(SECTORS[0]);
  const [newSectorShock, setNewSectorShock] = useState("-15");
  const [newFactorChoice, setNewFactorChoice] = useState(FACTOR_BENCHMARKS[0].ticker);
  const [newFactorShock, setNewFactorShock] = useState("-10");

  const [savedScenarios, setSavedScenarios] = useState<SavedStressScenario[]>([]);
  const [scenarioName, setScenarioName] = useState("");
  const [savingScenario, setSavingScenario] = useState(false);
  const [scenarioError, setScenarioError] = useState<string | null>(null);

  function handlePortfolioChange(id: number) {
    setSelectedPortfolioId(id);
    setUrlState({ p: String(id) });
  }

  useEffect(() => {
    getStressTestPresets()
      .then(setCatalog)
      .catch((err) => setCatalogError(err instanceof ApiError ? err.message : "Couldn't load scenario catalog."));
    loadSavedScenarios();
  }, []);

  function loadSavedScenarios() {
    listStressScenarios()
      .then((r) => setSavedScenarios(r.scenarios))
      .catch(() => setSavedScenarios([]));
  }

  function runPreset(presetKey: string) {
    if (selectedPortfolioId === null) return;
    setPresetResults((prev) => ({ ...prev, [presetKey]: { data: null, loading: true, error: null } }));
    runStressTest("preset", presetKey, selectedPortfolioId)
      .then((r) =>
        setPresetResults((prev) => ({
          ...prev,
          [presetKey]: { data: r.result as StressTestPresetResult, loading: false, error: null },
        })),
      )
      .catch((err) =>
        setPresetResults((prev) => ({
          ...prev,
          [presetKey]: { data: null, loading: false, error: err instanceof ApiError ? err.message : "Failed to run." },
        })),
      );
  }

  function runReplay(replayKey: string) {
    if (selectedPortfolioId === null) return;
    setReplayResults((prev) => ({ ...prev, [replayKey]: { data: null, loading: true, error: null } }));
    runStressTest("replay", replayKey, selectedPortfolioId)
      .then((r) =>
        setReplayResults((prev) => ({
          ...prev,
          [replayKey]: { data: r.result as StressTestReplayResult, loading: false, error: null },
        })),
      )
      .catch((err) =>
        setReplayResults((prev) => ({
          ...prev,
          [replayKey]: { data: null, loading: false, error: err instanceof ApiError ? err.message : "Failed to run." },
        })),
      );
  }

  function addSectorComponent() {
    const shockPct = Number(newSectorShock);
    if (!Number.isFinite(shockPct)) return;
    setCustomComponents((prev) => [
      ...prev,
      { kind: "sector", sector: newSectorChoice, shock_pct: shockPct, label: `${newSectorChoice} ${shockPct >= 0 ? "+" : ""}${shockPct}%` },
    ]);
  }

  function addFactorComponent() {
    const shockPct = Number(newFactorShock);
    if (!Number.isFinite(shockPct)) return;
    const benchLabel = FACTOR_BENCHMARKS.find((b) => b.ticker === newFactorChoice)?.label ?? newFactorChoice;
    setCustomComponents((prev) => [
      ...prev,
      { kind: "factor", benchmark_ticker: newFactorChoice, shock_pct: shockPct, label: `${benchLabel} ${shockPct >= 0 ? "+" : ""}${shockPct}%` },
    ]);
  }

  function removeComponent(index: number) {
    setCustomComponents((prev) => prev.filter((_, i) => i !== index));
  }

  function handleRunCustom() {
    if (selectedPortfolioId === null) return;
    setCustomResult({ data: null, loading: true, error: null });
    runCustomScenario(customComponents, selectedPortfolioId)
      .then((r) => setCustomResult({ data: r.result, loading: false, error: null }))
      .catch((err) =>
        setCustomResult({ data: null, loading: false, error: err instanceof ApiError ? err.message : "Failed to run." }),
      );
  }

  async function handleSaveScenario(e: React.FormEvent) {
    e.preventDefault();
    const name = scenarioName.trim();
    if (!name || customComponents.length === 0) return;
    setSavingScenario(true);
    setScenarioError(null);
    try {
      await saveStressScenario(name, customComponents);
      setScenarioName("");
      loadSavedScenarios();
    } catch (err) {
      setScenarioError(err instanceof ApiError ? err.message : "Failed to save scenario.");
    } finally {
      setSavingScenario(false);
    }
  }

  function handleLoadScenario(scenario: SavedStressScenario) {
    setCustomComponents(scenario.shock_config);
    setCustomResult({ data: null, loading: false, error: null });
  }

  async function handleDeleteScenario(id: number) {
    try {
      await deleteStressScenario(id);
      loadSavedScenarios();
    } catch (err) {
      setScenarioError(err instanceof ApiError ? err.message : "Failed to delete scenario.");
    }
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <div>
        <h1 className="text-2xl font-semibold text-slate-900">Stress Test</h1>
        <p className="mt-1 text-sm text-slate-500">
          Estimated impact of market shocks on your real holdings — every result is an estimate, and shows exactly
          how it was computed directly underneath it, never just a number.
        </p>
      </div>

      <div className="mt-6">
        <PortfolioSwitcher
          selectedPortfolioId={selectedPortfolioId}
          onChange={handlePortfolioChange}
          initialPreferId={portfolioParam ? Number(portfolioParam) : undefined}
        />
      </div>

      {catalogError && <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{catalogError}</p>}

      {/* Preset shocks */}
      <div className="mt-6 rounded-lg border border-slate-200 bg-white p-5">
        <h2 className="text-sm font-semibold text-slate-900">Preset Shocks</h2>
        <p className="mt-1 text-xs text-slate-500">
          Beta-based estimates: your portfolio&apos;s historical beta to a real-world proxy, applied to a
          hypothetical move.
        </p>
        <div className="mt-3 flex flex-wrap gap-2">
          {(catalog?.presets ?? []).map((preset) => (
            <button
              key={preset.preset_key}
              onClick={() => runPreset(preset.preset_key)}
              disabled={selectedPortfolioId === null || presetResults[preset.preset_key]?.loading}
              className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
            >
              {presetResults[preset.preset_key]?.loading ? "Running…" : preset.label}
            </button>
          ))}
        </div>

        <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
          {Object.entries(presetResults).map(([key, state]) => {
            const preset = catalog?.presets.find((p) => p.preset_key === key);
            if (!preset) return null;
            return (
              <div key={key} className="rounded-md border border-slate-200 p-4">
                <h3 className="text-sm font-semibold text-slate-800">{preset.label}</h3>
                {state.loading && <SkeletonBlock className="mt-2 h-16" />}
                {state.error && <p className="mt-2 text-xs text-red-700">{state.error}</p>}
                {state.data && !state.loading && (
                  <>
                    <div className="mt-2 flex items-baseline gap-2">
                      <span
                        className={`text-2xl font-semibold ${
                          (state.data.estimated_dollar_impact ?? 0) < 0 ? "text-red-600" : "text-emerald-600"
                        }`}
                      >
                        {fmtDollars(state.data.estimated_dollar_impact)}
                      </span>
                      <span className="text-sm text-slate-500">{fmtPct(state.data.estimated_pct_impact)}</span>
                    </div>
                    {state.data.beta !== null && (
                      <p className="mt-1 text-xs text-slate-500">
                        Beta to {state.data.benchmark_ticker}: {state.data.beta.toFixed(2)} · {state.data.data_start}
                        {" – "}
                        {state.data.data_end}
                      </p>
                    )}
                    {state.data.excluded_from_beta.length > 0 && (
                      <p className="mt-1 text-xs text-slate-400">
                        Excluded (no price history): {state.data.excluded_from_beta.join(", ")}
                      </p>
                    )}
                    <MethodBox method={state.data.method} />
                  </>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Historical replays */}
      <div className="mt-6 rounded-lg border border-slate-200 bg-white p-5">
        <h2 className="text-sm font-semibold text-slate-900">Historical Replays</h2>
        <p className="mt-1 text-xs text-slate-500">
          Each holding&apos;s own actual return over a real historical crisis window, applied to its current value —
          realized history, not a model.
        </p>
        <div className="mt-3 flex flex-wrap gap-2">
          {(catalog?.replays ?? []).map((replay) => (
            <button
              key={replay.replay_key}
              onClick={() => runReplay(replay.replay_key)}
              disabled={selectedPortfolioId === null || replayResults[replay.replay_key]?.loading}
              className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
            >
              {replayResults[replay.replay_key]?.loading ? "Running…" : replay.label}
            </button>
          ))}
        </div>

        <div className="mt-4 flex flex-col gap-4">
          {Object.entries(replayResults).map(([key, state]) => {
            const replay = catalog?.replays.find((r) => r.replay_key === key);
            if (!replay) return null;
            return (
              <div key={key} className="rounded-md border border-slate-200 p-4">
                <h3 className="text-sm font-semibold text-slate-800">
                  {replay.label} ({replay.window_start} – {replay.window_end})
                </h3>
                {state.loading && <SkeletonBlock className="mt-2 h-24" />}
                {state.error && <p className="mt-2 text-xs text-red-700">{state.error}</p>}
                {state.data && !state.loading && (
                  <>
                    <div className="mt-2 flex items-baseline gap-2">
                      <span
                        className={`text-2xl font-semibold ${
                          (state.data.estimated_dollar_impact ?? 0) < 0 ? "text-red-600" : "text-emerald-600"
                        }`}
                      >
                        {fmtDollars(state.data.estimated_dollar_impact)}
                      </span>
                      <span className="text-sm text-slate-500">{fmtPct(state.data.estimated_pct_impact)}</span>
                    </div>
                    <div className="mt-3 overflow-x-auto">
                      <table className="min-w-full text-xs">
                        <thead>
                          <tr className="border-b border-slate-200 text-left uppercase tracking-wide text-slate-500">
                            <th className="py-1 pr-3">Ticker</th>
                            <th className="py-1 pr-3 text-right">Market Value</th>
                            <th className="py-1 pr-3 text-right">%</th>
                            <th className="py-1 text-right">$</th>
                          </tr>
                        </thead>
                        <tbody>
                          {state.data.holdings.map((h) => (
                            <tr
                              key={h.ticker}
                              className={`border-b border-slate-100 last:border-0 ${
                                h.method_used === "excluded_no_history_for_window" ? "text-slate-400" : ""
                              }`}
                            >
                              <td className="py-1 pr-3 font-medium">
                                {h.ticker}
                                {h.method_used === "excluded_no_history_for_window" && (
                                  <span className="ml-1 rounded-full bg-slate-100 px-1.5 py-0.5 text-[10px]">
                                    excluded
                                  </span>
                                )}
                              </td>
                              <td className="py-1 pr-3 text-right">${h.market_value.toLocaleString()}</td>
                              <td className="py-1 pr-3 text-right">{fmtPct(h.estimated_pct_impact)}</td>
                              <td className="py-1 text-right">{fmtDollars(h.estimated_dollar_impact)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <MethodBox method={state.data.method} />
                  </>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Custom scenario builder */}
      <div className="mt-6 rounded-lg border border-slate-200 bg-white p-5">
        <h2 className="text-sm font-semibold text-slate-900">Custom Scenario</h2>
        <p className="mt-1 text-xs text-slate-500">
          Combine sector and factor moves. Sector moves apply directly to holdings classified in that sector; factor
          moves apply via your portfolio&apos;s beta. Impacts are summed — this doesn&apos;t model correlation
          between components.
        </p>

        <div className="mt-3 flex flex-wrap items-end gap-2">
          <select
            value={newSectorChoice}
            onChange={(e) => setNewSectorChoice(e.target.value)}
            className="rounded-md border border-slate-300 px-2 py-1.5 text-sm"
          >
            {SECTORS.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
          <input
            type="number"
            value={newSectorShock}
            onChange={(e) => setNewSectorShock(e.target.value)}
            className="w-20 rounded-md border border-slate-300 px-2 py-1.5 text-sm"
          />
          <button
            onClick={addSectorComponent}
            className="rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50"
          >
            Add sector move
          </button>

          <select
            value={newFactorChoice}
            onChange={(e) => setNewFactorChoice(e.target.value)}
            className="ml-4 rounded-md border border-slate-300 px-2 py-1.5 text-sm"
          >
            {FACTOR_BENCHMARKS.map((b) => (
              <option key={b.ticker} value={b.ticker}>
                {b.label}
              </option>
            ))}
          </select>
          <input
            type="number"
            value={newFactorShock}
            onChange={(e) => setNewFactorShock(e.target.value)}
            className="w-20 rounded-md border border-slate-300 px-2 py-1.5 text-sm"
          />
          <button
            onClick={addFactorComponent}
            className="rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50"
          >
            Add factor move
          </button>
        </div>

        {customComponents.length > 0 && (
          <div className="mt-3 flex flex-col gap-1">
            {customComponents.map((c, i) => (
              <div
                key={i}
                className="flex items-center justify-between rounded-md border border-slate-200 bg-slate-50 px-3 py-1.5 text-sm"
              >
                <span>{c.label}</span>
                <button onClick={() => removeComponent(i)} className="text-xs text-red-600 hover:underline">
                  Remove
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="mt-4 flex flex-wrap items-center gap-3">
          <button
            onClick={handleRunCustom}
            disabled={selectedPortfolioId === null || customComponents.length === 0 || customResult.loading}
            className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
          >
            {customResult.loading ? "Running…" : "Run scenario"}
          </button>

          <form onSubmit={handleSaveScenario} className="flex items-center gap-2">
            <input
              value={scenarioName}
              onChange={(e) => setScenarioName(e.target.value)}
              placeholder="Scenario name"
              className="w-40 rounded-md border border-slate-300 px-2 py-1.5 text-sm"
            />
            <button
              type="submit"
              disabled={savingScenario || !scenarioName.trim() || customComponents.length === 0}
              className="rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
            >
              {savingScenario ? "Saving…" : "Save"}
            </button>
          </form>
        </div>

        {scenarioError && <p className="mt-2 text-xs text-red-700">{scenarioError}</p>}

        {customResult.error && <p className="mt-3 text-xs text-red-700">{customResult.error}</p>}
        {customResult.data && !customResult.loading && (
          <div className="mt-4 rounded-md border border-slate-200 p-4">
            <div className="flex items-baseline gap-2">
              <span
                className={`text-2xl font-semibold ${
                  (customResult.data.total_estimated_dollar_impact ?? 0) < 0 ? "text-red-600" : "text-emerald-600"
                }`}
              >
                {fmtDollars(customResult.data.total_estimated_dollar_impact)}
              </span>
              <span className="text-sm text-slate-500">{fmtPct(customResult.data.total_estimated_pct_impact)}</span>
            </div>
            <div className="mt-2 flex flex-col gap-1">
              {customResult.data.components.map((c, i) => (
                <p key={i} className="text-xs text-slate-600">
                  <strong>{c.label}:</strong> {fmtDollars(c.estimated_dollar_impact)} ({fmtPct(c.estimated_pct_impact)})
                </p>
              ))}
            </div>
            <MethodBox method={customResult.data.method} />
          </div>
        )}

        {savedScenarios.length > 0 && (
          <div className="mt-5">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Saved Scenarios</h3>
            <div className="mt-2 flex flex-col gap-1">
              {savedScenarios.map((s) => (
                <div
                  key={s.id}
                  className="flex items-center justify-between rounded-md border border-slate-200 bg-white px-3 py-1.5 text-sm"
                >
                  <span>{s.name}</span>
                  <div className="flex gap-2">
                    <button onClick={() => handleLoadScenario(s)} className="text-xs text-slate-700 hover:underline">
                      Load
                    </button>
                    <button
                      onClick={() => handleDeleteScenario(s.id)}
                      className="text-xs text-red-600 hover:underline"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
