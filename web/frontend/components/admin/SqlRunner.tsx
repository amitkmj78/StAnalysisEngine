"use client";

import { useEffect, useState } from "react";

import { ApiError, getAdminSqlTables, runAdminSqlQuery } from "@/lib/api";
import type { AdminSqlQueryResponse, AdminSqlTable } from "@/lib/types";

const DATABASES = [
  { id: "stanalysisengine", label: "StAnalysisEngine", defaultSql: "SELECT * FROM users LIMIT 100" },
  { id: "crawlsearch", label: "CrawlSearch", defaultSql: "SELECT * FROM pages LIMIT 100" },
] as const;

export default function SqlRunner() {
  const [database, setDatabase] = useState<string>(DATABASES[0].id);
  const [tables, setTables] = useState<AdminSqlTable[] | null>(null);
  const [tablesError, setTablesError] = useState<string | null>(null);
  const [expandedTable, setExpandedTable] = useState<string | null>(null);

  const [sql, setSql] = useState<string>(DATABASES[0].defaultSql);
  const [result, setResult] = useState<AdminSqlQueryResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setTables(null);
    setTablesError(null);
    getAdminSqlTables(database)
      .then((res) => setTables(res.tables))
      .catch((err) => setTablesError(err instanceof ApiError ? err.message : "Failed to load tables."));
  }, [database]);

  function switchDatabase(id: string) {
    setDatabase(id);
    setResult(null);
    setError(null);
    setSql(DATABASES.find((d) => d.id === id)?.defaultSql ?? "");
  }

  async function runQuery() {
    setRunning(true);
    setError(null);
    try {
      setResult(await runAdminSqlQuery(sql, database));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Query failed.");
      setResult(null);
    } finally {
      setRunning(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      runQuery();
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex gap-1 rounded-lg border border-slate-200 bg-white p-1">
        {DATABASES.map((d) => (
          <button
            key={d.id}
            onClick={() => switchDatabase(d.id)}
            className={`rounded-md px-3 py-1.5 text-sm font-medium ${
              database === d.id ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-50"
            }`}
          >
            {d.label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[240px_1fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-3">
          <h2 className="px-1 text-xs font-semibold uppercase tracking-wide text-slate-500">Tables</h2>
        {tablesError && <p className="mt-2 px-1 text-xs text-red-600">{tablesError}</p>}
        {!tables && !tablesError && <p className="mt-2 px-1 text-xs text-slate-400">Loading…</p>}
        <div className="mt-2 flex flex-col">
          {tables?.map((t) => (
            <div key={t.table_name}>
              <button
                onClick={() => {
                  setExpandedTable(expandedTable === t.table_name ? null : t.table_name);
                  setSql(`SELECT * FROM ${t.table_name} LIMIT 100`);
                }}
                className="flex w-full items-center justify-between rounded px-2 py-1.5 text-left text-sm text-slate-700 hover:bg-slate-50"
              >
                <span className="font-mono">{t.table_name}</span>
                <span className="text-xs text-slate-400">{t.approx_row_count ?? "—"}</span>
              </button>
              {expandedTable === t.table_name && (
                <div className="mb-1 ml-2 border-l border-slate-100 pl-2 text-xs text-slate-500">
                  {t.columns.map((c) => (
                    <div key={c.name} className="py-0.5">
                      <span className="font-mono">{c.name}</span> <span className="text-slate-400">{c.type}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="flex flex-col gap-3">
        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <textarea
            value={sql}
            onChange={(e) => setSql(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={5}
            spellCheck={false}
            className="w-full rounded-md border border-slate-300 px-3 py-2 font-mono text-sm text-slate-800 focus:border-slate-500 focus:outline-none"
          />
          <div className="mt-2 flex items-center gap-3">
            <button
              onClick={runQuery}
              disabled={running || !sql.trim()}
              className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
            >
              {running ? "Running…" : "Run Query"}
            </button>
            <span className="text-xs text-slate-400">
              Read-only — enforced at the database level, not just checked here. ⌘/Ctrl + Enter to run.
            </span>
          </div>
        </div>

        {error && <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

        {result && !error && (
          <div className="rounded-lg border border-slate-200 bg-white">
            <div className="flex items-center justify-between border-b border-slate-200 px-3 py-2 text-xs text-slate-500">
              <span>
                {result.row_count} row{result.row_count === 1 ? "" : "s"}
                {result.truncated && " (truncated at 500)"}
              </span>
            </div>
            {result.rows.length === 0 ? (
              <p className="px-3 py-6 text-sm text-slate-500">No rows returned.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-medium uppercase tracking-wide text-slate-500">
                      {result.columns.map((c) => (
                        <th key={c} className="whitespace-nowrap px-3 py-2 font-mono">
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.rows.map((row, i) => (
                      <tr key={i} className="border-b border-slate-100 last:border-0">
                        {row.map((cell, j) => (
                          <td key={j} className="whitespace-nowrap px-3 py-2 font-mono text-slate-700">
                            {cell === null ? <span className="text-slate-300">null</span> : String(cell)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
      </div>
    </div>
  );
}
