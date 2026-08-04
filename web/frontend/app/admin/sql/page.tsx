import { redirect } from "next/navigation";

import SqlRunner from "@/components/admin/SqlRunner";
import { isAdmin } from "@/lib/admin";
import { getSession } from "@/lib/session";

export default async function AdminSqlPage() {
  const user = await getSession();

  // Defense in depth — proxy.ts already gates /admin/*, this is the second check.
  if (!isAdmin(user?.email)) {
    redirect("/predict");
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">SQL Runner</h1>
      <p className="mt-1 text-sm text-slate-500">
        Read-only ad-hoc queries across every table. Enforced by a genuine Postgres read-only transaction, not
        just a text check — writes are rejected by the database itself, even ones smuggled inside a CTE.
      </p>

      <div className="mt-6">
        <SqlRunner />
      </div>
    </div>
  );
}
