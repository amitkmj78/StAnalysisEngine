import { redirect } from "next/navigation";

import ActivityLog from "@/components/admin/ActivityLog";
import { isAdmin } from "@/lib/admin";
import { getSession } from "@/lib/session";

export default async function AdminActivityPage() {
  const user = await getSession();

  // Defense in depth — proxy.ts already gates /admin/*, this is the second check.
  if (!isAdmin(user?.email)) {
    redirect("/predict");
  }

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">User Activity</h1>
      <p className="mt-1 text-sm text-slate-500">
        Read-only log of API activity across all users, most recent first.
      </p>

      <div className="mt-6">
        <ActivityLog />
      </div>
    </div>
  );
}
