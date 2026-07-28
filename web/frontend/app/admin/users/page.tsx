import { redirect } from "next/navigation";

import UserApprovalPanel from "@/components/admin/UserApprovalPanel";
import { isAdmin } from "@/lib/admin";
import { getSession } from "@/lib/session";

export default async function AdminUsersPage() {
  const user = await getSession();

  // Defense in depth — proxy.ts already gates /admin/*, this is the second check.
  if (!isAdmin(user?.email)) {
    redirect("/predict");
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">User Approvals</h1>
      <p className="mt-1 text-sm text-slate-500">
        New signups can&apos;t sign in until approved here — nobody gets access without you.
      </p>

      <div className="mt-6">
        <UserApprovalPanel currentUserEmail={user!.email} />
      </div>
    </div>
  );
}
