import { redirect } from "next/navigation";

import PitPriceControls from "@/components/admin/PitPriceControls";
import PublishSignalsControls from "@/components/admin/PublishSignalsControls";
import SchedulerControls from "@/components/admin/SchedulerControls";
import { isAdmin } from "@/lib/admin";
import { getSession } from "@/lib/session";

export default async function AdminSchedulerPage() {
  const user = await getSession();

  // Defense in depth — proxy.ts already gates /admin/*, this is the second check.
  if (!isAdmin(user?.email)) {
    redirect("/predict");
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">Scheduler</h1>
      <p className="mt-1 text-sm text-slate-500">
        Controls for the background jobs that run automatically without anyone visiting a page.
      </p>

      <div className="mt-6 flex flex-col gap-4">
        <SchedulerControls />
        <PublishSignalsControls />
        <PitPriceControls />
      </div>
    </div>
  );
}
