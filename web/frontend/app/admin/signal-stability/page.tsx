import { redirect } from "next/navigation";

import SignalStabilityPanel from "@/components/admin/SignalStabilityPanel";
import { isAdmin } from "@/lib/admin";
import { getSession } from "@/lib/session";

export default async function AdminSignalStabilityPage() {
  const user = await getSession();

  // Defense in depth — proxy.ts already gates /admin/*, this is the second check.
  if (!isAdmin(user?.email)) {
    redirect("/predict");
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">Signal Stability</h1>
      <p className="mt-1 text-sm text-slate-500">
        Day-over-day BUY/HOLD/SELL flip analysis from the Quant Signal point-in-time history — how often each
        ticker&apos;s signal changes, and whether a flip looks like boundary noise, the model chasing a big
        same-day price move, or a cleaner shift in view.
      </p>

      <div className="mt-6">
        <SignalStabilityPanel />
      </div>
    </div>
  );
}
