import { redirect } from "next/navigation";

import CrawlSearchControl from "@/components/admin/CrawlSearchControl";
import IntegrationsPanel from "@/components/admin/IntegrationsPanel";
import PriceProviderControls from "@/components/admin/PriceProviderControls";
import { isAdmin } from "@/lib/admin";
import { getSession } from "@/lib/session";

export default async function AdminIntegrationsPage() {
  const user = await getSession();

  // Defense in depth — proxy.ts already gates /admin/*, this is the second check.
  if (!isAdmin(user?.email)) {
    redirect("/predict");
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">Integrations</h1>
      <p className="mt-1 text-sm text-slate-500">
        Every external service this app depends on — LLM providers, market data, and search. &quot;Configured&quot;
        just means the API key/URL is set; &quot;Test&quot; makes a real live call right now, since a configured
        key can still be expired, rate-limited, or out of credits (this is exactly how the OpenAI billing
        exhaustion and the retired Groq model were found this session — only by checking manually).
      </p>

      <div className="mt-6 flex flex-col gap-4">
        <PriceProviderControls />
        <CrawlSearchControl />
        <IntegrationsPanel />
      </div>
    </div>
  );
}
