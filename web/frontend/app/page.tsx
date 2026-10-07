import { redirect } from "next/navigation";

import { getSession } from "@/lib/session";

export default async function Home() {
  const user = await getSession();
  // Stock Detail is the home screen now -- SPY (the market) until a
  // logged-in user's own default (e.g. a top holding) is worth wiring up.
  redirect(user ? "/stock/SPY" : "/login");
}
