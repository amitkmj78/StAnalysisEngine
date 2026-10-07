import { cookies } from "next/headers";
import { redirect } from "next/navigation";

import { getHomeTicker, getSession, SESSION_COOKIE_NAME } from "@/lib/session";

export default async function Home() {
  const user = await getSession();
  if (!user) redirect("/login");

  // Stock Detail is the home screen -- open on whatever ticker the user set
  // as their default (via "Set as home" on that page itself), or SPY if
  // they've never set one.
  const token = (await cookies()).get(SESSION_COOKIE_NAME)?.value ?? "";
  const ticker = await getHomeTicker(token);
  redirect(`/stock/${ticker}`);
}
