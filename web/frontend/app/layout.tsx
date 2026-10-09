import type { Metadata } from "next";
import { Fraunces, Geist, Geist_Mono, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

import { isAdmin } from "@/lib/admin";
import { getSession } from "@/lib/session";
import MarketNewsTicker from "@/components/MarketNewsTicker";
import RegimeBanner from "@/components/RegimeBanner";
import SiteHeader from "@/components/SiteHeader";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const fraunces = Fraunces({ subsets: ["latin"], weight: ["500", "600", "700"], variable: "--font-pf-display" });
const plexMono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-pf-mono" });

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "StAnalysisEngine",
  description: "AI-assisted price prediction and stock ranking, with the accuracy shown next to the claim.",
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const user = await getSession();

  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} ${fraunces.variable} ${plexMono.variable} h-full antialiased`}>
      {/* Browser extensions such as Grammarly add attributes to body before React loads; ignore just those. */}
      <body className="min-h-full flex flex-col bg-slate-50 text-slate-900" suppressHydrationWarning>
        {user && <SiteHeader email={user.email} isAdmin={isAdmin(user.email)} />}
        {user && <RegimeBanner />}
        {user && (
          <div className="px-4 pt-4">
            <MarketNewsTicker />
          </div>
        )}
        <main className="flex-1">{children}</main>
      </body>
    </html>
  );
}
