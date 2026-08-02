import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

import { isAdmin } from "@/lib/admin";
import { getSession } from "@/lib/session";
import SiteHeader from "@/components/SiteHeader";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

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
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col bg-slate-50 text-slate-900">
        {user && <SiteHeader email={user.email} isAdmin={isAdmin(user.email)} />}
        <main className="flex-1">{children}</main>
      </body>
    </html>
  );
}
