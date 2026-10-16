"use client";

import { useEffect, useState } from "react";

// PLT-4: registers the service worker globally (it was previously only
// registered lazily from app/alerts/settings/page.tsx when a user opted
// into push -- installability needs an active SW controlling the page on
// first visit, not just after that one opt-in), and shows a minimal
// install prompt when the browser offers one. A dismissed prompt doesn't
// come back until the next page load -- there's no "don't ask again"
// persistence here, deliberately simple for a first version.
interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: "accepted" | "dismissed" }>;
}

export default function InstallPrompt() {
  const [deferredPrompt, setDeferredPrompt] = useState<BeforeInstallPromptEvent | null>(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register("/sw.js").catch(() => {});
    }

    function handleBeforeInstallPrompt(e: Event) {
      e.preventDefault();
      setDeferredPrompt(e as BeforeInstallPromptEvent);
    }
    window.addEventListener("beforeinstallprompt", handleBeforeInstallPrompt);
    return () => window.removeEventListener("beforeinstallprompt", handleBeforeInstallPrompt);
  }, []);

  if (!deferredPrompt || dismissed) return null;

  return (
    <div className="fixed bottom-4 left-1/2 z-50 flex -translate-x-1/2 items-center gap-3 rounded-full border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 px-4 py-2 shadow-lg">
      <span className="text-sm text-slate-700 dark:text-slate-200">Install StAnalysisEngine for quicker access?</span>
      <button
        onClick={async () => {
          await deferredPrompt.prompt();
          setDeferredPrompt(null);
        }}
        className="rounded-full bg-slate-900 px-3 py-1 text-xs font-semibold text-white hover:bg-slate-800"
      >
        Install
      </button>
      <button
        onClick={() => setDismissed(true)}
        className="text-xs text-slate-500 dark:text-slate-400 hover:underline"
      >
        Not now
      </button>
    </div>
  );
}
