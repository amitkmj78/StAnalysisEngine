// ALX-3: web push, plus (PLT-4) the minimum needed for PWA installability --
// an active service worker controlling the page, with a fetch handler.
// Deliberately NOT offline caching: this is a network-passthrough fetch
// handler only, so the app is installable but not usable offline. Claiming
// offline support this app doesn't have would be worse than not having a
// fetch handler at all.

self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", (event) => {
  event.respondWith(fetch(event.request));
});

self.addEventListener("push", (event) => {
  let data = {};
  try {
    data = event.data ? event.data.json() : {};
  } catch {
    data = { title: "StAnalysisEngine", body: event.data ? event.data.text() : "" };
  }

  const title = data.title || "StAnalysisEngine";
  const options = {
    body: data.body || "",
    data: { url: data.url || "/" },
  };

  event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const url = (event.notification.data && event.notification.data.url) || "/";
  event.waitUntil(self.clients.openWindow(url));
});
