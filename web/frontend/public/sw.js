// ALX-3: web push service worker. Minimal by design -- this app has no
// other service-worker use (no offline caching, no PWA install flow),
// so this file does exactly one thing: show a Notification for each
// push event, and route a click on it to the ticker page the alert was
// about (falling back to the app root).

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
