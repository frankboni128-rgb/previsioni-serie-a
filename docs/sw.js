// xgol: l'app funziona anche senza rete mostrando gli ultimi dati scaricati
const CACHE = "xgol-v1";
const BASE = ["./", "./index.html", "./logo.webp", "./fondo.webp", "./icona-192.png", "./manifest.json"];
self.addEventListener("install", e => { e.waitUntil(caches.open(CACHE).then(c => c.addAll(BASE))); self.skipWaiting(); });
self.addEventListener("activate", e => { e.waitUntil(caches.keys().then(k => Promise.all(k.filter(x => x !== CACHE).map(x => caches.delete(x))))); self.clients.claim(); });
self.addEventListener("fetch", e => {
  if (e.request.method !== "GET") return;
  // prima la rete (dati sempre freschi), se manca la rete la copia salvata
  e.respondWith(fetch(e.request).then(r => {
    const copia = r.clone(); caches.open(CACHE).then(c => c.put(e.request.url.split("?")[0], copia)); return r;
  }).catch(() => caches.match(e.request.url.split("?")[0])));
});
