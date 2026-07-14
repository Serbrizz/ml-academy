/* Service worker minimale per ML Academy — abilita installazione PWA e cache offline dei file dell'app */

const CACHE_NAME = 'ml-academy-v8';
const APP_SHELL = [
  './',
  './index.html',
  './styles.css',
  './app.js',
  './glossary.js',
  './cheatsheets.js',
  './pdf-library.js',
  './manifest.json',
  './lessons/index.js',
  './lessons/01-intro.js',
  './lessons/02-math.js',
  './lessons/03-numpy-pandas.js',
  './lessons/04-eda.js',
  './lessons/05-linear-regression.js',
  './lessons/06-logistic.js',
  './lessons/07-knn.js',
  './lessons/08-trees.js',
  './lessons/09-svm.js',
  './lessons/10-eval.js',
  './lessons/11-kmeans.js',
  './lessons/12-pca.js',
  './lessons/13-neural-nets.js',
  './lessons/14-project.js',
  './lessons/15-gradient-boosting.js',
  './lessons/16-feature-engineering.js',
  './lessons/17-cnn.js',
  './lessons/18-mlops.js',
  './lessons/19-fraud-project.js',
  './ROADMAP.md',
];

// Installa: precaching dei file locali
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(APP_SHELL).catch(err => {
      console.warn('Precache parziale:', err);
    }))
  );
  self.skipWaiting();
});

// Attiva: pulizia vecchie cache
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys => Promise.all(
      keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
    ))
  );
  self.clients.claim();
});

// Fetch: strategy = network-first per HTML, cache-first per il resto
self.addEventListener('fetch', event => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);

  // Non toccare Pyodide / CDN esterni: lascia che browser cache-i normalmente
  if (url.origin !== self.location.origin) return;

  if (req.destination === 'document' || url.pathname.endsWith('.html')) {
    // Network first
    event.respondWith(
      fetch(req).then(resp => {
        const copy = resp.clone();
        caches.open(CACHE_NAME).then(c => c.put(req, copy));
        return resp;
      }).catch(() => caches.match(req).then(r => r || caches.match('./index.html')))
    );
  } else {
    // Cache first
    event.respondWith(
      caches.match(req).then(cached => {
        if (cached) return cached;
        return fetch(req).then(resp => {
          if (resp && resp.status === 200 && resp.type === 'basic') {
            const copy = resp.clone();
            caches.open(CACHE_NAME).then(c => c.put(req, copy));
          }
          return resp;
        });
      })
    );
  }
});
