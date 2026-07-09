/* PDF library — IndexedDB storage + PDF.js viewer */

(function() {
  const DB_NAME = 'ml_academy_pdfs';
  const STORE = 'pdfs';
  const META_KEY = 'ml_academy_pdf_meta_v1';
  let db = null;
  let pdfJsLib = null;

  function openDB() {
    return new Promise((resolve, reject) => {
      if (db) return resolve(db);
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = e => {
        const d = e.target.result;
        if (!d.objectStoreNames.contains(STORE)) d.createObjectStore(STORE, { keyPath: 'id' });
      };
      req.onsuccess = e => { db = e.target.result; resolve(db); };
      req.onerror = e => reject(e);
    });
  }

  function getMeta() { try { return JSON.parse(localStorage.getItem(META_KEY)) || []; } catch { return []; } }
  function setMeta(m) { localStorage.setItem(META_KEY, JSON.stringify(m)); }

  async function loadPdfJs() {
    if (pdfJsLib) return pdfJsLib;
    // pdf.js is loaded as a module in index.html; if window.pdfjsLib isn't there use a fallback
    if (window.pdfjsLib) { pdfJsLib = window.pdfjsLib; return pdfJsLib; }
    try {
      const mod = await import('https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/build/pdf.min.mjs');
      mod.GlobalWorkerOptions.workerSrc = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/build/pdf.worker.min.mjs';
      pdfJsLib = mod;
      return pdfJsLib;
    } catch (e) {
      console.error('Impossibile caricare pdf.js', e);
      return null;
    }
  }

  async function add(file) {
    const d = await openDB();
    const id = 'pdf_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
    const buf = await file.arrayBuffer();
    await new Promise((resolve, reject) => {
      const tx = d.transaction(STORE, 'readwrite');
      tx.objectStore(STORE).put({ id, data: buf });
      tx.oncomplete = resolve; tx.onerror = reject;
    });
    const meta = getMeta();
    meta.push({ id, name: file.name, size: file.size, addedAt: Date.now(), lessonId: null });
    setMeta(meta);
  }

  async function remove(id) {
    const d = await openDB();
    await new Promise((resolve, reject) => {
      const tx = d.transaction(STORE, 'readwrite');
      tx.objectStore(STORE).delete(id);
      tx.oncomplete = resolve; tx.onerror = reject;
    });
    setMeta(getMeta().filter(m => m.id !== id));
  }

  async function getData(id) {
    const d = await openDB();
    return new Promise((resolve, reject) => {
      const tx = d.transaction(STORE, 'readonly');
      const req = tx.objectStore(STORE).get(id);
      req.onsuccess = () => resolve(req.result ? req.result.data : null);
      req.onerror = reject;
    });
  }

  function getAll() { return getMeta(); }
  function getForLesson(lessonId) { return getMeta().filter(m => m.lessonId === lessonId); }
  function setLesson(id, lessonId) {
    const meta = getMeta();
    const m = meta.find(x => x.id === id);
    if (m) { m.lessonId = lessonId || null; setMeta(meta); }
  }

  async function clear() {
    if (db) db.close(); db = null;
    return new Promise((resolve, reject) => {
      const req = indexedDB.deleteDatabase(DB_NAME);
      req.onsuccess = () => { localStorage.removeItem(META_KEY); resolve(); };
      req.onerror = reject;
    });
  }

  async function openViewer(id, container) {
    const meta = getMeta().find(m => m.id === id);
    if (!meta) return;
    const data = await getData(id);
    if (!data) { alert('PDF non trovato'); return; }
    const lib = await loadPdfJs();
    if (!lib) {
      container.innerHTML = '<p style="color: var(--danger);">Impossibile caricare il visualizzatore PDF. Verifica la connessione a internet.</p>';
      return;
    }
    container.innerHTML = `<div class="pdf-viewer" id="pv">
      <div class="pdf-viewer-header">
        <div class="pdf-viewer-title">📄 ${meta.name}</div>
        <div class="pdf-page-nav">
          <button id="pv-prev">← pag. precedente</button>
          <span id="pv-page">1 / 1</span>
          <button id="pv-next">pag. successiva →</button>
          <button id="pv-close" style="margin-left: 20px;">Chiudi</button>
        </div>
      </div>
      <div id="pv-canvas-wrap"></div>
    </div>`;
    const wrap = document.getElementById('pv-canvas-wrap');
    const pageInfo = document.getElementById('pv-page');
    const prevBtn = document.getElementById('pv-prev');
    const nextBtn = document.getElementById('pv-next');
    const closeBtn = document.getElementById('pv-close');

    let pdfDoc = null;
    let pageNum = 1;
    let rendering = false;

    async function renderPage(n) {
      if (rendering) return;
      rendering = true;
      wrap.innerHTML = '';
      const page = await pdfDoc.getPage(n);
      const scale = 1.4;
      const viewport = page.getViewport({ scale });
      const canvas = document.createElement('canvas');
      canvas.width = viewport.width; canvas.height = viewport.height;
      wrap.appendChild(canvas);
      const ctx = canvas.getContext('2d');
      await page.render({ canvasContext: ctx, viewport }).promise;
      pageInfo.textContent = `${n} / ${pdfDoc.numPages}`;
      prevBtn.disabled = n <= 1;
      nextBtn.disabled = n >= pdfDoc.numPages;
      rendering = false;
    }
    prevBtn.addEventListener('click', () => { if (pageNum > 1) { pageNum--; renderPage(pageNum); } });
    nextBtn.addEventListener('click', () => { if (pageNum < pdfDoc.numPages) { pageNum++; renderPage(pageNum); } });
    closeBtn.addEventListener('click', () => { container.innerHTML = ''; });

    try {
      pdfDoc = await lib.getDocument({ data: new Uint8Array(data) }).promise;
      renderPage(pageNum);
    } catch (e) {
      container.innerHTML = `<p style="color: var(--danger);">Errore nel caricare il PDF: ${e.message}</p>`;
    }
  }

  window.pdfLibrary = { add, remove, getAll, getForLesson, setLesson, clear, openViewer };
})();
