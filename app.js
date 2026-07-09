/* ML Academy - core application */

const STORAGE = {
  progress: 'ml_progress_v2',
  notes:    'ml_notes_v2',
  code:     'ml_code_v2',
  streak:   'ml_streak_v2',
  bookmarks:'ml_bookmarks_v2',
  activity: 'ml_activity_v2',
};

let pyodide = null;
let pyodideLoading = null;

const store = {
  get(key, def) {
    try {
      const raw = localStorage.getItem(key);
      if (raw === null) return def === undefined ? {} : def;
      return JSON.parse(raw);
    } catch { return def === undefined ? {} : def; }
  },
  set(key, val) { localStorage.setItem(key, JSON.stringify(val)); },
};

function todayStr() { return new Date().toISOString().slice(0, 10); }
function daysBetween(a, b) { return Math.round((new Date(b) - new Date(a)) / 86400000); }

function recordActivity(xp) {
  if (xp === undefined) xp = 5;
  const streak = store.get(STORAGE.streak, { current: 0, longest: 0, lastDate: null, xp: 0 });
  const activity = store.get(STORAGE.activity, {});
  const today = todayStr();
  if (streak.lastDate !== today) {
    if (streak.lastDate && daysBetween(streak.lastDate, today) === 1) streak.current += 1;
    else if (streak.lastDate && daysBetween(streak.lastDate, today) > 1) streak.current = 1;
    else streak.current = 1;
    streak.lastDate = today;
    streak.longest = Math.max(streak.longest, streak.current);
  }
  streak.xp = (streak.xp || 0) + xp;
  activity[today] = (activity[today] || 0) + xp;
  store.set(STORAGE.streak, streak);
  store.set(STORAGE.activity, activity);
  renderStreak();
}

function checkStreakOnLoad() {
  const streak = store.get(STORAGE.streak, { current: 0, longest: 0, lastDate: null, xp: 0 });
  if (streak.lastDate && daysBetween(streak.lastDate, todayStr()) > 1) {
    streak.current = 0;
    store.set(STORAGE.streak, streak);
  }
  renderStreak();
}

function renderStreak() {
  const s = store.get(STORAGE.streak, { current: 0, xp: 0 });
  const cEl = document.getElementById('streak-count');
  const xEl = document.getElementById('streak-xp');
  if (cEl) cEl.textContent = s.current || 0;
  if (xEl) xEl.textContent = (s.xp || 0) + ' XP';
}

function isCompleted(id) { return !!store.get(STORAGE.progress)[id]; }
function setCompleted(id, val) {
  const p = store.get(STORAGE.progress);
  if (val) { p[id] = { at: Date.now() }; recordActivity(25); }
  else delete p[id];
  store.set(STORAGE.progress, p);
}

async function ensurePyodide() {
  if (pyodide) return pyodide;
  if (pyodideLoading) return pyodideLoading;
  const stateEl = document.getElementById('pyodide-state');
  if (stateEl) { stateEl.textContent = 'caricamento...'; stateEl.className = 'loading'; }
  pyodideLoading = (async () => {
    pyodide = await loadPyodide();
    if (stateEl) stateEl.textContent = 'installo pacchetti...';
    await pyodide.loadPackage(['numpy', 'pandas', 'scikit-learn', 'matplotlib']);
    await pyodide.runPythonAsync([
      "import matplotlib",
      "matplotlib.use('AGG')",
      "import matplotlib.pyplot as plt",
      "import io, base64",
      "def _mpl_show():",
      "    buf = io.BytesIO()",
      "    plt.gcf().savefig(buf, format='png', bbox_inches='tight', dpi=110)",
      "    plt.close('all')",
      "    buf.seek(0)",
      "    return base64.b64encode(buf.read()).decode('ascii')",
      ""
    ].join('\n'));
    if (stateEl) { stateEl.textContent = 'pronto'; stateEl.className = 'ready'; }
    return pyodide;
  })();
  return pyodideLoading;
}

async function runPythonCode(code, outEl) {
  outEl.className = 'code-output';
  outEl.textContent = 'Avvio Python...';
  try {
    const py = await ensurePyodide();
    py.runPython([
      "import sys, io",
      "_stdout_buf = io.StringIO()",
      "sys.stdout = _stdout_buf",
      "sys.stderr = _stdout_buf"
    ].join('\n'));
    const wrapped = code + [
      "",
      "try:",
      "    import matplotlib.pyplot as _plt",
      "    __figure_data = _mpl_show() if _plt.get_fignums() else None",
      "except Exception:",
      "    __figure_data = None",
      ""
    ].join('\n');
    await py.runPythonAsync(wrapped);
    const output = py.runPython('_stdout_buf.getvalue()');
    const figData = py.runPython('__figure_data');
    outEl.className = 'code-output success';
    outEl.textContent = output || '(nessun output di testo)';
    if (figData) {
      const img = document.createElement('img');
      img.src = 'data:image/png;base64,' + figData;
      img.className = 'output-image';
      outEl.appendChild(img);
    }
    return { ok: true };
  } catch (e) {
    outEl.className = 'code-output error';
    outEl.textContent = String(e.message || e);
    return { ok: false };
  } finally {
    try { pyodide && pyodide.runPython('sys.stdout = sys.__stdout__; sys.stderr = sys.__stderr__'); } catch {}
  }
}

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
}

function renderMath(container) {
  if (window.renderMathInElement) {
    renderMathInElement(container, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false },
        { left: '\\(', right: '\\)', display: false },
        { left: '\\[', right: '\\]', display: true },
      ],
      throwOnError: false,
    });
  }
}

function attachGlossaryTooltips(container) {
  if (!window.GLOSSARY) return;
  container.querySelectorAll('p, li').forEach(node => {
    if (node.closest('code, .code, .exercise, .quiz-question')) return;
    let html = node.innerHTML;
    Object.entries(window.GLOSSARY).forEach(([term, def]) => {
      const escTerm = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const re = new RegExp('\\b(' + escTerm + ')\\b(?![^<]*>)', 'i');
      if (re.test(html) && !html.includes('glossary-tooltip')) {
        html = html.replace(re, '<span class="glossary-tooltip" title="' + def.replace(/"/g, '&quot;') + '">$1</span>');
      }
    });
    node.innerHTML = html;
  });
}

function renderCodeRunner(container, initialCode, opts) {
  opts = opts || {};
  const wrap = el('div', 'code-runner');
  const ta = el('textarea', 'code-editor');
  const codeKey = opts.lessonId && opts.exId ? (opts.lessonId + '::' + opts.exId) : null;
  const savedCode = codeKey ? store.get(STORAGE.code)[codeKey] : null;
  ta.value = savedCode || initialCode;
  ta.spellcheck = false;
  const tb = el('div', 'code-toolbar');
  const runBtn = el('button', 'run-btn', 'Esegui');
  const resetBtn = el('button', 'reset-btn', 'Reset');
  tb.appendChild(runBtn); tb.appendChild(resetBtn);
  let checkBtn = null;
  if (opts.check) {
    checkBtn = el('button', 'check-btn', 'Verifica soluzione');
    tb.appendChild(checkBtn);
  }
  const savedBadge = el('span', 'saved-badge', codeKey ? 'codice salvato' : '');
  tb.appendChild(savedBadge);
  const out = el('div', 'code-output');
  out.textContent = '(nessun output, clicca Esegui)';
  wrap.appendChild(ta); wrap.appendChild(tb); wrap.appendChild(out);
  container.appendChild(wrap);

  let saveTimer;
  ta.addEventListener('input', () => {
    if (!codeKey) return;
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      const all = store.get(STORAGE.code);
      all[codeKey] = ta.value;
      store.set(STORAGE.code, all);
      savedBadge.textContent = 'salvato';
      setTimeout(() => { savedBadge.textContent = 'codice salvato'; }, 1500);
    }, 500);
  });
  runBtn.addEventListener('click', async () => {
    runBtn.disabled = true; runBtn.textContent = 'Eseguo...';
    await runPythonCode(ta.value, out);
    runBtn.disabled = false; runBtn.textContent = 'Esegui';
    recordActivity(2);
  });
  resetBtn.addEventListener('click', () => {
    if (confirm('Ripristinare il codice iniziale?')) {
      ta.value = initialCode;
      if (codeKey) {
        const all = store.get(STORAGE.code);
        delete all[codeKey];
        store.set(STORAGE.code, all);
      }
    }
  });
  if (checkBtn && opts.check) {
    checkBtn.addEventListener('click', async () => {
      checkBtn.disabled = true; checkBtn.textContent = 'Verifico...';
      const codeWithCheck = ta.value + '\n\n# --- verifica ---\n' + opts.check;
      await runPythonCode(codeWithCheck, out);
      checkBtn.disabled = false; checkBtn.textContent = 'Verifica soluzione';
      if (out.textContent.match(/\bOK\b/)) recordActivity(10);
    });
  }
  ta.addEventListener('keydown', (e) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const s = ta.selectionStart, en = ta.selectionEnd;
      ta.value = ta.value.substring(0, s) + '    ' + ta.value.substring(en);
      ta.selectionStart = ta.selectionEnd = s + 4;
    }
  });
}

function renderQuiz(container, question, idx) {
  const q = el('div', 'quiz-question');
  q.appendChild(el('div', 'q-text', '<strong>Domanda ' + (idx + 1) + '.</strong> ' + question.q));
  const opts = el('div', 'quiz-options');
  question.options.forEach((opt, i) => {
    const o = el('div', 'quiz-option', opt);
    o.addEventListener('click', () => {
      if (opts.querySelector('.correct')) return;
      opts.querySelectorAll('.quiz-option').forEach(x => x.classList.remove('selected'));
      o.classList.add('selected');
      const fb = q.querySelector('.quiz-feedback');
      if (i === question.answer) {
        o.classList.add('correct');
        fb.className = 'quiz-feedback visible correct';
        fb.innerHTML = '<strong>Corretto.</strong> ' + (question.explain || '');
        recordActivity(3);
      } else {
        o.classList.add('wrong');
        opts.children[question.answer].classList.add('correct');
        fb.className = 'quiz-feedback visible wrong';
        fb.innerHTML = '<strong>Non corretto.</strong> ' + (question.explain || '');
      }
    });
    opts.appendChild(o);
  });
  q.appendChild(opts);
  q.appendChild(el('div', 'quiz-feedback'));
  container.appendChild(q);
}

function renderSection(panelEl, section, ctx) {
  panelEl.innerHTML = '';
  if (!section) { panelEl.innerHTML = '<p style="color:var(--muted)">Nessun contenuto qui.</p>'; return; }
  section.forEach((block, i) => {
    if (block.type === 'md') {
      const d = el('div'); d.innerHTML = block.content; panelEl.appendChild(d);
    } else if (block.type === 'code') {
      const p = el('pre', 'code'); p.textContent = block.content; panelEl.appendChild(p);
    } else if (block.type === 'run') {
      renderCodeRunner(panelEl, block.content, { lessonId: ctx.lessonId, exId: 'run-' + ctx.section + '-' + i });
    } else if (block.type === 'exercise') {
      const ex = el('div', 'exercise');
      ex.innerHTML = '<span class="exercise-tag tag-' + (block.difficulty || 'easy') + '">' + (block.difficulty || 'easy') + '</span><h4>' + block.title + '</h4><div>' + block.prompt + '</div>';
      renderCodeRunner(ex, block.starter || '# scrivi qui\n', { check: block.check, lessonId: ctx.lessonId, exId: 'ex-' + block.title });
      if (block.solution) {
        const revealBtn = el('button', 'reveal-btn', 'Mostra soluzione');
        const sol = el('div', 'solution');
        sol.innerHTML = '<h4>Soluzione</h4>';
        const pre = el('pre', 'code'); pre.textContent = block.solution; sol.appendChild(pre);
        if (block.solutionNote) { const n = el('div'); n.innerHTML = block.solutionNote; sol.appendChild(n); }
        revealBtn.addEventListener('click', () => sol.classList.toggle('visible'));
        ex.appendChild(revealBtn); ex.appendChild(sol);
      }
      panelEl.appendChild(ex);
    } else if (block.type === 'quiz') {
      block.questions.forEach((q, i) => renderQuiz(panelEl, q, i));
    } else if (block.type === 'callout') {
      const c = el('div', 'callout ' + (block.variant || 'note'));
      if (block.title) c.appendChild(el('div', 'callout-title', block.title));
      const b = el('div'); b.innerHTML = block.content; c.appendChild(b);
      panelEl.appendChild(c);
    }
  });
  renderMath(panelEl);
  attachGlossaryTooltips(panelEl);
}

function renderNotesPanel(container, lessonId) {
  const panel = el('div', 'notes-panel');
  panel.innerHTML = '<h4>Le tue note per questa lezione</h4>';
  const ta = el('textarea', 'notes-textarea');
  ta.placeholder = 'Scrivi qui appunti, dubbi, domande. Vengono salvati automaticamente.';
  const notes = store.get(STORAGE.notes);
  ta.value = notes[lessonId] || '';
  const toolbar = el('div', 'notes-toolbar');
  const status = el('span', 'notes-save-status', ta.value ? 'salvate' : '');
  const exportBtn = el('button', 'notes-export', 'Esporta in .md');
  toolbar.appendChild(status); toolbar.appendChild(exportBtn);
  panel.appendChild(ta); panel.appendChild(toolbar);
  container.appendChild(panel);

  let timer;
  ta.addEventListener('input', () => {
    clearTimeout(timer);
    status.textContent = 'salvo...';
    timer = setTimeout(() => {
      const all = store.get(STORAGE.notes);
      all[lessonId] = ta.value;
      store.set(STORAGE.notes, all);
      status.textContent = 'salvate';
      recordActivity(1);
    }, 400);
  });
  exportBtn.addEventListener('click', () => {
    const lesson = window.LESSONS.find(l => l.id === lessonId);
    const md = '# Note su ' + lesson.title + '\n\nData: ' + new Date().toLocaleDateString('it-IT') + '\n\n' + (ta.value || '(nessuna nota)') + '\n';
    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url;
    a.download = 'note-' + lessonId + '.md';
    a.click();
    URL.revokeObjectURL(url);
  });
}

function pageHome() {
  const c = document.getElementById('page-container');
  const total = window.LESSONS.length;
  const done = window.LESSONS.filter(l => isCompleted(l.id)).length;
  const streak = store.get(STORAGE.streak, {});
  c.innerHTML =
    '<div class="home-header">' +
    '<h2>Benvenuto in ML Academy</h2>' +
    '<p>Un percorso completo verso il machine learning, dalla teoria al deployment. Studia 30 minuti al giorno e in 6 mesi puoi essere job-ready.</p>' +
    '</div>' +
    '<div class="dashboard-row">' +
    '<div class="dashboard-card"><h3>La tua attivita</h3><div class="contrib-graph" id="contrib-graph"></div><p style="font-size:11px;color:var(--muted);margin-top:10px;">Ultimi 30 giorni. Colore piu intenso = piu tempo passato sull\'app.</p></div>' +
    '<div class="dashboard-card"><h3>I tuoi numeri</h3>' +
    '<div class="big-metric">' + done + ' <span style="font-size:16px;color:var(--muted);">/ ' + total + '</span></div>' +
    '<div class="metric-label">Lezioni completate</div>' +
    '<hr style="border:none;border-top:1px solid var(--border);margin:14px 0;">' +
    '<div class="big-metric" style="color:var(--fire);">' + (streak.current || 0) + '</div>' +
    '<div class="metric-label">Giorni di streak (record: ' + (streak.longest || 0) + ')</div>' +
    '</div></div>' +
    '<h3 style="margin-top:30px;">Continua da dove hai lasciato</h3>' +
    '<div class="home-grid" id="lesson-cards"></div>';
  const grid = document.getElementById('lesson-cards');
  window.LESSONS.forEach((l, i) => {
    const completed = isCompleted(l.id);
    const card = el('div', 'home-card');
    card.innerHTML =
      '<div class="home-card-tag">' + (l.tag || 'Lezione ' + (i + 1)) + '</div>' +
      '<h3>' + l.title + '</h3>' +
      '<p>' + (l.summary || '') + '</p>' +
      '<div class="home-card-progress">' +
      '<div class="mini-bar"><div class="mini-bar-fill" style="width:' + (completed ? 100 : 0) + '%"></div></div>' +
      '<span>' + (completed ? 'fatto' : 'da fare') + '</span></div>';
    card.addEventListener('click', () => { location.hash = '#/lesson/' + l.id; });
    grid.appendChild(card);
  });
  renderContribGraph('contrib-graph');
}

function renderContribGraph(id) {
  const graph = document.getElementById(id);
  if (!graph) return;
  const activity = store.get(STORAGE.activity, {});
  const today = new Date();
  for (let i = 29; i >= 0; i--) {
    const d = new Date(today); d.setDate(today.getDate() - i);
    const key = d.toISOString().slice(0, 10);
    const xp = activity[key] || 0;
    let level = 0;
    if (xp > 0) level = 1;
    if (xp > 15) level = 2;
    if (xp > 40) level = 3;
    if (xp > 80) level = 4;
    const day = el('div', 'contrib-day level-' + level + (i === 0 ? ' today' : ''));
    day.title = key + ': ' + xp + ' XP';
    graph.appendChild(day);
  }
}

function pageLesson(lessonId) {
  const lesson = window.LESSONS.find(l => l.id === lessonId);
  const c = document.getElementById('page-container');
  if (!lesson) { c.innerHTML = '<h2>Lezione non trovata</h2>'; return; }
  const idx = window.LESSONS.indexOf(lesson);
  c.innerHTML =
    '<div id="lesson-header">' +
    '<div class="breadcrumb"><a href="#/home">Home</a> - Lezione ' + (idx + 1) + ' di ' + window.LESSONS.length + ' - ' + (lesson.tag || '') + '</div>' +
    '<h2 id="lesson-title">' + lesson.title + '</h2>' +
    '<p id="lesson-summary">' + (lesson.summary || '') + '</p>' +
    '<div class="tabs">' +
    '<button class="tab-btn active" data-tab="teoria">Teoria</button>' +
    '<button class="tab-btn" data-tab="esempi">Esempi</button>' +
    '<button class="tab-btn" data-tab="esercizi">Esercizi</button>' +
    '<button class="tab-btn" data-tab="test">Test finale</button>' +
    '<button class="tab-btn" data-tab="risorse">Risorse</button>' +
    '</div></div>' +
    '<section id="tab-teoria" class="tab-panel active"></section>' +
    '<section id="tab-esempi" class="tab-panel"></section>' +
    '<section id="tab-esercizi" class="tab-panel"></section>' +
    '<section id="tab-test" class="tab-panel"></section>' +
    '<section id="tab-risorse" class="tab-panel"></section>' +
    '<div id="notes-container"></div>' +
    '<footer id="lesson-nav">' +
    '<button id="prev-lesson">Precedente</button>' +
    '<button id="mark-complete">' + (isCompleted(lesson.id) ? 'Completata' : 'Segna come completata') + '</button>' +
    '<button id="next-lesson">Prossima</button>' +
    '</footer>';
  renderSection(document.getElementById('tab-teoria'),   lesson.teoria,   { lessonId: lessonId, section: 'teoria' });
  renderSection(document.getElementById('tab-esempi'),   lesson.esempi,   { lessonId: lessonId, section: 'esempi' });
  renderSection(document.getElementById('tab-esercizi'), lesson.esercizi, { lessonId: lessonId, section: 'esercizi' });
  renderSection(document.getElementById('tab-test'),     lesson.test,     { lessonId: lessonId, section: 'test' });
  renderRisorse(document.getElementById('tab-risorse'), lessonId);
  renderNotesPanel(document.getElementById('notes-container'), lessonId);

  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    });
  });
  document.getElementById('prev-lesson').disabled = idx === 0;
  document.getElementById('next-lesson').disabled = idx === window.LESSONS.length - 1;
  document.getElementById('prev-lesson').addEventListener('click', () => { if (idx > 0) location.hash = '#/lesson/' + window.LESSONS[idx - 1].id; });
  document.getElementById('next-lesson').addEventListener('click', () => { if (idx < window.LESSONS.length - 1) location.hash = '#/lesson/' + window.LESSONS[idx + 1].id; });
  const mc = document.getElementById('mark-complete');
  if (isCompleted(lesson.id)) mc.classList.add('done');
  mc.addEventListener('click', () => {
    setCompleted(lesson.id, !isCompleted(lesson.id));
    pageLesson(lessonId);
    renderSidebar();
  });
  recordActivity(3);
}

function renderRisorse(container, lessonId) {
  const pdfs = window.pdfLibrary ? window.pdfLibrary.getForLesson(lessonId) : [];
  container.innerHTML =
    '<h3>PDF associati a questa lezione</h3>' +
    '<p style="color:var(--muted);">Aggiungi PDF dalla <a href="#/library">Libreria</a> e associali qui.</p>' +
    '<div id="lesson-pdf-list"></div>' +
    '<h3 style="margin-top:30px;">Link utili esterni</h3>' +
    '<ul>' +
    '<li><a href="https://scikit-learn.org/stable/user_guide.html" target="_blank">Scikit-learn user guide</a></li>' +
    '<li><a href="https://www.kaggle.com/learn" target="_blank">Kaggle Learn</a> - micro-corsi gratuiti</li>' +
    '<li><a href="https://d2l.ai/" target="_blank">Dive into Deep Learning</a> - libro gratis online</li>' +
    '</ul>';
  const list = document.getElementById('lesson-pdf-list');
  if (pdfs.length === 0) {
    list.innerHTML = '<p style="color:var(--muted);font-style:italic;">Nessun PDF associato. Vai alla Libreria per aggiungerne.</p>';
  } else {
    pdfs.forEach(pdf => {
      const item = el('div', 'pdf-item');
      item.innerHTML = '<div class="pdf-icon">PDF</div>' +
        '<div class="pdf-info"><div class="pdf-name">' + pdf.name + '</div>' +
        '<div class="pdf-meta">' + (pdf.size / 1024 / 1024).toFixed(2) + ' MB - aggiunto ' + new Date(pdf.addedAt).toLocaleDateString('it-IT') + '</div></div>' +
        '<div class="pdf-actions"><button class="view-btn">Apri</button></div>';
      item.querySelector('.view-btn').addEventListener('click', () => window.pdfLibrary.openViewer(pdf.id, container));
      list.appendChild(item);
    });
  }
}

function pageRoadmap() {
  const c = document.getElementById('page-container');
  c.innerHTML =
    '<div class="roadmap-page"><h2>Roadmap job-ready</h2>' +
    '<p style="color:var(--muted);">Piano di studio 6 mesi (30 min al giorno) per arrivare a un livello Junior ML Engineer.</p>' +
    '<div class="roadmap-content" id="roadmap-content"><p style="color:var(--muted);">Caricamento...</p></div></div>';
  fetch('ROADMAP.md').then(r => r.text()).then(md => {
    document.getElementById('roadmap-content').innerHTML = simpleMarkdown(md);
    renderMath(document.getElementById('roadmap-content'));
  }).catch(() => {
    document.getElementById('roadmap-content').innerHTML = '<p>ROADMAP.md non trovato nella root del sito.</p>';
  });
}

function simpleMarkdown(md) {
  let html = md
    .replace(/^### (.*)$/gm, '<h3>$1</h3>')
    .replace(/^## (.*)$/gm, '<h2>$1</h2>')
    .replace(/^# (.*)$/gm, '<h1>$1</h1>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
  html = html.replace(/^(\|.+\|)\n(\|[-:| ]+\|)\n((?:\|.+\|\n?)+)/gm, (m, header, sep, rows) => {
    const hs = header.split('|').filter(x => x.trim()).map(s => '<th>' + s.trim() + '</th>').join('');
    const rs = rows.trim().split('\n').map(r => '<tr>' + r.split('|').filter(x => x.trim()).map(s => '<td>' + s.trim() + '</td>').join('') + '</tr>').join('');
    return '<table><thead><tr>' + hs + '</tr></thead><tbody>' + rs + '</tbody></table>';
  });
  html = html.replace(/^(- .+(\n- .+)*)/gm, m => '<ul>' + m.split('\n').map(l => l.replace(/^- (.+)/, '<li>$1</li>')).join('') + '</ul>');
  html = html.replace(/^(\d+\. .+(\n\d+\. .+)*)/gm, m => '<ol>' + m.split('\n').map(l => l.replace(/^\d+\. (.+)/, '<li>$1</li>')).join('') + '</ol>');
  html = html.replace(/^(?!<[huol]|<table|<li|<\/)([^\n]+)$/gm, '<p>$1</p>');
  return html;
}

function pageGlossary() {
  const c = document.getElementById('page-container');
  const entries = Object.entries(window.GLOSSARY || {}).sort();
  c.innerHTML =
    '<h2>Glossario ML</h2>' +
    '<p style="color:var(--muted);">' + entries.length + ' termini. Nelle lezioni sono evidenziati automaticamente con tooltip.</p>' +
    '<input type="search" id="glo-search" placeholder="Filtra..." style="width:100%;max-width:400px;padding:10px;background:var(--panel-2);color:var(--text);border:1px solid var(--border);border-radius:6px;margin:20px 0;">' +
    '<div class="glossary-list" id="glo-list"></div>';
  const list = document.getElementById('glo-list');
  const filter = document.getElementById('glo-search');
  function paint(q) {
    q = q || '';
    list.innerHTML = '';
    entries.filter(([k, v]) => !q || k.toLowerCase().includes(q.toLowerCase()) || v.toLowerCase().includes(q.toLowerCase()))
      .forEach(([term, def]) => {
        const item = el('div', 'glossary-item');
        item.innerHTML = '<div class="glossary-term">' + term + '</div><div class="glossary-def">' + def + '</div>';
        list.appendChild(item);
      });
    renderMath(list);
  }
  paint();
  filter.addEventListener('input', () => paint(filter.value));
}

function pageStats() {
  const c = document.getElementById('page-container');
  const total = window.LESSONS.length;
  const done = window.LESSONS.filter(l => isCompleted(l.id)).length;
  const streak = store.get(STORAGE.streak, {});
  const activity = store.get(STORAGE.activity, {});
  const notesCount = Object.keys(store.get(STORAGE.notes)).filter(k => store.get(STORAGE.notes)[k]).length;
  const codeCount = Object.keys(store.get(STORAGE.code)).length;
  const activeDays = Object.keys(activity).length;
  const badges = [
    { name: 'Primo passo', desc: '1 lezione', earned: done >= 1 },
    { name: 'Costante', desc: '5 lezioni', earned: done >= 5 },
    { name: 'Completista', desc: 'Tutte le lezioni', earned: done >= total },
    { name: 'Tre giorni', desc: 'Streak 3+', earned: (streak.longest || 0) >= 3 },
    { name: 'Una settimana', desc: 'Streak 7+', earned: (streak.longest || 0) >= 7 },
    { name: 'Un mese', desc: 'Streak 30+', earned: (streak.longest || 0) >= 30 },
    { name: 'Studente', desc: '100 XP', earned: (streak.xp || 0) >= 100 },
    { name: 'Dedicato', desc: '500 XP', earned: (streak.xp || 0) >= 500 },
    { name: 'Riflessivo', desc: 'Note su 3+ lezioni', earned: notesCount >= 3 },
  ];
  c.innerHTML =
    '<h2>Le tue statistiche</h2>' +
    '<div class="stats-grid">' +
    '<div class="stat-card"><div class="stat-value" style="color:var(--fire);">' + (streak.current || 0) + '</div><div class="stat-label">Streak attuale</div></div>' +
    '<div class="stat-card"><div class="stat-value">' + (streak.longest || 0) + '</div><div class="stat-label">Streak record</div></div>' +
    '<div class="stat-card"><div class="stat-value" style="color:var(--accent-2);">' + (streak.xp || 0) + '</div><div class="stat-label">XP totali</div></div>' +
    '<div class="stat-card"><div class="stat-value">' + done + '/' + total + '</div><div class="stat-label">Lezioni</div></div>' +
    '<div class="stat-card"><div class="stat-value">' + activeDays + '</div><div class="stat-label">Giorni attivi</div></div>' +
    '<div class="stat-card"><div class="stat-value">' + notesCount + '</div><div class="stat-label">Lezioni con note</div></div>' +
    '<div class="stat-card"><div class="stat-value">' + codeCount + '</div><div class="stat-label">Esercizi salvati</div></div>' +
    '</div>' +
    '<h3>Badge</h3>' +
    '<div class="badges">' +
    badges.map(b => '<div class="badge ' + (b.earned ? 'earned' : 'locked') + '"><strong>' + b.name + '</strong> - ' + b.desc + '</div>').join('') +
    '</div>' +
    '<h3 style="margin-top:40px;">Attivita ultimi 30 giorni</h3>' +
    '<div class="contrib-graph" id="stats-contrib"></div>';
  renderContribGraph('stats-contrib');
}

function pageLibrary() {
  const c = document.getElementById('page-container');
  c.innerHTML =
    '<h2>Libreria PDF</h2>' +
    '<p style="color:var(--muted);">Carica i tuoi PDF (libri, dispense, appunti). Restano salvati nel browser (IndexedDB) e si aprono direttamente nell\'app. Puoi anche associarli a lezioni specifiche.</p>' +
    '<div class="pdf-drop-zone" id="drop-zone">' +
    '<div class="drop-icon">PDF</div>' +
    '<p><strong>Trascina un PDF qui</strong> oppure clicca per selezionare</p>' +
    '<input type="file" id="pdf-input" accept="application/pdf" multiple style="display:none;">' +
    '</div>' +
    '<div class="pdf-list" id="pdf-list"></div>' +
    '<div id="pdf-viewer-container"></div>';
  const dropZone = document.getElementById('drop-zone');
  const input = document.getElementById('pdf-input');
  dropZone.addEventListener('click', () => input.click());
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', async e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    for (const f of e.dataTransfer.files) if (f.type === 'application/pdf') await window.pdfLibrary.add(f);
    renderPdfList();
  });
  input.addEventListener('change', async () => {
    for (const f of input.files) if (f.type === 'application/pdf') await window.pdfLibrary.add(f);
    input.value = '';
    renderPdfList();
  });
  renderPdfList();
}

function renderPdfList() {
  const list = document.getElementById('pdf-list');
  if (!list) return;
  const pdfs = window.pdfLibrary.getAll();
  if (pdfs.length === 0) { list.innerHTML = '<p style="color:var(--muted);text-align:center;padding:20px;">Nessun PDF caricato.</p>'; return; }
  list.innerHTML = '';
  pdfs.forEach(pdf => {
    const item = el('div', 'pdf-item');
    const options = ['<option value="">- Nessuna lezione -</option>']
      .concat(window.LESSONS.map(l => '<option value="' + l.id + '"' + (pdf.lessonId === l.id ? ' selected' : '') + '>' + l.title + '</option>'))
      .join('');
    item.innerHTML =
      '<div class="pdf-icon">PDF</div>' +
      '<div class="pdf-info"><div class="pdf-name">' + pdf.name + '</div>' +
      '<div class="pdf-meta">' + (pdf.size / 1024 / 1024).toFixed(2) + ' MB - ' + new Date(pdf.addedAt).toLocaleDateString('it-IT') + '</div>' +
      '<div style="margin-top:6px;">Associata a: <select class="pdf-assoc" style="background:var(--panel-2);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:4px;">' + options + '</select></div></div>' +
      '<div class="pdf-actions"><button class="view-btn">Apri</button><button class="del-btn">Elimina</button></div>';
    item.querySelector('.view-btn').addEventListener('click', () => window.pdfLibrary.openViewer(pdf.id, document.getElementById('pdf-viewer-container')));
    item.querySelector('.del-btn').addEventListener('click', () => {
      if (confirm('Eliminare "' + pdf.name + '"?')) window.pdfLibrary.remove(pdf.id).then(renderPdfList);
    });
    item.querySelector('.pdf-assoc').addEventListener('change', e => window.pdfLibrary.setLesson(pdf.id, e.target.value));
    list.appendChild(item);
  });
}

function pageCheatsheet() {
  const c = document.getElementById('page-container');
  c.innerHTML =
    '<div class="cheatsheet-page">' +
    '<h2>Cheat sheet</h2>' +
    '<p style="color:var(--muted);">Riepilogo essenziale di formule, decisioni e codice minimo. Usa "Stampa in PDF" del browser per averlo su carta.</p>' +
    '<button class="print-btn" onclick="window.print()">Stampa questa pagina</button>' +
    '<div id="cheatsheet-content"></div></div>';
  const container = document.getElementById('cheatsheet-content');
  window.CHEATSHEETS.forEach(cs => {
    const s = el('div', 'cheatsheet-section');
    s.innerHTML = '<h3>' + cs.title + '</h3>' + cs.body;
    container.appendChild(s);
  });
  renderMath(container);
}

function pageSearch(query) {
  const c = document.getElementById('page-container');
  c.innerHTML = '<h2>Ricerca: "' + query + '"</h2><div class="search-results" id="sr"></div>';
  const results = [];
  const q = query.toLowerCase();
  window.LESSONS.forEach(l => {
    ['teoria', 'esempi', 'esercizi', 'test'].forEach(sec => {
      (l[sec] || []).forEach(block => {
        const content = JSON.stringify(block).toLowerCase();
        if (content.includes(q)) {
          const idx = content.indexOf(q);
          const start = Math.max(0, idx - 60);
          const raw = JSON.stringify(block).slice(start, idx + q.length + 60);
          const safe = raw.replace(/[<>]/g, '');
          const escQ = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
          const snippet = safe.replace(new RegExp(escQ, 'ig'), m => '<mark>' + m + '</mark>');
          results.push({ lesson: l, section: sec, snippet: snippet });
        }
      });
    });
  });
  const sr = document.getElementById('sr');
  if (results.length === 0) {
    sr.innerHTML = '<p style="color:var(--muted);">Nessun risultato.</p>';
    return;
  }
  results.slice(0, 30).forEach(r => {
    const it = el('div', 'search-result-item');
    it.innerHTML = '<div class="search-result-title">' + r.lesson.title + ' - ' + r.section + '</div><div class="search-result-snippet">...' + r.snippet + '...</div>';
    it.addEventListener('click', () => { location.hash = '#/lesson/' + r.lesson.id; });
    sr.appendChild(it);
  });
}

function renderSidebar() {
  const list = document.getElementById('lesson-list');
  list.innerHTML = '';
  window.LESSONS.forEach((l, i) => {
    const it = el('div', 'lesson-item' + (isCompleted(l.id) ? ' completed' : ''));
    it.innerHTML = '<span class="lesson-num">' + (i + 1) + '</span><span class="lesson-title-text">' + l.title + '</span>';
    it.addEventListener('click', () => { location.hash = '#/lesson/' + l.id; });
    list.appendChild(it);
  });
  updateActiveNav();
}

function updateActiveNav() {
  const hash = location.hash || '#/home';
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const active = document.querySelector('.nav-item[href="' + hash + '"]');
  if (active) active.classList.add('active');
  const parts = hash.slice(2).split('/');
  document.querySelectorAll('.lesson-item').forEach((it, i) => {
    it.classList.toggle('active', parts[0] === 'lesson' && window.LESSONS[i] && window.LESSONS[i].id === parts[1]);
  });
}

function router() {
  const hash = (location.hash || '#/home').slice(2);
  const parts = hash.split('/');
  const route = parts[0];
  const args = parts.slice(1);
  if (route === 'home' || route === '') pageHome();
  else if (route === 'lesson') pageLesson(args[0]);
  else if (route === 'roadmap') pageRoadmap();
  else if (route === 'library') pageLibrary();
  else if (route === 'glossary') pageGlossary();
  else if (route === 'stats') pageStats();
  else if (route === 'cheatsheet') pageCheatsheet();
  else if (route === 'search') pageSearch(decodeURIComponent(args.join('/')));
  else pageHome();
  updateActiveNav();
  window.scrollTo(0, 0);
}

// ------------------------ Mode toggles ------------------------
const MODE_KEY = 'ml_mode_v1';

function loadModePrefs() {
  try {
    return JSON.parse(localStorage.getItem(MODE_KEY)) || { mobile: false, readonly: false };
  } catch { return { mobile: false, readonly: false }; }
}
function saveModePrefs(prefs) {
  localStorage.setItem(MODE_KEY, JSON.stringify(prefs));
}
function applyMode(prefs) {
  document.body.classList.toggle('mobile-mode', !!prefs.mobile);
  document.body.classList.toggle('readonly-mode', !!prefs.readonly);
}

function openSidebar() { document.body.classList.add('sidebar-open'); }
function closeSidebar() { document.body.classList.remove('sidebar-open'); }

// Inserisce un bottone "Tab" dentro ogni code toolbar in modalita mobile
function ensureMobileTabButtons() {
  if (!document.body.classList.contains('mobile-mode')) {
    document.querySelectorAll('.tab-btn-mobile').forEach(b => b.remove());
    return;
  }
  document.querySelectorAll('.code-toolbar').forEach(tb => {
    if (tb.querySelector('.tab-btn-mobile')) return;
    const btn = document.createElement('button');
    btn.className = 'tab-btn-mobile';
    btn.textContent = 'Tab';
    btn.type = 'button';
    btn.title = 'Inserisci indentazione (4 spazi)';
    btn.addEventListener('click', () => {
      const ta = tb.parentElement.querySelector('.code-editor');
      if (!ta) return;
      const s = ta.selectionStart, en = ta.selectionEnd;
      ta.value = ta.value.substring(0, s) + '    ' + ta.value.substring(en);
      ta.selectionStart = ta.selectionEnd = s + 4;
      ta.focus();
      ta.dispatchEvent(new Event('input', { bubbles: true }));
    });
    const badge = tb.querySelector('.saved-badge');
    if (badge) tb.insertBefore(btn, badge); else tb.appendChild(btn);
  });
}

const _codeObserver = new MutationObserver(() => ensureMobileTabButtons());

document.addEventListener('DOMContentLoaded', () => {
  const prefs = loadModePrefs();
  applyMode(prefs);
  const mobileToggle = document.getElementById('mobile-mode-toggle');
  const readonlyToggle = document.getElementById('readonly-toggle');
  if (mobileToggle) mobileToggle.checked = !!prefs.mobile;
  if (readonlyToggle) readonlyToggle.checked = !!prefs.readonly;

  checkStreakOnLoad();
  renderSidebar();
  window.addEventListener('hashchange', router);
  router();
  ensureMobileTabButtons();
  const pageContainer = document.getElementById('page-container');
  if (pageContainer) _codeObserver.observe(pageContainer, { childList: true, subtree: true });

  const search = document.getElementById('global-search');
  search.addEventListener('keydown', e => {
    if (e.key === 'Enter' && search.value.trim()) {
      location.hash = '#/search/' + encodeURIComponent(search.value.trim());
      if (document.body.classList.contains('mobile-mode')) closeSidebar();
    }
  });

  document.getElementById('reset-all').addEventListener('click', () => {
    if (confirm('Cancellare TUTTI i dati: progressi, note, codice salvato, streak, PDF? Non e reversibile.')) {
      Object.values(STORAGE).forEach(k => localStorage.removeItem(k));
      if (window.pdfLibrary) window.pdfLibrary.clear().then(() => location.reload());
      else location.reload();
    }
  });

  if (mobileToggle) {
    mobileToggle.addEventListener('change', () => {
      const p = loadModePrefs();
      p.mobile = mobileToggle.checked;
      saveModePrefs(p);
      applyMode(p);
      ensureMobileTabButtons();
      if (!p.mobile) closeSidebar();
    });
  }
  if (readonlyToggle) {
    readonlyToggle.addEventListener('change', () => {
      const p = loadModePrefs();
      p.readonly = readonlyToggle.checked;
      saveModePrefs(p);
      applyMode(p);
    });
  }

  const menuBtn = document.getElementById('mobile-menu-toggle');
  const overlay = document.getElementById('sidebar-overlay');
  if (menuBtn) menuBtn.addEventListener('click', () => {
    if (document.body.classList.contains('sidebar-open')) closeSidebar();
    else openSidebar();
  });
  if (overlay) overlay.addEventListener('click', closeSidebar);

  window.addEventListener('hashchange', () => {
    if (document.body.classList.contains('mobile-mode')) closeSidebar();
  });

  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('sw.js').catch(() => {});
    });
  }
});
