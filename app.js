/* ML Academy - main app logic */

const STORAGE_KEY = 'ml_academy_progress_v1';
let currentLessonIdx = 0;
let pyodide = null;
let pyodideLoading = null;

// ---------- Progress ----------
function loadProgress() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; }
  catch { return {}; }
}
function saveProgress(p) { localStorage.setItem(STORAGE_KEY, JSON.stringify(p)); }
function isCompleted(id) { return !!loadProgress()[id]; }
function setCompleted(id, val) {
  const p = loadProgress();
  if (val) p[id] = { completedAt: Date.now() }; else delete p[id];
  saveProgress(p);
  updateProgressUI();
}

function updateProgressUI() {
  const total = LESSONS.length;
  const done = LESSONS.filter(l => isCompleted(l.id)).length;
  document.getElementById('progress-count').textContent = done;
  document.getElementById('progress-total').textContent = total;
  document.getElementById('progress-fill').style.width = (100 * done / total) + '%';
  renderLessonList();
}

// ---------- Pyodide ----------
async function ensurePyodide() {
  if (pyodide) return pyodide;
  if (pyodideLoading) return pyodideLoading;
  const stateEl = document.getElementById('pyodide-state');
  stateEl.textContent = 'caricamento...';
  stateEl.className = 'loading';
  pyodideLoading = (async () => {
    pyodide = await loadPyodide();
    stateEl.textContent = 'installo pacchetti...';
    await pyodide.loadPackage(['numpy', 'pandas', 'scikit-learn', 'matplotlib']);
    // Configure matplotlib to a non-interactive backend
    await pyodide.runPythonAsync(`
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import io, base64
def _mpl_show():
    buf = io.BytesIO()
    plt.gcf().savefig(buf, format='png', bbox_inches='tight', dpi=110)
    plt.close('all')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')
`);
    stateEl.textContent = 'pronto';
    stateEl.className = 'ready';
    return pyodide;
  })();
  return pyodideLoading;
}

async function runPythonCode(code, outEl) {
  outEl.className = 'code-output';
  outEl.textContent = 'Avvio Python...';
  try {
    const py = await ensurePyodide();
    // Redirect stdout
    py.runPython(`
import sys, io
_stdout_buf = io.StringIO()
sys.stdout = _stdout_buf
sys.stderr = _stdout_buf
`);
    let hasFigure = false;
    const wrapped = code + `
try:
    import matplotlib.pyplot as _plt
    if _plt.get_fignums():
        __figure_data = _mpl_show()
    else:
        __figure_data = None
except Exception:
    __figure_data = None
`;
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
    return { ok: true, output };
  } catch (e) {
    outEl.className = 'code-output error';
    outEl.textContent = String(e.message || e);
    return { ok: false, error: String(e) };
  } finally {
    try { pyodide && pyodide.runPython('sys.stdout = sys.__stdout__; sys.stderr = sys.__stderr__'); } catch {}
  }
}

// ---------- Rendering helpers ----------
function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
}

function renderCodeRunner(container, initialCode, opts = {}) {
  const wrap = el('div', 'code-runner');
  const ta = el('textarea', 'code-editor');
  ta.value = initialCode;
  ta.spellcheck = false;
  const tb = el('div', 'code-toolbar');
  const runBtn = el('button', 'run-btn', 'Esegui');
  const resetBtn = el('button', 'reset-btn', 'Reset');
  tb.appendChild(runBtn);
  tb.appendChild(resetBtn);
  let checkBtn = null;
  if (opts.check) {
    checkBtn = el('button', 'check-btn', 'Verifica soluzione');
    tb.appendChild(checkBtn);
  }
  const out = el('div', 'code-output');
  out.textContent = '(nessun output ancora — clicca Esegui)';
  wrap.appendChild(ta);
  wrap.appendChild(tb);
  wrap.appendChild(out);
  container.appendChild(wrap);

  runBtn.addEventListener('click', async () => {
    runBtn.disabled = true;
    runBtn.textContent = 'Eseguo...';
    await runPythonCode(ta.value, out);
    runBtn.disabled = false;
    runBtn.textContent = 'Esegui';
  });
  resetBtn.addEventListener('click', () => { ta.value = initialCode; });

  if (checkBtn && opts.check) {
    checkBtn.addEventListener('click', async () => {
      checkBtn.disabled = true;
      checkBtn.textContent = 'Verifico...';
      const codeWithCheck = ta.value + '\n\n# --- verifica ---\n' + opts.check;
      const r = await runPythonCode(codeWithCheck, out);
      checkBtn.disabled = false;
      checkBtn.textContent = 'Verifica soluzione';
    });
  }

  // Tab key support
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
  q.appendChild(el('div', 'q-text', `<strong>Domanda ${idx + 1}.</strong> ${question.q}`));
  const opts = el('div', 'quiz-options');
  question.options.forEach((opt, i) => {
    const o = el('div', 'quiz-option', opt);
    o.addEventListener('click', () => {
      opts.querySelectorAll('.quiz-option').forEach(x => x.classList.remove('selected', 'correct', 'wrong'));
      o.classList.add('selected');
      const fb = q.querySelector('.quiz-feedback');
      if (i === question.answer) {
        o.classList.add('correct');
        fb.className = 'quiz-feedback visible correct';
        fb.innerHTML = '<strong>Corretto.</strong> ' + (question.explain || '');
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

// Content rendering: builds one tab-panel from a lesson section
function renderSection(panelEl, section) {
  panelEl.innerHTML = '';
  if (!section) { panelEl.innerHTML = '<p style="color:var(--muted)">Nessun contenuto per questa sezione.</p>'; return; }
  section.forEach(block => {
    if (block.type === 'md') {
      const d = el('div');
      d.innerHTML = block.content;
      panelEl.appendChild(d);
    } else if (block.type === 'code') {
      const p = el('pre', 'code');
      p.textContent = block.content;
      panelEl.appendChild(p);
    } else if (block.type === 'run') {
      renderCodeRunner(panelEl, block.content);
    } else if (block.type === 'exercise') {
      const ex = el('div', 'exercise');
      ex.innerHTML = `<span class="exercise-tag tag-${block.difficulty || 'easy'}">${block.difficulty || 'easy'}</span>
        <h4>${block.title}</h4>
        <div>${block.prompt}</div>`;
      renderCodeRunner(ex, block.starter || '# scrivi qui la tua soluzione\n', { check: block.check });
      if (block.solution) {
        const revealBtn = el('button', 'reveal-btn', 'Mostra soluzione');
        const sol = el('div', 'solution');
        sol.innerHTML = '<h4>Soluzione</h4>';
        const pre = el('pre', 'code');
        pre.textContent = block.solution;
        sol.appendChild(pre);
        if (block.solutionNote) {
          const n = el('div');
          n.innerHTML = block.solutionNote;
          sol.appendChild(n);
        }
        revealBtn.addEventListener('click', () => sol.classList.toggle('visible'));
        ex.appendChild(revealBtn);
        ex.appendChild(sol);
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
  // Render math
  if (window.renderMathInElement) {
    renderMathInElement(panelEl, {
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

// ---------- Lesson list & navigation ----------
function renderLessonList() {
  const list = document.getElementById('lesson-list');
  list.innerHTML = '';
  LESSONS.forEach((l, i) => {
    const it = el('div', 'lesson-item' + (i === currentLessonIdx ? ' active' : '') + (isCompleted(l.id) ? ' completed' : ''));
    it.innerHTML = `<span class="lesson-num">${i + 1}</span><span class="lesson-title-wrap"><span class="lesson-title">${l.title}</span><span class="lesson-tag">${l.tag || ''}</span></span>`;
    it.addEventListener('click', () => selectLesson(i));
    list.appendChild(it);
  });
}

function selectLesson(idx) {
  currentLessonIdx = Math.max(0, Math.min(LESSONS.length - 1, idx));
  const lesson = LESSONS[currentLessonIdx];
  document.getElementById('lesson-index').textContent = `Lezione ${currentLessonIdx + 1} di ${LESSONS.length} · ${lesson.tag || ''}`;
  document.getElementById('lesson-title').textContent = lesson.title;
  document.getElementById('lesson-summary').textContent = lesson.summary || '';
  renderSection(document.getElementById('tab-teoria'), lesson.teoria);
  renderSection(document.getElementById('tab-esempi'), lesson.esempi);
  renderSection(document.getElementById('tab-esercizi'), lesson.esercizi);
  renderSection(document.getElementById('tab-test'), lesson.test);
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === 'teoria'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-teoria').classList.add('active');
  document.getElementById('prev-lesson').disabled = currentLessonIdx === 0;
  document.getElementById('next-lesson').disabled = currentLessonIdx === LESSONS.length - 1;
  const mc = document.getElementById('mark-complete');
  if (isCompleted(lesson.id)) { mc.textContent = 'Completata ✓'; mc.classList.add('done'); }
  else { mc.textContent = 'Segna come completata'; mc.classList.remove('done'); }
  renderLessonList();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ---------- Event wiring ----------
document.addEventListener('DOMContentLoaded', () => {
  updateProgressUI();
  selectLesson(0);

  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    });
  });

  document.getElementById('prev-lesson').addEventListener('click', () => selectLesson(currentLessonIdx - 1));
  document.getElementById('next-lesson').addEventListener('click', () => selectLesson(currentLessonIdx + 1));
  document.getElementById('mark-complete').addEventListener('click', () => {
    const id = LESSONS[currentLessonIdx].id;
    setCompleted(id, !isCompleted(id));
    selectLesson(currentLessonIdx);
  });
  document.getElementById('reset-progress').addEventListener('click', () => {
    if (confirm('Sei sicuro di voler azzerare tutti i progressi?')) {
      localStorage.removeItem(STORAGE_KEY);
      updateProgressUI();
      selectLesson(currentLessonIdx);
    }
  });
});
