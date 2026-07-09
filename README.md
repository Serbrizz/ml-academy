# ML Academy — Corso interattivo di Machine Learning (v2)

Web app didattica completa, offline-capable, per imparare il machine learning da zero fino a un livello job-ready. Include 14 lezioni con teoria, esempi eseguibili, esercizi con verifica automatica e test finali. Python (NumPy, Pandas, scikit-learn, matplotlib) gira direttamente nel browser via Pyodide.

## Novità versione 2

- **Multi-pagina** con router: home, roadmap, libreria PDF, glossario, cheat sheet, statistiche
- **Roadmap 6 mesi** dettagliata mese per mese verso il livello Junior ML Engineer (`ROADMAP.md`)
- **Streak giornaliera** con XP e contribution graph tipo GitHub, badge sbloccabili
- **Libreria PDF**: carica libri e dispense nel browser (IndexedDB), viewer integrato, associa PDF a lezioni
- **Autosave** del codice degli esercizi (non perdi il lavoro tra sessioni)
- **Note personali** per ogni lezione, esportabili in Markdown
- **Ricerca full-text** in tutte le lezioni
- **Glossario** con 90 termini, tooltip automatici nelle lezioni
- **Cheat sheet** stampabili con formule e codice essenziale
- **Statistiche** e badge motivazionali

## Struttura file

```
ml-academy/
├── index.html           # struttura pagina e caricamento script
├── styles.css           # tema dark
├── app.js               # core: router, streak, note, ricerca, ecc.
├── glossary.js          # 90 termini di ML con definizione
├── cheatsheets.js       # cheat sheet stampabili
├── pdf-library.js       # gestione PDF via IndexedDB + PDF.js
├── lessons/
│   ├── index.js         # inizializza array LESSONS
│   ├── 01-intro.js      # Lezione 1: Cos'e il ML
│   ├── 02-math.js       # Lezione 2: Ripasso matematico
│   ├── ...
│   └── 14-project.js    # Lezione 14: progetto finale
├── ROADMAP.md           # piano di studio 6 mesi
├── README.md            # questo file
└── .nojekyll            # per GitHub Pages
```

## Come pushare l'update sul repo esistente

Assumendo che tu abbia già `Serbrizz/ml-academy` con la versione 1, e che tu abbia la cartella locale `ml-academy` con Git configurato:

### Passo 1 — Copia i nuovi file

Copia tutti i file di questa cartella `outputs/` dentro la tua cartella locale `ml-academy`, **sostituendo** i vecchi. In PowerShell:

```powershell
cd $HOME\Documents\ml-academy
# per prudenza, rimuovi il vecchio lessons.js (ora e stato splittato in lessons/)
Remove-Item -ErrorAction SilentlyContinue lessons.js
# copia da dove hai i nuovi file (adatta il percorso)
Copy-Item "C:\percorso\dei\file\nuovi\*" -Destination . -Recurse -Force
```

Il percorso di origine è la cartella outputs della sessione:
```
C:\Users\BRIZZIP2\AppData\Local\Claude-3p\local-agent-mode-sessions\608d2558\00000000\local_612ad815-5777-4fe2-8125-528e46a0f732\outputs
```

### Passo 2 — Verifica cosa cambia

```powershell
git status
```

Dovresti vedere:
- File modificati: `index.html`, `styles.css`, `app.js`, `README.md`
- File nuovi: `pdf-library.js`, `glossary.js`, `cheatsheets.js`, `ROADMAP.md`, `lessons/` (cartella)
- File eliminato: `lessons.js` (rimpiazzato dalla cartella lessons/)

### Passo 3 — Commit e push

```powershell
git add -A
git commit -m "v2: multi-page, streak, PDF library, glossario, roadmap"
git push
```

`git add -A` include anche le rimozioni. Dopo 1-2 minuti GitHub Pages ridispiega automaticamente e vedrai la nuova versione su `https://serbrizz.github.io/ml-academy/`.

### Se non ricordi come è configurato Git

Verifica remote:
```powershell
git remote -v
```
Deve mostrare `origin https://github.com/Serbrizz/ml-academy.git`. Se non c'è, aggiungilo:
```powershell
git remote add origin https://github.com/Serbrizz/ml-academy.git
```

## Come lanciare in locale

Serve un server HTTP (il browser blocca `fetch` da `file://`, e la roadmap si carica via fetch):

```bash
cd cartella-del-progetto
python -m http.server 8080
```

Poi apri `http://localhost:8080`. Al primo caricamento Pyodide scarica ~50MB e diventa poi cached. La streak si aggiorna automaticamente ad ogni giorno di utilizzo.

## Come funziona la streak

Ogni azione (aprire una lezione, eseguire codice, rispondere a un quiz, salvare note) conta come attività del giorno e dà XP. La streak sale se studi ogni giorno consecutivo, si azzera se salti più di un giorno. Basta anche una singola azione per mantenerla — l'idea è la costanza, non la durata.

Guadagni XP così:
- Eseguire un esempio o esercizio: 2 XP
- Rispondere correttamente a un quiz: 3 XP
- Risolvere un esercizio (output OK): 10 XP
- Completare una lezione: 25 XP
- Scrivere note: 1 XP
- Aprire una lezione: 3 XP

## Come importare un PDF

Vai su **Libreria PDF** dal menu, trascina il file (o clicca per selezionarlo). Il PDF resta nel tuo browser (IndexedDB), non viene caricato da nessuna parte. Puoi associarlo a una lezione specifica dal menu a tendina — così apparirà nella tab "Risorse" di quella lezione.

## Limiti tecnici da conoscere

- **Pyodide non funziona su `file://`**: serve un server anche in locale (`python -m http.server`). Su GitHub Pages funziona senza problemi.
- **IndexedDB (PDF) ha limite di ~50-100MB** nella maggior parte dei browser. Se carichi troppi PDF, alcuni verranno rifiutati.
- **Le note e la streak sono per browser**: se apri l'app da un dispositivo diverso, non vedi i tuoi progressi. Puoi esportare le note in `.md` per portarle altrove.
- **Reset**: il pulsante in fondo alla sidebar cancella *tutto* (progressi, note, codice, streak, PDF). Usalo con cautela.

## Prossimi sviluppi previsti

- Fase 2: approfondire le 14 lezioni esistenti (più teoria, più esercizi, capitoli intermedi)
- Fase 3: nuove lezioni su gradient boosting, deep learning (PyTorch), NLP, CV, MLOps
- Fase 4: progetti end-to-end guidati (Kaggle house prices, image classification, ecc.)

## Licenza

Uso libero, personale e non. Attribuzione gradita ma non obbligatoria.
