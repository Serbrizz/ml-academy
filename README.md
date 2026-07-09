# ML Academy — Corso interattivo di Machine Learning

Una web app didattica completa, self-hosted e offline-capable, per imparare il machine learning partendo dai fondamenti fino alle reti neurali. Pensata per chi ha una base matematica (fisica, ingegneria, matematica) e vuole imparare seriamente.

## Cosa contiene

- **14 lezioni** strutturate per costruire progressivamente competenza
- Ogni lezione ha 4 sezioni:
  - **Teoria** con formule LaTeX renderizzate
  - **Esempi** eseguibili direttamente nel browser
  - **Esercizi** con verifica automatica del codice
  - **Test finale** (quiz + notebook pratico)
- Python (con NumPy, Pandas, scikit-learn, matplotlib) girato **direttamente nel browser** via Pyodide — zero installazioni
- Progress tracking salvato in localStorage
- Funziona 100% offline dopo il primo caricamento

## Come lanciarla in locale

Serve un semplice server HTTP (i browser bloccano `fetch` da `file://`). Il modo più semplice:

```bash
cd cartella-del-progetto
python -m http.server 8080
```

Poi apri `http://localhost:8080` nel browser. Il primo caricamento richiede 30-60 secondi (scarica Pyodide + librerie ~50MB). Da lì in poi tutto è in cache.

## Deploy su GitHub Pages

1. Crea un repository su GitHub (es. `ml-academy`)
2. Carica questi file nella root:
   - `index.html`
   - `styles.css`
   - `app.js`
   - `lessons.js`
   - `.nojekyll` (impedisce a Jekyll di elaborare i file)
3. Vai su **Settings → Pages**, seleziona:
   - Source: `Deploy from a branch`
   - Branch: `main` (o `master`), folder: `/ (root)`
4. Dopo un minuto l'app sarà disponibile su `https://<tuo-username>.github.io/ml-academy/`

### Comandi rapidi da terminale

```bash
git init
git add index.html styles.css app.js lessons.js README.md .nojekyll
git commit -m "ML Academy: corso interattivo"
git branch -M main
git remote add origin https://github.com/<tuo-username>/ml-academy.git
git push -u origin main
```

## Struttura file

```
ml-academy/
├── index.html      # struttura della pagina
├── styles.css      # tema dark
├── app.js          # logica: navigazione, Pyodide, verifica esercizi
├── lessons.js      # contenuto di tutte le 14 lezioni
├── README.md       # questo file
└── .nojekyll       # per GitHub Pages
```

Per modificare o aggiungere contenuto, edita `lessons.js`. La struttura di una lezione è documentata in cima al file.

## Le 14 lezioni

1. Cos'è il Machine Learning
2. Ripasso matematico (algebra lineare, calcolo, probabilità)
3. NumPy e Pandas
4. Esplorazione dei dati e visualizzazione
5. Regressione lineare
6. Regressione logistica
7. K-Nearest Neighbors
8. Alberi di decisione e Random Forest
9. Support Vector Machines
10. Valutazione dei modelli (CV, metriche, tuning)
11. K-Means e clustering
12. PCA e riduzione dimensionale
13. Introduzione alle reti neurali
14. Progetto finale end-to-end

## Requisiti tecnici

- Browser moderno (Chrome, Firefox, Safari, Edge — versioni recenti)
- Connessione la prima volta per scaricare Pyodide (poi cache-abilitato)
- ~150MB di spazio browser cache

## Note sulla sicurezza

Il codice Python gira in un sandbox WebAssembly (Pyodide) — è isolato dal sistema. Puoi scrivere qualunque codice senza rischi.

## Licenza

Uso libero, personale e non. Attribuzione gradita ma non obbligatoria.
