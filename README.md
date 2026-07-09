# ML Academy — Corso interattivo di Machine Learning (v3)

Web app didattica offline-capable per imparare il machine learning da zero fino a un livello job-ready. Include 16 lezioni con teoria, esempi eseguibili, esercizi con verifica automatica e test finali. Python (NumPy, Pandas, scikit-learn, matplotlib) gira direttamente nel browser via Pyodide.

## Novità versione 3

- **Lezione 5 (Regressione lineare) approfondita**: da 15 a 90 min. Aggiunte derivazione closed-form, statistica inferenziale (t-test, p-value, intervalli confidenza), diagnostica residui, multicollinearità, Ridge/Lasso/Elastic Net da zero, robust regression con Huber, feature engineering intelligente. 12 esercizi progressivi + bonus expert.
- **Lezione 10 (Valutazione) approfondita**: da 20 a 90 min. Aggiunte nested CV, ROC vs PR-AUC, calibrazione probabilità (Platt, isotonic), model comparison con paired t-test e Wilcoxon, walk-forward validation per time series, gestione classi sbilanciate. 10 esercizi + bonus.
- **Nuova Lezione 14 (Gradient Boosting)**: XGBoost/LightGBM/CatBoost e HistGradientBoosting, algoritmo da zero, tuning con early stopping, feature importance, permutation importance, gestione sbilanciamento. 12 esercizi + bonus Kaggle-level.
- **Nuova Lezione 15 (Feature Engineering avanzato)**: encoding categoriche (one-hot, target, frequency), interazioni, binning, time features (encoding ciclico), aggregation, gestione missing values, robust scaling, target leakage detection. 12 esercizi + bonus domain-aware.
- **Glossario**: +19 termini (109 totali) — gradient boosting, target encoding, SHAP, calibrazione, MCC, PR-AUC, ecc.
- **Cheat sheet**: +2 schede stampabili per le nuove lezioni.

## Statistiche corso

- **16 lezioni** strutturate progressivamente
- **106 esercizi** con verifica automatica del codice
- **69 domande** di quiz interattivi
- **70 esempi** Python eseguibili nel browser
- **109 termini** nel glossario ML
- **16 cheat sheet** stampabili con formule e codice essenziale

## Struttura file

```
ml-academy/
├── index.html                        # struttura pagina
├── styles.css                        # tema dark
├── app.js                            # core: router, streak, note, ricerca
├── glossary.js                       # 109 termini
├── cheatsheets.js                    # 16 schede
├── pdf-library.js                    # gestione PDF (IndexedDB + PDF.js)
├── lessons/
│   ├── index.js
│   ├── 01-intro.js
│   ├── 02-math.js
│   ├── 03-numpy-pandas.js
│   ├── 04-eda.js
│   ├── 05-linear-regression.js       # APPROFONDITA
│   ├── 06-logistic.js
│   ├── 07-knn.js
│   ├── 08-trees.js
│   ├── 09-svm.js
│   ├── 10-eval.js                    # APPROFONDITA
│   ├── 11-kmeans.js
│   ├── 12-pca.js
│   ├── 13-neural-nets.js
│   ├── 14-project.js                 # progetto finale (in coda)
│   ├── 15-gradient-boosting.js       # NUOVA
│   └── 16-feature-engineering.js     # NUOVA
├── ROADMAP.md                        # piano 6 mesi
├── README.md
└── .nojekyll                         # per GitHub Pages
```

Nota: nell'index.html le lezioni 15 e 16 sono caricate prima del progetto finale, quindi nell'ordine in-app diventano lezione 14 (Gradient Boosting) e 15 (Feature Engineering), col progetto come 16.

## Come aggiornare il repo esistente

Dalla cartella locale `ml-academy` con Git configurato:

```powershell
cd $HOME\Documents\ml-academy

# Cancella tutto tranne .git
Get-ChildItem -Force | Where-Object { $_.Name -ne '.git' } | Remove-Item -Recurse -Force

# Copia i file aggiornati (sostituisci il percorso)
Copy-Item "C:\Users\BRIZZIP2\AppData\Local\Claude-3p\local-agent-mode-sessions\608d2558\00000000\local_612ad815-5777-4fe2-8125-528e46a0f732\outputs\*" -Destination . -Recurse -Force

git add -A
git status                           # verifica le modifiche
git commit -m "v3: approfondimento + gradient boosting + feature engineering"
git push
```

Dopo 1-2 minuti il sito è live su `https://serbrizz.github.io/ml-academy/`. Ricarica con Ctrl+F5 per bypassare la cache.

## Come lanciare in locale

```bash
cd cartella-del-progetto
python -m http.server 8080
```

Poi apri `http://localhost:8080`.

## Contenuto delle lezioni approfondite

### Lezione 5 — Regressione lineare (nuova versione)
Copertura: dalla derivazione delle normal equations alla statistica inferenziale (test t sui coefficienti, p-value, intervalli di confidenza), diagnostica visiva dei residui, multicollinearità (numero di condizione), Ridge/Lasso/Elastic Net implementati da zero, robust regression con Huber loss, weighted least squares per rumore eteroschedastico, feature engineering intelligente per non-linearità. Include analisi delle ipotesi di Gauss-Markov e sezione "quando NON usare regressione lineare".

### Lezione 10 — Valutazione dei modelli (nuova versione)
Copertura: train/val/test split, k-fold, stratified, group, time series (walk-forward). Nested CV con esempi eseguibili per stima onesta della performance con tuning. Metriche complete: accuracy, precision, recall, F1, F-beta, MCC, ROC-AUC vs PR-AUC (quando usare l'una o l'altra). Calibrazione delle probabilità con Platt scaling e isotonic regression, reliability diagrams, Brier score. Model comparison con paired t-test e Wilcoxon. Sezione dedicata al data leakage in azione.

### Lezione 14 — Gradient Boosting (nuova)
Copertura: dall'intuizione di "correggere gli errori" alla derivazione del gradiente per la loss quadratica (residui = gradiente negativo). Implementazione from scratch con 3-500 alberi. Confronto XGBoost/LightGBM/CatBoost/HistGradientBoosting (in Pyodide si usa HistGB, ma i concetti sono trasferibili). Iperparametri chiave in ordine di importanza. Early stopping. Feature importance vs permutation importance (perché le native sono biased). Gestione categoriche e sbilanciamento con sample_weight.

### Lezione 15 — Feature Engineering avanzato (nuova)
Copertura: encoding categoriche (one-hot, ordinal, target con smoothing e out-of-fold, frequency, embedding), interazioni tra feature, binning con KBinsDiscretizer, time features con encoding ciclico (sin/cos), aggregation features per entità (groupby con statistiche multiple), missing values con multiple strategie (median, KNN, iterative, indicator), scaling robusto agli outlier, log/power transforms, detect target leakage con permutation importance, feature selection.

## Come continuare oltre

Le prossime sessioni possono aggiungere:
- **Deep learning con PyTorch** (CNN, RNN, Transformer intuition) — richiede rete per scaricare torch
- **Interpretabilità con SHAP** — implementazione dedicata
- **MLOps essentials** — MLflow, FastAPI, Docker
- **Time series in profondità** — Prophet, ARIMA, deep learning per sequenze
- **Progetti end-to-end** guidati su dataset reali

## Licenza

Uso libero, personale e non. Attribuzione gradita ma non obbligatoria.
