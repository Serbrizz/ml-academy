# ML Academy — Corso interattivo di Machine Learning (v4)

Web app didattica offline-capable per imparare il machine learning da zero fino a un livello job-ready. **19 lezioni** con teoria, esempi eseguibili, esercizi con verifica automatica e test finali. Python (NumPy, Pandas, scikit-learn, matplotlib) gira direttamente nel browser via Pyodide.

## Novità versione 4

- **Lezione 13 (Reti neurali) approfondita**: da 12 a 36KB. MLP from scratch con backprop completo, ottimizzatori (SGD/Momentum/Adam), regolarizzazione (dropout, weight decay), diagnosi problemi di training. 12 esercizi + bonus MLP completo from scratch.
- **Nuova Lezione 14 (CNN e Deep Learning pratico)**: convoluzione, pooling, architetture moderne, transfer learning. Codice PyTorch di riferimento da eseguire in locale. 10 esercizi + bonus.
- **Nuova Lezione 17 (MLOps e Deployment)**: MLflow, FastAPI, Docker, CI/CD, monitoring, data drift con KS test e PSI. 10 esercizi (alcuni in Pyodide, altri con codice da eseguire in locale).
- **Nuova Lezione 18 (Progetto Fraud Detection End-to-End)**: classificazione fortemente sbilanciata con feature engineering, threshold cost-sensitive, PR-AUC, monitoring. 10 esercizi + bonus ROI business.
- **Glossario**: +32 termini (141 totali) — CNN, transfer learning, MLflow, drift, PSI, cost-sensitive, ecc.
- **Cheat sheet**: +4 nuove schede (Reti neurali, CNN, MLOps, Imbalanced classification) per un totale di 20.

## Statistiche corso

- **19 lezioni** strutturate progressivamente
- **152 esercizi** con verifica automatica del codice
- **85 domande** di quiz interattivi
- **89 esempi** Python eseguibili nel browser
- **141 termini** nel glossario ML
- **20 cheat sheet** stampabili

## Ordine attuale delle lezioni

1. Cos'è il Machine Learning
2. Ripasso matematico
3. NumPy e Pandas
4. Esplorazione dei dati e visualizzazione
5. Regressione lineare (approfondita v3)
6. Regressione logistica
7. K-Nearest Neighbors
8. Alberi di decisione e Random Forest
9. Support Vector Machines
10. Valutazione dei modelli (approfondita v3)
11. K-Means e clustering
12. PCA e riduzione dimensionale
13. Reti neurali: fondamenta profonde (approfondita v4)
14. CNN e Deep Learning pratico (nuova v4)
15. Gradient Boosting (v3)
16. Feature Engineering avanzato (v3)
17. MLOps e Deployment (nuova v4)
18. Progetto: Fraud Detection End-to-End (nuova v4)
19. Progetto finale: un modello end-to-end

## Struttura file

```
ml-academy/
├── index.html
├── styles.css
├── app.js
├── glossary.js
├── cheatsheets.js
├── pdf-library.js
├── lessons/
│   ├── index.js
│   ├── 01-intro.js ... 19-fraud-project.js
├── ROADMAP.md
├── README.md
└── .nojekyll
```

## Come aggiornare il repo esistente

Dalla cartella locale `ml-academy` con Git configurato:

```powershell
cd $HOME\Documents\ml-academy

# Cancella tutto tranne .git
Get-ChildItem -Force | Where-Object { $_.Name -ne '.git' } | Remove-Item -Recurse -Force

# Copia i file aggiornati (sostituisci il percorso con la tua cartella outputs)
Copy-Item "C:\Users\BRIZZIP2\AppData\Local\Claude-3p\local-agent-mode-sessions\608d2558\00000000\local_612ad815-5777-4fe2-8125-528e46a0f732\outputs\*" -Destination . -Recurse -Force

git add -A
git status
git commit -m "v4: reti neurali approfondite + CNN + MLOps + progetto fraud"
git push
```

Dopo 1-2 minuti il sito è live su `https://serbrizz.github.io/ml-academy/`. Ricarica con Ctrl+F5 per bypassare la cache.

## Note tecniche sulle nuove lezioni

**CNN (lezione 14)**: Pyodide non ha PyTorch installato. Il codice PyTorch nelle lezioni è marcato come "da eseguire in locale" — installa PyTorch nel tuo ambiente con `pip install torch torchvision` (~1GB per la versione CPU-only). I concetti fondamentali e gli esercizi sono comunque tutti implementati in NumPy, quindi girano nell'app.

**MLOps (lezione 17)**: Il codice di FastAPI, Docker, MLflow è didattico, ovviamente non esegui un server web dentro Pyodide. Gli esercizi in-app si concentrano su joblib serialization, drift detection (KS test, PSI), sistemi di logging — tutte cose che girano nel browser. Il codice di serving/deployment lo prendi come reference per il tuo lavoro in locale.

**Fraud Detection (lezione 18)**: Il progetto usa dataset simulato con distribuzioni realistiche (importi esponenziali, ore di picco notturne per frodi, ecc). Tutto esegue in Pyodide.

## Come funziona la streak

Ogni azione (aprire una lezione, eseguire codice, rispondere a un quiz, salvare note) conta come attività del giorno e dà XP. La streak sale se studi ogni giorno consecutivo, si azzera se salti più di un giorno.

- Aprire una lezione: 3 XP
- Eseguire un esempio o esercizio: 2 XP
- Rispondere correttamente a un quiz: 3 XP
- Risolvere un esercizio (output OK): 10 XP
- Completare una lezione: 25 XP
- Scrivere note: 1 XP

## Contenuto totale ora disponibile

Con 19 lezioni e 152 esercizi hai coperto la maggior parte del percorso Junior ML Engineer descritto nella ROADMAP.md. Cosa manca ancora (per sessioni future):

- **NLP con Transformers**: tokenizzazione, embeddings, fine-tuning BERT con HuggingFace
- **Time series in profondità**: Prophet, ARIMA/SARIMA, deep learning per sequenze
- **SHAP e interpretabilità avanzata** (breve introduzione già in Gradient Boosting e Fraud)
- **2 altri progetti end-to-end** (House Prices Kaggle, Customer Churn)
- **Reinforcement Learning intro** (Q-learning, policy gradients)

## Licenza

Uso libero, personale e non. Attribuzione gradita ma non obbligatoria.
