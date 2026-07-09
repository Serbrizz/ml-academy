/* Lezione: Progetto finale: un modello end-to-end */
window.LESSONS.push({
  "id": "l14-project",
  "title": "Progetto finale: un modello end-to-end",
  "tag": "Progetto",
  "summary": "Mettere tutto insieme: EDA, feature engineering, modelli, tuning, valutazione.",
  "teoria": [
    {
      "type": "md",
      "content": "\n<h3>14.1 Il workflow di un progetto ML</h3>\n<p>Ora che conosci i pezzi, ecco l'ordine in cui li combini per un progetto reale.</p>\n<ol>\n<li><strong>Definisci il problema</strong>: cosa vuoi predire? Regressione o classificazione? Che metrica ti interessa davvero? (Accuracy non è sempre la risposta giusta)</li>\n<li><strong>Raccogli e ispeziona i dati</strong>: EDA, statistiche descrittive, grafici, valori mancanti, outlier.</li>\n<li><strong>Split train/val/test</strong>: <em>subito</em>, prima di guardare i dati troppo a lungo. Isola il test.</li>\n<li><strong>Baseline</strong>: modello stupido (majority class, media, regressione lineare). Ti dice il minimo accettabile.</li>\n<li><strong>Preprocessing e feature engineering</strong>: dentro Pipeline. Standardizza, encoding categoriche, feature derivate.</li>\n<li><strong>Prova più modelli</strong>: lineare, tree, ensemble, magari MLP. Con CV per stimare performance.</li>\n<li><strong>Tuning</strong>: grid search o random search sui migliori 1-2 modelli.</li>\n<li><strong>Valutazione finale</strong>: sul test set, una volta sola. Riporta con intervallo di confidenza.</li>\n<li><strong>Interpretazione</strong>: feature importance, esempi predetti male, casi limite. Un modello che non capisci è pericoloso.</li>\n</ol>\n\n<h3>14.2 Errori comuni</h3>\n<ul>\n<li><strong>Data leakage</strong>: statistiche calcolate su tutto il dataset, uso di feature che nascondono $y$</li>\n<li><strong>Guardare il test durante lo sviluppo</strong>: overfitting sul test</li>\n<li><strong>Ignorare l'imbalance</strong>: accuracy 99% con classe rara è fake</li>\n<li><strong>Trascurare i baseline</strong>: se un modello mediocre batte il tuo modello complesso, hai un bug</li>\n<li><strong>Cercare complessità prima di robustezza</strong>: prima fai funzionare la lineare, poi passi ai transformer</li>\n</ul>\n\n<h3>14.3 Prossimi passi</h3>\n<p>Fatta questa base, per continuare:</p>\n<ul>\n<li>Kaggle: 3-5 competizioni tabulari per consolidare. Poi immagini/testo.</li>\n<li>Corso di deep learning di Andrew Ng o fast.ai.</li>\n<li>Libro \"Hands-On Machine Learning\" di Géron dalla copertina alla copertina.</li>\n<li>Un progetto personale con dati che ti interessano davvero. La motivazione è il carburante.</li>\n</ul>\n"
    },
    {
      "type": "callout",
      "variant": "tip",
      "title": "Ultima raccomandazione",
      "content": "Il ML si impara costruendo modelli, non leggendo di modelli. Ogni ora di libro deve essere seguita da due ore di codice. E non aver paura di fallire: un modello che crolla ti insegna più di dieci che funzionano subito."
    }
  ],
  "esempi": [
    {
      "type": "md",
      "content": "<h3>Progetto completo su breast_cancer</h3><p>Un template end-to-end che puoi copiare e adattare a qualsiasi problema tabulare.</p>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nimport pandas as pd\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.svm import SVC\nfrom sklearn.metrics import classification_report, roc_auc_score, confusion_matrix\n\n# 1) Carica e ispeziona\ndata = load_breast_cancer()\nX, y = data.data, data.target\nprint(f\"Shape: {X.shape}, classi: {np.bincount(y)}\")\n\n# 2) Split\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n\n# 3) Baseline\nfrom sklearn.dummy import DummyClassifier\ndummy = DummyClassifier(strategy='most_frequent').fit(Xt, yt)\nprint(f\"Baseline (majority): {dummy.score(Xe, ye):.4f}\")\n\n# 4) Prova tre modelli con CV\nmodels = {\n    'Logistic':  Pipeline([('s', StandardScaler()), ('m', LogisticRegression(max_iter=5000))]),\n    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=0),\n    'SVM-RBF':   Pipeline([('s', StandardScaler()), ('m', SVC(probability=True))]),\n}\ncv = StratifiedKFold(5, shuffle=True, random_state=0)\nfor name, m in models.items():\n    s = cross_val_score(m, Xt, yt, cv=cv, scoring='roc_auc')\n    print(f\"{name:15s} CV AUC: {s.mean():.4f} +/- {s.std():.4f}\")\n\n# 5) Tuning del migliore\ngrid = {'m__C':[0.01, 0.1, 1, 10, 100]}\ngs = GridSearchCV(models['Logistic'], grid, cv=cv, scoring='roc_auc').fit(Xt, yt)\nprint(f\"\\nMigliori parametri: {gs.best_params_}\")\n\n# 6) Valutazione finale\nfinal = gs.best_estimator_\nyp = final.predict(Xe); yprob = final.predict_proba(Xe)[:,1]\nprint(f\"\\nTest AUC:      {roc_auc_score(ye, yprob):.4f}\")\nprint(f\"Test Accuracy: {(yp==ye).mean():.4f}\")\nprint(\"\\nMatrice di confusione:\\n\", confusion_matrix(ye, yp))\nprint(\"\\nReport:\\n\", classification_report(ye, yp, digits=3))"
    }
  ],
  "esercizi": [
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Progetto — California housing",
      "prompt": "<p>Task di regressione. Prevedi il target sulla base di 8 feature. Devi ottenere <strong>R² test > 0.75</strong>. Sei libero di scegliere modello, preprocessing, tuning.</p>",
      "starter": "import numpy as np\nfrom sklearn.datasets import fetch_california_housing\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\nfrom sklearn.linear_model import Ridge\n\ntry:\n    d = fetch_california_housing()\n    X, y = d.data, d.target\nexcept Exception as e:\n    # fallback se il download non è possibile in Pyodide\n    np.random.seed(0)\n    X = np.random.randn(2000, 8)\n    y = (2*X[:,0] - X[:,1] + 0.5*X[:,0]*X[:,2] + np.random.randn(2000)*0.5)\n\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=42)\n\nr2 = None\nprint(\"R²:\", r2)",
      "check": "if r2 is None: print(\"FAIL: None\")\nelif r2 > 0.75: print(f\"OK — R² = {r2:.4f}\")\nelse: print(f\"FAIL — R² = {r2:.4f}\")",
      "solution": "m = GradientBoostingRegressor(n_estimators=300, max_depth=4, random_state=0).fit(Xt, yt)\nr2 = m.score(Xe, ye)\n# Random Forest funziona quasi identico:\n# m = RandomForestRegressor(n_estimators=200, random_state=0).fit(Xt, yt)"
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Progetto — pipeline completa da zero",
      "prompt": "<p>Sul dataset wine, costruisci una pipeline che comprenda: standardizzazione, PCA (mantieni il 95% di varianza), e un classificatore a tua scelta. Fai tuning con GridSearchCV. Target: accuracy test > 0.95. Salva in <code class=\"inline\">acc</code>.</p>",
      "starter": "from sklearn.datasets import load_wine\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.svm import SVC\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import GridSearchCV, train_test_split\n\nX, y = load_wine(return_X_y=True)\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)\n\nacc = None",
      "check": "if acc is None: print(\"FAIL: None\")\nelif acc > 0.95: print(f\"OK — {acc:.4f}\")\nelse: print(f\"FAIL — {acc:.4f}\")",
      "solution": "pipe = Pipeline([('s', StandardScaler()),\n                 ('pca', PCA(n_components=0.95)),\n                 ('c', LogisticRegression(max_iter=5000))])\ngs = GridSearchCV(pipe, {'c__C':[0.1, 1, 10]}, cv=5).fit(Xt, yt)\nacc = gs.score(Xe, ye)"
    }
  ],
  "test": [
    {
      "type": "md",
      "content": "<h3>Test finale — riflessione</h3><p>Non c'è codice: rispondi alle domande. Sono le più importanti che ti farai iniziando un progetto reale.</p>"
    },
    {
      "type": "quiz",
      "questions": [
        {
          "q": "Un modello ha CV accuracy 0.95 ma sul test cala a 0.85. La spiegazione più probabile è:",
          "options": [
            "Test set piccolo (varianza alta)",
            "Data leakage durante training",
            "Hai fatto tuning contro il test set — hai overfittato sul test",
            "Tutte plausibili, andrebbero verificate una per una"
          ],
          "answer": 3,
          "explain": "Diagnosticare senza dati aggiuntivi richiede investigazione, non conclusioni affrettate."
        },
        {
          "q": "Il primo passo di un progetto è:",
          "options": [
            "Scegliere il modello",
            "Definire chiaramente cosa vuoi predire e con che metrica",
            "Cercare dati",
            "Installare le librerie"
          ],
          "answer": 1,
          "explain": "Senza definizione del problema, tutto il resto è casuale."
        },
        {
          "q": "Un baseline stupido è utile perché:",
          "options": [
            "Fa perdere tempo",
            "Stabilisce il floor che qualsiasi modello serio deve battere",
            "È divertente",
            "Non lo è"
          ],
          "answer": 1,
          "explain": "Se il tuo Deep Learning fa uguale alla media, hai un bug o un problema di segnale."
        },
        {
          "q": "Il test set va guardato:",
          "options": [
            "Ogni volta che iteri",
            "Solo alla fine, una sola volta",
            "Ogni epoch",
            "Solo se train val hanno grandi gap"
          ],
          "answer": 1,
          "explain": "Guardare il test durante lo sviluppo lo trasforma in un secondo val set."
        }
      ]
    },
    {
      "type": "md",
      "content": "<h3>Congratulazioni</h3><p>Se sei arrivato qui e hai completato gli esercizi, hai una base solida di machine learning classico. I prossimi passi naturali sono: (1) fare 3-5 progetti Kaggle su dati tabulari, (2) imparare gradient boosting avanzato (XGBoost, LightGBM), (3) iniziare deep learning con PyTorch, (4) leggere il libro di Géron per approfondire.</p><p><strong>Il ML si impara facendo. Adesso inizia a farlo.</strong></p>"
    }
  ]
});
