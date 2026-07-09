/* Lezione: Valutazione dei modelli */
window.LESSONS.push({
  "id": "l10-eval",
  "title": "Valutazione dei modelli",
  "tag": "Workflow",
  "summary": "Train/val/test, k-fold CV, metriche, learning curves. Fare la valutazione bene è più difficile del training.",
  "teoria": [
    {
      "type": "md",
      "content": "\n<h3>10.1 Il pericolo: valutare troppo poco</h3>\n<p>La domanda che il ML cerca di rispondere è: <em>come funzionerà su dati mai visti?</em> Se sbagli la valutazione, tutto il resto è inutile: pubblicherai modelli che sembrano ottimi e falliscono in produzione.</p>\n\n<h3>10.2 Train/val/test split</h3>\n<p>Il minimo sindacale:</p>\n<ul>\n<li><strong>Training</strong> (60-80%): per addestrare</li>\n<li><strong>Validation</strong> (10-20%): per scegliere iperparametri e modello</li>\n<li><strong>Test</strong> (10-20%): un solo utilizzo, alla fine, per stimare la generalizzazione</li>\n</ul>\n<p>Il test set deve essere sacro. Se lo guardi 100 volte e scegli il modello che fa meglio, hai fatto overfitting sul test.</p>\n\n<h3>10.3 K-fold cross-validation</h3>\n<p>Split del training in $K$ fold. Per ciascuno: addestra sui $K-1$ fold, valida sul fold rimanente. Il punteggio è la media. Vantaggi: usa tutti i dati per addestrare e per validare, riduce la varianza della stima.</p>\n<p>Varianti:</p>\n<ul>\n<li><strong>Stratified K-fold</strong>: mantiene la proporzione di classi in ogni fold (obbligatorio con classi sbilanciate)</li>\n<li><strong>Leave-one-out</strong>: $K = n$. Massima varianza, minimo bias. Costoso.</li>\n<li><strong>Time series split</strong>: per dati temporali, mai fare split casuale. Split cronologico.</li>\n<li><strong>Group K-fold</strong>: per dati raggruppati (es. pazienti con più misure) — non contaminare tra train e val</li>\n</ul>\n\n<h3>10.4 La regola d'oro del CV</h3>\n<p>Ogni operazione che coinvolge $y$ (o statistiche dei dati) deve essere fatta <strong>dentro</strong> ogni fold, non prima. Esempi di leakage:</p>\n<ul>\n<li>Scaling calcolato su tutti i dati &rarr; leakage</li>\n<li>Feature selection basata su correlazione con $y$ prima dello split &rarr; leakage</li>\n<li>Imputazione con la media di tutto il dataset &rarr; leakage</li>\n</ul>\n<p>Usa sempre <code class=\"inline\">Pipeline</code> di scikit-learn: garantisce che ogni step sia refittato dentro il fold.</p>\n\n<h3>10.5 Bias-variance tradeoff</h3>\n<p>Per un dato punto $x_0$, decomponendo l'errore atteso:</p>\n<p>$$\\mathbb{E}[(y_0 - \\hat f(x_0))^2] = \\underbrace{[\\mathbb{E}\\hat f(x_0) - f(x_0)]^2}_{\\text{bias}^2} + \\underbrace{\\mathrm{Var}(\\hat f(x_0))}_{\\text{variance}} + \\sigma^2$$</p>\n<p>Modelli semplici: alto bias, bassa variance (underfit). Modelli complessi: basso bias, alta variance (overfit). L'ottimo è nel mezzo.</p>\n\n<h3>10.6 Learning curves</h3>\n<p>Grafico dell'errore in funzione della dimensione del training set. Ti dicono se hai un problema di bias (curve piatte, entrambe alte) o di variance (grande gap tra train e val). Guida per decidere se raccogliere più dati o cambiare modello.</p>\n\n<h3>10.7 Curve ROC e Precision-Recall</h3>\n<p>Per classificazione binaria con score continuo:</p>\n<ul>\n<li><strong>ROC</strong>: TPR (recall) vs FPR al variare della soglia. Area = AUC.</li>\n<li><strong>PR</strong>: Precision vs Recall. Preferibile con classi sbilanciate perché non \"ricompensa\" i negativi facili.</li>\n</ul>\n\n<h3>10.8 Regola definitiva</h3>\n<p>Il test set si guarda <strong>una volta sola</strong>, alla fine. Se lo consulti durante lo sviluppo, non è più un test: è un secondo validation. Ricomincia da capo con nuovi dati.</p>\n"
    }
  ],
  "esempi": [
    {
      "type": "md",
      "content": "<h3>Esempio 1: pipeline corretta con CV</h3>"
    },
    {
      "type": "run",
      "content": "from sklearn.datasets import load_breast_cancer\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.model_selection import cross_val_score, StratifiedKFold\nimport numpy as np\n\nX, y = load_breast_cancer(return_X_y=True)\npipe = Pipeline([('s', StandardScaler()), ('lr', LogisticRegression(max_iter=1000))])\ncv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\nscores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')\nprint(f\"CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}\")\nprint(f\"Fold: {scores.round(4)}\")"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 2: learning curve</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.datasets import make_classification\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.model_selection import learning_curve\n\nX, y = make_classification(n_samples=1000, n_features=20, random_state=0)\nsizes = np.linspace(0.1, 1.0, 8)\n\nfor ax_title, model in [('Tree depth=None (high var)', DecisionTreeClassifier(random_state=0)),\n                         ('Tree depth=2 (high bias)', DecisionTreeClassifier(max_depth=2, random_state=0))]:\n    ts, train_s, val_s = learning_curve(model, X, y, train_sizes=sizes, cv=5)\n    print(f\"\\n{ax_title}\")\n    for t, tr, va in zip(ts, train_s.mean(1), val_s.mean(1)):\n        print(f\"  n={t:4d}: train {tr:.3f}, val {va:.3f}\")"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 3: GridSearchCV con pipeline</h3>"
    },
    {
      "type": "run",
      "content": "from sklearn.datasets import load_wine\nfrom sklearn.svm import SVC\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.model_selection import GridSearchCV, train_test_split\n\nX, y = load_wine(return_X_y=True)\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)\n\npipe = Pipeline([('s', StandardScaler()), ('svm', SVC())])\ngrid = {'svm__C':[0.1,1,10], 'svm__gamma':['scale',0.01,0.1], 'svm__kernel':['rbf','linear']}\ngs = GridSearchCV(pipe, grid, cv=5, scoring='accuracy').fit(Xt, yt)\n\nprint(f\"Migliori parametri: {gs.best_params_}\")\nprint(f\"Migliore CV score:  {gs.best_score_:.4f}\")\nprint(f\"Score sul test:     {gs.score(Xe, ye):.4f}\")"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 4: leakage in azione</h3><p>Facciamo un errore di proposito e vediamo cosa succede.</p>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import cross_val_score\n\nX, y = load_breast_cancer(return_X_y=True)\n\n# SBAGLIATO: scaling PRIMA di CV\nX_scaled = StandardScaler().fit_transform(X)  # usa TUTTI i dati\nscores_bad = cross_val_score(LogisticRegression(max_iter=1000), X_scaled, y, cv=5)\nprint(f\"Con leakage (scaling globale): {scores_bad.mean():.4f}\")\n\n# CORRETTO: scaling DENTRO la pipeline\nfrom sklearn.pipeline import Pipeline\npipe = Pipeline([('s', StandardScaler()), ('lr', LogisticRegression(max_iter=1000))])\nscores_good = cross_val_score(pipe, X, y, cv=5)\nprint(f\"Senza leakage (in pipeline):   {scores_good.mean():.4f}\")\nprint(\"\\nQui la differenza è piccola, ma con dataset piccoli o feature engineering complesso può essere enorme e ingannevole.\")"
    }
  ],
  "esercizi": [
    {
      "type": "exercise",
      "difficulty": "easy",
      "title": "Es. 10.1 — CV manuale",
      "prompt": "<p>Implementa una funzione <code class=\"inline\">my_cv(model, X, y, k)</code> che ritorna la lista degli accuracy sui $k$ fold. Non serve stratified, va bene k-fold semplice.</p>",
      "starter": "import numpy as np\ndef my_cv(model, X, y, k=5):\n    n = len(X)\n    idx = np.arange(n)\n    np.random.default_rng(0).shuffle(idx)\n    folds = np.array_split(idx, k)\n    scores = []\n    # TODO: per ogni fold, addestra sul resto e valida sul fold\n    return scores\n\nfrom sklearn.datasets import load_iris\nfrom sklearn.linear_model import LogisticRegression\nX, y = load_iris(return_X_y=True)\nprint(my_cv(LogisticRegression(max_iter=1000), X, y, k=5))",
      "check": "import numpy as np\nscores = my_cv(LogisticRegression(max_iter=1000), X, y, k=5)\nif len(scores) == 5 and all(s > 0.8 for s in scores):\n    print(f\"OK — mean {np.mean(scores):.4f}\")\nelse:\n    print(f\"FAIL — {scores}\")",
      "solution": "for i in range(k):\n    val = folds[i]\n    train = np.concatenate([folds[j] for j in range(k) if j != i])\n    m = model.__class__(**model.get_params()).fit(X[train], y[train])\n    scores.append(m.score(X[val], y[val]))\nreturn scores"
    },
    {
      "type": "exercise",
      "difficulty": "medium",
      "title": "Es. 10.2 — Curva ROC manuale",
      "prompt": "<p>Data <code class=\"inline\">y_true</code> e <code class=\"inline\">y_score</code>, calcola manualmente TPR e FPR per ogni soglia in <code class=\"inline\">y_score</code>. Salva i due array in <code class=\"inline\">tpr, fpr</code>.</p>",
      "starter": "import numpy as np\nnp.random.seed(0)\ny_true = np.array([0,0,1,1,1,0,1,0,1,0])\ny_score = np.array([0.1,0.4,0.35,0.8,0.7,0.2,0.6,0.3,0.9,0.15])\n\nthresholds = np.sort(y_score)[::-1]\ntpr = []\nfpr = []\n# TODO\nprint(list(zip(thresholds, tpr, fpr)))",
      "check": "import numpy as np\nfrom sklearn.metrics import roc_auc_score\n# controlla monotonicità e range\nif len(tpr) == len(thresholds) and len(fpr) == len(thresholds):\n    if all(0 <= x <= 1 for x in tpr+fpr):\n        print(\"OK — arrays coerenti\")\n    else: print(\"FAIL — valori fuori [0,1]\")\nelse:\n    print(\"FAIL — lunghezze\")",
      "solution": "P = (y_true == 1).sum(); N = (y_true == 0).sum()\nfor t in thresholds:\n    pred = (y_score >= t).astype(int)\n    tp = ((pred==1)&(y_true==1)).sum()\n    fp = ((pred==1)&(y_true==0)).sum()\n    tpr.append(tp/P); fpr.append(fp/N)"
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Es. 10.3 — Nested CV",
      "prompt": "<p>Implementa una nested cross-validation: outer 5-fold per stimare la performance, inner 3-fold per selezionare l'iperparametro <code class=\"inline\">C</code> di una logistica tra $\\{0.01, 0.1, 1, 10\\}$. Salva l'accuracy media in <code class=\"inline\">nested_score</code>.</p>",
      "starter": "import numpy as np\nfrom sklearn.datasets import load_iris\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold\n\nX, y = load_iris(return_X_y=True)\n\n# TODO\nnested_score = None\nprint(\"Nested CV:\", nested_score)",
      "check": "if nested_score is not None and nested_score > 0.9:\n    print(f\"OK — {nested_score:.4f}\")\nelse:\n    print(f\"FAIL — {nested_score}\")",
      "solution": "pipe = Pipeline([('s', StandardScaler()), ('lr', LogisticRegression(max_iter=5000))])\ninner = StratifiedKFold(3, shuffle=True, random_state=0)\nouter = StratifiedKFold(5, shuffle=True, random_state=0)\ngs = GridSearchCV(pipe, {'lr__C':[0.01,0.1,1,10]}, cv=inner)\nnested_score = cross_val_score(gs, X, y, cv=outer).mean()"
    }
  ],
  "test": [
    {
      "type": "quiz",
      "questions": [
        {
          "q": "Perché non calcolare la media dell'imputazione su tutto il dataset prima dello split?",
          "options": [
            "Rallenta",
            "Data leakage: il test \"vede\" statistiche del training e viceversa",
            "Non si può in Python",
            "Nessun motivo"
          ],
          "answer": 1,
          "explain": "La media contiene informazioni del test set."
        },
        {
          "q": "La stratified K-fold serve per:",
          "options": [
            "Rendere il CV più veloce",
            "Mantenere la proporzione di classi in ogni fold",
            "Ridurre il bias",
            "Nessuna delle precedenti"
          ],
          "answer": 1,
          "explain": "Cruciale con classi sbilanciate."
        },
        {
          "q": "Con learning curve, se train e val hanno errore alto e vicini:",
          "options": [
            "Overfitting: modello troppo complesso",
            "Underfitting: modello troppo semplice",
            "Serve più dati",
            "Nulla di anomalo"
          ],
          "answer": 1,
          "explain": "Entrambi alti = alto bias = underfit."
        },
        {
          "q": "Perché usare Pipeline invece di applicare scaling manualmente?",
          "options": [
            "Estetica",
            "Garantisce che tutti i preprocessing siano ripetuti dentro ogni fold, evitando leakage",
            "Solo per velocità",
            "Solo per Grid Search"
          ],
          "answer": 1,
          "explain": "Il refit corretto dentro CV."
        }
      ]
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Test pratico — valutazione seria",
      "prompt": "<p>Sul dataset breast_cancer, costruisci una pipeline con scaling + Logistic + grid search su C. Fai stratified 10-fold CV. Salva la media dell'accuracy in <code class=\"inline\">cv_acc</code> e la deviazione standard in <code class=\"inline\">cv_std</code>.</p>",
      "starter": "from sklearn.datasets import load_breast_cancer\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV\n\nX, y = load_breast_cancer(return_X_y=True)\n\ncv_acc = None\ncv_std = None\nprint(cv_acc, cv_std)",
      "check": "if cv_acc is not None and cv_acc > 0.97:\n    print(f\"OK — {cv_acc:.4f} +/- {cv_std:.4f}\")\nelse:\n    print(f\"FAIL — {cv_acc}\")",
      "solution": "pipe = Pipeline([('s', StandardScaler()), ('lr', LogisticRegression(max_iter=5000))])\ninner = StratifiedKFold(3, shuffle=True, random_state=0)\nouter = StratifiedKFold(10, shuffle=True, random_state=0)\ngs = GridSearchCV(pipe, {'lr__C':[0.01,0.1,1,10]}, cv=inner)\nscores = cross_val_score(gs, X, y, cv=outer)\ncv_acc = scores.mean(); cv_std = scores.std()"
    }
  ]
});
