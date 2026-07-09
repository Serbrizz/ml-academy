/* Lezione: Regressione lineare */
window.LESSONS.push({
  "id": "l05-linear-regression",
  "title": "Regressione lineare",
  "tag": "Supervisionato",
  "summary": "Il modello più semplice, il più importante da capire davvero: fondamento di tutto il ML.",
  "teoria": [
    {
      "type": "md",
      "content": "\n<h3>5.1 Il modello</h3>\n<p>Data $X \\in \\mathbb{R}^{n\\times d}$ e $y \\in \\mathbb{R}^n$, cerchiamo un vettore di pesi $w \\in \\mathbb{R}^d$ e un bias $b \\in \\mathbb{R}$ tali che:</p>\n<p>$$\\hat y_i = w^T x_i + b$$</p>\n<p>La loss classica è l'errore quadratico:</p>\n<p>$$\\mathcal{L}(w, b) = \\frac{1}{n}\\sum_{i=1}^n (\\hat y_i - y_i)^2 = \\frac{1}{n}\\|Xw + b\\mathbf{1} - y\\|^2$$</p>\n\n<h3>5.2 Soluzione in forma chiusa</h3>\n<p>Aggiungendo una colonna di 1 a $X$ (per assorbire il bias in $w$), abbiamo $\\hat y = Xw$. Il gradiente della loss è $\\nabla_w \\mathcal{L} = \\frac{2}{n} X^T(Xw - y)$. Ponendolo uguale a zero:</p>\n<p>$$X^T X w = X^T y \\quad\\Rightarrow\\quad \\hat w = (X^T X)^{-1} X^T y$$</p>\n<p>Queste sono le <em>normal equations</em>. Se $X^T X$ è singolare (feature collineari) usa la pseudoinversa $\\hat w = X^+ y$, calcolata via SVD.</p>\n\n<h3>5.3 Interpretazione probabilistica</h3>\n<p>Assumiamo $y_i = w^T x_i + \\varepsilon_i$ con $\\varepsilon_i \\sim \\mathcal{N}(0, \\sigma^2)$ i.i.d. La log-likelihood è:</p>\n<p>$$\\log p(y|X, w) = -\\frac{1}{2\\sigma^2}\\sum_i (y_i - w^T x_i)^2 + \\text{const}$$</p>\n<p>Massimizzarla equivale a minimizzare l'MSE. <strong>Least squares = MLE con rumore gaussiano.</strong></p>\n\n<h3>5.4 Ipotesi classiche (Gauss-Markov)</h3>\n<ul>\n<li>Linearità della relazione</li>\n<li>Errori a media zero, varianza costante (omoschedasticità)</li>\n<li>Errori incorrelati</li>\n<li>Features non collineari</li>\n</ul>\n<p>Sotto queste ipotesi, l'OLS (Ordinary Least Squares) è il miglior stimatore lineare non distorto (BLUE).</p>\n\n<h3>5.5 Metriche</h3>\n<p><strong>MSE</strong>: $\\frac{1}{n}\\sum (y_i - \\hat y_i)^2$. Sensibile agli outlier per il quadrato.</p>\n<p><strong>RMSE</strong>: $\\sqrt{\\text{MSE}}$. Stessa unità di $y$, più interpretabile.</p>\n<p><strong>MAE</strong>: $\\frac{1}{n}\\sum |y_i - \\hat y_i|$. Robusta agli outlier.</p>\n<p><strong>$R^2$</strong>: $1 - \\frac{\\sum (y_i-\\hat y_i)^2}{\\sum (y_i-\\bar y)^2}$. Frazione di varianza spiegata. Non è correlazione al quadrato in generale, e può essere negativo se il modello è peggiore della media.</p>\n\n<h3>5.6 Regolarizzazione: Ridge e Lasso</h3>\n<p>Se il numero di feature è grande o le feature sono correlate, la soluzione OLS ha varianza elevata (piccole variazioni di dati &rarr; grandi variazioni di $w$). Aggiungiamo un termine di penalità:</p>\n<p><strong>Ridge (L2)</strong>: $\\mathcal{L}_{\\text{ridge}} = \\|Xw - y\\|^2 + \\lambda \\|w\\|_2^2$. Soluzione: $\\hat w = (X^T X + \\lambda I)^{-1} X^T y$. Riduce tutti i coefficienti verso zero senza mai azzerarli.</p>\n<p><strong>Lasso (L1)</strong>: $\\mathcal{L}_{\\text{lasso}} = \\|Xw - y\\|^2 + \\lambda \\|w\\|_1$. Nessuna forma chiusa. Produce soluzioni sparse: alcuni coefficienti diventano esattamente zero &rarr; <em>selezione delle feature</em>.</p>\n<p><strong>Elastic Net</strong>: combina L1 e L2.</p>\n\n<h3>5.7 Feature engineering per non-linearità</h3>\n<p>La linearità è nei parametri, non nelle feature. Puoi trasformare le feature per catturare non-linearità:</p>\n<ul>\n<li><strong>Polinomiali</strong>: $x, x^2, x^3, x_1 x_2$</li>\n<li><strong>Log</strong>: $\\log x$ per variabili con coda pesante</li>\n<li><strong>Categoriche</strong>: one-hot encoding</li>\n</ul>\n<p>Il modello resta \"lineare\" in senso ottimizzativo, ma può fittare curve.</p>\n"
    },
    {
      "type": "callout",
      "variant": "note",
      "title": "Perché matter ancora",
      "content": "La regressione lineare è veloce, interpretabile, ha intervalli di confidenza analitici, è un ottimo baseline. Il 30% delle applicazioni ML in produzione sono di fatto regressioni lineari o logistiche ben tarate."
    }
  ],
  "esempi": [
    {
      "type": "md",
      "content": "<h3>Esempio 1: OLS con scikit-learn</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import mean_squared_error, r2_score\n\nnp.random.seed(0)\nn = 200\nX = np.random.randn(n, 3)\nw_true, b_true = np.array([2.0, -1.5, 0.5]), 3.0\ny = X @ w_true + b_true + 0.3*np.random.randn(n)\n\nX_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)\nmodel = LinearRegression().fit(X_tr, y_tr)\n\nprint(f\"pesi stimati: {model.coef_.round(3)} (veri {w_true})\")\nprint(f\"bias stimato: {model.intercept_:.3f} (vero {b_true})\")\n\ny_pred = model.predict(X_te)\nprint(f\"\\nMSE test: {mean_squared_error(y_te, y_pred):.4f}\")\nprint(f\"R^2 test: {r2_score(y_te, y_pred):.4f}\")"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 2: gradient descent from scratch</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nnp.random.seed(0)\nX = np.random.randn(100, 2)\ny = 3*X[:,0] - 2*X[:,1] + 1 + 0.1*np.random.randn(100)\n\n# aggiungi bias\nX_aug = np.hstack([X, np.ones((100,1))])\nw = np.zeros(3)\neta, epochs = 0.1, 200\n\nfor ep in range(epochs):\n    y_hat = X_aug @ w\n    grad = 2/len(X_aug) * X_aug.T @ (y_hat - y)\n    w = w - eta * grad\n    if ep % 40 == 0:\n        loss = np.mean((y_hat - y)**2)\n        print(f\"epoch {ep:3d}: loss={loss:.5f}, w={w.round(3)}\")\n\nprint(f\"\\nFinale: {w.round(3)} — atteso [3, -2, 1]\")"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 3: Ridge vs Lasso su feature correlate</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nfrom sklearn.linear_model import LinearRegression, Ridge, Lasso\n\nnp.random.seed(0)\nn = 100\nx1 = np.random.randn(n)\nx2 = x1 + 0.01*np.random.randn(n)   # quasi identiche a x1\nx3 = np.random.randn(n)\nX = np.column_stack([x1, x2, x3])\ny = 2*x1 + x3 + 0.5*np.random.randn(n)  # x2 inutile\n\nfor M, name in [(LinearRegression(), 'OLS'), (Ridge(alpha=1.0), 'Ridge'), (Lasso(alpha=0.1), 'Lasso')]:\n    M.fit(X, y)\n    print(f\"{name:6s}: {M.coef_.round(3)}\")\nprint(\"\\nOLS: coefficienti instabili per collinearità\")\nprint(\"Ridge: li riduce ma li tiene tutti\")\nprint(\"Lasso: azzera x2, seleziona feature\")"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 4: regressione polinomiale</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.preprocessing import PolynomialFeatures\nfrom sklearn.pipeline import make_pipeline\n\nnp.random.seed(0)\nx = np.linspace(-3, 3, 40).reshape(-1,1)\ny = np.sin(x).ravel() + 0.1*np.random.randn(40)\n\nfig, ax = plt.subplots(figsize=(8,4))\nax.scatter(x, y, s=20, alpha=0.6, label='dati')\nxs = np.linspace(-3, 3, 200).reshape(-1,1)\nfor deg in [1, 3, 7, 15]:\n    m = make_pipeline(PolynomialFeatures(deg), LinearRegression()).fit(x, y)\n    ax.plot(xs, m.predict(xs), label=f'grado {deg}')\nax.legend(); ax.set_title('Regressione polinomiale'); ax.grid(alpha=0.3)\nplt.show()"
    }
  ],
  "esercizi": [
    {
      "type": "exercise",
      "difficulty": "easy",
      "title": "Es. 5.1 — Fit e metriche",
      "prompt": "<p>Fitta una regressione lineare sui dati generati, poi calcola RMSE e $R^2$ sul test. Salvali in <code class=\"inline\">rmse</code> e <code class=\"inline\">r2</code>.</p>",
      "starter": "import numpy as np\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import mean_squared_error, r2_score\n\nnp.random.seed(1)\nX = np.random.randn(200, 4)\ny = X @ np.array([1.5, -1, 0.5, 2]) + 1 + 0.5*np.random.randn(200)\nX_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)\n\n# TODO\nrmse = None\nr2 = None\nprint(f\"RMSE: {rmse}, R2: {r2}\")",
      "check": "import numpy as np\nm = LinearRegression().fit(X_tr, y_tr)\nyp = m.predict(X_te)\n_rmse = np.sqrt(np.mean((yp - y_te)**2))\n_r2 = r2_score(y_te, yp)\nif rmse is not None and abs(rmse - _rmse) < 1e-4 and abs(r2 - _r2) < 1e-4:\n    print(f\"OK — RMSE={rmse:.4f}, R2={r2:.4f}\")\nelse:\n    print(f\"FAIL — atteso RMSE={_rmse:.4f}, R2={_r2:.4f}\")",
      "solution": "m = LinearRegression().fit(X_tr, y_tr)\nyp = m.predict(X_te)\nrmse = np.sqrt(np.mean((yp - y_te)**2))\nr2 = r2_score(y_te, yp)"
    },
    {
      "type": "exercise",
      "difficulty": "medium",
      "title": "Es. 5.2 — Ridge da zero",
      "prompt": "<p>Implementa la soluzione ridge in forma chiusa $\\hat w = (X^T X + \\lambda I)^{-1} X^T y$ (senza bias, per semplicità). Salva in <code class=\"inline\">w_ridge</code>. Confronta con <code class=\"inline\">sklearn.linear_model.Ridge</code>.</p>",
      "starter": "import numpy as np\nfrom sklearn.linear_model import Ridge\n\nnp.random.seed(0)\nX = np.random.randn(100, 5)\ny = X @ np.array([1,2,3,4,5]) + np.random.randn(100)\nlam = 2.0\n\n# TODO\nw_ridge = None\nprint(\"mio  :\", w_ridge)\nprint(\"sklearn:\", Ridge(alpha=lam, fit_intercept=False).fit(X, y).coef_)",
      "check": "import numpy as np\nsk = Ridge(alpha=lam, fit_intercept=False).fit(X, y).coef_\nif w_ridge is not None and np.allclose(w_ridge, sk, atol=1e-6):\n    print(\"OK — identico a sklearn\")\nelse:\n    print(\"FAIL\")",
      "solution": "w_ridge = np.linalg.solve(X.T @ X + lam*np.eye(X.shape[1]), X.T @ y)"
    },
    {
      "type": "exercise",
      "difficulty": "medium",
      "title": "Es. 5.3 — Overfit polinomiale",
      "prompt": "<p>Sui punti <code class=\"inline\">x, y</code>, trova il grado polinomiale $d \\in \\{1,2,...,10\\}$ con miglior RMSE sul test. Salva in <code class=\"inline\">best_deg</code>.</p>",
      "starter": "import numpy as np\nfrom sklearn.preprocessing import PolynomialFeatures\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.model_selection import train_test_split\n\nnp.random.seed(0)\nx = np.linspace(-2, 2, 50).reshape(-1,1)\ny = np.sin(x).ravel() + 0.1*np.random.randn(50)\nxt, xe, yt, ye = train_test_split(x, y, test_size=0.3, random_state=42)\n\n# TODO\nbest_deg = None\nprint(\"miglior grado:\", best_deg)",
      "check": "import numpy as np\nresults = {}\nfor d in range(1, 11):\n    m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(xt, yt)\n    results[d] = np.sqrt(np.mean((m.predict(xe) - ye)**2))\n_atteso = min(results, key=results.get)\nif best_deg == _atteso:\n    print(f\"OK — grado ottimo {best_deg}\")\nelse:\n    print(f\"FAIL — atteso {_atteso}, tuo {best_deg}\")",
      "solution": "import numpy as np\nr = {}\nfor d in range(1,11):\n    m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(xt, yt)\n    r[d] = np.sqrt(np.mean((m.predict(xe) - ye)**2))\nbest_deg = min(r, key=r.get)"
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Es. 5.4 — Diabetes dataset",
      "prompt": "<p>Sul dataset diabetes di scikit-learn, addestra <code class=\"inline\">LinearRegression</code>, <code class=\"inline\">Ridge(alpha=1)</code> e <code class=\"inline\">Lasso(alpha=0.1)</code>. Stampa i tre $R^2$ sul test e conta i coefficienti non-nulli del Lasso. Salva quest'ultimo in <code class=\"inline\">n_nonzero</code>.</p>",
      "starter": "from sklearn.datasets import load_diabetes\nfrom sklearn.linear_model import LinearRegression, Ridge, Lasso\nfrom sklearn.model_selection import train_test_split\nimport numpy as np\n\nX, y = load_diabetes(return_X_y=True)\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42)\n\n# TODO\nn_nonzero = None",
      "check": "import numpy as np\nlin_r2 = LinearRegression().fit(Xt, yt).score(Xe, ye)\nrid_r2 = Ridge(alpha=1).fit(Xt, yt).score(Xe, ye)\nlas = Lasso(alpha=0.1, max_iter=5000).fit(Xt, yt)\nlas_r2 = las.score(Xe, ye)\n_nz = int(np.sum(las.coef_ != 0))\nprint(f\"OLS R2: {lin_r2:.4f} | Ridge R2: {rid_r2:.4f} | Lasso R2: {las_r2:.4f}\")\nprint(f\"Lasso non-zero: {_nz} feature su {len(las.coef_)}\")\nif n_nonzero == _nz:\n    print(\"OK\")\nelse:\n    print(f\"FAIL — atteso n_nonzero={_nz}\")",
      "solution": "las = Lasso(alpha=0.1, max_iter=5000).fit(Xt, yt)\nn_nonzero = int((las.coef_ != 0).sum())"
    }
  ],
  "test": [
    {
      "type": "quiz",
      "questions": [
        {
          "q": "Least squares equivale a MLE sotto quale ipotesi sul rumore?",
          "options": [
            "Uniforme",
            "Gaussiano con varianza costante",
            "Laplace",
            "Poisson"
          ],
          "answer": 1,
          "explain": "Rumore normale &rarr; log-likelihood = -MSE + costante."
        },
        {
          "q": "La differenza principale tra Ridge e Lasso è:",
          "options": [
            "Ridge è più veloce",
            "Lasso produce soluzioni sparse (coefficienti esattamente zero)",
            "Ridge non ha regolarizzazione",
            "Lasso non converge"
          ],
          "answer": 1,
          "explain": "La penalità L1 induce sparsità geometricamente (spigoli del rombo)."
        },
        {
          "q": "$R^2 = 0.85$ significa che il modello:",
          "options": [
            "Ha 85% di accuracy",
            "Spiega l'85% della varianza di $y$",
            "Ha errore medio 0.85",
            "È perfetto"
          ],
          "answer": 1,
          "explain": "$R^2 = 1 - \\text{SSR}/\\text{SST}$: frazione di varianza spiegata."
        },
        {
          "q": "Se le feature sono altamente collineari, OLS soffre di:",
          "options": [
            "Bias elevato",
            "Varianza elevata / coefficienti instabili",
            "Non converge",
            "Sparsità"
          ],
          "answer": 1,
          "explain": "$X^T X$ è mal condizionata; piccoli cambi di dati muovono molto $\\hat w$."
        }
      ]
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Test pratico — pipeline completa",
      "prompt": "<p>Sul dataset California housing (versione semplificata simulata), costruisci un modello che minimizza RMSE. Puoi usare: standardizzazione, feature polinomiali, ridge, lasso. Target: <strong>RMSE test &lt; 0.6</strong>.</p>",
      "starter": "import numpy as np\nfrom sklearn.linear_model import Ridge\nfrom sklearn.preprocessing import StandardScaler, PolynomialFeatures\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.model_selection import train_test_split\n\nnp.random.seed(0)\nn = 500\nX = np.random.randn(n, 5)\ny = (2*X[:,0] - 1.5*X[:,1] + 0.5*X[:,0]*X[:,2] + X[:,3]**2 + 0.3*np.random.randn(n))\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42)\n\n# TODO: costruisci una pipeline, fit, calcola RMSE su test\nrmse_test = None\nprint(\"RMSE test:\", rmse_test)",
      "check": "if rmse_test is None:\n    print(\"FAIL: None\")\nelif rmse_test < 0.6:\n    print(f\"OK — RMSE {rmse_test:.4f}\")\nelse:\n    print(f\"FAIL — RMSE {rmse_test:.4f}, target < 0.6\")",
      "solution": "pipe = Pipeline([('scaler', StandardScaler()),\n                 ('poly', PolynomialFeatures(2)),\n                 ('reg', Ridge(alpha=1.0))])\npipe.fit(Xt, yt)\nrmse_test = np.sqrt(np.mean((pipe.predict(Xe) - ye)**2))"
    }
  ]
});
