/* Lezione: PCA e riduzione dimensionale */
window.LESSONS.push({
  "id": "l12-pca",
  "title": "PCA e riduzione dimensionale",
  "tag": "Non supervisionato",
  "summary": "Trovare le direzioni di massima varianza. Applicazioni: visualizzazione, compressione, denoising.",
  "teoria": [
    {
      "type": "md",
      "content": "\n<h3>12.1 Il problema</h3>\n<p>Dati $X \\in \\mathbb{R}^{n\\times d}$ ad alta dimensione, vogliamo proiettarli in $\\mathbb{R}^k$ con $k \\ll d$ preservando quanta più informazione possibile. \"Informazione\" qui significa <em>varianza</em>: le direzioni lungo cui i dati variano di più.</p>\n\n<h3>12.2 Formulazione</h3>\n<p>Standardizziamo (media zero, opzionalmente varianza unitaria). Cerchiamo un vettore unitario $v$ tale che la varianza dei punti proiettati $v^T x_i$ sia massima:</p>\n<p>$$\\max_{\\|v\\|=1} v^T \\Sigma v \\quad\\text{dove}\\quad \\Sigma = \\tfrac{1}{n}X^T X$$</p>\n<p>La soluzione è l'autovettore di $\\Sigma$ associato al più grande autovalore. Il valore del massimo è l'autovalore stesso: quello è la varianza spiegata dalla prima componente principale.</p>\n<p>Ripeti nel sottospazio ortogonale &rarr; le prime $k$ componenti sono i primi $k$ autovettori di $\\Sigma$.</p>\n\n<h3>12.3 Via SVD</h3>\n<p>Con $X$ centrata, la SVD $X = U\\Sigma V^T$ dà:</p>\n<ul>\n<li>Le colonne di $V$ sono le direzioni principali (loadings)</li>\n<li>$X V$ sono i punti proiettati (scores)</li>\n<li>$\\sigma_i^2 / n$ è la varianza spiegata dalla $i$-esima componente</li>\n</ul>\n<p>SVD è numericamente più stabile del calcolo esplicito di $X^T X$.</p>\n\n<h3>12.4 Varianza spiegata</h3>\n<p>Fraction of variance explained by first $k$ components:</p>\n<p>$$\\text{VE}(k) = \\frac{\\sum_{i=1}^k \\sigma_i^2}{\\sum_{i=1}^d \\sigma_i^2}$$</p>\n<p>Si sceglie $k$ per catturare, per esempio, il 95% della varianza.</p>\n\n<h3>12.5 Interpretazione</h3>\n<ul>\n<li>PCA è <strong>lineare</strong>: le componenti sono combinazioni lineari delle feature originali</li>\n<li>Le componenti sono <strong>ortogonali</strong></li>\n<li>La <strong>ricostruzione</strong> approssimata è $\\hat X = XV_k V_k^T$: proiezione poi anti-proiezione, l'errore quadratico medio è minimo tra tutte le proiezioni di rango $k$</li>\n</ul>\n\n<h3>12.6 Cosa fa PCA <strong>bene</strong> e cosa NO</h3>\n<p>Bene: visualizzazione 2D/3D, denoising, decorrelazione delle feature, preprocessing.</p>\n<p>Non bene: se le direzioni di massima varianza <em>non</em> coincidono con quelle utili per la classificazione. PCA non guarda $y$. Per un'alternativa supervisionata: Linear Discriminant Analysis (LDA).</p>\n\n<h3>12.7 Standardizzazione: obbligatoria</h3>\n<p>Se le feature hanno scale diverse, quelle con varianza grande dominano PCA anche se non sono le più informative. <strong>Sempre</strong> standardizza prima.</p>\n\n<h3>12.8 Alternative non lineari</h3>\n<ul>\n<li><strong>t-SNE</strong>: preserva similarità locali. Ottimo per visualizzazione, non per feature engineering.</li>\n<li><strong>UMAP</strong>: come t-SNE ma più veloce e preserva anche struttura globale.</li>\n<li><strong>Autoencoder</strong>: rete neurale che comprime e decomprime. Non lineare, flessibile.</li>\n</ul>\n"
    },
    {
      "type": "callout",
      "variant": "note",
      "title": "Legame con la fisica",
      "content": "La matrice di covarianza è simmetrica e semidefinita positiva: diagonalizzabile con base ortonormale, autovalori non negativi. Esattamente come il tensore d'inerzia in meccanica classica. Le componenti principali sono gli \"assi principali di inerzia\" della nuvola di punti."
    }
  ],
  "esempi": [
    {
      "type": "md",
      "content": "<h3>Esempio 1: PCA su Iris (visualizzazione 2D)</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.datasets import load_iris\nfrom sklearn.decomposition import PCA\nfrom sklearn.preprocessing import StandardScaler\n\niris = load_iris()\nX = StandardScaler().fit_transform(iris.data)\n\npca = PCA(n_components=2).fit(X)\nZ = pca.transform(X)\nprint(f\"Varianza spiegata: {pca.explained_variance_ratio_}\")\nprint(f\"Totale: {pca.explained_variance_ratio_.sum():.3f}\")\n\nfig, ax = plt.subplots(figsize=(7,5))\nfor i, name in enumerate(iris.target_names):\n    ax.scatter(Z[iris.target==i, 0], Z[iris.target==i, 1], label=name, s=40, edgecolor='k', alpha=0.7)\nax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(); ax.set_title('Iris in 2D via PCA')\nplt.show()"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 2: quante componenti tenere?</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nfrom sklearn.datasets import load_digits\nfrom sklearn.decomposition import PCA\nfrom sklearn.preprocessing import StandardScaler\n\nX = load_digits().data  # 64 feature (immagini 8x8)\nX = StandardScaler().fit_transform(X)\n\npca = PCA().fit(X)\ncum = np.cumsum(pca.explained_variance_ratio_)\nfor thresh in [0.80, 0.90, 0.95, 0.99]:\n    k = np.searchsorted(cum, thresh) + 1\n    print(f\"Per {thresh*100:.0f}% varianza servono {k} componenti (su 64)\")"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 3: PCA from scratch via SVD</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nfrom sklearn.datasets import load_iris\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\n\nX = StandardScaler().fit_transform(load_iris().data)\nn = len(X)\n\nU, S, Vt = np.linalg.svd(X, full_matrices=False)\nprint(f\"Valori singolari: {S.round(3)}\")\nprint(f\"Varianza spiegata: {(S**2 / (n-1) / (S**2/(n-1)).sum()).round(4)}\")\n\n# proiezione sulle prime 2 componenti\nZ_mio = X @ Vt[:2].T\nZ_sk = PCA(n_components=2).fit_transform(X)\n# le colonne coincidono a meno di segno\nprint(f\"\\nMax diff (in valore assoluto): {min(np.abs(np.abs(Z_mio) - np.abs(Z_sk)).max(), 1):.2e}\")"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 4: PCA come denoiser</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.datasets import load_digits\nfrom sklearn.decomposition import PCA\n\ndigits = load_digits()\nX = digits.data\nnp.random.seed(0)\nX_noisy = X + 4 * np.random.randn(*X.shape)\n\npca = PCA(n_components=25).fit(X)\nX_denoised = pca.inverse_transform(pca.transform(X_noisy))\n\nfig, axes = plt.subplots(3, 6, figsize=(10, 5))\nfor i in range(6):\n    axes[0,i].imshow(X[i].reshape(8,8), cmap='gray_r'); axes[0,i].axis('off')\n    axes[1,i].imshow(X_noisy[i].reshape(8,8), cmap='gray_r'); axes[1,i].axis('off')\n    axes[2,i].imshow(X_denoised[i].reshape(8,8), cmap='gray_r'); axes[2,i].axis('off')\nfor r, t in enumerate(['originale', 'rumoroso', 'denoised (PCA 25 comp)']):\n    axes[r,0].set_title(t, loc='left', fontsize=9)\nplt.tight_layout(); plt.show()"
    }
  ],
  "esercizi": [
    {
      "type": "exercise",
      "difficulty": "easy",
      "title": "Es. 12.1 — Varianza spiegata",
      "prompt": "<p>Su breast_cancer standardizzato, calcola quante componenti servono per catturare almeno il 90% della varianza. Salva in <code class=\"inline\">k</code>.</p>",
      "starter": "import numpy as np\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.decomposition import PCA\nfrom sklearn.preprocessing import StandardScaler\n\nX = StandardScaler().fit_transform(load_breast_cancer().data)\n\n# TODO\nk = None\nprint(\"k:\", k)",
      "check": "import numpy as np\npca = PCA().fit(X)\ncum = np.cumsum(pca.explained_variance_ratio_)\n_atteso = int(np.searchsorted(cum, 0.90) + 1)\nif k == _atteso:\n    print(f\"OK — {k} componenti per 90%\")\nelse:\n    print(f\"FAIL — atteso {_atteso}\")",
      "solution": "pca = PCA().fit(X)\ncum = np.cumsum(pca.explained_variance_ratio_)\nk = int(np.searchsorted(cum, 0.90) + 1)"
    },
    {
      "type": "exercise",
      "difficulty": "medium",
      "title": "Es. 12.2 — PCA manuale",
      "prompt": "<p>Implementa PCA a mano: centra i dati, calcola $\\Sigma = X^T X / (n-1)$, trova autovettori, proietta sui primi 2. Salva la matrice risultato in <code class=\"inline\">Z</code>.</p>",
      "starter": "import numpy as np\nnp.random.seed(0)\nX = np.random.randn(100, 5) @ np.random.randn(5, 5)\n\n# TODO: Z shape (100, 2)\nZ = None\nprint(Z.shape if Z is not None else \"None\")",
      "check": "import numpy as np\nfrom sklearn.decomposition import PCA\nZ_sk = PCA(n_components=2).fit_transform(X - X.mean(0))\n# confronto a meno di segno\nif Z is None: print(\"FAIL: None\")\nelif Z.shape == (100, 2):\n    diff = min(np.abs(np.abs(Z) - np.abs(Z_sk)).max(), 1e10)\n    if diff < 1e-6: print(\"OK\")\n    else: print(f\"FAIL — max diff (abs) {diff}\")\nelse:\n    print(f\"FAIL — shape {Z.shape}\")",
      "solution": "Xc = X - X.mean(0)\nSigma = Xc.T @ Xc / (len(X)-1)\nvals, vecs = np.linalg.eigh(Sigma)\n# eigh ritorna in ordine crescente, prendo gli ultimi 2\nV = vecs[:, [-1, -2]]\nZ = Xc @ V"
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Es. 12.3 — Ricostruzione ed errore",
      "prompt": "<p>Su digits, per $k \\in \\{5, 10, 20, 40\\}$ calcola l'errore di ricostruzione MSE. Salva un dizionario <code class=\"inline\">errs</code> con chiave $k$.</p>",
      "starter": "import numpy as np\nfrom sklearn.datasets import load_digits\nfrom sklearn.decomposition import PCA\n\nX = load_digits().data\n\nerrs = {}\n# TODO\nprint(errs)",
      "check": "import numpy as np\n_atteso = {}\nfor k in [5,10,20,40]:\n    p = PCA(n_components=k).fit(X)\n    Xr = p.inverse_transform(p.transform(X))\n    _atteso[k] = float(((Xr - X)**2).mean())\nok = all(abs(errs.get(k,0) - _atteso[k]) < 1e-6 for k in [5,10,20,40])\nprint(\"OK\" if ok else f\"FAIL: atteso {_atteso}, tuo {errs}\")",
      "solution": "for k in [5, 10, 20, 40]:\n    p = PCA(n_components=k).fit(X)\n    Xr = p.inverse_transform(p.transform(X))\n    errs[k] = float(((Xr - X)**2).mean())"
    }
  ],
  "test": [
    {
      "type": "quiz",
      "questions": [
        {
          "q": "La prima componente principale è la direzione:",
          "options": [
            "Con media massima",
            "Con varianza massima dei dati proiettati",
            "Ortogonale a $y$",
            "Con densità massima"
          ],
          "answer": 1,
          "explain": "Per definizione: massimizza la varianza."
        },
        {
          "q": "PCA è supervisionato?",
          "options": [
            "Sì, usa le label",
            "No, non guarda $y$",
            "Dipende",
            "Solo per classificazione"
          ],
          "answer": 1,
          "explain": "Solo la struttura di $X$."
        },
        {
          "q": "Non standardizzare prima di PCA:",
          "options": [
            "Non ha effetto",
            "Le feature con scala grande dominano le componenti principali",
            "Rende PCA più veloce",
            "Rende PCA più accurato"
          ],
          "answer": 1,
          "explain": "La varianza dipende dall'unità di misura."
        },
        {
          "q": "Se una componente ha varianza spiegata 0.65, significa:",
          "options": [
            "Predice il 65% dei valori",
            "Contiene il 65% della varianza totale dei dati",
            "Ha 65% di accuracy",
            "È il 65-esimo autovettore"
          ],
          "answer": 1,
          "explain": "$\\sigma^2 / \\sum \\sigma^2$."
        }
      ]
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Test pratico — PCA + classificatore",
      "prompt": "<p>Su digits, applica PCA per ridurre a $k$ tale che si catturi il 95% di varianza, poi addestra una LogisticRegression. Salva il numero di componenti in <code class=\"inline\">k95</code> e l'accuracy test in <code class=\"inline\">acc</code> (target > 0.94).</p>",
      "starter": "import numpy as np\nfrom sklearn.datasets import load_digits\nfrom sklearn.decomposition import PCA\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.model_selection import train_test_split\n\nX, y = load_digits(return_X_y=True)\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)\n\nk95 = None\nacc = None",
      "check": "if k95 is None or acc is None: print(\"FAIL: None\")\nelif acc > 0.94: print(f\"OK — {k95} componenti, acc {acc:.4f}\")\nelse: print(f\"FAIL — {k95} comp, acc {acc:.4f}\")",
      "solution": "sc = StandardScaler().fit(Xt)\nXt_s = sc.transform(Xt); Xe_s = sc.transform(Xe)\np = PCA(n_components=0.95).fit(Xt_s)\nk95 = p.n_components_\nlr = LogisticRegression(max_iter=5000).fit(p.transform(Xt_s), yt)\nacc = lr.score(p.transform(Xe_s), ye)"
    }
  ]
});
