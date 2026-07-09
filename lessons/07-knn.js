/* Lezione: K-Nearest Neighbors */
window.LESSONS.push({
  "id": "l07-knn",
  "title": "K-Nearest Neighbors",
  "tag": "Supervisionato",
  "summary": "Il modello più intuitivo del ML: predici come i tuoi vicini. Non-parametrico, potente, con difetti.",
  "teoria": [
    {
      "type": "md",
      "content": "\n<h3>7.1 L'idea</h3>\n<p>Per predire $y$ su un nuovo $x$: trova i $k$ punti del training più vicini a $x$, e restituisci la loro etichetta più comune (classificazione) o la loro media (regressione). Fine. Non c'è \"training\" nel senso tradizionale: il modello <em>è</em> il dataset.</p>\n<p>Formalmente, sia $\\mathcal{N}_k(x)$ l'insieme dei $k$ vicini di $x$ nel training:</p>\n<p>$$\\hat y(x) = \\arg\\max_c \\sum_{i \\in \\mathcal{N}_k(x)} \\mathbb{1}[y_i = c] \\quad\\text{(classificazione)}$$</p>\n\n<h3>7.2 Distanza</h3>\n<p>La nozione di \"vicino\" richiede una metrica. Le più comuni:</p>\n<ul>\n<li><strong>Euclidea</strong>: $\\|x-x'\\|_2 = \\sqrt{\\sum_j (x_j - x'_j)^2}$</li>\n<li><strong>Manhattan</strong>: $\\|x-x'\\|_1 = \\sum_j |x_j - x'_j|$</li>\n<li><strong>Cosine</strong>: $1 - \\frac{x \\cdot x'}{\\|x\\|\\|x'\\|}$ — utile per testi</li>\n<li><strong>Mahalanobis</strong>: tiene conto della covarianza delle feature</li>\n</ul>\n\n<h3>7.3 Scaling è obbligatorio</h3>\n<p>La distanza euclidea è dominata dalle feature con range grande. Se una feature va da 0 a 1000 e un'altra da 0 a 1, la prima decide tutto. <strong>Standardizza sempre</strong> (o usa min-max normalization) prima di KNN.</p>\n\n<h3>7.4 Effetto di $k$</h3>\n<ul>\n<li>$k = 1$: modello ad altissima varianza; segue perfettamente il training, quindi overfitta.</li>\n<li>$k$ grande: modello ad alto bias; media su troppi punti diversi, sottofitta.</li>\n<li>Il valore ottimale si trova con cross-validation.</li>\n</ul>\n<p>Rule of thumb: $k \\approx \\sqrt{n}$, ma la CV vince sempre.</p>\n\n<h3>7.5 KNN pesato</h3>\n<p>Invece di dare peso uguale a tutti i vicini, si pesa inversamente alla distanza:</p>\n<p>$$\\hat y(x) = \\frac{\\sum_{i \\in \\mathcal{N}_k} \\frac{1}{d(x, x_i)} y_i}{\\sum_{i \\in \\mathcal{N}_k} \\frac{1}{d(x, x_i)}}$$</p>\n\n<h3>7.6 La maledizione della dimensionalità</h3>\n<p>In alta dimensione, \"vicino\" perde significato: tutti i punti sono a distanza simile. Precisamente, se hai $n$ punti in $[0,1]^d$ uniformi, la distanza minima e massima tendono a coincidere per $d \\to \\infty$. KNN è quasi inutile in $d > 20$ senza riduzione dimensionale.</p>\n\n<h3>7.7 Efficienza</h3>\n<p>Predizione naïve: $O(nd)$ per punto. Per dataset grandi si usano strutture dati:</p>\n<ul>\n<li><strong>KD-tree</strong>: efficiente per $d$ piccolo (fino a 20)</li>\n<li><strong>Ball tree</strong>: meglio in alta dimensione</li>\n<li><strong>Approssimati</strong>: HNSW, LSH per $n$ o $d$ enormi</li>\n</ul>\n<p>Scikit-learn sceglie automaticamente con <code class=\"inline\">algorithm='auto'</code>.</p>\n"
    },
    {
      "type": "callout",
      "variant": "note",
      "title": "Quando usare KNN",
      "content": "Ottimo baseline. Ottimo per dataset piccoli, poche feature, con confini di decisione irregolari. Da evitare in produzione con dataset grandi (lento a predire) o alta dimensionalità."
    }
  ],
  "esempi": [
    {
      "type": "md",
      "content": "<h3>Esempio 1: KNN su Iris</h3>"
    },
    {
      "type": "run",
      "content": "from sklearn.datasets import load_iris\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.model_selection import train_test_split, cross_val_score\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nimport numpy as np\n\nX, y = load_iris(return_X_y=True)\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)\n\npipe = Pipeline([('s', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5))])\npipe.fit(Xt, yt)\nprint(f\"Accuracy k=5: {pipe.score(Xe, ye):.4f}\")\n\n# CV per trovare k ottimale\nfor k in [1, 3, 5, 7, 15, 25]:\n    p = Pipeline([('s', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=k))])\n    scores = cross_val_score(p, Xt, yt, cv=5)\n    print(f\"k={k:2d}: CV mean {scores.mean():.4f} +/- {scores.std():.4f}\")"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 2: KNN from scratch</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nfrom collections import Counter\n\ndef knn_predict(X_train, y_train, X_new, k=3):\n    preds = []\n    for x in X_new:\n        # distanze da tutti i punti di train\n        d = np.sqrt(((X_train - x)**2).sum(axis=1))\n        # indici dei k più vicini\n        idx = np.argsort(d)[:k]\n        # voto di maggioranza\n        preds.append(Counter(y_train[idx]).most_common(1)[0][0])\n    return np.array(preds)\n\n# test\nfrom sklearn.datasets import load_iris\nfrom sklearn.model_selection import train_test_split\nX, y = load_iris(return_X_y=True)\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)\n\npred = knn_predict(Xt, yt, Xe, k=5)\nprint(f\"Accuracy: {np.mean(pred == ye):.4f}\")"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 3: effetto di $k$ visualizzato</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.datasets import make_moons\nfrom sklearn.neighbors import KNeighborsClassifier\n\nX, y = make_moons(n_samples=200, noise=0.25, random_state=0)\nfig, axes = plt.subplots(1, 3, figsize=(13,4))\nfor ax, k in zip(axes, [1, 15, 50]):\n    m = KNeighborsClassifier(n_neighbors=k).fit(X, y)\n    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.4, X[:,0].max()+0.4, 200),\n                         np.linspace(X[:,1].min()-0.4, X[:,1].max()+0.4, 200))\n    Z = m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)\n    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu_r')\n    ax.scatter(X[:,0], X[:,1], c=y, cmap='RdBu_r', edgecolor='k', s=25)\n    ax.set_title(f'k={k}')\nplt.tight_layout(); plt.show()\nprint(\"k=1: overfit (bordo frastagliato)\")\nprint(\"k=15: bilanciato\")\nprint(\"k=50: underfit (bordo troppo liscio)\")"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 4: importanza dello scaling</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nfrom sklearn.datasets import make_classification\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\n\nnp.random.seed(0)\nX, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, random_state=0)\n# distruggi la scala: una feature x 1000\nX_bad = X.copy(); X_bad[:,1] *= 1000\n\nXt, Xe, yt, ye = train_test_split(X_bad, y, test_size=0.3, random_state=0, stratify=y)\nprint(\"Senza scaling:\", KNeighborsClassifier(5).fit(Xt, yt).score(Xe, ye))\n\nsc = StandardScaler().fit(Xt)\nXt_s, Xe_s = sc.transform(Xt), sc.transform(Xe)\nprint(\"Con scaling: \", KNeighborsClassifier(5).fit(Xt_s, yt).score(Xe_s, ye))"
    }
  ],
  "esercizi": [
    {
      "type": "exercise",
      "difficulty": "easy",
      "title": "Es. 7.1 — KNN a mano per un punto",
      "prompt": "<p>Dato il training e un nuovo punto <code class=\"inline\">q = [1, 1]</code>, trova i 3 vicini più vicini con distanza euclidea. Salva gli <strong>indici</strong> in <code class=\"inline\">indices</code>.</p>",
      "starter": "import numpy as np\nX = np.array([[0,0],[2,2],[3,3],[1,2],[5,5],[0.5,0.5]])\nq = np.array([1, 1])\n\n# TODO\nindices = None\nprint(\"Indici:\", indices)",
      "check": "import numpy as np\nd = np.sqrt(((X - q)**2).sum(1))\n_atteso = np.argsort(d)[:3]\nif indices is not None and set(list(indices)) == set(list(_atteso)):\n    print(f\"OK — vicini {list(indices)}\")\nelse:\n    print(f\"FAIL — atteso {list(_atteso)}\")",
      "solution": "d = np.sqrt(((X - q)**2).sum(1))\nindices = np.argsort(d)[:3]"
    },
    {
      "type": "exercise",
      "difficulty": "medium",
      "title": "Es. 7.2 — KNN regressore",
      "prompt": "<p>Implementa <code class=\"inline\">knn_regress(X, y, x, k)</code> che ritorna la media dei $k$ target dei vicini di <code class=\"inline\">x</code>. Testalo.</p>",
      "starter": "import numpy as np\ndef knn_regress(X, y, x, k):\n    # TODO\n    return None\n\nXtr = np.array([[0],[1],[2],[3],[4],[5]])\nytr = np.array([0., 1., 4., 9., 16., 25.])\nprint(knn_regress(Xtr, ytr, np.array([2.5]), 2))  # atteso (4+9)/2 = 6.5",
      "check": "import numpy as np\nXtr = np.array([[0],[1],[2],[3],[4],[5]])\nytr = np.array([0., 1., 4., 9., 16., 25.])\nr1 = knn_regress(Xtr, ytr, np.array([2.5]), 2)\nr2 = knn_regress(Xtr, ytr, np.array([0.1]), 3)\nif abs(r1 - 6.5) < 1e-6 and abs(r2 - (0+1+4)/3) < 1e-6:\n    print(\"OK\")\nelse:\n    print(f\"FAIL: r1={r1} (atteso 6.5), r2={r2} (atteso {(0+1+4)/3:.4f})\")",
      "solution": "def knn_regress(X, y, x, k):\n    d = np.linalg.norm(X - x, axis=1)\n    idx = np.argsort(d)[:k]\n    return y[idx].mean()"
    },
    {
      "type": "exercise",
      "difficulty": "medium",
      "title": "Es. 7.3 — Trova k ottimale",
      "prompt": "<p>Con cross-validation a 5 fold su Iris, trova il $k \\in \\{1,3,5,\\dots,29\\}$ che massimizza l'accuracy. Salva in <code class=\"inline\">best_k</code>.</p>",
      "starter": "from sklearn.datasets import load_iris\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.model_selection import cross_val_score\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nimport numpy as np\n\nX, y = load_iris(return_X_y=True)\n\n# TODO\nbest_k = None\nprint(\"k ottimo:\", best_k)",
      "check": "from sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.model_selection import cross_val_score\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nscores = {}\nfor k in range(1, 30, 2):\n    p = Pipeline([('s', StandardScaler()), ('k', KNeighborsClassifier(k))])\n    scores[k] = cross_val_score(p, X, y, cv=5).mean()\n_atteso_top = sorted(scores, key=scores.get, reverse=True)[:3]\nif best_k in _atteso_top:\n    print(f\"OK — k={best_k} tra i top-3\")\nelse:\n    print(f\"FAIL — top-3 attesi {_atteso_top}, tuo {best_k}\")",
      "solution": "scores = {}\nfor k in range(1, 30, 2):\n    p = Pipeline([('s', StandardScaler()), ('k', KNeighborsClassifier(k))])\n    scores[k] = cross_val_score(p, X, y, cv=5).mean()\nbest_k = max(scores, key=scores.get)"
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Es. 7.4 — Distanza pesata",
      "prompt": "<p>Implementa <code class=\"inline\">knn_weighted(X_tr, y_tr, X_te, k)</code>: la predizione per ogni test è la media pesata inversamente alla distanza dei $k$ vicini. Attenzione ai zero-division: se un vicino è a distanza 0, usa quello direttamente.</p>",
      "starter": "import numpy as np\ndef knn_weighted(X_tr, y_tr, X_te, k=3):\n    # TODO\n    return None\n\n# test\nX_tr = np.array([[0],[1],[2],[3]], dtype=float)\ny_tr = np.array([10., 20., 30., 40.])\nX_te = np.array([[0.5],[2.5]])\nprint(knn_weighted(X_tr, y_tr, X_te, 2))\n# per 0.5: vicini 0 e 1, distanze 0.5 e 0.5 -> media (10+20)/2 = 15\n# per 2.5: vicini 2 e 3, distanze 0.5 e 0.5 -> media (30+40)/2 = 35",
      "check": "import numpy as np\nres = knn_weighted(X_tr, y_tr, X_te, 2)\nif res is not None and np.allclose(res, [15., 35.]):\n    print(\"OK\")\nelse:\n    print(f\"FAIL — atteso [15, 35], tuo {res}\")",
      "solution": "def knn_weighted(X_tr, y_tr, X_te, k=3):\n    out = []\n    for x in X_te:\n        d = np.linalg.norm(X_tr - x, axis=1)\n        idx = np.argsort(d)[:k]\n        d_k = d[idx]\n        y_k = y_tr[idx]\n        if (d_k == 0).any():\n            out.append(y_k[d_k == 0].mean())\n        else:\n            w = 1 / d_k\n            out.append((w * y_k).sum() / w.sum())\n    return np.array(out)"
    }
  ],
  "test": [
    {
      "type": "quiz",
      "questions": [
        {
          "q": "Con $k=1$, l'accuracy sul training è:",
          "options": [
            "0%",
            "Sempre 100%",
            "Circa 50%",
            "Dipende dai dati"
          ],
          "answer": 1,
          "explain": "Ogni punto è il vicino più vicino di se stesso — memorizzazione perfetta."
        },
        {
          "q": "La maledizione della dimensionalità colpisce KNN perché:",
          "options": [
            "La distanza euclidea è lenta",
            "Le distanze si concentrano — vicini/lontani indistinguibili",
            "La memoria non basta",
            "Il gradiente svanisce"
          ],
          "answer": 1,
          "explain": "In alta dimensione tutto è quasi equidistante."
        },
        {
          "q": "KNN è definito \"lazy learning\" perché:",
          "options": [
            "Non fa nulla",
            "Non ha una fase di training vera e propria",
            "È lento a implementarsi",
            "Non usa gradienti"
          ],
          "answer": 1,
          "explain": "Il calcolo è tutto rimandato al momento della predizione."
        },
        {
          "q": "La distanza cosine è più adatta di quella euclidea quando:",
          "options": [
            "I dati sono numeri interi",
            "Interessa la direzione più che la magnitudine (es. testi TF-IDF)",
            "Ci sono pochi dati",
            "Ci sono outlier"
          ],
          "answer": 1,
          "explain": "Cosine ignora la norma dei vettori."
        }
      ]
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Test pratico — pipeline KNN",
      "prompt": "<p>Sul dataset digits (cifre 8x8), costruisci un classificatore KNN con scaling e trova la miglior combinazione di $k \\in \\{3,5,7,9\\}$ e weights $\\in \\{$\"uniform\",\"distance\"$\\}$ via GridSearchCV. Salva la migliore accuracy test in <code class=\"inline\">acc_test</code> (target &gt; 0.98).</p>",
      "starter": "from sklearn.datasets import load_digits\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.model_selection import GridSearchCV, train_test_split\n\nX, y = load_digits(return_X_y=True)\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)\n\n# TODO\nacc_test = None\nprint(\"Accuracy test:\", acc_test)",
      "check": "if acc_test is None:\n    print(\"FAIL: None\")\nelif acc_test > 0.98:\n    print(f\"OK — {acc_test:.4f}\")\nelse:\n    print(f\"FAIL — {acc_test:.4f}, target > 0.98\")",
      "solution": "pipe = Pipeline([('s', StandardScaler()), ('k', KNeighborsClassifier())])\ngrid = {'k__n_neighbors':[3,5,7,9], 'k__weights':['uniform','distance']}\ngs = GridSearchCV(pipe, grid, cv=5).fit(Xt, yt)\nacc_test = gs.score(Xe, ye)"
    }
  ]
});
