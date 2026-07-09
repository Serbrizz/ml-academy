/* Lezione: K-Means e clustering */
window.LESSONS.push({
  "id": "l11-kmeans",
  "title": "K-Means e clustering",
  "tag": "Non supervisionato",
  "summary": "Il primo modello non supervisionato: raggruppare senza etichette.",
  "teoria": [
    {
      "type": "md",
      "content": "\n<h3>11.1 Il problema del clustering</h3>\n<p>Dato $\\{x_i\\}_{i=1}^n$ senza etichette, vogliamo raggrupparli in $K$ cluster tali che punti nello stesso cluster siano \"simili\" e punti in cluster diversi siano \"diversi\". La definizione di simile dipende dal problema: è una scelta di modello, non un dato.</p>\n\n<h3>11.2 K-Means: formulazione</h3>\n<p>Cerchiamo $K$ centroidi $\\mu_1, \\dots, \\mu_K \\in \\mathbb{R}^d$ e un assegnamento $c : \\{1,\\dots,n\\} \\to \\{1,\\dots,K\\}$ che minimizzano:</p>\n<p>$$J(\\mu, c) = \\sum_{i=1}^n \\|x_i - \\mu_{c(i)}\\|^2$$</p>\n<p>Cioè la somma delle distanze al quadrato di ogni punto dal proprio centroide (inertia).</p>\n\n<h3>11.3 L'algoritmo (Lloyd)</h3>\n<ol>\n<li>Inizializza $\\mu_1, \\dots, \\mu_K$ (random o k-means++)</li>\n<li><strong>Assign step</strong>: assegna ogni punto al centroide più vicino: $c(i) = \\arg\\min_k \\|x_i - \\mu_k\\|^2$</li>\n<li><strong>Update step</strong>: aggiorna ogni centroide come media dei punti assegnati: $\\mu_k = \\frac{1}{|C_k|}\\sum_{i \\in C_k} x_i$</li>\n<li>Ripeti finché nessun punto cambia cluster</li>\n</ol>\n<p>È un algoritmo EM: ogni step riduce $J$, quindi converge in tempo finito (a un minimo locale, non necessariamente globale).</p>\n\n<h3>11.4 Scelta di $K$: metodo del gomito</h3>\n<p>Grafica $J$ in funzione di $K$. Aumentando $K$, $J$ decresce sempre (con $K=n$ è zero). Cerca il \"gomito\": il punto dopo il quale la riduzione rallenta drasticamente.</p>\n\n<h3>11.5 Silhouette score</h3>\n<p>Metrica alternativa. Per ogni punto $i$:</p>\n<ul>\n<li>$a_i$ = distanza media dai punti dello stesso cluster</li>\n<li>$b_i$ = distanza media dai punti del cluster più vicino</li>\n<li>$s_i = (b_i - a_i) / \\max(a_i, b_i) \\in [-1, 1]$</li>\n</ul>\n<p>Valori vicini a 1: cluster ben definito; vicini a 0: sul confine; negativi: probabilmente nel cluster sbagliato.</p>\n\n<h3>11.6 Limitazioni</h3>\n<p>K-Means assume cluster:</p>\n<ul>\n<li>Sferici (usa distanza euclidea)</li>\n<li>Di dimensione simile</li>\n<li>Con densità simile</li>\n</ul>\n<p>Fallisce su cluster allungati, densità variabile, forme complesse. Per questi casi: DBSCAN (density-based), Gaussian Mixture Models, Spectral Clustering.</p>\n\n<h3>11.7 k-means++</h3>\n<p>Inizializzazione intelligente: scegli il primo centroide a caso, poi ogni successivo con probabilità proporzionale alla distanza al quadrato dal centroide più vicino già scelto. Molto meno soggetta a minimi locali cattivi.</p>\n\n<h3>11.8 Altri metodi di clustering</h3>\n<ul>\n<li><strong>Gerarchico</strong>: costruisce un albero (dendrogramma). Bottom-up (agglomerativo) o top-down (divisivo).</li>\n<li><strong>DBSCAN</strong>: raggruppa punti densi. Non serve specificare $K$. Gestisce outlier.</li>\n<li><strong>GMM</strong>: fit di una miscela di gaussiane. Cluster ellittici, probabilità soft di appartenenza.</li>\n</ul>\n"
    }
  ],
  "esempi": [
    {
      "type": "md",
      "content": "<h3>Esempio 1: K-Means su dati sintetici</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.cluster import KMeans\nfrom sklearn.datasets import make_blobs\n\nX, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.7, random_state=0)\nkm = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)\n\nfig, ax = plt.subplots(figsize=(7,5))\nax.scatter(X[:,0], X[:,1], c=km.labels_, cmap='tab10', s=25, edgecolor='k', alpha=0.7)\nax.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], c='red', marker='X', s=200, edgecolor='k', label='centroidi')\nax.legend(); ax.set_title(f'K-Means con K=4, inertia={km.inertia_:.1f}')\nplt.show()"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 2: metodo del gomito</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nfrom sklearn.cluster import KMeans\nfrom sklearn.datasets import make_blobs\n\nX, _ = make_blobs(n_samples=300, centers=4, random_state=0)\ninertias = []\nfor k in range(1, 11):\n    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)\n    inertias.append(km.inertia_)\n    print(f\"K={k:2d}: inertia = {km.inertia_:8.2f}\")\nprint(\"\\nIl 'gomito' è a K=4, coerente con il vero numero di cluster.\")"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 3: K-Means from scratch</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nnp.random.seed(0)\n\ndef kmeans_scratch(X, k, n_iter=100):\n    n, d = X.shape\n    # init: scegli k punti a caso\n    centroids = X[np.random.choice(n, k, replace=False)].copy()\n    for it in range(n_iter):\n        # assign\n        dists = ((X[:,None,:] - centroids[None,:,:])**2).sum(-1)\n        labels = np.argmin(dists, axis=1)\n        # update\n        new_centroids = np.array([X[labels==j].mean(axis=0) if (labels==j).any() else centroids[j] for j in range(k)])\n        if np.allclose(new_centroids, centroids):\n            print(f\"converged in {it} steps\")\n            break\n        centroids = new_centroids\n    return centroids, labels\n\nfrom sklearn.datasets import make_blobs\nX, _ = make_blobs(n_samples=300, centers=3, random_state=0)\nc, l = kmeans_scratch(X, k=3)\nprint(\"centroidi:\", c.round(2))"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 4: quando K-Means fallisce</h3>"
    },
    {
      "type": "run",
      "content": "import matplotlib.pyplot as plt\nfrom sklearn.cluster import KMeans, DBSCAN\nfrom sklearn.datasets import make_moons\n\nX, _ = make_moons(n_samples=300, noise=0.05, random_state=0)\nfig, axes = plt.subplots(1, 2, figsize=(11,4))\nkm = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)\ndb = DBSCAN(eps=0.2, min_samples=5).fit(X)\naxes[0].scatter(X[:,0], X[:,1], c=km.labels_, cmap='coolwarm', edgecolor='k'); axes[0].set_title('K-Means: fallisce sui half-moon')\naxes[1].scatter(X[:,0], X[:,1], c=db.labels_, cmap='coolwarm', edgecolor='k'); axes[1].set_title('DBSCAN: cattura le lune')\nplt.tight_layout(); plt.show()"
    }
  ],
  "esercizi": [
    {
      "type": "exercise",
      "difficulty": "easy",
      "title": "Es. 11.1 — Fit e conteggio",
      "prompt": "<p>Su dati blobs con 5 centri, fitta un KMeans e conta quanti punti in ciascun cluster. Salva l'array dei conteggi in <code class=\"inline\">counts</code>.</p>",
      "starter": "import numpy as np\nfrom sklearn.cluster import KMeans\nfrom sklearn.datasets import make_blobs\n\nX, _ = make_blobs(n_samples=500, centers=5, random_state=0)\n\n# TODO\ncounts = None\nprint(counts)",
      "check": "import numpy as np\nif counts is not None and len(counts) == 5 and sum(counts) == 500:\n    print(f\"OK — {counts}\")\nelse:\n    print(\"FAIL\")",
      "solution": "km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(X)\n_, counts = np.unique(km.labels_, return_counts=True)"
    },
    {
      "type": "exercise",
      "difficulty": "medium",
      "title": "Es. 11.2 — Silhouette",
      "prompt": "<p>Trova il $K \\in \\{2,3,4,5,6,7,8\\}$ che massimizza il silhouette score sul dataset. Salva in <code class=\"inline\">best_k</code>.</p>",
      "starter": "import numpy as np\nfrom sklearn.cluster import KMeans\nfrom sklearn.datasets import make_blobs\nfrom sklearn.metrics import silhouette_score\n\nX, _ = make_blobs(n_samples=300, centers=4, random_state=0)\n\n# TODO\nbest_k = None\nprint(\"k ottimo:\", best_k)",
      "check": "import numpy as np\nfrom sklearn.cluster import KMeans\nfrom sklearn.metrics import silhouette_score\nscores = {k: silhouette_score(X, KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)) for k in range(2,9)}\n_atteso = max(scores, key=scores.get)\nif best_k == _atteso:\n    print(f\"OK — k={best_k}, silhouette={scores[best_k]:.4f}\")\nelse:\n    print(f\"FAIL — atteso {_atteso}\")",
      "solution": "scores = {k: silhouette_score(X, KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)) for k in range(2,9)}\nbest_k = max(scores, key=scores.get)"
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Es. 11.3 — Inerzia manuale",
      "prompt": "<p>Data una matrice <code class=\"inline\">X</code>, dei centroidi <code class=\"inline\">centers</code> e le assegnazioni <code class=\"inline\">labels</code>, calcola l'inerzia (SSE): $\\sum_i \\|x_i - \\mu_{c(i)}\\|^2$.</p>",
      "starter": "import numpy as np\nnp.random.seed(0)\nX = np.random.randn(100, 3)\ncenters = np.random.randn(4, 3)\nlabels = np.random.randint(0, 4, 100)\n\n# TODO\ninertia = None\nprint(inertia)",
      "check": "import numpy as np\n_atteso = sum(((X[labels==k] - centers[k])**2).sum() for k in range(4))\nif inertia is not None and abs(inertia - _atteso) < 1e-6:\n    print(f\"OK — {inertia:.4f}\")\nelse:\n    print(f\"FAIL — atteso {_atteso:.4f}\")",
      "solution": "inertia = sum(((X[labels==k] - centers[k])**2).sum() for k in range(4))\n# oppure: inertia = ((X - centers[labels])**2).sum()"
    }
  ],
  "test": [
    {
      "type": "quiz",
      "questions": [
        {
          "q": "K-Means converge sempre a:",
          "options": [
            "Un minimo globale",
            "Un minimo locale (dipende dall'inizializzazione)",
            "Un massimo",
            "Nessuno dei precedenti"
          ],
          "answer": 1,
          "explain": "È monotono decrescente ma non convesso — locale, non globale."
        },
        {
          "q": "K-Means su cluster a mezza luna:",
          "options": [
            "Funziona bene",
            "Fallisce (assume forme sferiche)",
            "Dipende da K",
            "Serve più iterazioni"
          ],
          "answer": 1,
          "explain": "La distanza euclidea impone confini sferici. DBSCAN è meglio."
        },
        {
          "q": "Il metodo del gomito serve per:",
          "options": [
            "Rimuovere outlier",
            "Scegliere K guardando dove l'inerzia smette di calare rapidamente",
            "Inizializzare centroidi",
            "Calcolare silhouette"
          ],
          "answer": 1,
          "explain": "Cerchi il \"cambio di pendenza\" nel grafico K vs inertia."
        },
        {
          "q": "k-means++ vs random init:",
          "options": [
            "Peggio",
            "Meglio: distribuisce i centroidi iniziali in modo intelligente riducendo il rischio di minimi locali cattivi",
            "Nessuna differenza",
            "Solo più veloce"
          ],
          "answer": 1,
          "explain": "Sceglie centroidi lontani tra loro."
        }
      ]
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Test pratico — clustering su Iris (unsupervised)",
      "prompt": "<p>Su Iris (senza usare le label), fitta un KMeans con K=3 e calcola quanto ben concorda con le vere classi via <em>adjusted rand index</em>. Salva in <code class=\"inline\">ari</code>. Target > 0.7.</p>",
      "starter": "from sklearn.datasets import load_iris\nfrom sklearn.cluster import KMeans\nfrom sklearn.metrics import adjusted_rand_score\nfrom sklearn.preprocessing import StandardScaler\n\niris = load_iris()\nX, y = iris.data, iris.target\n\nari = None",
      "check": "if ari is None: print(\"FAIL: None\")\nelif ari > 0.7: print(f\"OK — ARI {ari:.4f}\")\nelse: print(f\"FAIL — ARI {ari:.4f}\")",
      "solution": "X_s = StandardScaler().fit_transform(X)\nlabels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X_s)\nari = adjusted_rand_score(y, labels)"
    }
  ]
});
