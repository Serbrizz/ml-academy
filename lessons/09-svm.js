/* Lezione: Support Vector Machines */
window.LESSONS.push({
  "id": "l09-svm",
  "title": "Support Vector Machines",
  "tag": "Supervisionato",
  "summary": "Massimizzare il margine, il kernel trick, la dualità. Uno dei modelli più belli teoricamente.",
  "teoria": [
    {
      "type": "md",
      "content": "\n<h3>9.1 L'intuizione: massimo margine</h3>\n<p>Con due classi linearmente separabili in $\\mathbb{R}^d$, ci sono infiniti iperpiani che le dividono. Quale scegliere? L'SVM sceglie quello che <strong>massimizza la distanza dai punti più vicini di ciascuna classe</strong>. Questa distanza è il <em>margine</em>.</p>\n<p>Un margine grande è associato a migliore generalizzazione (teoria di Vapnik-Chervonenkis).</p>\n\n<h3>9.2 Formulazione primale (hard margin)</h3>\n<p>Con etichette $y_i \\in \\{-1, +1\\}$, l'iperpiano è $w^T x + b = 0$. La distanza dal punto $x_i$ è $|w^T x_i + b|/\\|w\\|$. Se normalizziamo affinché i punti più vicini soddisfino $|w^T x + b| = 1$, il margine è $2/\\|w\\|$.</p>\n<p>Massimizzare il margine equivale a minimizzare $\\|w\\|^2$:</p>\n<p>$$\\min_{w,b} \\tfrac{1}{2}\\|w\\|^2 \\quad\\text{sotto vincolo}\\quad y_i(w^T x_i + b) \\geq 1\\ \\forall i$$</p>\n<p>È un problema di programmazione quadratica convessa.</p>\n\n<h3>9.3 Soft margin</h3>\n<p>Se i dati non sono linearmente separabili, introduciamo variabili di slack $\\xi_i \\geq 0$:</p>\n<p>$$\\min_{w,b,\\xi} \\tfrac{1}{2}\\|w\\|^2 + C\\sum_i \\xi_i \\quad\\text{s.t.}\\quad y_i(w^T x_i + b) \\geq 1 - \\xi_i$$</p>\n<p>Il parametro $C$ controlla il trade-off: $C$ grande &rarr; poca tolleranza agli errori (rischio overfitting); $C$ piccolo &rarr; margine largo, più errori tollerati.</p>\n\n<h3>9.4 Formulazione duale</h3>\n<p>Con i moltiplicatori di Lagrange, si arriva alla forma duale in cui i dati appaiono solo tramite prodotti scalari $x_i^T x_j$:</p>\n<p>$$\\max_\\alpha \\sum_i \\alpha_i - \\tfrac{1}{2}\\sum_{i,j}\\alpha_i \\alpha_j y_i y_j x_i^T x_j \\quad\\text{s.t.}\\ 0 \\leq \\alpha_i \\leq C,\\ \\sum_i \\alpha_i y_i = 0$$</p>\n<p>Solo gli $x_i$ con $\\alpha_i > 0$ contribuiscono: quelli sono i <em>support vector</em>. In pratica, il modello dipende solo da un sottoinsieme (a volte molto piccolo) dei dati.</p>\n\n<h3>9.5 Il kernel trick</h3>\n<p>Poiché i dati appaiono solo come prodotti scalari, possiamo <em>sostituire</em> $x^T x'$ con $K(x, x') = \\phi(x)^T \\phi(x')$ dove $\\phi$ mappa in uno spazio ad alta (anche infinita) dimensione. Senza mai calcolare $\\phi$ esplicitamente.</p>\n<p>Kernel comuni:</p>\n<ul>\n<li><strong>Lineare</strong>: $K(x,x') = x^T x'$</li>\n<li><strong>Polinomiale</strong>: $K(x,x') = (\\gamma x^T x' + r)^d$</li>\n<li><strong>RBF (Gaussiano)</strong>: $K(x,x') = e^{-\\gamma \\|x-x'\\|^2}$ — mappa in uno spazio di dimensione infinita</li>\n<li><strong>Sigmoid</strong>: $\\tanh(\\gamma x^T x' + r)$</li>\n</ul>\n<p>Con RBF, l'SVM può approssimare qualsiasi funzione — è un modello universale.</p>\n\n<h3>9.6 Parametri chiave (in scikit-learn)</h3>\n<ul>\n<li><code class=\"inline\">C</code>: regolarizzazione (inversa)</li>\n<li><code class=\"inline\">gamma</code>: larghezza del kernel RBF. Grande $\\gamma$ &rarr; kernel stretto &rarr; overfitting</li>\n<li><code class=\"inline\">kernel</code>: 'linear', 'poly', 'rbf', 'sigmoid'</li>\n</ul>\n<p>La combinazione (C, gamma) va tarata con GridSearchCV. È la coppia più importante.</p>\n\n<h3>9.7 Regressione: SVR</h3>\n<p>Versione regressiva: cerca una funzione $f(x) = w^T x + b$ tale che $|y_i - f(x_i)| \\leq \\varepsilon$ per la maggior parte dei punti. Chiamata $\\varepsilon$-tube regression.</p>\n"
    },
    {
      "type": "callout",
      "variant": "note",
      "title": "Punti forti/deboli",
      "content": "SVM è potente su dataset di dimensione medio-piccola (fino a decine di migliaia). Su dataset molto grandi il costo O(n² o n³) del training diventa proibitivo — meglio SGD o gradient boosting."
    }
  ],
  "esempi": [
    {
      "type": "md",
      "content": "<h3>Esempio 1: SVM lineare</h3>"
    },
    {
      "type": "run",
      "content": "from sklearn.svm import SVC\nfrom sklearn.datasets import make_classification\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nimport numpy as np\n\nX, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, random_state=0)\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)\n\npipe = Pipeline([('s', StandardScaler()), ('svm', SVC(kernel='linear', C=1.0))]).fit(Xt, yt)\nprint(f\"Accuracy: {pipe.score(Xe, ye):.4f}\")\nprint(f\"Numero di support vectors: {pipe.named_steps['svm'].n_support_}\")"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 2: kernel RBF vs lineare su dati non lineari</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.svm import SVC\nfrom sklearn.datasets import make_circles\n\nX, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=0)\n\nfig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\nfor ax, kernel in zip(axes, ['linear', 'rbf']):\n    m = SVC(kernel=kernel, C=1.0, gamma='scale').fit(X, y)\n    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.3, X[:,0].max()+0.3, 200),\n                         np.linspace(X[:,1].min()-0.3, X[:,1].max()+0.3, 200))\n    Z = m.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)\n    ax.contourf(xx, yy, Z, levels=20, cmap='RdBu_r', alpha=0.5)\n    ax.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)\n    ax.scatter(X[:,0], X[:,1], c=y, cmap='RdBu_r', edgecolor='k')\n    ax.set_title(f'kernel={kernel}, acc train={m.score(X,y):.3f}')\nplt.tight_layout(); plt.show()"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 3: effetto di C e gamma</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nfrom sklearn.svm import SVC\nfrom sklearn.datasets import make_moons\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.model_selection import cross_val_score\n\nX, y = make_moons(n_samples=300, noise=0.3, random_state=0)\nprint(f\"{'C':>6} {'gamma':>6} {'CV mean':>10}\")\nfor C in [0.1, 1.0, 10.0]:\n    for g in [0.1, 1.0, 10.0]:\n        p = Pipeline([('s', StandardScaler()), ('svm', SVC(C=C, gamma=g))])\n        s = cross_val_score(p, X, y, cv=5).mean()\n        print(f\"{C:>6.1f} {g:>6.1f} {s:>10.4f}\")"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 4: kernel trick manuale</h3><p>Costruiamo il kernel polinomiale di grado 2 esplicitamente vs implicitamente.</p>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nnp.random.seed(0)\nX = np.random.randn(5, 2)\n\n# esplicito: mappa a phi(x) = (1, sqrt(2)x1, sqrt(2)x2, x1^2, sqrt(2)x1x2, x2^2)\ndef phi(x):\n    return np.array([1, np.sqrt(2)*x[0], np.sqrt(2)*x[1], x[0]**2, np.sqrt(2)*x[0]*x[1], x[1]**2])\n\nPhi = np.array([phi(x) for x in X])\nK_explicit = Phi @ Phi.T\n\n# implicito: K(x, x') = (1 + x^T x')^2\nK_implicit = (1 + X @ X.T)**2\n\nprint(\"Max differenza:\", np.abs(K_explicit - K_implicit).max())\nprint(\"=> stesso kernel, ma implicito non richiede di calcolare phi!\")"
    }
  ],
  "esercizi": [
    {
      "type": "exercise",
      "difficulty": "easy",
      "title": "Es. 9.1 — SVM lineare vs logistica",
      "prompt": "<p>Su breast_cancer, addestra un SVM lineare (<code class=\"inline\">SVC(kernel=\"linear\")</code>) e una LogisticRegression. Confronta accuracy test. Salva in <code class=\"inline\">acc_svm, acc_lr</code>.</p>",
      "starter": "from sklearn.datasets import load_breast_cancer\nfrom sklearn.svm import SVC\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.model_selection import train_test_split\n\nX, y = load_breast_cancer(return_X_y=True)\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)\n\nacc_svm = None\nacc_lr = None\nprint(acc_svm, acc_lr)",
      "check": "if acc_svm is not None and acc_lr is not None and acc_svm > 0.9 and acc_lr > 0.9:\n    print(f\"OK — SVM {acc_svm:.4f}, LR {acc_lr:.4f}\")\nelse:\n    print(\"FAIL — devono essere entrambe > 0.9\")",
      "solution": "p1 = Pipeline([('s', StandardScaler()), ('c', SVC(kernel='linear'))]).fit(Xt, yt)\np2 = Pipeline([('s', StandardScaler()), ('c', LogisticRegression(max_iter=5000))]).fit(Xt, yt)\nacc_svm = p1.score(Xe, ye); acc_lr = p2.score(Xe, ye)"
    },
    {
      "type": "exercise",
      "difficulty": "medium",
      "title": "Es. 9.2 — Kernel gaussiano manuale",
      "prompt": "<p>Implementa <code class=\"inline\">rbf_kernel(X1, X2, gamma)</code> senza usare sklearn. Deve restituire una matrice $K \\in \\mathbb{R}^{n_1 \\times n_2}$ con $K_{ij} = e^{-\\gamma \\|x_i - x_j\\|^2}$.</p>",
      "starter": "import numpy as np\ndef rbf_kernel(X1, X2, gamma):\n    # TODO\n    return None\n\nX1 = np.array([[0,0],[1,1]])\nX2 = np.array([[0,0],[2,2],[1,0]])\nprint(rbf_kernel(X1, X2, gamma=0.5))",
      "check": "import numpy as np\nfrom sklearn.metrics.pairwise import rbf_kernel as sk_rbf\nX1 = np.random.randn(4, 3); X2 = np.random.randn(5, 3)\nK_mine = rbf_kernel(X1, X2, 0.7)\nK_sk = sk_rbf(X1, X2, gamma=0.7)\nif K_mine is not None and np.allclose(K_mine, K_sk):\n    print(\"OK — identico a sklearn\")\nelse:\n    print(\"FAIL\")",
      "solution": "def rbf_kernel(X1, X2, gamma):\n    sq_dist = ((X1[:,None,:] - X2[None,:,:])**2).sum(-1)\n    return np.exp(-gamma * sq_dist)"
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Es. 9.3 — Grid search SVM RBF",
      "prompt": "<p>Su make_moons rumoroso, fai grid search su $C \\in \\{0.1,1,10\\}$ e $\\gamma \\in \\{0.1,1,10\\}$ con CV a 5 fold. Salva la miglior CV score in <code class=\"inline\">best_cv</code>.</p>",
      "starter": "import numpy as np\nfrom sklearn.svm import SVC\nfrom sklearn.datasets import make_moons\nfrom sklearn.model_selection import GridSearchCV\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\n\nX, y = make_moons(n_samples=500, noise=0.3, random_state=0)\n\n# TODO\nbest_cv = None\nprint(best_cv)",
      "check": "if best_cv is not None and best_cv > 0.85:\n    print(f\"OK — {best_cv:.4f}\")\nelse:\n    print(f\"FAIL — {best_cv}\")",
      "solution": "pipe = Pipeline([('s', StandardScaler()), ('svm', SVC())])\ngrid = {'svm__C':[0.1,1,10], 'svm__gamma':[0.1,1,10]}\ngs = GridSearchCV(pipe, grid, cv=5).fit(X, y)\nbest_cv = gs.best_score_"
    }
  ],
  "test": [
    {
      "type": "quiz",
      "questions": [
        {
          "q": "Un support vector è:",
          "options": [
            "Un vettore casuale",
            "Un punto del training con $\\alpha_i > 0$, cioè che tocca o viola il margine",
            "Un iperpiano",
            "Un'etichetta"
          ],
          "answer": 1,
          "explain": "Solo i punti che contribuiscono alla soluzione del duale."
        },
        {
          "q": "Il kernel trick permette di:",
          "options": [
            "Velocizzare gradient descent",
            "Calcolare prodotti scalari in spazi ad alta dimensione senza mappare esplicitamente",
            "Ridurre la dimensione",
            "Regolarizzare"
          ],
          "answer": 1,
          "explain": "La formula duale usa solo prodotti scalari; li sostituiamo con $K$."
        },
        {
          "q": "In SVM, un $C$ molto grande implica:",
          "options": [
            "Margine largo, molti errori tollerati",
            "Margine stretto, pochi errori tollerati, rischio overfitting",
            "Kernel RBF automatico",
            "Nessun effetto"
          ],
          "answer": 1,
          "explain": "$C$ pesa la penalità sugli errori: grande = severi."
        },
        {
          "q": "Il kernel RBF mappa in uno spazio di dimensione:",
          "options": [
            "1",
            "Uguale a $d$",
            "Uguale a $n$",
            "Infinita"
          ],
          "answer": 3,
          "explain": "Espansione di Taylor di $e^x$ produce infiniti termini."
        }
      ]
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Test pratico — digits con SVM",
      "prompt": "<p>Sul dataset digits, addestra un SVM con kernel RBF, standardizzazione e grid search su C, gamma. Target accuracy test > 0.98. Salva in <code class=\"inline\">acc</code>.</p>",
      "starter": "from sklearn.datasets import load_digits\nfrom sklearn.svm import SVC\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.model_selection import GridSearchCV, train_test_split\n\nX, y = load_digits(return_X_y=True)\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)\n\nacc = None",
      "check": "if acc is None: print(\"FAIL: None\")\nelif acc > 0.98: print(f\"OK — {acc:.4f}\")\nelse: print(f\"FAIL — {acc:.4f}\")",
      "solution": "pipe = Pipeline([('s', StandardScaler()), ('svm', SVC(kernel='rbf'))])\ngrid = {'svm__C':[1,10,100], 'svm__gamma':['scale', 0.01, 0.001]}\ngs = GridSearchCV(pipe, grid, cv=5).fit(Xt, yt)\nacc = gs.score(Xe, ye)"
    }
  ]
});
