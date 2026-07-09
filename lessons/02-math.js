/* Lezione: Ripasso matematico */
window.LESSONS.push({
  "id": "l02-math",
  "title": "Ripasso matematico",
  "tag": "Fondamenti",
  "summary": "Algebra lineare, calcolo, probabilità, ottimizzazione — visti dalla lente del ML.",
  "teoria": [
    {
      "type": "md",
      "content": "\n<h3>2.1 Vettori, matrici, tensori</h3>\n<p>Un dataset è tipicamente una matrice $X \\in \\mathbb{R}^{n \\times d}$: $n$ righe (esempi), $d$ colonne (features). Ogni riga $x_i \\in \\mathbb{R}^d$ è un vettore di features. Le operazioni fondamentali sono:</p>\n<ul>\n<li><strong>Prodotto scalare</strong>: $x \\cdot w = \\sum_j x_j w_j = x^T w$. Misura quanto due vettori \"puntano nella stessa direzione\".</li>\n<li><strong>Norma</strong>: $\\|x\\|_2 = \\sqrt{\\sum_j x_j^2}$. La versione $\\ell_1$ è $\\|x\\|_1 = \\sum_j |x_j|$.</li>\n<li><strong>Prodotto matrice-vettore</strong>: $Xw \\in \\mathbb{R}^n$. Ogni riga è $x_i^T w$: le predizioni di un modello lineare.</li>\n<li><strong>Trasposta</strong>: $(X^T)_{ij} = X_{ji}$. Utile per pratica di broadcasting e per la formula $X^T X$ che ricorre ovunque.</li>\n</ul>\n\n<h4>Autovalori e SVD</h4>\n<p>Data $A \\in \\mathbb{R}^{d\\times d}$ simmetrica semi-definita positiva (come una matrice di covarianza), esiste una decomposizione $A = V \\Lambda V^T$ con $V$ ortogonale e $\\Lambda$ diagonale con autovalori $\\lambda_1 \\geq \\dots \\geq \\lambda_d \\geq 0$. Questo è il fondamento di PCA.</p>\n<p>Per matrici rettangolari $X \\in \\mathbb{R}^{n\\times d}$: <em>Singular Value Decomposition</em> $X = U \\Sigma V^T$ con $U, V$ ortogonali e $\\Sigma$ diagonale (allargata) con valori singolari $\\sigma_1 \\geq \\dots \\geq 0$. SVD è forse lo strumento più utile di algebra lineare numerica.</p>\n\n<h3>2.2 Calcolo differenziale in $\\mathbb{R}^d$</h3>\n<p>Il <strong>gradiente</strong> di $f : \\mathbb{R}^d \\to \\mathbb{R}$ è il vettore delle derivate parziali:</p>\n<p>$$\\nabla f(x) = \\left(\\frac{\\partial f}{\\partial x_1}, \\dots, \\frac{\\partial f}{\\partial x_d}\\right)^T$$</p>\n<p>Punta nella direzione di massima crescita. La discesa del gradiente muove $x$ nella direzione $-\\nabla f$ perché è quella di massima decrescita locale.</p>\n<p>Formule utili (che ti serviranno spesso):</p>\n<ul>\n<li>$\\nabla_x (a^T x) = a$</li>\n<li>$\\nabla_x (x^T A x) = (A + A^T) x$, che per $A$ simmetrica dà $2Ax$</li>\n<li>$\\nabla_x \\|Xx - y\\|^2 = 2 X^T (Xx - y)$ &larr; questa è il gradiente della loss quadratica</li>\n</ul>\n<p>L'<strong>Hessiana</strong> $H = \\nabla^2 f$ è la matrice delle derivate seconde. $f$ è convessa se e solo se $H \\succeq 0$ ovunque.</p>\n\n<h3>2.3 Probabilità in versione ML-friendly</h3>\n<p>Una <strong>variabile aleatoria</strong> $X$ ha una distribuzione. Concetti chiave:</p>\n<ul>\n<li><strong>Attesa</strong>: $\\mathbb{E}[X] = \\int x\\, p(x)\\, dx$. Linearità: $\\mathbb{E}[aX+bY]=a\\mathbb{E}[X]+b\\mathbb{E}[Y]$.</li>\n<li><strong>Varianza</strong>: $\\mathrm{Var}(X) = \\mathbb{E}[(X-\\mathbb{E}X)^2]$.</li>\n<li><strong>Probabilità condizionata</strong>: $p(y|x) = p(x,y)/p(x)$.</li>\n<li><strong>Bayes</strong>: $p(y|x) = \\dfrac{p(x|y)\\, p(y)}{p(x)}$. Il ML \"generativo\" modella $p(x|y)$; quello \"discriminativo\" modella direttamente $p(y|x)$.</li>\n<li><strong>Gaussiana</strong>: $\\mathcal{N}(x; \\mu, \\sigma^2) = \\frac{1}{\\sqrt{2\\pi}\\sigma} e^{-(x-\\mu)^2/(2\\sigma^2)}$. Multivariata: $\\mathcal{N}(x; \\mu, \\Sigma) \\propto \\exp(-\\frac{1}{2}(x-\\mu)^T \\Sigma^{-1}(x-\\mu))$.</li>\n</ul>\n\n<h4>Massima verosimiglianza</h4>\n<p>Se assumiamo che i dati $\\{y_i\\}$ siano i.i.d. da $p(y|\\theta)$, il principio MLE dice: scegli $\\theta$ che massimizza $\\prod_i p(y_i|\\theta)$, o equivalentemente:</p>\n<p>$$\\hat{\\theta}_{\\text{MLE}} = \\arg\\max_\\theta \\sum_i \\log p(y_i | \\theta)$$</p>\n<p>Molte loss standard (MSE, cross-entropy) derivano da MLE sotto specifiche ipotesi.</p>\n\n<h3>2.4 Ottimizzazione: gradient descent</h3>\n<p>Iterazione base:</p>\n<p>$$\\theta_{t+1} = \\theta_t - \\eta\\, \\nabla_\\theta \\mathcal{L}(\\theta_t)$$</p>\n<p>dove $\\eta > 0$ è il <em>learning rate</em>. Convergenza garantita se $\\mathcal{L}$ è convessa e $\\eta$ è piccolo abbastanza (regola di Lipschitz).</p>\n<p><strong>Stochastic Gradient Descent (SGD)</strong>: al posto del gradiente esatto sul dataset intero, si usa il gradiente su un mini-batch. Molto più veloce, converge in media alla soluzione se $\\eta$ decresce.</p>\n"
    },
    {
      "type": "callout",
      "variant": "tip",
      "title": "Trucco di calcolo",
      "content": "Per derivare formule matriciali, tratta $x^T A y$ come un numero, differenziato una variabile alla volta. Se qualcosa \"non torna dimensionalmente\", non torna proprio. È come l'analisi dimensionale in fisica."
    }
  ],
  "esempi": [
    {
      "type": "md",
      "content": "<h3>Esempio 1: operazioni con NumPy</h3><p>Il codice sotto mostra le operazioni fondamentali. Cambia i numeri e riesegui per prenderci confidenza.</p>"
    },
    {
      "type": "run",
      "content": "import numpy as np\n\n# vettori\nx = np.array([1.0, 2.0, 3.0])\nw = np.array([0.5, -1.0, 2.0])\n\nprint(\"prodotto scalare:\", x @ w)         # = 1*0.5 + 2*(-1) + 3*2 = 4.5\nprint(\"norma L2 di x:\", np.linalg.norm(x))\nprint(\"norma L1 di x:\", np.linalg.norm(x, 1))\n\n# matrici\nX = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])  # 4x3\nprint(\"X @ w =\", X @ w)                    # 4 predizioni lineari\n\n# X^T X è simmetrica 3x3\nXtX = X.T @ X\nprint(\"X^T X:\\n\", XtX)\n\n# autovalori\nvals, vecs = np.linalg.eig(XtX)\nprint(\"autovalori:\", np.round(np.sort(vals)[::-1], 3))"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 2: gradient descent a mano</h3><p>Minimizziamo $f(x) = (x-3)^2 + 1$ senza scikit-learn. Il minimo è ovviamente $x^*=3$.</p>"
    },
    {
      "type": "run",
      "content": "def f(x):     return (x-3)**2 + 1\ndef grad(x):  return 2*(x-3)\n\nx = 0.0\neta = 0.1\nfor step in range(30):\n    x = x - eta * grad(x)\n    if step % 5 == 0:\n        print(f\"step {step:2d}: x={x:.5f}, f(x)={f(x):.5f}\")\nprint(f\"convergenza a x = {x:.6f} (atteso 3.0)\")"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 3: MLE di una gaussiana</h3><p>Dati $y_i \\sim \\mathcal{N}(\\mu, \\sigma^2)$, la stima MLE è $\\hat\\mu = \\bar y$, $\\hat\\sigma^2 = \\frac{1}{n}\\sum (y_i - \\bar y)^2$. Verifichiamolo:</p>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nnp.random.seed(1)\ny = np.random.normal(loc=5.0, scale=2.0, size=1000)\n\nmu_hat = y.mean()\nsigma2_hat = np.mean((y - mu_hat)**2)\nprint(f\"mu_hat = {mu_hat:.3f} (vero 5.0)\")\nprint(f\"sigma_hat = {np.sqrt(sigma2_hat):.3f} (vero 2.0)\")"
    }
  ],
  "esercizi": [
    {
      "type": "exercise",
      "difficulty": "easy",
      "title": "Es. 2.1 — Prodotto scalare a mano",
      "prompt": "<p>Implementa il prodotto scalare tra due liste Python senza usare numpy. Deve funzionare per vettori di qualunque lunghezza uguale.</p>",
      "starter": "def dot(a, b):\n    # TODO\n    return None\n\nprint(dot([1,2,3], [4,5,6]))  # atteso 32",
      "check": "if dot([1,2,3],[4,5,6]) == 32 and dot([1.0,-1.0],[2.0,3.0]) == -1.0:\n    print(\"OK\")\nelse:\n    print(\"FAIL\")",
      "solution": "def dot(a, b):\n    return sum(x*y for x, y in zip(a, b))"
    },
    {
      "type": "exercise",
      "difficulty": "medium",
      "title": "Es. 2.2 — Gradient descent multivariato",
      "prompt": "<p>Minimizza $f(x, y) = (x-1)^2 + 3(y+2)^2$ partendo da $(0, 0)$ con learning rate $0.1$ per 50 passi. Salva il risultato in <code class=\"inline\">xy_min</code> (tupla di 2 valori).</p>",
      "starter": "import numpy as np\ndef f(v):    return (v[0]-1)**2 + 3*(v[1]+2)**2\ndef grad(v): return np.array([2*(v[0]-1), 6*(v[1]+2)])\n\nv = np.array([0.0, 0.0])\n# TODO: loop di 50 iterazioni\n\nxy_min = tuple(v)\nprint(\"min trovato:\", xy_min)",
      "check": "import numpy as np\ntarget = np.array([1.0, -2.0])\nif np.allclose(xy_min, target, atol=1e-3):\n    print(f\"OK — trovato {xy_min}, atteso (1, -2)\")\nelse:\n    print(f\"FAIL — trovato {xy_min}, atteso circa (1, -2)\")",
      "solution": "for _ in range(50):\n    v = v - 0.1 * grad(v)\nxy_min = tuple(v)"
    },
    {
      "type": "exercise",
      "difficulty": "medium",
      "title": "Es. 2.3 — Autovalori e direzione dominante",
      "prompt": "<p>Data una matrice di covarianza $\\Sigma = \\begin{pmatrix}4 & 2\\\\ 2 & 3\\end{pmatrix}$, trova l'autovettore associato all'autovalore più grande. Salvalo in <code class=\"inline\">v_max</code> (array di 2 elementi).</p>",
      "starter": "import numpy as np\nSigma = np.array([[4, 2],[2, 3]])\n\n# TODO: trova autovalori e autovettori, seleziona il maggiore\nv_max = np.array([0.0, 0.0])\nprint(\"autovettore dominante:\", v_max)",
      "check": "import numpy as np\nvals, vecs = np.linalg.eigh(Sigma)\ntarget = vecs[:, np.argmax(vals)]\n# autovettore definito a meno di segno\nif np.allclose(v_max, target, atol=1e-4) or np.allclose(v_max, -target, atol=1e-4):\n    print(\"OK — direzione corretta\")\nelse:\n    print(f\"FAIL — tuo {v_max}, atteso ±{target}\")",
      "solution": "vals, vecs = np.linalg.eigh(Sigma)\nv_max = vecs[:, np.argmax(vals)]"
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Es. 2.4 — MLE per Poisson",
      "prompt": "<p>Dati $k_1, \\dots, k_n$ da $\\mathrm{Poisson}(\\lambda)$, dimostra prima su carta che $\\hat\\lambda_{\\text{MLE}} = \\bar k$. Poi verifica numericamente su un campione simulato che il valore stimato converge al vero $\\lambda$ al crescere di $n$.</p>",
      "starter": "import numpy as np\nnp.random.seed(0)\nvero_lambda = 3.7\nfor n in [10, 100, 1000, 10000]:\n    sample = np.random.poisson(vero_lambda, size=n)\n    # TODO: calcola stima MLE\n    stima = 0.0\n    print(f\"n={n:5d}  stima={stima:.4f}  err={abs(stima-vero_lambda):.4f}\")",
      "check": "# ricomputiamo per verifica\nnp.random.seed(0)\n_last = None\nfor n in [10, 100, 1000, 10000]:\n    _last = np.random.poisson(vero_lambda, size=n).mean()\nprint(f\"\\n(l'ultima stima con n=10000 deve essere vicina a {vero_lambda})\")\nif abs(_last - vero_lambda) < 0.1:\n    print(\"OK — convergenza verificata\")",
      "solution": "stima = sample.mean()"
    }
  ],
  "test": [
    {
      "type": "quiz",
      "questions": [
        {
          "q": "Data $f(x) = \\|Ax-b\\|^2$, il gradiente rispetto a $x$ è:",
          "options": [
            "$Ax-b$",
            "$A^T(Ax-b)$",
            "$2 A^T(Ax-b)$",
            "$2A(Ax-b)$"
          ],
          "answer": 2,
          "explain": "Espandi $(Ax-b)^T(Ax-b)$ e differenzia; il fattore 2 viene dai due termini simmetrici."
        },
        {
          "q": "La SVD $X = U\\Sigma V^T$ vale:",
          "options": [
            "Solo per matrici quadrate",
            "Solo per matrici simmetriche",
            "Per qualsiasi matrice reale",
            "Solo per matrici invertibili"
          ],
          "answer": 2,
          "explain": "La SVD esiste per qualsiasi matrice reale $n\\times d$."
        },
        {
          "q": "Se $\\mathrm{Var}(X) = 4$, quanto vale $\\mathrm{Var}(3X + 5)$?",
          "options": [
            "4",
            "9",
            "12",
            "36"
          ],
          "answer": 3,
          "explain": "$\\mathrm{Var}(aX+b) = a^2 \\mathrm{Var}(X) = 9 \\cdot 4 = 36$."
        },
        {
          "q": "Un learning rate troppo grande produce tipicamente:",
          "options": [
            "Convergenza più lenta",
            "Convergenza più veloce e stabile",
            "Oscillazioni o divergenza",
            "Overfitting"
          ],
          "answer": 2,
          "explain": "Se $\\eta$ è troppo grande, ogni passo \"salta oltre\" il minimo e la loss può crescere."
        }
      ]
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Test pratico — regressione lineare a mano",
      "prompt": "<p>Implementa la <strong>closed-form solution</strong> della regressione lineare: dato $X$ e $y$, calcola $\\hat w = (X^T X)^{-1} X^T y$. Confronta con scikit-learn.</p><p>Nota: aggiungi una colonna di 1 per il bias.</p>",
      "starter": "import numpy as np\nnp.random.seed(42)\nn = 100\nX = np.random.randn(n, 3)\nw_true = np.array([1.5, -2.0, 0.7])\nb_true = 0.3\ny = X @ w_true + b_true + 0.1*np.random.randn(n)\n\n# aggiungi colonna di 1 per il bias\nX_aug = np.hstack([X, np.ones((n,1))])\n\n# TODO: calcola w_hat = (X_aug^T X_aug)^{-1} X_aug^T y\nw_hat = None\nprint(\"w_hat:\", w_hat)\nprint(\"veri:\", list(w_true) + [b_true])",
      "check": "import numpy as np\nfrom sklearn.linear_model import LinearRegression\nsk = LinearRegression().fit(X, y)\nsk_full = np.concatenate([sk.coef_, [sk.intercept_]])\nif w_hat is None:\n    print(\"FAIL: w_hat è None\")\nelif np.allclose(w_hat, sk_full, atol=1e-6):\n    print(\"OK — soluzione identica a scikit-learn\")\n    print(\"w_hat:\", np.round(w_hat, 4))\nelse:\n    print(f\"FAIL:\\ntuo    {np.round(w_hat,4)}\\nsklearn{np.round(sk_full,4)}\")",
      "solution": "w_hat = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y"
    }
  ]
});
