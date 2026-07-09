/* Lezione: Regressione logistica */
window.LESSONS.push({
  "id": "l06-logistic",
  "title": "Regressione logistica",
  "tag": "Supervisionato",
  "summary": "Il modello lineare per la classificazione. Sigmoid, cross-entropy, softmax.",
  "teoria": [
    {
      "type": "md",
      "content": "\n<h3>6.1 Dal lineare al probabilistico</h3>\n<p>Per la classificazione binaria vogliamo $P(y=1|x)$. Un modello lineare $w^T x + b$ produce numeri reali — bisogna schiacciarli in $[0,1]$. La funzione <em>sigmoide</em> fa esattamente questo:</p>\n<p>$$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$</p>\n<p>Proprietà utili: $\\sigma(0) = 0.5$, $\\sigma(-z) = 1 - \\sigma(z)$, $\\sigma'(z) = \\sigma(z)(1-\\sigma(z))$.</p>\n<p>Il modello di regressione logistica è quindi:</p>\n<p>$$P(y=1 | x; w, b) = \\sigma(w^T x + b)$$</p>\n\n<h3>6.2 Interpretazione: log-odds</h3>\n<p>Se $p = P(y=1|x)$, gli <em>odds</em> sono $p / (1-p)$ e i <em>log-odds</em> sono $\\log[p/(1-p)]$. Con la sigmoide si dimostra facilmente che:</p>\n<p>$$\\log \\frac{p}{1-p} = w^T x + b$$</p>\n<p>Cioè la regressione logistica modella i log-odds come funzione lineare di $x$. Un coefficiente $w_j$ è l'incremento nei log-odds per unità di $x_j$.</p>\n\n<h3>6.3 La loss: binary cross-entropy</h3>\n<p>Sotto Bernoulli, la log-likelihood di un esempio è $y \\log p + (1-y)\\log(1-p)$. Massimizzarla equivale a minimizzare:</p>\n<p>$$\\mathcal{L}(w,b) = -\\frac{1}{n}\\sum_i \\left[y_i \\log \\hat p_i + (1-y_i)\\log(1-\\hat p_i)\\right]$$</p>\n<p>Questa è la <em>binary cross-entropy</em> (o log-loss). Non ha soluzione chiusa, ma è <strong>convessa</strong>, quindi la discesa del gradiente trova il minimo globale.</p>\n\n<h3>6.4 Il gradiente</h3>\n<p>Con un po' di algebra si trova (per un esempio $x_i, y_i$):</p>\n<p>$$\\nabla_w \\mathcal{L}_i = (\\hat p_i - y_i)\\, x_i$$</p>\n<p>Sorprendentemente semplice ed identico in forma al gradiente della regressione lineare! È il motivo per cui è così comoda.</p>\n\n<h3>6.5 Confine di decisione</h3>\n<p>Si predice classe 1 se $\\hat p \\geq 0.5$, cioè se $w^T x + b \\geq 0$. Il confine di decisione è quindi un <strong>iperpiano</strong> in $\\mathbb{R}^d$. Modello lineare &rarr; confine lineare.</p>\n\n<h3>6.6 Multi-classe: softmax</h3>\n<p>Per $K$ classi, generalizziamo con $K$ vettori di pesi:</p>\n<p>$$P(y=k | x) = \\frac{e^{w_k^T x}}{\\sum_{j=1}^K e^{w_j^T x}}$$</p>\n<p>La loss è la <em>categorical cross-entropy</em>: $-\\sum_i \\log P(y=y_i | x_i)$. Se il target è one-hot $t_i \\in \\{0,1\\}^K$: $-\\sum_i \\sum_k t_{ik} \\log P(y=k|x_i)$.</p>\n\n<h3>6.7 Regolarizzazione</h3>\n<p>Aggiungiamo $\\lambda \\|w\\|^2$ (L2) o $\\lambda\\|w\\|_1$ (L1). In scikit-learn il parametro <code class=\"inline\">C = 1/\\lambda</code>: valori piccoli &rarr; regolarizzazione forte.</p>\n\n<h3>6.8 Metriche di classificazione</h3>\n<p>Non basta l'accuracy, soprattutto con classi sbilanciate. Considera:</p>\n<ul>\n<li><strong>Matrice di confusione</strong>: TP, FP, FN, TN</li>\n<li><strong>Precision</strong>: $\\frac{TP}{TP+FP}$ — di quelli che ho detto \"positivo\", quanti lo erano?</li>\n<li><strong>Recall</strong>: $\\frac{TP}{TP+FN}$ — dei positivi veri, quanti ho trovato?</li>\n<li><strong>F1</strong>: media armonica di precision e recall</li>\n<li><strong>ROC-AUC</strong>: area sotto la curva ROC; misura la separabilità indipendentemente dalla soglia</li>\n</ul>\n"
    },
    {
      "type": "callout",
      "variant": "tip",
      "title": "Trick numerico",
      "content": "Non implementare mai $\\log(\\sigma(z))$ come <code class=\"inline\">np.log(1/(1+np.exp(-z)))</code>: overflow per $z$ molto negativo. Usa <code class=\"inline\">-np.logaddexp(0, -z)</code> o formulazioni stabili come quelle di scikit-learn."
    }
  ],
  "esempi": [
    {
      "type": "md",
      "content": "<h3>Esempio 1: dataset breast cancer</h3>"
    },
    {
      "type": "run",
      "content": "from sklearn.datasets import load_breast_cancer\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.metrics import classification_report, roc_auc_score\n\nX, y = load_breast_cancer(return_X_y=True)\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)\n\npipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))])\npipe.fit(Xt, yt)\n\ny_pred = pipe.predict(Xe)\ny_prob = pipe.predict_proba(Xe)[:, 1]\n\nprint(classification_report(ye, y_pred, digits=3))\nprint(f\"ROC-AUC: {roc_auc_score(ye, y_prob):.4f}\")"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 2: gradient descent per logistica</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\n\nnp.random.seed(0)\nn = 400\nX = np.random.randn(n, 2)\ntrue_w = np.array([2.0, -1.0])\ny = (X @ true_w + 0.5*np.random.randn(n) > 0).astype(int)\n\nX_aug = np.hstack([X, np.ones((n,1))])   # bias in ultima colonna\nw = np.zeros(3)\n\ndef sigmoid(z): return 1/(1+np.exp(-z))\n\nfor step in range(200):\n    p = sigmoid(X_aug @ w)\n    grad = X_aug.T @ (p - y) / n\n    w = w - 0.5 * grad\n    if step % 40 == 0:\n        loss = -np.mean(y*np.log(p+1e-9) + (1-y)*np.log(1-p+1e-9))\n        acc = np.mean((p > 0.5) == y)\n        print(f\"step {step:3d}: loss={loss:.4f}, acc={acc:.3f}, w={w.round(3)}\")"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 3: confine di decisione visualizzato</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.datasets import make_moons\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import PolynomialFeatures\nfrom sklearn.pipeline import make_pipeline\n\nX, y = make_moons(n_samples=200, noise=0.2, random_state=0)\n\nfig, axes = plt.subplots(1, 2, figsize=(12,4))\nfor ax, poly_deg in zip(axes, [1, 5]):\n    model = make_pipeline(PolynomialFeatures(poly_deg), LogisticRegression(max_iter=1000))\n    model.fit(X, y)\n    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 200),\n                         np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 200))\n    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1].reshape(xx.shape)\n    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu_r', levels=20)\n    ax.contour(xx, yy, Z, levels=[0.5], colors='k')\n    ax.scatter(X[:,0], X[:,1], c=y, cmap='RdBu_r', edgecolor='k', s=30)\n    ax.set_title(f'Logistic + poly grado {poly_deg}')\nplt.tight_layout(); plt.show()"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 4: soglia e trade-off precision/recall</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.metrics import precision_score, recall_score\n\nX, y = load_breast_cancer(return_X_y=True)\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)\nm = Pipeline([('s', StandardScaler()), ('c', LogisticRegression(max_iter=1000))]).fit(Xt, yt)\nprob = m.predict_proba(Xe)[:,1]\n\nprint(f\"{'soglia':>7} {'precision':>10} {'recall':>8}\")\nfor t in [0.2, 0.3, 0.5, 0.7, 0.9]:\n    pred = (prob >= t).astype(int)\n    p = precision_score(ye, pred, zero_division=0)\n    r = recall_score(ye, pred)\n    print(f\"{t:>7.2f} {p:>10.3f} {r:>8.3f}\")"
    }
  ],
  "esercizi": [
    {
      "type": "exercise",
      "difficulty": "easy",
      "title": "Es. 6.1 — Sigmoid manuale",
      "prompt": "<p>Implementa sigmoide senza usare scipy. Deve funzionare su scalari e array numpy. Salvala come funzione <code class=\"inline\">sigmoid</code>.</p>",
      "starter": "import numpy as np\ndef sigmoid(z):\n    # TODO\n    return None\n\nprint(sigmoid(0), sigmoid(np.array([-5, 0, 5])))",
      "check": "import numpy as np\nif abs(sigmoid(0) - 0.5) < 1e-9 and np.allclose(sigmoid(np.array([-5,0,5])), [1/(1+np.exp(5)), 0.5, 1/(1+np.exp(-5))]):\n    print(\"OK\")\nelse:\n    print(\"FAIL\")",
      "solution": "def sigmoid(z):\n    return 1 / (1 + np.exp(-z))"
    },
    {
      "type": "exercise",
      "difficulty": "medium",
      "title": "Es. 6.2 — Cross-entropy",
      "prompt": "<p>Implementa <code class=\"inline\">binary_ce(y_true, y_prob)</code> che ritorna la media di $-[y\\log p + (1-y)\\log(1-p)]$. Attenzione a clippare $p$ in $[\\epsilon, 1-\\epsilon]$.</p>",
      "starter": "import numpy as np\ndef binary_ce(y_true, y_prob, eps=1e-9):\n    # TODO\n    return 0.0\n\nprint(binary_ce(np.array([1,0,1,1]), np.array([0.9,0.1,0.8,0.6])))",
      "check": "import numpy as np\nfrom sklearn.metrics import log_loss\ny = np.array([1,0,1,1]); p = np.array([0.9,0.1,0.8,0.6])\nif abs(binary_ce(y, p) - log_loss(y, p)) < 1e-6:\n    print(\"OK — identica a sklearn\")\nelse:\n    print(\"FAIL\")",
      "solution": "def binary_ce(y_true, y_prob, eps=1e-9):\n    p = np.clip(y_prob, eps, 1-eps)\n    return -np.mean(y_true*np.log(p) + (1-y_true)*np.log(1-p))"
    },
    {
      "type": "exercise",
      "difficulty": "medium",
      "title": "Es. 6.3 — Soglia ottima per F1",
      "prompt": "<p>Data una serie di probabilità e labels, trova la soglia in <code class=\"inline\">np.linspace(0,1,101)</code> che massimizza F1 sul test. Salva in <code class=\"inline\">best_thresh</code> (float).</p>",
      "starter": "import numpy as np\nfrom sklearn.metrics import f1_score\nnp.random.seed(0)\ny_true = np.random.randint(0,2,size=500)\nprob = np.clip(0.5 + 0.4*(2*y_true-1) + 0.3*np.random.randn(500), 0, 1)\n\n# TODO\nbest_thresh = None\nprint(\"miglior soglia:\", best_thresh)",
      "check": "import numpy as np\nfrom sklearn.metrics import f1_score\nscores = {t: f1_score(y_true, (prob>=t).astype(int), zero_division=0)\n          for t in np.linspace(0,1,101)}\n_atteso = max(scores, key=scores.get)\nif best_thresh is not None and abs(best_thresh - _atteso) < 0.02:\n    print(f\"OK — soglia {best_thresh:.2f}, F1={scores[_atteso]:.3f}\")\nelse:\n    print(f\"FAIL — atteso {_atteso:.2f}\")",
      "solution": "import numpy as np\nfrom sklearn.metrics import f1_score\nbest_thresh = max(np.linspace(0,1,101),\n                  key=lambda t: f1_score(y_true, (prob>=t).astype(int), zero_division=0))"
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Es. 6.4 — Multi-classe su Iris",
      "prompt": "<p>Addestra una regressione logistica multi-classe su Iris. Standardizza, valuta test accuracy e ROC-AUC (one-vs-rest). Salvali in <code class=\"inline\">acc</code> e <code class=\"inline\">auc</code>.</p>",
      "starter": "from sklearn.datasets import load_iris\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.metrics import roc_auc_score\nimport numpy as np\n\nX, y = load_iris(return_X_y=True)\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)\n\n# TODO\nacc = None\nauc = None\nprint(f\"Accuracy: {acc}, AUC: {auc}\")",
      "check": "pipe = Pipeline([('s', StandardScaler()), ('c', LogisticRegression(max_iter=1000))]).fit(Xt, yt)\n_acc = pipe.score(Xe, ye)\n_prob = pipe.predict_proba(Xe)\n_auc = roc_auc_score(ye, _prob, multi_class='ovr')\nok = acc is not None and abs(acc-_acc)<1e-4 and abs(auc-_auc)<1e-3\nprint(\"OK\" if ok else f\"FAIL — atteso acc={_acc:.4f}, auc={_auc:.4f}\")",
      "solution": "pipe = Pipeline([('s', StandardScaler()), ('c', LogisticRegression(max_iter=1000))]).fit(Xt, yt)\nacc = pipe.score(Xe, ye)\nauc = roc_auc_score(ye, pipe.predict_proba(Xe), multi_class='ovr')"
    }
  ],
  "test": [
    {
      "type": "quiz",
      "questions": [
        {
          "q": "La sigmoide $\\sigma(z)$ mappa $\\mathbb{R}$ in:",
          "options": [
            "$[-1,1]$",
            "$[0,1]$",
            "$[0,\\infty)$",
            "$\\mathbb{R}$"
          ],
          "answer": 1,
          "explain": "È il suo scopo: schiacciare reali in probabilità."
        },
        {
          "q": "Con classi molto sbilanciate (99% classe 0), un classificatore che predice sempre 0 ha:",
          "options": [
            "Alta accuracy ma recall nullo sui positivi",
            "Alta precision e recall",
            "Bassa accuracy",
            "F1 alto"
          ],
          "answer": 0,
          "explain": "Accuracy 99% ma il modello è inutile: mai identifica positivi."
        },
        {
          "q": "La cross-entropy vs MSE per classificazione binaria:",
          "options": [
            "Sono equivalenti",
            "CE ha gradienti migliori (non satura come MSE)",
            "MSE è preferibile",
            "Nessuna delle due funziona"
          ],
          "answer": 1,
          "explain": "MSE con sigmoide dà gradienti piccoli quando il modello è molto sbagliato; CE no."
        },
        {
          "q": "In scikit-learn, per regolarizzare più fortemente <code class=\"inline\">LogisticRegression</code>:",
          "options": [
            "Aumenta C",
            "Diminuisci C",
            "Cambia solver",
            "Aumenta max_iter"
          ],
          "answer": 1,
          "explain": "C = 1/λ: piccolo C significa forte penalità."
        }
      ]
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Test pratico — classificatore completo",
      "prompt": "<p>Sul dataset breast_cancer, costruisci un classificatore che raggiunga <strong>ROC-AUC test &gt; 0.995</strong>. Puoi usare qualsiasi combinazione di scaling, feature poly, regolarizzazione. Salva l'AUC in <code class=\"inline\">auc_test</code>.</p>",
      "starter": "from sklearn.datasets import load_breast_cancer\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler, PolynomialFeatures\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import roc_auc_score\n\nX, y = load_breast_cancer(return_X_y=True)\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)\n\n# TODO\nauc_test = None\nprint(\"AUC:\", auc_test)",
      "check": "if auc_test is None:\n    print(\"FAIL: None\")\nelif auc_test > 0.995:\n    print(f\"OK — AUC {auc_test:.4f}\")\nelse:\n    print(f\"FAIL — AUC {auc_test:.4f}, target > 0.995\")",
      "solution": "pipe = Pipeline([('s', StandardScaler()), ('c', LogisticRegression(C=1.0, max_iter=5000))]).fit(Xt, yt)\nauc_test = roc_auc_score(ye, pipe.predict_proba(Xe)[:,1])"
    }
  ]
});
