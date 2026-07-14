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
    },
    {
      "type": "md",
      "content": "\n<h3>2.5 Condizionamento e stabilità numerica</h3>\n<p>Il <strong>numero di condizionamento</strong> di una matrice $A$ è $\\kappa(A) = \\|A\\|\\,\\|A^{-1}\\|$. Per matrici simmetriche positive, $\\kappa(A) = \\sigma_{\\max}/\\sigma_{\\min}$. Indica quanto gli errori di input vengono amplificati: se $\\kappa(A) = 10^6$, perdiamo ~6 cifre decimali di precisione.</p>\n<p><strong>Perché $X^TX$ è pericolosa?</strong> Nella soluzione normale $(X^TX)\\hat{w} = X^Ty$, il numero di condizionamento si <em>quadra</em>: $\\kappa(X^TX) = \\kappa(X)^2$. Con features correlate o mal scalate, $X^TX$ può essere quasi-singolare e l'inversione amplifica gli errori numerici di floating point.</p>\n<p><strong>SVD come alternativa stabile.</strong> Da $X = U\\Sigma V^T$, la pseudoinversa è $X^+ = V\\Sigma^+ U^T$ e la soluzione di minimi quadrati diventa $\\hat{w} = V\\Sigma^+ U^T y$. NumPy usa SVD internamente in <code class=\"inline\">np.linalg.lstsq</code>. Non invertire mai $X^TX$ direttamente; usa <code class=\"inline\">lstsq</code> o <code class=\"inline\">scipy.linalg.solve</code>.</p>\n<ul>\n<li>$\\kappa \\approx 1$: matrice ben condizionata (features ortogonali, normalizzate)</li>\n<li>$\\kappa \\gg 1$: matrice mal condizionata (features correlate o scale molto diverse)</li>\n<li>Ridge regression aggiunge $\\lambda I$ prima di invertire: $\\kappa(X^TX + \\lambda I) &lt; \\kappa(X^TX)$, migliorando la stabilità</li>\n</ul>\n\n<h3>2.6 Distribuzioni di probabilità utili nel ML</h3>\n<p><strong>Bernoulli</strong> $\\mathrm{Ber}(p)$: $P(X=1)=p$, $P(X=0)=1-p$. $\\mathbb{E}[X]=p$, $\\mathrm{Var}(X)=p(1-p)$. Modella classificazione binaria; la cross-entropy binaria deriva dal suo MLE.</p>\n<p><strong>Binomiale</strong> $\\mathrm{Bin}(n,p)$: numero di successi in $n$ prove Bernoulli. $P(X=k)=\\binom{n}{k}p^k(1-p)^{n-k}$. $\\mathbb{E}[X]=np$, $\\mathrm{Var}(X)=np(1-p)$.</p>\n<p><strong>Poisson</strong> $\\mathrm{Poi}(\\lambda)$: conta eventi rari in un intervallo. PMF:</p>\n<p>$$P(X=k) = \\frac{\\lambda^k e^{-\\lambda}}{k!}, \\quad k = 0,1,2,\\dots$$</p>\n<p>$\\mathbb{E}[X]=\\lambda$, $\\mathrm{Var}(X)=\\lambda$ (media = varianza: firma distintiva della Poisson). Log-verosimiglianza: $\\ell(\\lambda)=\\sum_i k_i \\log\\lambda - n\\lambda$; derivando: $\\hat{\\lambda}_{\\text{MLE}}=\\bar{k}$.</p>\n<p><strong>Normale multivariata</strong> $\\mathcal{N}(\\mu, \\Sigma)$ con $\\mu \\in \\mathbb{R}^d$, $\\Sigma \\in \\mathbb{R}^{d\\times d}$ simmetrica definita positiva:</p>\n<p>$$p(x) = \\frac{1}{(2\\pi)^{d/2}|\\Sigma|^{1/2}} \\exp\\!\\left(-\\frac{1}{2}(x-\\mu)^T \\Sigma^{-1}(x-\\mu)\\right)$$</p>\n<p>Le curve di livello sono ellissoidi. Gli assi sono gli autovettori di $\\Sigma$; le semiampiezze sono $\\sqrt{\\lambda_i}$. Se $\\Sigma$ è diagonale, le componenti sono indipendenti.</p>\n<p><strong>Esponenziale</strong> $\\mathrm{Exp}(\\lambda)$: $p(x)=\\lambda e^{-\\lambda x}$ per $x\\geq 0$. $\\mathbb{E}[X]=1/\\lambda$, $\\mathrm{Var}(X)=1/\\lambda^2$. Modella tempi di attesa; $\\hat{\\lambda}_{\\text{MLE}}=1/\\bar{x}$.</p>\n\n<h3>2.7 Convessità</h3>\n<p><strong>Definizione formale.</strong> $f : \\mathcal{C} \\to \\mathbb{R}$ è <em>convessa</em> se per ogni $x, y \\in \\mathcal{C}$ e $\\lambda \\in [0,1]$:</p>\n<p>$$f(\\lambda x + (1-\\lambda)y) \\leq \\lambda f(x) + (1-\\lambda) f(y)$$</p>\n<p>Geometricamente: il segmento che unisce due punti del grafico sta <em>sopra</em> (o uguale) al grafico. È <strong>strettamente convessa</strong> se vale $&lt;$ per $\\lambda \\in (0,1)$ e $x \\neq y$. È <strong>quasiconvessa</strong> se gli insiemi di sottolivello $\\{x : f(x) \\leq c\\}$ sono convessi (condizione più debole).</p>\n<p><strong>Implicazione cruciale per ottimizzazione:</strong> per una funzione convessa, <em>ogni minimo locale è anche globale</em>. La discesa del gradiente su una funzione convessa converge (con learning rate appropriato) all'ottimo globale. Condizione equivalente via Hessiana: $f$ convessa $\\Leftrightarrow$ $\\nabla^2 f(x) \\succeq 0$ ovunque.</p>\n<p><strong>Esempi rilevanti nel ML:</strong></p>\n<ul>\n<li><strong>MSE</strong> $\\mathcal{L}(w) = \\|Xw-y\\|^2$: convessa (Hessiana $= 2X^TX \\succeq 0$); strettamente convessa se $X$ ha rango pieno.</li>\n<li><strong>Cross-entropy</strong> $-\\sum_i y_i \\log \\hat{p}_i$: convessa in $w$ per regressione logistica.</li>\n<li><strong>Norma $\\ell_1$</strong> $\\|w\\|_1 = \\sum_j |w_j|$: convessa ma <em>non differenziabile</em> in $w_j=0$. Richiede il <strong>subgradiente</strong>: $\\partial|w_j| = \\mathrm{sign}(w_j)$ se $w_j \\neq 0$, qualsiasi valore in $[-1,1]$ se $w_j=0$.</li>\n<li><strong>Reti neurali</strong>: non convesse in generale &rarr; ottimi locali, saddle point; GD può comunque trovare soluzioni buone in pratica.</li>\n</ul>\n\n<h3>2.8 Regola della catena multivariata e grafo computazionale</h3>\n<p>Per una composizione $L = g(f(w))$ con $f : \\mathbb{R}^d \\to \\mathbb{R}^k$ e $g : \\mathbb{R}^k \\to \\mathbb{R}$, la regola della catena multivariata è:</p>\n<p>$$\\frac{\\partial L}{\\partial w_i} = \\sum_{j=1}^k \\frac{\\partial L}{\\partial f_j} \\cdot \\frac{\\partial f_j}{\\partial w_i}$$</p>\n<p>In forma matriciale, il <strong>Jacobiano</strong> di $f$ è $J_f \\in \\mathbb{R}^{k\\times d}$ con $(J_f)_{ji} = \\partial f_j / \\partial w_i$. La regola diventa $\\nabla_w L = J_f^T \\nabla_f L$.</p>\n<p>Il <strong>grafo computazionale</strong> è un DAG dove ogni nodo è un'operazione intermedia. Il <em>forward pass</em> calcola le attivazioni da input a output; il <em>backward pass</em> (backpropagation) applica la regola della catena all'indietro, riusando i valori del forward pass. PyTorch/JAX automatizzano questo con differenziazione automatica (<em>autograd</em>).</p>\n<p><strong>Esempio concreto</strong> (regressione logistica): con $z = Xw$, $\\hat{y} = \\sigma(z)$, $L = -\\sum_i y_i \\log \\hat{y}_i$:</p>\n<ul>\n<li>$\\partial L / \\partial \\hat{y}_i = -y_i/\\hat{y}_i$</li>\n<li>$\\partial \\hat{y} / \\partial z = \\sigma(z)(1-\\sigma(z))$</li>\n<li>$\\partial z / \\partial w = X^T$</li>\n<li>Risultato via chain rule: $\\nabla_w L = X^T (\\hat{y} - y)$</li>\n</ul>\n\n<h3>2.9 SGD e varianti</h3>\n<p><strong>Full batch GD</strong>: usa tutti gli $n$ esempi per calcolare $\\nabla \\mathcal{L}$. Gradiente esatto, ma costoso per $n$ grande. <strong>Stocastico (SGD)</strong>: un esempio alla volta; veloce ma rumoroso. <strong>Mini-batch</strong>: compromesso standard (batch size 32–256); sfrutta parallelismo GPU.</p>\n<p><strong>Momentum</strong>: accumula una media esponenzialmente pesata delle direzioni passate, smorzando le oscillazioni:</p>\n<p>$$v_t = \\beta\\, v_{t-1} + \\nabla_\\theta \\mathcal{L}(\\theta_t), \\qquad \\theta_{t+1} = \\theta_t - \\eta\\, v_t$$</p>\n<p>con $\\beta \\approx 0.9$ tipicamente. Intuizione: come una palla che rotola su un piano inclinato, accumulando velocità nella direzione giusta e smorzando i rimbalzi trasversali.</p>\n<p><strong>Adam</strong> (Adaptive Moment Estimation): stima il primo momento (media del gradiente) e il secondo momento (varianza non centrata), adattando il learning rate per ogni parametro:</p>\n<p>$$m_t = \\beta_1 m_{t-1} + (1-\\beta_1)g_t, \\quad v_t = \\beta_2 v_{t-1} + (1-\\beta_2)g_t^2$$</p>\n<p>$$\\hat{m}_t = \\frac{m_t}{1-\\beta_1^t}, \\quad \\hat{v}_t = \\frac{v_t}{1-\\beta_2^t}, \\quad \\theta_{t+1} = \\theta_t - \\eta\\, \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t}+\\epsilon}$$</p>\n<p>Default: $\\beta_1=0.9$, $\\beta_2=0.999$, $\\epsilon=10^{-8}$. Adam è insensibile alla scala dei gradienti e funziona bene con scarsa tuning del learning rate.</p>\n<p><strong>Learning rate schedule</strong>: $\\eta$ costante spesso non è ottimale. Strategie comuni:</p>\n<ul>\n<li><strong>Step decay</strong>: dimezza $\\eta$ ogni $k$ epoche.</li>\n<li><strong>Cosine annealing</strong>: $\\eta_t = \\eta_{\\min} + \\frac{1}{2}(\\eta_{\\max}-\\eta_{\\min})\\left(1+\\cos\\frac{\\pi t}{T}\\right)$. Scende dolcemente, spesso con warm restart.</li>\n<li><strong>Warmup</strong>: inizia con $\\eta$ piccolo e lo aumenta per le prime epoche (importante per Transformer).</li>\n</ul>\n\n<h3>2.10 Norme e regolarizzazione</h3>\n<p>Aggiungere una penalità alla loss contrasta l'overfitting vincolando la magnitudine dei pesi.</p>\n<p><strong>Regolarizzazione $\\ell_2$ (Ridge):</strong></p>\n<p>$$\\mathcal{L}_{\\text{ridge}}(w) = \\|Xw-y\\|^2 + \\lambda\\|w\\|_2^2$$</p>\n<p>Gradiente: $2X^T(Xw-y) + 2\\lambda w$. Soluzione closed-form: $\\hat{w} = (X^TX+\\lambda I)^{-1}X^Ty$. Effetto: <em>shrinkage uniforme</em> verso zero — pesi piccoli ma raramente esattamente zero. Aggiungere $\\lambda I$ migliora anche il condizionamento numerico.</p>\n<p><strong>Regolarizzazione $\\ell_1$ (Lasso):</strong></p>\n<p>$$\\mathcal{L}_{\\text{lasso}}(w) = \\|Xw-y\\|^2 + \\lambda\\|w\\|_1$$</p>\n<p>La norma $\\|w\\|_1$ è non differenziabile in $w_j=0$; il <strong>subgradiente</strong> è $\\mathrm{sign}(w_j)$ per $w_j \\neq 0$ e qualsiasi valore in $[-1,1]$ per $w_j=0$. Questo crea un \"pull\" esatto verso zero: molti pesi diventano <em>esattamente</em> zero &rarr; soluzione <strong>sparsa</strong> (selezione automatica delle features).</p>\n<p><strong>Intuizione geometrica:</strong> i contour della loss si intersecano con la \"palla\" $\\ell_1$ (rombo in 2D) preferibilmente sugli <em>spigoli</em>, dove qualche coordinata è zero. La palla $\\ell_2$ (cerchio) non ha spigoli: l'intersezione raramente cade sugli assi coordinati.</p>\n<p><strong>Elastic Net</strong>: combina entrambe, $\\lambda_1 \\|w\\|_1 + \\lambda_2 \\|w\\|_2^2$, ottenendo sparsità e stabilità numerica insieme.</p>\n"
    },
    {
      "type": "callout",
      "variant": "warn",
      "title": "Non invertire mai X^TX direttamente",
      "content": "Usa sempre np.linalg.lstsq o scipy.linalg.solve invece di np.linalg.inv(X.T @ X). lstsq usa SVD internamente e gestisce correttamente matrici mal condizionate o rango-deficienti, evitando errori numerici silenti."
    },
    {
      "type": "callout",
      "variant": "tip",
      "title": "Adam vs SGD: quale scegliere?",
      "content": "Adam converge più in fretta e richiede meno tuning del learning rate. SGD con momentum tende a generalizzare leggermente meglio in alcuni contesti (spesso preferito per CNN vision). Regola pratica: usa Adam per prototipare rapidamente e per reti NLP/Transformer; SGD+momentum per architetture deep come ResNet quando la generalizzazione è prioritaria."
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
