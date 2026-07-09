/* ML Academy — contenuto lezioni */
const LESSONS = [];

// =====================================================================
// LEZIONE 1 — Cos'è il Machine Learning
// =====================================================================
LESSONS.push({
  id: 'l01-intro',
  title: 'Cos\'è il Machine Learning',
  tag: 'Fondamenti',
  summary: 'Definizioni, tipi di apprendimento, terminologia essenziale.',
  teoria: [
    { type: 'md', content: `
<h3>1.1 Definizione operativa</h3>
<p>Il machine learning è la disciplina che studia algoritmi capaci di <strong>migliorare le proprie prestazioni su un compito attraverso l'esperienza</strong>. La definizione classica di Tom Mitchell (1997) è quasi una formula: un programma apprende dall'esperienza $E$ rispetto a un compito $T$ misurato da $P$ se le prestazioni su $T$, misurate da $P$, migliorano con $E$.</p>
<p>In termini fisici puoi pensarci così: non stai scrivendo l'equazione del moto a mano, stai facendo <em>fit di una funzione</em> a partire dai dati. La differenza rispetto al fit classico è che le funzioni che consideriamo hanno milioni di parametri e non hanno una struttura fisica imposta a priori.</p>

<h3>1.2 Il paradigma: dai dati alla funzione</h3>
<p>In programmazione tradizionale scrivi: <code class="inline">regole + dati &rarr; risultati</code>. Nel machine learning inverti: <code class="inline">dati + risultati &rarr; regole</code>. L'algoritmo di apprendimento è quindi un <em>meta-programma</em> che produce un programma (il modello) a partire da esempi.</p>

<h3>1.3 I tre grandi paradigmi</h3>
<p><strong>Apprendimento supervisionato.</strong> Hai coppie $(x_i, y_i)$ dove $x_i \\in \\mathbb{R}^d$ è un vettore di features e $y_i$ è l'etichetta. Vuoi imparare $f : \\mathbb{R}^d \\to \\mathcal{Y}$ tale che $f(x) \\approx y$ su nuovi dati. Se $\\mathcal{Y}$ è discreto è <em>classificazione</em>; se è continuo è <em>regressione</em>.</p>
<p><strong>Apprendimento non supervisionato.</strong> Hai solo $x_i$, senza etichette. Vuoi scoprire struttura: cluster, direzioni principali di variabilità, anomalie.</p>
<p><strong>Reinforcement learning.</strong> Un agente interagisce con un ambiente, osserva stati $s_t$, sceglie azioni $a_t$, riceve ricompense $r_t$. Vuoi imparare una policy $\\pi(a|s)$ che massimizza la ricompensa cumulativa attesa $\\mathbb{E}\\left[\\sum_t \\gamma^t r_t\\right]$.</p>

<h3>1.4 Formalizzazione dell'apprendimento supervisionato</h3>
<p>Assumiamo che i dati provengano da una distribuzione congiunta ignota $\\mathcal{D}$ su $\\mathcal{X} \\times \\mathcal{Y}$. Data una funzione di loss $\\ell(\\hat{y}, y)$, definiamo il <em>rischio atteso</em>:</p>
<p>$$R(f) = \\mathbb{E}_{(x,y) \\sim \\mathcal{D}}\\left[\\ell(f(x), y)\\right]$$</p>
<p>Non conosciamo $\\mathcal{D}$, quindi minimizziamo il <em>rischio empirico</em> su un training set $\\{(x_i, y_i)\\}_{i=1}^n$:</p>
<p>$$\\hat{R}(f) = \\frac{1}{n}\\sum_{i=1}^n \\ell(f(x_i), y_i)$$</p>
<p>Il grande tema del ML è: quando il minimo di $\\hat{R}$ è vicino al minimo di $R$? Questa è la questione della <em>generalizzazione</em>.</p>

<h3>1.5 Terminologia essenziale</h3>
<ul>
<li><strong>Feature</strong> (o predittore, variabile indipendente): una componente di $x$.</li>
<li><strong>Label</strong> (o target, variabile dipendente): $y$.</li>
<li><strong>Modello</strong>: la funzione $f_\\theta$ parametrizzata da $\\theta$.</li>
<li><strong>Training</strong>: il processo di trovare $\\theta$ ottimale.</li>
<li><strong>Inference</strong>: usare il modello addestrato per fare previsioni.</li>
<li><strong>Overfitting</strong>: il modello memorizza il training set ma non generalizza. $\\hat{R}$ piccolo, $R$ grande.</li>
<li><strong>Underfitting</strong>: il modello è troppo semplice per catturare il pattern. Entrambi grandi.</li>
</ul>
` },
    { type: 'callout', variant: 'note', title: 'Analogia con la fisica', content: 'Overfitting è come fittare un polinomio di grado 20 su 10 punti sperimentali: passi per tutti i punti ma il modello non ha significato fisico. Il bias-variance tradeoff che vedremo dopo è essenzialmente la stessa questione della scelta del grado del polinomio, generalizzata.' },
  ],
  esempi: [
    { type: 'md', content: '<h3>Esempio: da dati a modello in 10 righe</h3><p>Prendiamo un problema banale (predire il prezzo di una casa dalla sua superficie) e vediamo l\'intero flusso.</p>' },
    { type: 'run', content: `import numpy as np
from sklearn.linear_model import LinearRegression

# Dati sintetici: superficie in mq -> prezzo in migliaia di euro
X = np.array([[50], [65], [80], [100], [120], [150], [180], [220]])
y = np.array([120, 155, 190, 235, 280, 340, 400, 480])

# Modello: cerchiamo f(x) = w*x + b
model = LinearRegression()
model.fit(X, y)

print(f"Coefficiente (w): {model.coef_[0]:.3f}")
print(f"Intercetta  (b): {model.intercept_:.3f}")
print(f"Predizione per 90 mq: {model.predict([[90]])[0]:.1f} k€")
print(f"R^2 sul training: {model.score(X, y):.4f}")` },
    { type: 'md', content: '<p>Nota: abbiamo scelto una famiglia di funzioni (retta), abbiamo trovato i parametri che minimizzano l\'errore quadratico, abbiamo valutato. Questo è il pattern che ripeteremo ossessivamente, cambiando solo la famiglia di funzioni e la loss.</p><h3>Esempio: classificazione binaria</h3>' },
    { type: 'run', content: `from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                            n_redundant=0, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

clf = LogisticRegression()
clf.fit(X_tr, y_tr)

print(f"Accuracy training: {clf.score(X_tr, y_tr):.3f}")
print(f"Accuracy test:     {clf.score(X_te, y_te):.3f}")
print("Prima predizione:", clf.predict(X_te[:1]), "vera:", y_te[:1])` },
  ],
  esercizi: [
    { type: 'exercise', difficulty: 'easy', title: 'Es. 1.1 — Classifica il problema',
      prompt: '<p>Per ognuno dei seguenti scenari, decidi se è <strong>regressione</strong>, <strong>classificazione binaria</strong>, <strong>classificazione multi-classe</strong>, <strong>clustering</strong>, o <strong>reinforcement learning</strong>. Modifica il dizionario nel codice e poi clicca "Verifica soluzione".</p><ol><li>Predire il consumo elettrico di una casa domani (kWh)</li><li>Riconoscere se un\'email è spam</li><li>Raggruppare clienti simili senza avere una categoria predefinita</li><li>Riconoscere la cifra scritta a mano (0-9)</li><li>Allenare un bot a giocare a scacchi</li></ol>',
      starter: `risposte = {
    1: "?",  # scegli tra: regressione, binaria, multi, clustering, rl
    2: "?",
    3: "?",
    4: "?",
    5: "?",
}
for k, v in risposte.items():
    print(f"{k}: {v}")`,
      check: `_atteso = {1:"regressione", 2:"binaria", 3:"clustering", 4:"multi", 5:"rl"}
_giusti = sum(1 for k in _atteso if risposte.get(k) == _atteso[k])
print(f"\\n--- {_giusti}/5 corrette ---")
for k,v in _atteso.items():
    ok = "OK" if risposte.get(k)==v else "NO"
    print(f"[{ok}] {k}: atteso '{v}', tuo '{risposte.get(k)}'")`,
      solution: `risposte = {1: "regressione", 2: "binaria", 3: "clustering", 4: "multi", 5: "rl"}` },

    { type: 'exercise', difficulty: 'easy', title: 'Es. 1.2 — Calcola rischio empirico',
      prompt: '<p>Data una lista di predizioni <code class="inline">y_pred</code> e valori veri <code class="inline">y_true</code>, calcola il rischio empirico con loss quadratica $\\ell(\\hat{y}, y) = (\\hat{y}-y)^2$. Il risultato deve essere in <code class="inline">rischio</code>.</p>',
      starter: `y_true = [3.0, 5.0, 2.5, 7.0, 4.5]
y_pred = [2.8, 5.2, 3.0, 6.5, 4.7]

# calcola qui il rischio empirico (media dei quadrati degli scarti)
rischio = None

print("Rischio empirico:", rischio)`,
      check: `_atteso = sum((a-b)**2 for a,b in zip(y_true,y_pred)) / len(y_true)
if rischio is None:
    print("FAIL: rischio è None")
elif abs(rischio - _atteso) < 1e-6:
    print(f"OK! Valore corretto: {rischio:.4f}")
else:
    print(f"FAIL: atteso {_atteso:.4f}, ottenuto {rischio}")`,
      solution: `rischio = sum((yp-yt)**2 for yp,yt in zip(y_pred,y_true)) / len(y_true)` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 1.3 — Overfitting visibile',
      prompt: '<p>Usa <code class="inline">numpy.polyfit</code> per fittare un polinomio di grado 1 e uno di grado 15 sugli stessi 10 punti rumorosi. Stampa gli errori MSE sul training e su un test set separato. Osserva la differenza.</p>',
      starter: `import numpy as np
np.random.seed(0)
x_tr = np.linspace(0, 1, 10)
y_tr = np.sin(2*np.pi*x_tr) + 0.1*np.random.randn(10)
x_te = np.linspace(0, 1, 100)
y_te = np.sin(2*np.pi*x_te)

# fitta polinomio grado 1 e grado 15, calcola MSE su train e test
# usa np.polyfit(x, y, deg) e np.polyval(coef, x)

# TODO
`,
      check: `import numpy as np
c1 = np.polyfit(x_tr, y_tr, 1); c15 = np.polyfit(x_tr, y_tr, 15)
tr1 = np.mean((np.polyval(c1,x_tr)-y_tr)**2); te1 = np.mean((np.polyval(c1,x_te)-y_te)**2)
tr15 = np.mean((np.polyval(c15,x_tr)-y_tr)**2); te15 = np.mean((np.polyval(c15,x_te)-y_te)**2)
print(f"grado 1  : MSE train {tr1:.4f}, test {te1:.4f}")
print(f"grado 15 : MSE train {tr15:.4f}, test {te15:.4f}")
print("\\nOsserva: grado 15 ha MSE train quasi zero ma test enorme -> OVERFITTING")`,
      solution: `import numpy as np
c1  = np.polyfit(x_tr, y_tr, 1)
c15 = np.polyfit(x_tr, y_tr, 15)
for c, deg in [(c1,1),(c15,15)]:
    tr = np.mean((np.polyval(c,x_tr)-y_tr)**2)
    te = np.mean((np.polyval(c,x_te)-y_te)**2)
    print(f"grado {deg}: MSE train {tr:.4f}, test {te:.4f}")` },
  ],
  test: [
    { type: 'md', content: '<h3>Test di comprensione</h3><p>Rispondi alle domande. Alla fine, l\'esercizio pratico verifica l\'applicazione.</p>' },
    { type: 'quiz', questions: [
      { q: 'Un algoritmo che raggruppa articoli di giornale in temi senza conoscere le categorie è un esempio di:', options: ['Supervisionato', 'Non supervisionato', 'Reinforcement learning', 'Semi-supervisionato'], answer: 1, explain: 'Non ci sono etichette, cerchiamo struttura: clustering, non supervisionato.' },
      { q: 'Il rischio empirico $\\hat{R}$ è:', options: ['La stessa cosa del rischio atteso $R$', 'Il valore di $R$ sul test set', 'La media della loss sul training set', 'La varianza delle predizioni'], answer: 2, explain: 'Per definizione, $\\hat{R} = \\frac{1}{n}\\sum_i \\ell(f(x_i), y_i)$ sul training set.' },
      { q: 'Un modello con training accuracy 99% e test accuracy 60% è tipicamente:', options: ['Underfit', 'Overfit', 'Ben calibrato', 'Non identificabile'], answer: 1, explain: 'Grande gap train-test è la firma dell\'overfitting.' },
      { q: 'Nella regressione lineare $y = wx + b$, i parametri appresi sono:', options: ['Solo $w$', 'Solo $b$', '$w$ e $b$', '$x$ e $y$'], answer: 2, explain: '$x, y$ sono i dati (fissi); $w, b$ sono i parametri da apprendere.' },
    ] },
    { type: 'md', content: '<h3>Notebook finale</h3><p>Applica quanto imparato: costruisci un modello che predice se il numero passato è pari usando la sua rappresentazione binaria. Segui lo scheletro.</p>' },
    { type: 'exercise', difficulty: 'medium', title: 'Test pratico',
      prompt: '<p>Crea un classificatore che, dati 100 interi tra 0 e 255 rappresentati come vettore binario di 8 bit, predica se il numero è pari. Usa <code class="inline">LogisticRegression</code> di scikit-learn. Deve raggiungere <strong>almeno 95% accuracy</strong> sul test.</p>',
      starter: `import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

np.random.seed(0)
nums = np.random.randint(0, 256, size=200)

def to_bits(n):
    return [(n >> i) & 1 for i in range(8)]

X = np.array([to_bits(n) for n in nums])
y = (nums % 2 == 0).astype(int)

# TODO: split, fit, valuta test accuracy in variable 'acc'
acc = 0.0
print("Test accuracy:", acc)`,
      check: `if acc >= 0.95:
    print(f"OK — accuracy {acc*100:.1f}%. Nota: il primo bit E' la parità, il modello lo scopre.")
else:
    print(f"FAIL — accuracy {acc*100:.1f}%, servono >= 95%.")`,
      solution: `X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
clf = LogisticRegression().fit(X_tr, y_tr)
acc = clf.score(X_te, y_te)
print("Test accuracy:", acc)` },
  ]
});

// =====================================================================
// LEZIONE 2 — Ripasso matematico per ML
// =====================================================================
LESSONS.push({
  id: 'l02-math',
  title: 'Ripasso matematico',
  tag: 'Fondamenti',
  summary: 'Algebra lineare, calcolo, probabilità, ottimizzazione — visti dalla lente del ML.',
  teoria: [
    { type: 'md', content: `
<h3>2.1 Vettori, matrici, tensori</h3>
<p>Un dataset è tipicamente una matrice $X \\in \\mathbb{R}^{n \\times d}$: $n$ righe (esempi), $d$ colonne (features). Ogni riga $x_i \\in \\mathbb{R}^d$ è un vettore di features. Le operazioni fondamentali sono:</p>
<ul>
<li><strong>Prodotto scalare</strong>: $x \\cdot w = \\sum_j x_j w_j = x^T w$. Misura quanto due vettori "puntano nella stessa direzione".</li>
<li><strong>Norma</strong>: $\\|x\\|_2 = \\sqrt{\\sum_j x_j^2}$. La versione $\\ell_1$ è $\\|x\\|_1 = \\sum_j |x_j|$.</li>
<li><strong>Prodotto matrice-vettore</strong>: $Xw \\in \\mathbb{R}^n$. Ogni riga è $x_i^T w$: le predizioni di un modello lineare.</li>
<li><strong>Trasposta</strong>: $(X^T)_{ij} = X_{ji}$. Utile per pratica di broadcasting e per la formula $X^T X$ che ricorre ovunque.</li>
</ul>

<h4>Autovalori e SVD</h4>
<p>Data $A \\in \\mathbb{R}^{d\\times d}$ simmetrica semi-definita positiva (come una matrice di covarianza), esiste una decomposizione $A = V \\Lambda V^T$ con $V$ ortogonale e $\\Lambda$ diagonale con autovalori $\\lambda_1 \\geq \\dots \\geq \\lambda_d \\geq 0$. Questo è il fondamento di PCA.</p>
<p>Per matrici rettangolari $X \\in \\mathbb{R}^{n\\times d}$: <em>Singular Value Decomposition</em> $X = U \\Sigma V^T$ con $U, V$ ortogonali e $\\Sigma$ diagonale (allargata) con valori singolari $\\sigma_1 \\geq \\dots \\geq 0$. SVD è forse lo strumento più utile di algebra lineare numerica.</p>

<h3>2.2 Calcolo differenziale in $\\mathbb{R}^d$</h3>
<p>Il <strong>gradiente</strong> di $f : \\mathbb{R}^d \\to \\mathbb{R}$ è il vettore delle derivate parziali:</p>
<p>$$\\nabla f(x) = \\left(\\frac{\\partial f}{\\partial x_1}, \\dots, \\frac{\\partial f}{\\partial x_d}\\right)^T$$</p>
<p>Punta nella direzione di massima crescita. La discesa del gradiente muove $x$ nella direzione $-\\nabla f$ perché è quella di massima decrescita locale.</p>
<p>Formule utili (che ti serviranno spesso):</p>
<ul>
<li>$\\nabla_x (a^T x) = a$</li>
<li>$\\nabla_x (x^T A x) = (A + A^T) x$, che per $A$ simmetrica dà $2Ax$</li>
<li>$\\nabla_x \\|Xx - y\\|^2 = 2 X^T (Xx - y)$ &larr; questa è il gradiente della loss quadratica</li>
</ul>
<p>L'<strong>Hessiana</strong> $H = \\nabla^2 f$ è la matrice delle derivate seconde. $f$ è convessa se e solo se $H \\succeq 0$ ovunque.</p>

<h3>2.3 Probabilità in versione ML-friendly</h3>
<p>Una <strong>variabile aleatoria</strong> $X$ ha una distribuzione. Concetti chiave:</p>
<ul>
<li><strong>Attesa</strong>: $\\mathbb{E}[X] = \\int x\\, p(x)\\, dx$. Linearità: $\\mathbb{E}[aX+bY]=a\\mathbb{E}[X]+b\\mathbb{E}[Y]$.</li>
<li><strong>Varianza</strong>: $\\mathrm{Var}(X) = \\mathbb{E}[(X-\\mathbb{E}X)^2]$.</li>
<li><strong>Probabilità condizionata</strong>: $p(y|x) = p(x,y)/p(x)$.</li>
<li><strong>Bayes</strong>: $p(y|x) = \\dfrac{p(x|y)\\, p(y)}{p(x)}$. Il ML "generativo" modella $p(x|y)$; quello "discriminativo" modella direttamente $p(y|x)$.</li>
<li><strong>Gaussiana</strong>: $\\mathcal{N}(x; \\mu, \\sigma^2) = \\frac{1}{\\sqrt{2\\pi}\\sigma} e^{-(x-\\mu)^2/(2\\sigma^2)}$. Multivariata: $\\mathcal{N}(x; \\mu, \\Sigma) \\propto \\exp(-\\frac{1}{2}(x-\\mu)^T \\Sigma^{-1}(x-\\mu))$.</li>
</ul>

<h4>Massima verosimiglianza</h4>
<p>Se assumiamo che i dati $\\{y_i\\}$ siano i.i.d. da $p(y|\\theta)$, il principio MLE dice: scegli $\\theta$ che massimizza $\\prod_i p(y_i|\\theta)$, o equivalentemente:</p>
<p>$$\\hat{\\theta}_{\\text{MLE}} = \\arg\\max_\\theta \\sum_i \\log p(y_i | \\theta)$$</p>
<p>Molte loss standard (MSE, cross-entropy) derivano da MLE sotto specifiche ipotesi.</p>

<h3>2.4 Ottimizzazione: gradient descent</h3>
<p>Iterazione base:</p>
<p>$$\\theta_{t+1} = \\theta_t - \\eta\\, \\nabla_\\theta \\mathcal{L}(\\theta_t)$$</p>
<p>dove $\\eta > 0$ è il <em>learning rate</em>. Convergenza garantita se $\\mathcal{L}$ è convessa e $\\eta$ è piccolo abbastanza (regola di Lipschitz).</p>
<p><strong>Stochastic Gradient Descent (SGD)</strong>: al posto del gradiente esatto sul dataset intero, si usa il gradiente su un mini-batch. Molto più veloce, converge in media alla soluzione se $\\eta$ decresce.</p>
` },
    { type: 'callout', variant: 'tip', title: 'Trucco di calcolo', content: 'Per derivare formule matriciali, tratta $x^T A y$ come un numero, differenziato una variabile alla volta. Se qualcosa "non torna dimensionalmente", non torna proprio. È come l\'analisi dimensionale in fisica.' },
  ],
  esempi: [
    { type: 'md', content: '<h3>Esempio 1: operazioni con NumPy</h3><p>Il codice sotto mostra le operazioni fondamentali. Cambia i numeri e riesegui per prenderci confidenza.</p>' },
    { type: 'run', content: `import numpy as np

# vettori
x = np.array([1.0, 2.0, 3.0])
w = np.array([0.5, -1.0, 2.0])

print("prodotto scalare:", x @ w)         # = 1*0.5 + 2*(-1) + 3*2 = 4.5
print("norma L2 di x:", np.linalg.norm(x))
print("norma L1 di x:", np.linalg.norm(x, 1))

# matrici
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])  # 4x3
print("X @ w =", X @ w)                    # 4 predizioni lineari

# X^T X è simmetrica 3x3
XtX = X.T @ X
print("X^T X:\\n", XtX)

# autovalori
vals, vecs = np.linalg.eig(XtX)
print("autovalori:", np.round(np.sort(vals)[::-1], 3))` },
    { type: 'md', content: '<h3>Esempio 2: gradient descent a mano</h3><p>Minimizziamo $f(x) = (x-3)^2 + 1$ senza scikit-learn. Il minimo è ovviamente $x^*=3$.</p>' },
    { type: 'run', content: `def f(x):     return (x-3)**2 + 1
def grad(x):  return 2*(x-3)

x = 0.0
eta = 0.1
for step in range(30):
    x = x - eta * grad(x)
    if step % 5 == 0:
        print(f"step {step:2d}: x={x:.5f}, f(x)={f(x):.5f}")
print(f"convergenza a x = {x:.6f} (atteso 3.0)")` },
    { type: 'md', content: '<h3>Esempio 3: MLE di una gaussiana</h3><p>Dati $y_i \\sim \\mathcal{N}(\\mu, \\sigma^2)$, la stima MLE è $\\hat\\mu = \\bar y$, $\\hat\\sigma^2 = \\frac{1}{n}\\sum (y_i - \\bar y)^2$. Verifichiamolo:</p>' },
    { type: 'run', content: `import numpy as np
np.random.seed(1)
y = np.random.normal(loc=5.0, scale=2.0, size=1000)

mu_hat = y.mean()
sigma2_hat = np.mean((y - mu_hat)**2)
print(f"mu_hat = {mu_hat:.3f} (vero 5.0)")
print(f"sigma_hat = {np.sqrt(sigma2_hat):.3f} (vero 2.0)")` },
  ],
  esercizi: [
    { type: 'exercise', difficulty: 'easy', title: 'Es. 2.1 — Prodotto scalare a mano',
      prompt: '<p>Implementa il prodotto scalare tra due liste Python senza usare numpy. Deve funzionare per vettori di qualunque lunghezza uguale.</p>',
      starter: `def dot(a, b):
    # TODO
    return None

print(dot([1,2,3], [4,5,6]))  # atteso 32`,
      check: `if dot([1,2,3],[4,5,6]) == 32 and dot([1.0,-1.0],[2.0,3.0]) == -1.0:
    print("OK")
else:
    print("FAIL")`,
      solution: `def dot(a, b):
    return sum(x*y for x, y in zip(a, b))` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 2.2 — Gradient descent multivariato',
      prompt: '<p>Minimizza $f(x, y) = (x-1)^2 + 3(y+2)^2$ partendo da $(0, 0)$ con learning rate $0.1$ per 50 passi. Salva il risultato in <code class="inline">xy_min</code> (tupla di 2 valori).</p>',
      starter: `import numpy as np
def f(v):    return (v[0]-1)**2 + 3*(v[1]+2)**2
def grad(v): return np.array([2*(v[0]-1), 6*(v[1]+2)])

v = np.array([0.0, 0.0])
# TODO: loop di 50 iterazioni

xy_min = tuple(v)
print("min trovato:", xy_min)`,
      check: `import numpy as np
target = np.array([1.0, -2.0])
if np.allclose(xy_min, target, atol=1e-3):
    print(f"OK — trovato {xy_min}, atteso (1, -2)")
else:
    print(f"FAIL — trovato {xy_min}, atteso circa (1, -2)")`,
      solution: `for _ in range(50):
    v = v - 0.1 * grad(v)
xy_min = tuple(v)` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 2.3 — Autovalori e direzione dominante',
      prompt: '<p>Data una matrice di covarianza $\\Sigma = \\begin{pmatrix}4 & 2\\\\ 2 & 3\\end{pmatrix}$, trova l\'autovettore associato all\'autovalore più grande. Salvalo in <code class="inline">v_max</code> (array di 2 elementi).</p>',
      starter: `import numpy as np
Sigma = np.array([[4, 2],[2, 3]])

# TODO: trova autovalori e autovettori, seleziona il maggiore
v_max = np.array([0.0, 0.0])
print("autovettore dominante:", v_max)`,
      check: `import numpy as np
vals, vecs = np.linalg.eigh(Sigma)
target = vecs[:, np.argmax(vals)]
# autovettore definito a meno di segno
if np.allclose(v_max, target, atol=1e-4) or np.allclose(v_max, -target, atol=1e-4):
    print("OK — direzione corretta")
else:
    print(f"FAIL — tuo {v_max}, atteso ±{target}")`,
      solution: `vals, vecs = np.linalg.eigh(Sigma)
v_max = vecs[:, np.argmax(vals)]` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 2.4 — MLE per Poisson',
      prompt: '<p>Dati $k_1, \\dots, k_n$ da $\\mathrm{Poisson}(\\lambda)$, dimostra prima su carta che $\\hat\\lambda_{\\text{MLE}} = \\bar k$. Poi verifica numericamente su un campione simulato che il valore stimato converge al vero $\\lambda$ al crescere di $n$.</p>',
      starter: `import numpy as np
np.random.seed(0)
vero_lambda = 3.7
for n in [10, 100, 1000, 10000]:
    sample = np.random.poisson(vero_lambda, size=n)
    # TODO: calcola stima MLE
    stima = 0.0
    print(f"n={n:5d}  stima={stima:.4f}  err={abs(stima-vero_lambda):.4f}")`,
      check: `# ricomputiamo per verifica
np.random.seed(0)
_last = None
for n in [10, 100, 1000, 10000]:
    _last = np.random.poisson(vero_lambda, size=n).mean()
print(f"\\n(l'ultima stima con n=10000 deve essere vicina a {vero_lambda})")
if abs(_last - vero_lambda) < 0.1:
    print("OK — convergenza verificata")`,
      solution: `stima = sample.mean()` },
  ],
  test: [
    { type: 'quiz', questions: [
      { q: 'Data $f(x) = \\|Ax-b\\|^2$, il gradiente rispetto a $x$ è:', options: ['$Ax-b$', '$A^T(Ax-b)$', '$2 A^T(Ax-b)$', '$2A(Ax-b)$'], answer: 2, explain: 'Espandi $(Ax-b)^T(Ax-b)$ e differenzia; il fattore 2 viene dai due termini simmetrici.' },
      { q: 'La SVD $X = U\\Sigma V^T$ vale:', options: ['Solo per matrici quadrate', 'Solo per matrici simmetriche', 'Per qualsiasi matrice reale', 'Solo per matrici invertibili'], answer: 2, explain: 'La SVD esiste per qualsiasi matrice reale $n\\times d$.' },
      { q: 'Se $\\mathrm{Var}(X) = 4$, quanto vale $\\mathrm{Var}(3X + 5)$?', options: ['4', '9', '12', '36'], answer: 3, explain: '$\\mathrm{Var}(aX+b) = a^2 \\mathrm{Var}(X) = 9 \\cdot 4 = 36$.' },
      { q: 'Un learning rate troppo grande produce tipicamente:', options: ['Convergenza più lenta', 'Convergenza più veloce e stabile', 'Oscillazioni o divergenza', 'Overfitting'], answer: 2, explain: 'Se $\\eta$ è troppo grande, ogni passo "salta oltre" il minimo e la loss può crescere.' },
    ] },
    { type: 'exercise', difficulty: 'hard', title: 'Test pratico — regressione lineare a mano',
      prompt: '<p>Implementa la <strong>closed-form solution</strong> della regressione lineare: dato $X$ e $y$, calcola $\\hat w = (X^T X)^{-1} X^T y$. Confronta con scikit-learn.</p><p>Nota: aggiungi una colonna di 1 per il bias.</p>',
      starter: `import numpy as np
np.random.seed(42)
n = 100
X = np.random.randn(n, 3)
w_true = np.array([1.5, -2.0, 0.7])
b_true = 0.3
y = X @ w_true + b_true + 0.1*np.random.randn(n)

# aggiungi colonna di 1 per il bias
X_aug = np.hstack([X, np.ones((n,1))])

# TODO: calcola w_hat = (X_aug^T X_aug)^{-1} X_aug^T y
w_hat = None
print("w_hat:", w_hat)
print("veri:", list(w_true) + [b_true])`,
      check: `import numpy as np
from sklearn.linear_model import LinearRegression
sk = LinearRegression().fit(X, y)
sk_full = np.concatenate([sk.coef_, [sk.intercept_]])
if w_hat is None:
    print("FAIL: w_hat è None")
elif np.allclose(w_hat, sk_full, atol=1e-6):
    print("OK — soluzione identica a scikit-learn")
    print("w_hat:", np.round(w_hat, 4))
else:
    print(f"FAIL:\\ntuo    {np.round(w_hat,4)}\\nsklearn{np.round(sk_full,4)}")`,
      solution: `w_hat = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y` },
  ]
});

// =====================================================================
// LEZIONE 3 — NumPy e Pandas
// =====================================================================
LESSONS.push({
  id: 'l03-numpy-pandas',
  title: 'NumPy e Pandas',
  tag: 'Strumenti',
  summary: 'Vettorializzazione, broadcasting, DataFrame — l\'ambiente in cui vive il ML.',
  teoria: [
    { type: 'md', content: `
<h3>3.1 Perché NumPy</h3>
<p>Python puro è lento per il calcolo numerico. NumPy sposta i loop dentro C compilato, con dati in memoria contigua. Un\'operazione vettoriale in NumPy è tipicamente <strong>50-500 volte più veloce</strong> del loop Python equivalente.</p>
<p>La regola d\'oro: <em>se stai scrivendo un for-loop su un array, quasi certamente c\'è un modo vettorializzato più elegante e più veloce.</em></p>

<h3>3.2 Array: shape, dtype, axis</h3>
<p>Un <code class="inline">ndarray</code> ha una <em>shape</em> (tupla di dimensioni), un <em>dtype</em> (tipo dei singoli elementi), e uno <em>strides</em> (come si passa da un elemento all'altro in memoria). Non devi pensare agli strides quasi mai, ma capire che array multidimensionali sono <em>views</em> su un blocco piatto di memoria aiuta.</p>
<p>Un dataset tipico ha shape <code class="inline">(n_samples, n_features)</code>. Un batch di immagini RGB ha shape <code class="inline">(batch, H, W, 3)</code>.</p>

<h3>3.3 Broadcasting</h3>
<p>Regola: quando fai un\'operazione tra array di shape diverse, NumPy le "estende" (senza copiare memoria) se sono <em>compatibili</em>. Sono compatibili se, allineando le shape a destra, ogni dimensione è uguale o vale 1.</p>
<pre class="code">A shape (3, 4)
b shape    (4,)     -> ok, b viene "ripetuto" sulle 3 righe
c shape (3, 1)      -> ok, c viene "ripetuto" sulle 4 colonne
d shape (2, 4)      -> ERRORE (2 vs 3)</pre>
<p>Broadcasting è come sono implementati praticamente tutti i preprocessing di dati. Esempio classico: standardizzare le colonne.</p>

<h3>3.4 Pandas: DataFrame</h3>
<p>Pandas costruisce sopra NumPy la nozione di <em>tabella con nomi</em>: righe con indice, colonne con nome, tipi eterogenei. È il coltellino svizzero per la fase esplorativa.</p>
<ul>
<li><code class="inline">df.loc[etichette]</code> selezione per etichetta</li>
<li><code class="inline">df.iloc[posizioni]</code> selezione per posizione intera</li>
<li><code class="inline">df.groupby('col').agg(...)</code> aggregazioni</li>
<li><code class="inline">df.merge(altro, on='...')</code> join tra tabelle</li>
<li><code class="inline">df.isna().sum()</code> conteggio valori mancanti</li>
</ul>

<h3>3.5 Split, indexing, boolean masks</h3>
<p>Il modo idiomatico di filtrare dati:</p>
<pre class="code">mask = df['age'] &gt; 30
adulti = df[mask]
df.loc[mask, 'salario'] *= 1.1</pre>
` },
    { type: 'callout', variant: 'tip', title: 'Regola pratica', content: 'Ogni volta che scrivi un for-loop che itera su righe di un DataFrame, fermati 30 secondi e chiediti: "posso fare la stessa cosa con un\'operazione vettoriale, un groupby, o apply?" Nel 90% dei casi la risposta è sì.' },
  ],
  esempi: [
    { type: 'md', content: '<h3>Esempio 1: broadcasting per standardizzare</h3><p>Sottraiamo la media colonna per colonna e dividiamo per la deviazione standard. Un\'operazione tipica nel preprocessing.</p>' },
    { type: 'run', content: `import numpy as np

np.random.seed(0)
X = np.random.randn(5, 3) * np.array([1, 10, 100]) + np.array([0, 50, 200])
print("Dati grezzi:\\n", np.round(X, 2))

mu = X.mean(axis=0)      # media per colonna, shape (3,)
sigma = X.std(axis=0)    # std per colonna, shape (3,)
X_std = (X - mu) / sigma # broadcasting: shape (5,3) - shape (3,)

print("\\nMedia dopo std:", np.round(X_std.mean(axis=0), 6))
print("Std   dopo std:", np.round(X_std.std(axis=0), 6))` },
    { type: 'md', content: '<h3>Esempio 2: fancy indexing e boolean mask</h3>' },
    { type: 'run', content: `import numpy as np
a = np.arange(10, 20)
print("array:", a)
print("indici pari:", a[[0, 2, 4, 6, 8]])
print("elementi > 15:", a[a > 15])
print("modifica in place:")
a[a > 15] = 0
print(a)` },
    { type: 'md', content: '<h3>Esempio 3: Pandas basics</h3>' },
    { type: 'run', content: `import pandas as pd
import numpy as np

df = pd.DataFrame({
    'nome': ['Anna','Bruno','Carlo','Diana','Elisa','Franco'],
    'eta': [23, 45, 34, 28, 51, 39],
    'citta': ['Roma','Milano','Roma','Torino','Milano','Roma'],
    'salario': [28000, 55000, 42000, 33000, 68000, 47000]
})
print(df)
print("\\nStatistiche:\\n", df.describe())
print("\\nSalario medio per citta:")
print(df.groupby('citta')['salario'].mean())
print("\\nPersone con salario > 40k:")
print(df[df.salario > 40000][['nome','salario']])` },
    { type: 'md', content: '<h3>Esempio 4: pivot e valori mancanti</h3>' },
    { type: 'run', content: `import pandas as pd
import numpy as np

df = pd.DataFrame({
    'prodotto': ['A','A','B','B','C','C'],
    'mese': ['gen','feb','gen','feb','gen','feb'],
    'vendite': [100, 120, 80, np.nan, 150, 170]
})
print("Dati lunghi:\\n", df)
pivot = df.pivot(index='prodotto', columns='mese', values='vendite')
print("\\nPivot:\\n", pivot)
print("\\nRiempo NaN con la media del prodotto:")
pivot['feb'] = pivot['feb'].fillna(pivot.mean(axis=1))
print(pivot)` },
  ],
  esercizi: [
    { type: 'exercise', difficulty: 'easy', title: 'Es. 3.1 — Vettorializza',
      prompt: '<p>Riscrivi questo loop in versione vettorializzata NumPy. Il risultato deve essere identico ma senza <code class="inline">for</code>.</p><pre class="code">out = []\nfor v in x:\n    out.append(v**2 + 3*v - 1)\nresult = np.array(out)</pre>',
      starter: `import numpy as np
x = np.arange(-5, 6)

# TODO: versione vettorializzata
result = None
print(result)`,
      check: `import numpy as np
_atteso = x**2 + 3*x - 1
if result is not None and np.allclose(result, _atteso):
    print("OK")
else:
    print(f"FAIL: atteso {_atteso}, tuo {result}")`,
      solution: `result = x**2 + 3*x - 1` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 3.2 — Distanze pairwise',
      prompt: '<p>Data una matrice <code class="inline">X</code> di shape (n, d), calcola la matrice <code class="inline">D</code> di shape (n, n) dove <code class="inline">D[i,j]</code> è la distanza euclidea tra la riga $i$ e la riga $j$. Usa broadcasting, no loop.</p><p><em>Hint:</em> $\\|x_i - x_j\\|^2 = \\|x_i\\|^2 + \\|x_j\\|^2 - 2 x_i^T x_j$.</p>',
      starter: `import numpy as np
np.random.seed(0)
X = np.random.randn(5, 3)

# TODO: matrice D di shape (5,5) senza loop
D = None
print(np.round(D, 3))`,
      check: `import numpy as np
from scipy.spatial.distance import cdist  # potrebbe non essere disponibile
_atteso = np.sqrt(((X[:,None,:] - X[None,:,:])**2).sum(-1))
if D is not None and np.allclose(D, _atteso, atol=1e-6):
    print("OK")
else:
    print("FAIL")`,
      solution: `D = np.sqrt(((X[:,None,:] - X[None,:,:])**2).sum(-1))
# oppure: sq = (X**2).sum(1); D = np.sqrt(sq[:,None]+sq[None,:] - 2*X@X.T)` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 3.3 — Analisi su DataFrame',
      prompt: '<p>Dato il DataFrame di partenza, trova per ogni <strong>città</strong> il salario massimo e il nome della persona che lo ha. Il risultato deve essere un DataFrame con colonne <code class="inline">[\'citta\', \'nome\', \'salario\']</code>, salvato in <code class="inline">top_citta</code>.</p>',
      starter: `import pandas as pd
df = pd.DataFrame({
    'nome': ['Anna','Bruno','Carlo','Diana','Elisa','Franco'],
    'eta': [23, 45, 34, 28, 51, 39],
    'citta': ['Roma','Milano','Roma','Torino','Milano','Roma'],
    'salario': [28000, 55000, 42000, 33000, 68000, 47000]
})

# TODO
top_citta = None
print(top_citta)`,
      check: `import pandas as pd
if top_citta is None:
    print("FAIL: None")
else:
    ok = (
        set(top_citta['citta']) == {'Roma','Milano','Torino'} and
        top_citta.loc[top_citta.citta=='Roma','nome'].iloc[0] == 'Franco' and
        top_citta.loc[top_citta.citta=='Milano','nome'].iloc[0] == 'Elisa'
    )
    print("OK" if ok else "FAIL — controlla nomi/città")`,
      solution: `idx = df.groupby('citta')['salario'].idxmax()
top_citta = df.loc[idx, ['citta','nome','salario']].reset_index(drop=True)` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 3.4 — Rolling mean',
      prompt: '<p>Data una serie temporale <code class="inline">ts</code>, calcola la media mobile su una finestra di 3 elementi senza usare <code class="inline">pandas.rolling</code>. Il risultato dev\'essere lungo <code class="inline">len(ts) - 2</code>.</p>',
      starter: `import numpy as np
ts = np.array([1., 2., 3., 4., 5., 6., 7., 8.])

# TODO
mm = None
print(mm)  # atteso [2, 3, 4, 5, 6, 7]`,
      check: `import numpy as np
atteso = np.array([2,3,4,5,6,7], dtype=float)
if mm is not None and np.allclose(mm, atteso):
    print("OK")
else:
    print(f"FAIL: atteso {atteso}, tuo {mm}")`,
      solution: `mm = (ts[:-2] + ts[1:-1] + ts[2:]) / 3
# alternativa: np.convolve(ts, np.ones(3)/3, mode='valid')` },
  ],
  test: [
    { type: 'quiz', questions: [
      { q: 'Qual è la shape di <code class="inline">np.zeros((3,4))@np.ones((4,))</code>?', options: ['(3,4)', '(3,)', '(4,)', 'Errore'], answer: 1, explain: 'Matrice (3,4) per vettore (4,) dà vettore (3,).' },
      { q: 'In Pandas, <code class="inline">df.iloc[2]</code> restituisce:', options: ['La riga con indice etichetta 2', 'La terza riga (posizione 2)', 'La seconda colonna', 'Errore se non c\'è etichetta 2'], answer: 1, explain: 'iloc è per posizione intera.' },
      { q: 'Broadcasting tra shape (5,1,4) e (3,4) produce shape:', options: ['(5,3,4)', '(5,4)', '(3,5,4)', 'Errore'], answer: 0, explain: 'Allineando a destra: (5,1,4) e (_,3,4). 1 si estende a 3, dimensione mancante diventa 5. Risultato (5,3,4).' },
      { q: 'Per iterare su righe di un DataFrame di 1M righe conviene:', options: ['for i, row in df.iterrows()', 'Convertire a lista e iterare', 'Vettorializzare o usare df.apply solo se necessario', 'Usare un ciclo while'], answer: 2, explain: 'iterrows è lentissimo; sempre preferire operazioni vettoriali.' },
    ] },
    { type: 'exercise', difficulty: 'hard', title: 'Test pratico — mini EDA',
      prompt: '<p>Ti passo un DataFrame di transazioni. Devi produrre queste 3 statistiche:</p><ol><li><code class="inline">stat_total</code>: totale speso da ogni cliente (Series)</li><li><code class="inline">stat_top_prod</code>: prodotto più venduto (stringa)</li><li><code class="inline">stat_gg_medio</code>: numero medio di transazioni al giorno (float)</li></ol>',
      starter: `import pandas as pd
df = pd.DataFrame({
    'giorno': ['2024-01-01','2024-01-01','2024-01-02','2024-01-02','2024-01-02','2024-01-03'],
    'cliente': ['A','B','A','C','B','A'],
    'prodotto': ['libro','penna','libro','laptop','libro','penna'],
    'importo': [15, 3, 12, 900, 18, 4]
})

stat_total = None
stat_top_prod = None
stat_gg_medio = None

print("Totale per cliente:\\n", stat_total)
print("Prodotto top:", stat_top_prod)
print("Transazioni/giorno:", stat_gg_medio)`,
      check: `ok = True
if not (stat_total is not None and stat_total.get('A')==27 and stat_total.get('B')==21 and stat_total.get('C')==900): ok=False
if stat_top_prod != 'libro': ok=False
if not (abs(stat_gg_medio - 2.0) < 1e-6): ok=False
print("OK" if ok else "FAIL — controlla i tre valori")`,
      solution: `stat_total = df.groupby('cliente')['importo'].sum()
stat_top_prod = df['prodotto'].value_counts().idxmax()
stat_gg_medio = df.groupby('giorno').size().mean()` },
  ]
});

// =====================================================================
// LEZIONE 4 — EDA e visualizzazione
// =====================================================================
LESSONS.push({
  id: 'l04-eda',
  title: 'Esplorazione dei dati e visualizzazione',
  tag: 'Workflow',
  summary: 'L\'80% del lavoro di un data scientist. Come guardare i dati prima di modellarli.',
  teoria: [
    { type: 'md', content: `
<h3>4.1 Perché EDA</h3>
<p>Prima di lanciare qualsiasi modello, devi <strong>guardare i dati</strong>. La Exploratory Data Analysis (Tukey, anni '60) è la fase in cui rispondi a domande come:</p>
<ul>
<li>Che tipo hanno le variabili? Quali sono numeriche, categoriche, ordinali?</li>
<li>Ci sono valori mancanti? Come sono distribuiti?</li>
<li>Le distribuzioni sono ragionevoli o ci sono outlier assurdi?</li>
<li>Ci sono correlazioni evidenti tra variabili? Con il target?</li>
<li>La classe target è bilanciata o sbilanciata?</li>
</ul>
<p>Se salti l'EDA, quasi certamente costruirai un modello che fallisce silenziosamente. Un modello mediocre su dati puliti batte un modello sofisticato su dati sporchi.</p>

<h3>4.2 Statistica descrittiva</h3>
<p>Le prime cose da guardare per una variabile numerica:</p>
<ul>
<li><strong>Tendenza centrale</strong>: media, mediana</li>
<li><strong>Dispersione</strong>: deviazione standard, IQR (differenza tra 3° e 1° quartile)</li>
<li><strong>Forma</strong>: skewness (asimmetria), kurtosi (code)</li>
<li><strong>Estremi</strong>: min, max, percentili</li>
</ul>
<p>La differenza media-mediana è un ottimo detector di skewness: se media &gg; mediana, la distribuzione ha coda a destra (es. redditi).</p>

<h3>4.3 Grafici essenziali</h3>
<ul>
<li><strong>Istogramma / KDE</strong>: distribuzione di una variabile numerica</li>
<li><strong>Boxplot</strong>: mediana, quartili, outlier</li>
<li><strong>Scatter plot</strong>: relazione tra due numeriche</li>
<li><strong>Bar chart</strong>: conteggi per categoria</li>
<li><strong>Heatmap di correlazioni</strong>: sguardo globale sulle relazioni lineari</li>
<li><strong>Pair plot</strong>: tutti gli scatter a coppie in una griglia</li>
</ul>

<h3>4.4 Gestione dei missing values</h3>
<p>Strategie principali:</p>
<ol>
<li><strong>Elimina</strong> le righe o colonne con troppi missing. Solo se sono pochi.</li>
<li><strong>Imputa</strong> con media/mediana (numeriche) o moda (categoriche).</li>
<li><strong>Imputa</strong> con un modello (KNN, regressione).</li>
<li><strong>Missing come informazione</strong>: aggiungi una feature indicatrice "era missing?".</li>
</ol>
<p>Attenzione al <em>data leakage</em>: calcola le statistiche di imputazione SOLO sul training set, non su tutto il dataset.</p>

<h3>4.5 Outlier: rimuovere o no?</h3>
<p>Un outlier può essere:</p>
<ul>
<li>Un errore di misurazione &rarr; rimuovilo</li>
<li>Un evento raro ma reale &rarr; tienilo (magari con un modello robusto)</li>
<li>La cosa che vuoi predire (frode, guasti) &rarr; è l'oggetto stesso del task</li>
</ul>
<p>Regola pratica: guarda l'outlier, capisci perché è lì, poi decidi. Non applicare mai una regola "rimuovi tutto oltre 3 sigma" senza aver capito i dati.</p>

<h3>4.6 Correlazione ≠ causalità</h3>
<p>La correlazione di Pearson misura la relazione <em>lineare</em>. Due variabili possono avere correlazione zero ed essere fortemente dipendenti (es. $y = x^2$ con $x$ centrato). Usa anche Spearman (rank correlation) per relazioni monotone non lineari.</p>
` },
    { type: 'callout', variant: 'warn', title: 'Attenzione', content: 'Ricorda il quartetto di Anscombe: quattro dataset con identiche statistiche riassuntive ma visualizzazioni totalmente diverse. La conclusione è che le statistiche riassuntive non bastano — devi vedere i grafici.' },
  ],
  esempi: [
    { type: 'md', content: '<h3>Esempio 1: dataset Iris — il classico</h3><p>Iris è il "dataset di riferimento" del ML. 150 fiori, 4 misure, 3 specie.</p>' },
    { type: 'run', content: `from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris(as_frame=True)
df = iris.frame
df['specie'] = df['target'].map(dict(enumerate(iris.target_names)))
print("Shape:", df.shape)
print("\\nPrime righe:\\n", df.head())
print("\\nStatistiche:\\n", df.describe())
print("\\nConteggio per specie:\\n", df['specie'].value_counts())` },
    { type: 'md', content: '<h3>Esempio 2: istogrammi e boxplot</h3>' },
    { type: 'run', content: `import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris(as_frame=True)
df = iris.frame

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
for ax, col in zip(axes.flatten(), df.columns[:-1]):
    for cls in [0,1,2]:
        ax.hist(df.loc[df.target==cls, col], alpha=0.5,
                bins=15, label=f'classe {cls}')
    ax.set_title(col); ax.legend(fontsize=8)
plt.tight_layout()
plt.show()` },
    { type: 'md', content: '<h3>Esempio 3: matrice di correlazione</h3>' },
    { type: 'run', content: `import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame
corr = df.drop('target', axis=1).corr()
print("Matrice di correlazione:\\n", corr.round(3))

fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=45, ha='right')
ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', color='k', fontsize=9)
plt.colorbar(im, ax=ax)
plt.tight_layout(); plt.show()` },
    { type: 'md', content: '<h3>Esempio 4: outlier detection con IQR</h3>' },
    { type: 'run', content: `import numpy as np
np.random.seed(0)
data = np.concatenate([np.random.normal(50, 5, 100), [120, 130, -10]])  # 3 outlier

q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
print(f"Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
print(f"Range accettabile: [{lo:.2f}, {hi:.2f}]")
outliers = data[(data < lo) | (data > hi)]
print(f"Outlier trovati: {outliers}")` },
  ],
  esercizi: [
    { type: 'exercise', difficulty: 'easy', title: 'Es. 4.1 — Statistiche di base',
      prompt: '<p>Data una serie di valori con qualche NaN, calcola media, mediana, e conta i NaN. Salva rispettivamente in <code class="inline">m, med, n_nan</code>.</p>',
      starter: `import numpy as np
import pandas as pd
s = pd.Series([1.2, 3.5, np.nan, 4.1, 2.8, np.nan, 5.0, 3.3, np.nan, 4.7])

m = None
med = None
n_nan = None
print(m, med, n_nan)`,
      check: `import numpy as np
if abs(m - s.mean()) < 1e-9 and abs(med - s.median()) < 1e-9 and n_nan == 3:
    print("OK")
else:
    print(f"FAIL: attesi mean={s.mean():.3f}, median={s.median():.3f}, nan=3")`,
      solution: `m = s.mean(); med = s.median(); n_nan = s.isna().sum()` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 4.2 — Skewness manuale',
      prompt: '<p>Implementa la skewness campionaria: $\\text{skew} = \\frac{1}{n}\\sum \\left(\\frac{x_i - \\bar x}{s}\\right)^3$. Usa il numpy std di default (ddof=0). Testa su dati normali (skew ~ 0) e su dati esponenziali (skew > 0).</p>',
      starter: `import numpy as np
def skew_manual(x):
    # TODO
    return 0.0

np.random.seed(0)
print("normale:      ", skew_manual(np.random.normal(0,1,10000)))
print("esponenziale: ", skew_manual(np.random.exponential(1,10000)))`,
      check: `import numpy as np
np.random.seed(0)
s1 = skew_manual(np.random.normal(0,1,10000))
s2 = skew_manual(np.random.exponential(1,10000))
if abs(s1) < 0.1 and s2 > 1.5:
    print("OK — skew normale ~ 0, esponenziale > 1.5")
else:
    print(f"FAIL — normale {s1:.3f}, esp {s2:.3f}")`,
      solution: `def skew_manual(x):
    x = np.asarray(x, float)
    z = (x - x.mean()) / x.std()
    return np.mean(z**3)` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 4.3 — Imputazione mediana per gruppo',
      prompt: '<p>Riempi i NaN nella colonna <code class="inline">salario</code> con la mediana <strong>del gruppo</strong> a cui appartiene la riga (colonna <code class="inline">reparto</code>). Salva il risultato in <code class="inline">df[\'salario\']</code>.</p>',
      starter: `import pandas as pd
import numpy as np
df = pd.DataFrame({
    'reparto': ['R&D','R&D','R&D','HR','HR','HR','R&D','HR'],
    'salario':[50000, np.nan, 60000, 30000, 32000, np.nan, 55000, np.nan]
})

# TODO
print(df)`,
      check: `import pandas as pd
ok = df['salario'].isna().sum()==0
# la mediana R&D è tra {50000,55000,60000} = 55000
# la mediana HR è tra {30000,32000} = 31000
ok = ok and df.loc[df.reparto=='R&D','salario'].iloc[1] == 55000
ok = ok and df.loc[df.reparto=='HR','salario'].iloc[2] == 31000
print("OK" if ok else "FAIL — imputazione errata")`,
      solution: `df['salario'] = df.groupby('reparto')['salario'].transform(lambda s: s.fillna(s.median()))` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 4.4 — Feature engineering',
      prompt: '<p>Dato un DataFrame con timestamp di transazioni per utente, crea una nuova colonna <code class="inline">giorni_da_ultimo</code>: quanti giorni sono passati tra ogni transazione e la precedente dello stesso utente. La prima transazione di ogni utente ha valore NaN.</p>',
      starter: `import pandas as pd
df = pd.DataFrame({
    'user': ['A','A','B','A','B','C','B'],
    'data': pd.to_datetime(['2024-01-05','2024-01-10','2024-01-06','2024-01-20','2024-01-15','2024-01-08','2024-01-25'])
}).sort_values(['user','data']).reset_index(drop=True)

# TODO: crea colonna 'giorni_da_ultimo'
print(df)`,
      check: `import pandas as pd, numpy as np
ok = True
if 'giorni_da_ultimo' not in df.columns: ok=False
else:
    a = df[df.user=='A']['giorni_da_ultimo'].tolist()
    b = df[df.user=='B']['giorni_da_ultimo'].tolist()
    if not (np.isnan(a[0]) and a[1]==5 and a[2]==10): ok=False
    if not (np.isnan(b[0]) and b[1]==9 and b[2]==10): ok=False
print("OK" if ok else "FAIL")`,
      solution: `df['giorni_da_ultimo'] = df.groupby('user')['data'].diff().dt.days` },
  ],
  test: [
    { type: 'quiz', questions: [
      { q: 'Se media &gt; mediana in una distribuzione, la skewness è tipicamente:', options: ['Negativa (coda a sinistra)', 'Zero', 'Positiva (coda a destra)', 'Non definita'], answer: 2, explain: 'Coda lunga a destra tira la media in alto rispetto alla mediana.' },
      { q: 'Il quartetto di Anscombe insegna che:', options: ['La media è la migliore statistica', 'Le statistiche riassuntive possono nascondere pattern importanti', 'La correlazione è sempre affidabile', 'I grafici sono superflui'], answer: 1, explain: '4 dataset con stesse statistiche hanno grafici totalmente diversi.' },
      { q: 'Calcolare la media per imputare i NaN usando tutto il dataset (train+test) è:', options: ['La pratica corretta', 'Data leakage: usa solo il training', 'Ininfluente', 'Consigliato in caso di pochi dati'], answer: 1, explain: 'Includere il test nel calcolo delle statistiche contamina la valutazione.' },
      { q: 'Correlazione di Pearson tra $x$ e $y = x^2$ con $x \\sim \\mathcal{N}(0,1)$:', options: ['~ 1', '~ -1', '~ 0', 'Dipende dal campione'], answer: 2, explain: 'Simmetria: la relazione è perfettamente non lineare ma con correlazione lineare nulla.' },
    ] },
    { type: 'exercise', difficulty: 'hard', title: 'Test pratico — EDA completa',
      prompt: '<p>Sul dataset Iris, calcola:</p><ol><li><code class="inline">v_max_corr</code>: la coppia di variabili numeriche con correlazione più alta (tupla di 2 nomi)</li><li><code class="inline">v_min_var</code>: nome della variabile con varianza minore</li><li><code class="inline">v_disc</code>: la variabile che meglio separa le classi (quella con maggior rapporto varianza tra-gruppi / varianza intra-gruppi)</li></ol>',
      starter: `from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
iris = load_iris(as_frame=True)
df = iris.frame.rename(columns={'target':'y'})
features = [c for c in df.columns if c != 'y']

v_max_corr = None
v_min_var = None
v_disc = None

print(v_max_corr, v_min_var, v_disc)`,
      check: `ok = True
if v_max_corr is None or set(v_max_corr) != {'petal length (cm)','petal width (cm)'}: ok=False
if v_min_var != 'sepal width (cm)': ok=False
if v_disc not in ('petal length (cm)','petal width (cm)'): ok=False
print("OK" if ok else f"FAIL: {v_max_corr}, {v_min_var}, {v_disc}")`,
      solution: `import numpy as np
# max correlation off-diagonal
c = df[features].corr().abs()
np.fill_diagonal(c.values, 0)
i, j = np.unravel_index(np.argmax(c.values), c.shape)
v_max_corr = (c.index[i], c.columns[j])

v_min_var = df[features].var().idxmin()

# rapporto discriminante
def f_ratio(col):
    mu = df[col].mean()
    between = df.groupby('y')[col].mean().sub(mu).pow(2).mul(df.groupby('y').size()).sum()
    within = df.groupby('y')[col].var(ddof=0).mul(df.groupby('y').size()).sum()
    return between / within
v_disc = max(features, key=f_ratio)` },
  ]
});

// =====================================================================
// LEZIONE 5 — Regressione lineare
// =====================================================================
LESSONS.push({
  id: 'l05-linear-regression',
  title: 'Regressione lineare',
  tag: 'Supervisionato',
  summary: 'Il modello più semplice, il più importante da capire davvero: fondamento di tutto il ML.',
  teoria: [
    { type: 'md', content: `
<h3>5.1 Il modello</h3>
<p>Data $X \\in \\mathbb{R}^{n\\times d}$ e $y \\in \\mathbb{R}^n$, cerchiamo un vettore di pesi $w \\in \\mathbb{R}^d$ e un bias $b \\in \\mathbb{R}$ tali che:</p>
<p>$$\\hat y_i = w^T x_i + b$$</p>
<p>La loss classica è l'errore quadratico:</p>
<p>$$\\mathcal{L}(w, b) = \\frac{1}{n}\\sum_{i=1}^n (\\hat y_i - y_i)^2 = \\frac{1}{n}\\|Xw + b\\mathbf{1} - y\\|^2$$</p>

<h3>5.2 Soluzione in forma chiusa</h3>
<p>Aggiungendo una colonna di 1 a $X$ (per assorbire il bias in $w$), abbiamo $\\hat y = Xw$. Il gradiente della loss è $\\nabla_w \\mathcal{L} = \\frac{2}{n} X^T(Xw - y)$. Ponendolo uguale a zero:</p>
<p>$$X^T X w = X^T y \\quad\\Rightarrow\\quad \\hat w = (X^T X)^{-1} X^T y$$</p>
<p>Queste sono le <em>normal equations</em>. Se $X^T X$ è singolare (feature collineari) usa la pseudoinversa $\\hat w = X^+ y$, calcolata via SVD.</p>

<h3>5.3 Interpretazione probabilistica</h3>
<p>Assumiamo $y_i = w^T x_i + \\varepsilon_i$ con $\\varepsilon_i \\sim \\mathcal{N}(0, \\sigma^2)$ i.i.d. La log-likelihood è:</p>
<p>$$\\log p(y|X, w) = -\\frac{1}{2\\sigma^2}\\sum_i (y_i - w^T x_i)^2 + \\text{const}$$</p>
<p>Massimizzarla equivale a minimizzare l'MSE. <strong>Least squares = MLE con rumore gaussiano.</strong></p>

<h3>5.4 Ipotesi classiche (Gauss-Markov)</h3>
<ul>
<li>Linearità della relazione</li>
<li>Errori a media zero, varianza costante (omoschedasticità)</li>
<li>Errori incorrelati</li>
<li>Features non collineari</li>
</ul>
<p>Sotto queste ipotesi, l'OLS (Ordinary Least Squares) è il miglior stimatore lineare non distorto (BLUE).</p>

<h3>5.5 Metriche</h3>
<p><strong>MSE</strong>: $\\frac{1}{n}\\sum (y_i - \\hat y_i)^2$. Sensibile agli outlier per il quadrato.</p>
<p><strong>RMSE</strong>: $\\sqrt{\\text{MSE}}$. Stessa unità di $y$, più interpretabile.</p>
<p><strong>MAE</strong>: $\\frac{1}{n}\\sum |y_i - \\hat y_i|$. Robusta agli outlier.</p>
<p><strong>$R^2$</strong>: $1 - \\frac{\\sum (y_i-\\hat y_i)^2}{\\sum (y_i-\\bar y)^2}$. Frazione di varianza spiegata. Non è correlazione al quadrato in generale, e può essere negativo se il modello è peggiore della media.</p>

<h3>5.6 Regolarizzazione: Ridge e Lasso</h3>
<p>Se il numero di feature è grande o le feature sono correlate, la soluzione OLS ha varianza elevata (piccole variazioni di dati &rarr; grandi variazioni di $w$). Aggiungiamo un termine di penalità:</p>
<p><strong>Ridge (L2)</strong>: $\\mathcal{L}_{\\text{ridge}} = \\|Xw - y\\|^2 + \\lambda \\|w\\|_2^2$. Soluzione: $\\hat w = (X^T X + \\lambda I)^{-1} X^T y$. Riduce tutti i coefficienti verso zero senza mai azzerarli.</p>
<p><strong>Lasso (L1)</strong>: $\\mathcal{L}_{\\text{lasso}} = \\|Xw - y\\|^2 + \\lambda \\|w\\|_1$. Nessuna forma chiusa. Produce soluzioni sparse: alcuni coefficienti diventano esattamente zero &rarr; <em>selezione delle feature</em>.</p>
<p><strong>Elastic Net</strong>: combina L1 e L2.</p>

<h3>5.7 Feature engineering per non-linearità</h3>
<p>La linearità è nei parametri, non nelle feature. Puoi trasformare le feature per catturare non-linearità:</p>
<ul>
<li><strong>Polinomiali</strong>: $x, x^2, x^3, x_1 x_2$</li>
<li><strong>Log</strong>: $\\log x$ per variabili con coda pesante</li>
<li><strong>Categoriche</strong>: one-hot encoding</li>
</ul>
<p>Il modello resta "lineare" in senso ottimizzativo, ma può fittare curve.</p>
` },
    { type: 'callout', variant: 'note', title: 'Perché matter ancora', content: 'La regressione lineare è veloce, interpretabile, ha intervalli di confidenza analitici, è un ottimo baseline. Il 30% delle applicazioni ML in produzione sono di fatto regressioni lineari o logistiche ben tarate.' },
  ],
  esempi: [
    { type: 'md', content: '<h3>Esempio 1: OLS con scikit-learn</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(0)
n = 200
X = np.random.randn(n, 3)
w_true, b_true = np.array([2.0, -1.5, 0.5]), 3.0
y = X @ w_true + b_true + 0.3*np.random.randn(n)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression().fit(X_tr, y_tr)

print(f"pesi stimati: {model.coef_.round(3)} (veri {w_true})")
print(f"bias stimato: {model.intercept_:.3f} (vero {b_true})")

y_pred = model.predict(X_te)
print(f"\\nMSE test: {mean_squared_error(y_te, y_pred):.4f}")
print(f"R^2 test: {r2_score(y_te, y_pred):.4f}")` },
    { type: 'md', content: '<h3>Esempio 2: gradient descent from scratch</h3>' },
    { type: 'run', content: `import numpy as np
np.random.seed(0)
X = np.random.randn(100, 2)
y = 3*X[:,0] - 2*X[:,1] + 1 + 0.1*np.random.randn(100)

# aggiungi bias
X_aug = np.hstack([X, np.ones((100,1))])
w = np.zeros(3)
eta, epochs = 0.1, 200

for ep in range(epochs):
    y_hat = X_aug @ w
    grad = 2/len(X_aug) * X_aug.T @ (y_hat - y)
    w = w - eta * grad
    if ep % 40 == 0:
        loss = np.mean((y_hat - y)**2)
        print(f"epoch {ep:3d}: loss={loss:.5f}, w={w.round(3)}")

print(f"\\nFinale: {w.round(3)} — atteso [3, -2, 1]")` },
    { type: 'md', content: '<h3>Esempio 3: Ridge vs Lasso su feature correlate</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso

np.random.seed(0)
n = 100
x1 = np.random.randn(n)
x2 = x1 + 0.01*np.random.randn(n)   # quasi identiche a x1
x3 = np.random.randn(n)
X = np.column_stack([x1, x2, x3])
y = 2*x1 + x3 + 0.5*np.random.randn(n)  # x2 inutile

for M, name in [(LinearRegression(), 'OLS'), (Ridge(alpha=1.0), 'Ridge'), (Lasso(alpha=0.1), 'Lasso')]:
    M.fit(X, y)
    print(f"{name:6s}: {M.coef_.round(3)}")
print("\\nOLS: coefficienti instabili per collinearità")
print("Ridge: li riduce ma li tiene tutti")
print("Lasso: azzera x2, seleziona feature")` },
    { type: 'md', content: '<h3>Esempio 4: regressione polinomiale</h3>' },
    { type: 'run', content: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(0)
x = np.linspace(-3, 3, 40).reshape(-1,1)
y = np.sin(x).ravel() + 0.1*np.random.randn(40)

fig, ax = plt.subplots(figsize=(8,4))
ax.scatter(x, y, s=20, alpha=0.6, label='dati')
xs = np.linspace(-3, 3, 200).reshape(-1,1)
for deg in [1, 3, 7, 15]:
    m = make_pipeline(PolynomialFeatures(deg), LinearRegression()).fit(x, y)
    ax.plot(xs, m.predict(xs), label=f'grado {deg}')
ax.legend(); ax.set_title('Regressione polinomiale'); ax.grid(alpha=0.3)
plt.show()` },
  ],
  esercizi: [
    { type: 'exercise', difficulty: 'easy', title: 'Es. 5.1 — Fit e metriche',
      prompt: '<p>Fitta una regressione lineare sui dati generati, poi calcola RMSE e $R^2$ sul test. Salvali in <code class="inline">rmse</code> e <code class="inline">r2</code>.</p>',
      starter: `import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(1)
X = np.random.randn(200, 4)
y = X @ np.array([1.5, -1, 0.5, 2]) + 1 + 0.5*np.random.randn(200)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

# TODO
rmse = None
r2 = None
print(f"RMSE: {rmse}, R2: {r2}")`,
      check: `import numpy as np
m = LinearRegression().fit(X_tr, y_tr)
yp = m.predict(X_te)
_rmse = np.sqrt(np.mean((yp - y_te)**2))
_r2 = r2_score(y_te, yp)
if rmse is not None and abs(rmse - _rmse) < 1e-4 and abs(r2 - _r2) < 1e-4:
    print(f"OK — RMSE={rmse:.4f}, R2={r2:.4f}")
else:
    print(f"FAIL — atteso RMSE={_rmse:.4f}, R2={_r2:.4f}")`,
      solution: `m = LinearRegression().fit(X_tr, y_tr)
yp = m.predict(X_te)
rmse = np.sqrt(np.mean((yp - y_te)**2))
r2 = r2_score(y_te, yp)` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 5.2 — Ridge da zero',
      prompt: '<p>Implementa la soluzione ridge in forma chiusa $\\hat w = (X^T X + \\lambda I)^{-1} X^T y$ (senza bias, per semplicità). Salva in <code class="inline">w_ridge</code>. Confronta con <code class="inline">sklearn.linear_model.Ridge</code>.</p>',
      starter: `import numpy as np
from sklearn.linear_model import Ridge

np.random.seed(0)
X = np.random.randn(100, 5)
y = X @ np.array([1,2,3,4,5]) + np.random.randn(100)
lam = 2.0

# TODO
w_ridge = None
print("mio  :", w_ridge)
print("sklearn:", Ridge(alpha=lam, fit_intercept=False).fit(X, y).coef_)`,
      check: `import numpy as np
sk = Ridge(alpha=lam, fit_intercept=False).fit(X, y).coef_
if w_ridge is not None and np.allclose(w_ridge, sk, atol=1e-6):
    print("OK — identico a sklearn")
else:
    print("FAIL")`,
      solution: `w_ridge = np.linalg.solve(X.T @ X + lam*np.eye(X.shape[1]), X.T @ y)` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 5.3 — Overfit polinomiale',
      prompt: '<p>Sui punti <code class="inline">x, y</code>, trova il grado polinomiale $d \\in \\{1,2,...,10\\}$ con miglior RMSE sul test. Salva in <code class="inline">best_deg</code>.</p>',
      starter: `import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

np.random.seed(0)
x = np.linspace(-2, 2, 50).reshape(-1,1)
y = np.sin(x).ravel() + 0.1*np.random.randn(50)
xt, xe, yt, ye = train_test_split(x, y, test_size=0.3, random_state=42)

# TODO
best_deg = None
print("miglior grado:", best_deg)`,
      check: `import numpy as np
results = {}
for d in range(1, 11):
    m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(xt, yt)
    results[d] = np.sqrt(np.mean((m.predict(xe) - ye)**2))
_atteso = min(results, key=results.get)
if best_deg == _atteso:
    print(f"OK — grado ottimo {best_deg}")
else:
    print(f"FAIL — atteso {_atteso}, tuo {best_deg}")`,
      solution: `import numpy as np
r = {}
for d in range(1,11):
    m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(xt, yt)
    r[d] = np.sqrt(np.mean((m.predict(xe) - ye)**2))
best_deg = min(r, key=r.get)` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 5.4 — Diabetes dataset',
      prompt: '<p>Sul dataset diabetes di scikit-learn, addestra <code class="inline">LinearRegression</code>, <code class="inline">Ridge(alpha=1)</code> e <code class="inline">Lasso(alpha=0.1)</code>. Stampa i tre $R^2$ sul test e conta i coefficienti non-nulli del Lasso. Salva quest\'ultimo in <code class="inline">n_nonzero</code>.</p>',
      starter: `from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_diabetes(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42)

# TODO
n_nonzero = None`,
      check: `import numpy as np
lin_r2 = LinearRegression().fit(Xt, yt).score(Xe, ye)
rid_r2 = Ridge(alpha=1).fit(Xt, yt).score(Xe, ye)
las = Lasso(alpha=0.1, max_iter=5000).fit(Xt, yt)
las_r2 = las.score(Xe, ye)
_nz = int(np.sum(las.coef_ != 0))
print(f"OLS R2: {lin_r2:.4f} | Ridge R2: {rid_r2:.4f} | Lasso R2: {las_r2:.4f}")
print(f"Lasso non-zero: {_nz} feature su {len(las.coef_)}")
if n_nonzero == _nz:
    print("OK")
else:
    print(f"FAIL — atteso n_nonzero={_nz}")`,
      solution: `las = Lasso(alpha=0.1, max_iter=5000).fit(Xt, yt)
n_nonzero = int((las.coef_ != 0).sum())` },
  ],
  test: [
    { type: 'quiz', questions: [
      { q: 'Least squares equivale a MLE sotto quale ipotesi sul rumore?', options: ['Uniforme', 'Gaussiano con varianza costante', 'Laplace', 'Poisson'], answer: 1, explain: 'Rumore normale &rarr; log-likelihood = -MSE + costante.' },
      { q: 'La differenza principale tra Ridge e Lasso è:', options: ['Ridge è più veloce', 'Lasso produce soluzioni sparse (coefficienti esattamente zero)', 'Ridge non ha regolarizzazione', 'Lasso non converge'], answer: 1, explain: 'La penalità L1 induce sparsità geometricamente (spigoli del rombo).' },
      { q: '$R^2 = 0.85$ significa che il modello:', options: ['Ha 85% di accuracy', 'Spiega l\'85% della varianza di $y$', 'Ha errore medio 0.85', 'È perfetto'], answer: 1, explain: '$R^2 = 1 - \\text{SSR}/\\text{SST}$: frazione di varianza spiegata.' },
      { q: 'Se le feature sono altamente collineari, OLS soffre di:', options: ['Bias elevato', 'Varianza elevata / coefficienti instabili', 'Non converge', 'Sparsità'], answer: 1, explain: '$X^T X$ è mal condizionata; piccoli cambi di dati muovono molto $\\hat w$.' },
    ] },
    { type: 'exercise', difficulty: 'hard', title: 'Test pratico — pipeline completa',
      prompt: '<p>Sul dataset California housing (versione semplificata simulata), costruisci un modello che minimizza RMSE. Puoi usare: standardizzazione, feature polinomiali, ridge, lasso. Target: <strong>RMSE test &lt; 0.6</strong>.</p>',
      starter: `import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

np.random.seed(0)
n = 500
X = np.random.randn(n, 5)
y = (2*X[:,0] - 1.5*X[:,1] + 0.5*X[:,0]*X[:,2] + X[:,3]**2 + 0.3*np.random.randn(n))
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42)

# TODO: costruisci una pipeline, fit, calcola RMSE su test
rmse_test = None
print("RMSE test:", rmse_test)`,
      check: `if rmse_test is None:
    print("FAIL: None")
elif rmse_test < 0.6:
    print(f"OK — RMSE {rmse_test:.4f}")
else:
    print(f"FAIL — RMSE {rmse_test:.4f}, target < 0.6")`,
      solution: `pipe = Pipeline([('scaler', StandardScaler()),
                 ('poly', PolynomialFeatures(2)),
                 ('reg', Ridge(alpha=1.0))])
pipe.fit(Xt, yt)
rmse_test = np.sqrt(np.mean((pipe.predict(Xe) - ye)**2))` },
  ]
});

// =====================================================================
// LEZIONE 6 — Regressione logistica e classificazione
// =====================================================================
LESSONS.push({
  id: 'l06-logistic',
  title: 'Regressione logistica',
  tag: 'Supervisionato',
  summary: 'Il modello lineare per la classificazione. Sigmoid, cross-entropy, softmax.',
  teoria: [
    { type: 'md', content: `
<h3>6.1 Dal lineare al probabilistico</h3>
<p>Per la classificazione binaria vogliamo $P(y=1|x)$. Un modello lineare $w^T x + b$ produce numeri reali — bisogna schiacciarli in $[0,1]$. La funzione <em>sigmoide</em> fa esattamente questo:</p>
<p>$$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$</p>
<p>Proprietà utili: $\\sigma(0) = 0.5$, $\\sigma(-z) = 1 - \\sigma(z)$, $\\sigma'(z) = \\sigma(z)(1-\\sigma(z))$.</p>
<p>Il modello di regressione logistica è quindi:</p>
<p>$$P(y=1 | x; w, b) = \\sigma(w^T x + b)$$</p>

<h3>6.2 Interpretazione: log-odds</h3>
<p>Se $p = P(y=1|x)$, gli <em>odds</em> sono $p / (1-p)$ e i <em>log-odds</em> sono $\\log[p/(1-p)]$. Con la sigmoide si dimostra facilmente che:</p>
<p>$$\\log \\frac{p}{1-p} = w^T x + b$$</p>
<p>Cioè la regressione logistica modella i log-odds come funzione lineare di $x$. Un coefficiente $w_j$ è l'incremento nei log-odds per unità di $x_j$.</p>

<h3>6.3 La loss: binary cross-entropy</h3>
<p>Sotto Bernoulli, la log-likelihood di un esempio è $y \\log p + (1-y)\\log(1-p)$. Massimizzarla equivale a minimizzare:</p>
<p>$$\\mathcal{L}(w,b) = -\\frac{1}{n}\\sum_i \\left[y_i \\log \\hat p_i + (1-y_i)\\log(1-\\hat p_i)\\right]$$</p>
<p>Questa è la <em>binary cross-entropy</em> (o log-loss). Non ha soluzione chiusa, ma è <strong>convessa</strong>, quindi la discesa del gradiente trova il minimo globale.</p>

<h3>6.4 Il gradiente</h3>
<p>Con un po' di algebra si trova (per un esempio $x_i, y_i$):</p>
<p>$$\\nabla_w \\mathcal{L}_i = (\\hat p_i - y_i)\\, x_i$$</p>
<p>Sorprendentemente semplice ed identico in forma al gradiente della regressione lineare! È il motivo per cui è così comoda.</p>

<h3>6.5 Confine di decisione</h3>
<p>Si predice classe 1 se $\\hat p \\geq 0.5$, cioè se $w^T x + b \\geq 0$. Il confine di decisione è quindi un <strong>iperpiano</strong> in $\\mathbb{R}^d$. Modello lineare &rarr; confine lineare.</p>

<h3>6.6 Multi-classe: softmax</h3>
<p>Per $K$ classi, generalizziamo con $K$ vettori di pesi:</p>
<p>$$P(y=k | x) = \\frac{e^{w_k^T x}}{\\sum_{j=1}^K e^{w_j^T x}}$$</p>
<p>La loss è la <em>categorical cross-entropy</em>: $-\\sum_i \\log P(y=y_i | x_i)$. Se il target è one-hot $t_i \\in \\{0,1\\}^K$: $-\\sum_i \\sum_k t_{ik} \\log P(y=k|x_i)$.</p>

<h3>6.7 Regolarizzazione</h3>
<p>Aggiungiamo $\\lambda \\|w\\|^2$ (L2) o $\\lambda\\|w\\|_1$ (L1). In scikit-learn il parametro <code class="inline">C = 1/\\lambda</code>: valori piccoli &rarr; regolarizzazione forte.</p>

<h3>6.8 Metriche di classificazione</h3>
<p>Non basta l\'accuracy, soprattutto con classi sbilanciate. Considera:</p>
<ul>
<li><strong>Matrice di confusione</strong>: TP, FP, FN, TN</li>
<li><strong>Precision</strong>: $\\frac{TP}{TP+FP}$ — di quelli che ho detto "positivo", quanti lo erano?</li>
<li><strong>Recall</strong>: $\\frac{TP}{TP+FN}$ — dei positivi veri, quanti ho trovato?</li>
<li><strong>F1</strong>: media armonica di precision e recall</li>
<li><strong>ROC-AUC</strong>: area sotto la curva ROC; misura la separabilità indipendentemente dalla soglia</li>
</ul>
` },
    { type: 'callout', variant: 'tip', title: 'Trick numerico', content: 'Non implementare mai $\\log(\\sigma(z))$ come <code class="inline">np.log(1/(1+np.exp(-z)))</code>: overflow per $z$ molto negativo. Usa <code class="inline">-np.logaddexp(0, -z)</code> o formulazioni stabili come quelle di scikit-learn.' },
  ],
  esempi: [
    { type: 'md', content: '<h3>Esempio 1: dataset breast cancer</h3>' },
    { type: 'run', content: `from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))])
pipe.fit(Xt, yt)

y_pred = pipe.predict(Xe)
y_prob = pipe.predict_proba(Xe)[:, 1]

print(classification_report(ye, y_pred, digits=3))
print(f"ROC-AUC: {roc_auc_score(ye, y_prob):.4f}")` },
    { type: 'md', content: '<h3>Esempio 2: gradient descent per logistica</h3>' },
    { type: 'run', content: `import numpy as np

np.random.seed(0)
n = 400
X = np.random.randn(n, 2)
true_w = np.array([2.0, -1.0])
y = (X @ true_w + 0.5*np.random.randn(n) > 0).astype(int)

X_aug = np.hstack([X, np.ones((n,1))])   # bias in ultima colonna
w = np.zeros(3)

def sigmoid(z): return 1/(1+np.exp(-z))

for step in range(200):
    p = sigmoid(X_aug @ w)
    grad = X_aug.T @ (p - y) / n
    w = w - 0.5 * grad
    if step % 40 == 0:
        loss = -np.mean(y*np.log(p+1e-9) + (1-y)*np.log(1-p+1e-9))
        acc = np.mean((p > 0.5) == y)
        print(f"step {step:3d}: loss={loss:.4f}, acc={acc:.3f}, w={w.round(3)}")` },
    { type: 'md', content: '<h3>Esempio 3: confine di decisione visualizzato</h3>' },
    { type: 'run', content: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

X, y = make_moons(n_samples=200, noise=0.2, random_state=0)

fig, axes = plt.subplots(1, 2, figsize=(12,4))
for ax, poly_deg in zip(axes, [1, 5]):
    model = make_pipeline(PolynomialFeatures(poly_deg), LogisticRegression(max_iter=1000))
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 200),
                         np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 200))
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1].reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu_r', levels=20)
    ax.contour(xx, yy, Z, levels=[0.5], colors='k')
    ax.scatter(X[:,0], X[:,1], c=y, cmap='RdBu_r', edgecolor='k', s=30)
    ax.set_title(f'Logistic + poly grado {poly_deg}')
plt.tight_layout(); plt.show()` },
    { type: 'md', content: '<h3>Esempio 4: soglia e trade-off precision/recall</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score

X, y = load_breast_cancer(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
m = Pipeline([('s', StandardScaler()), ('c', LogisticRegression(max_iter=1000))]).fit(Xt, yt)
prob = m.predict_proba(Xe)[:,1]

print(f"{'soglia':>7} {'precision':>10} {'recall':>8}")
for t in [0.2, 0.3, 0.5, 0.7, 0.9]:
    pred = (prob >= t).astype(int)
    p = precision_score(ye, pred, zero_division=0)
    r = recall_score(ye, pred)
    print(f"{t:>7.2f} {p:>10.3f} {r:>8.3f}")` },
  ],
  esercizi: [
    { type: 'exercise', difficulty: 'easy', title: 'Es. 6.1 — Sigmoid manuale',
      prompt: '<p>Implementa sigmoide senza usare scipy. Deve funzionare su scalari e array numpy. Salvala come funzione <code class="inline">sigmoid</code>.</p>',
      starter: `import numpy as np
def sigmoid(z):
    # TODO
    return None

print(sigmoid(0), sigmoid(np.array([-5, 0, 5])))`,
      check: `import numpy as np
if abs(sigmoid(0) - 0.5) < 1e-9 and np.allclose(sigmoid(np.array([-5,0,5])), [1/(1+np.exp(5)), 0.5, 1/(1+np.exp(-5))]):
    print("OK")
else:
    print("FAIL")`,
      solution: `def sigmoid(z):
    return 1 / (1 + np.exp(-z))` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 6.2 — Cross-entropy',
      prompt: '<p>Implementa <code class="inline">binary_ce(y_true, y_prob)</code> che ritorna la media di $-[y\\log p + (1-y)\\log(1-p)]$. Attenzione a clippare $p$ in $[\\epsilon, 1-\\epsilon]$.</p>',
      starter: `import numpy as np
def binary_ce(y_true, y_prob, eps=1e-9):
    # TODO
    return 0.0

print(binary_ce(np.array([1,0,1,1]), np.array([0.9,0.1,0.8,0.6])))`,
      check: `import numpy as np
from sklearn.metrics import log_loss
y = np.array([1,0,1,1]); p = np.array([0.9,0.1,0.8,0.6])
if abs(binary_ce(y, p) - log_loss(y, p)) < 1e-6:
    print("OK — identica a sklearn")
else:
    print("FAIL")`,
      solution: `def binary_ce(y_true, y_prob, eps=1e-9):
    p = np.clip(y_prob, eps, 1-eps)
    return -np.mean(y_true*np.log(p) + (1-y_true)*np.log(1-p))` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 6.3 — Soglia ottima per F1',
      prompt: '<p>Data una serie di probabilità e labels, trova la soglia in <code class="inline">np.linspace(0,1,101)</code> che massimizza F1 sul test. Salva in <code class="inline">best_thresh</code> (float).</p>',
      starter: `import numpy as np
from sklearn.metrics import f1_score
np.random.seed(0)
y_true = np.random.randint(0,2,size=500)
prob = np.clip(0.5 + 0.4*(2*y_true-1) + 0.3*np.random.randn(500), 0, 1)

# TODO
best_thresh = None
print("miglior soglia:", best_thresh)`,
      check: `import numpy as np
from sklearn.metrics import f1_score
scores = {t: f1_score(y_true, (prob>=t).astype(int), zero_division=0)
          for t in np.linspace(0,1,101)}
_atteso = max(scores, key=scores.get)
if best_thresh is not None and abs(best_thresh - _atteso) < 0.02:
    print(f"OK — soglia {best_thresh:.2f}, F1={scores[_atteso]:.3f}")
else:
    print(f"FAIL — atteso {_atteso:.2f}")`,
      solution: `import numpy as np
from sklearn.metrics import f1_score
best_thresh = max(np.linspace(0,1,101),
                  key=lambda t: f1_score(y_true, (prob>=t).astype(int), zero_division=0))` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 6.4 — Multi-classe su Iris',
      prompt: '<p>Addestra una regressione logistica multi-classe su Iris. Standardizza, valuta test accuracy e ROC-AUC (one-vs-rest). Salvali in <code class="inline">acc</code> e <code class="inline">auc</code>.</p>',
      starter: `from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import numpy as np

X, y = load_iris(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# TODO
acc = None
auc = None
print(f"Accuracy: {acc}, AUC: {auc}")`,
      check: `pipe = Pipeline([('s', StandardScaler()), ('c', LogisticRegression(max_iter=1000))]).fit(Xt, yt)
_acc = pipe.score(Xe, ye)
_prob = pipe.predict_proba(Xe)
_auc = roc_auc_score(ye, _prob, multi_class='ovr')
ok = acc is not None and abs(acc-_acc)<1e-4 and abs(auc-_auc)<1e-3
print("OK" if ok else f"FAIL — atteso acc={_acc:.4f}, auc={_auc:.4f}")`,
      solution: `pipe = Pipeline([('s', StandardScaler()), ('c', LogisticRegression(max_iter=1000))]).fit(Xt, yt)
acc = pipe.score(Xe, ye)
auc = roc_auc_score(ye, pipe.predict_proba(Xe), multi_class='ovr')` },
  ],
  test: [
    { type: 'quiz', questions: [
      { q: 'La sigmoide $\\sigma(z)$ mappa $\\mathbb{R}$ in:', options: ['$[-1,1]$', '$[0,1]$', '$[0,\\infty)$', '$\\mathbb{R}$'], answer: 1, explain: 'È il suo scopo: schiacciare reali in probabilità.' },
      { q: 'Con classi molto sbilanciate (99% classe 0), un classificatore che predice sempre 0 ha:', options: ['Alta accuracy ma recall nullo sui positivi', 'Alta precision e recall', 'Bassa accuracy', 'F1 alto'], answer: 0, explain: 'Accuracy 99% ma il modello è inutile: mai identifica positivi.' },
      { q: 'La cross-entropy vs MSE per classificazione binaria:', options: ['Sono equivalenti', 'CE ha gradienti migliori (non satura come MSE)', 'MSE è preferibile', 'Nessuna delle due funziona'], answer: 1, explain: 'MSE con sigmoide dà gradienti piccoli quando il modello è molto sbagliato; CE no.' },
      { q: 'In scikit-learn, per regolarizzare più fortemente <code class="inline">LogisticRegression</code>:', options: ['Aumenta C', 'Diminuisci C', 'Cambia solver', 'Aumenta max_iter'], answer: 1, explain: 'C = 1/λ: piccolo C significa forte penalità.' },
    ] },
    { type: 'exercise', difficulty: 'hard', title: 'Test pratico — classificatore completo',
      prompt: '<p>Sul dataset breast_cancer, costruisci un classificatore che raggiunga <strong>ROC-AUC test &gt; 0.995</strong>. Puoi usare qualsiasi combinazione di scaling, feature poly, regolarizzazione. Salva l\'AUC in <code class="inline">auc_test</code>.</p>',
      starter: `from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# TODO
auc_test = None
print("AUC:", auc_test)`,
      check: `if auc_test is None:
    print("FAIL: None")
elif auc_test > 0.995:
    print(f"OK — AUC {auc_test:.4f}")
else:
    print(f"FAIL — AUC {auc_test:.4f}, target > 0.995")`,
      solution: `pipe = Pipeline([('s', StandardScaler()), ('c', LogisticRegression(C=1.0, max_iter=5000))]).fit(Xt, yt)
auc_test = roc_auc_score(ye, pipe.predict_proba(Xe)[:,1])` },
  ]
});

// =====================================================================
// LEZIONE 7 — K-Nearest Neighbors
// =====================================================================
LESSONS.push({
  id: 'l07-knn',
  title: 'K-Nearest Neighbors',
  tag: 'Supervisionato',
  summary: 'Il modello più intuitivo del ML: predici come i tuoi vicini. Non-parametrico, potente, con difetti.',
  teoria: [
    { type: 'md', content: `
<h3>7.1 L'idea</h3>
<p>Per predire $y$ su un nuovo $x$: trova i $k$ punti del training più vicini a $x$, e restituisci la loro etichetta più comune (classificazione) o la loro media (regressione). Fine. Non c'è "training" nel senso tradizionale: il modello <em>è</em> il dataset.</p>
<p>Formalmente, sia $\\mathcal{N}_k(x)$ l'insieme dei $k$ vicini di $x$ nel training:</p>
<p>$$\\hat y(x) = \\arg\\max_c \\sum_{i \\in \\mathcal{N}_k(x)} \\mathbb{1}[y_i = c] \\quad\\text{(classificazione)}$$</p>

<h3>7.2 Distanza</h3>
<p>La nozione di "vicino" richiede una metrica. Le più comuni:</p>
<ul>
<li><strong>Euclidea</strong>: $\\|x-x'\\|_2 = \\sqrt{\\sum_j (x_j - x'_j)^2}$</li>
<li><strong>Manhattan</strong>: $\\|x-x'\\|_1 = \\sum_j |x_j - x'_j|$</li>
<li><strong>Cosine</strong>: $1 - \\frac{x \\cdot x'}{\\|x\\|\\|x'\\|}$ — utile per testi</li>
<li><strong>Mahalanobis</strong>: tiene conto della covarianza delle feature</li>
</ul>

<h3>7.3 Scaling è obbligatorio</h3>
<p>La distanza euclidea è dominata dalle feature con range grande. Se una feature va da 0 a 1000 e un'altra da 0 a 1, la prima decide tutto. <strong>Standardizza sempre</strong> (o usa min-max normalization) prima di KNN.</p>

<h3>7.4 Effetto di $k$</h3>
<ul>
<li>$k = 1$: modello ad altissima varianza; segue perfettamente il training, quindi overfitta.</li>
<li>$k$ grande: modello ad alto bias; media su troppi punti diversi, sottofitta.</li>
<li>Il valore ottimale si trova con cross-validation.</li>
</ul>
<p>Rule of thumb: $k \\approx \\sqrt{n}$, ma la CV vince sempre.</p>

<h3>7.5 KNN pesato</h3>
<p>Invece di dare peso uguale a tutti i vicini, si pesa inversamente alla distanza:</p>
<p>$$\\hat y(x) = \\frac{\\sum_{i \\in \\mathcal{N}_k} \\frac{1}{d(x, x_i)} y_i}{\\sum_{i \\in \\mathcal{N}_k} \\frac{1}{d(x, x_i)}}$$</p>

<h3>7.6 La maledizione della dimensionalità</h3>
<p>In alta dimensione, "vicino" perde significato: tutti i punti sono a distanza simile. Precisamente, se hai $n$ punti in $[0,1]^d$ uniformi, la distanza minima e massima tendono a coincidere per $d \\to \\infty$. KNN è quasi inutile in $d > 20$ senza riduzione dimensionale.</p>

<h3>7.7 Efficienza</h3>
<p>Predizione naïve: $O(nd)$ per punto. Per dataset grandi si usano strutture dati:</p>
<ul>
<li><strong>KD-tree</strong>: efficiente per $d$ piccolo (fino a 20)</li>
<li><strong>Ball tree</strong>: meglio in alta dimensione</li>
<li><strong>Approssimati</strong>: HNSW, LSH per $n$ o $d$ enormi</li>
</ul>
<p>Scikit-learn sceglie automaticamente con <code class="inline">algorithm='auto'</code>.</p>
` },
    { type: 'callout', variant: 'note', title: 'Quando usare KNN', content: 'Ottimo baseline. Ottimo per dataset piccoli, poche feature, con confini di decisione irregolari. Da evitare in produzione con dataset grandi (lento a predire) o alta dimensionalità.' },
  ],
  esempi: [
    { type: 'md', content: '<h3>Esempio 1: KNN su Iris</h3>' },
    { type: 'run', content: `from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

X, y = load_iris(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

pipe = Pipeline([('s', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5))])
pipe.fit(Xt, yt)
print(f"Accuracy k=5: {pipe.score(Xe, ye):.4f}")

# CV per trovare k ottimale
for k in [1, 3, 5, 7, 15, 25]:
    p = Pipeline([('s', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=k))])
    scores = cross_val_score(p, Xt, yt, cv=5)
    print(f"k={k:2d}: CV mean {scores.mean():.4f} +/- {scores.std():.4f}")` },
    { type: 'md', content: '<h3>Esempio 2: KNN from scratch</h3>' },
    { type: 'run', content: `import numpy as np
from collections import Counter

def knn_predict(X_train, y_train, X_new, k=3):
    preds = []
    for x in X_new:
        # distanze da tutti i punti di train
        d = np.sqrt(((X_train - x)**2).sum(axis=1))
        # indici dei k più vicini
        idx = np.argsort(d)[:k]
        # voto di maggioranza
        preds.append(Counter(y_train[idx]).most_common(1)[0][0])
    return np.array(preds)

# test
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
X, y = load_iris(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

pred = knn_predict(Xt, yt, Xe, k=5)
print(f"Accuracy: {np.mean(pred == ye):.4f}")` },
    { type: 'md', content: '<h3>Esempio 3: effetto di $k$ visualizzato</h3>' },
    { type: 'run', content: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier

X, y = make_moons(n_samples=200, noise=0.25, random_state=0)
fig, axes = plt.subplots(1, 3, figsize=(13,4))
for ax, k in zip(axes, [1, 15, 50]):
    m = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.4, X[:,0].max()+0.4, 200),
                         np.linspace(X[:,1].min()-0.4, X[:,1].max()+0.4, 200))
    Z = m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu_r')
    ax.scatter(X[:,0], X[:,1], c=y, cmap='RdBu_r', edgecolor='k', s=25)
    ax.set_title(f'k={k}')
plt.tight_layout(); plt.show()
print("k=1: overfit (bordo frastagliato)")
print("k=15: bilanciato")
print("k=50: underfit (bordo troppo liscio)")` },
    { type: 'md', content: '<h3>Esempio 4: importanza dello scaling</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(0)
X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, random_state=0)
# distruggi la scala: una feature x 1000
X_bad = X.copy(); X_bad[:,1] *= 1000

Xt, Xe, yt, ye = train_test_split(X_bad, y, test_size=0.3, random_state=0, stratify=y)
print("Senza scaling:", KNeighborsClassifier(5).fit(Xt, yt).score(Xe, ye))

sc = StandardScaler().fit(Xt)
Xt_s, Xe_s = sc.transform(Xt), sc.transform(Xe)
print("Con scaling: ", KNeighborsClassifier(5).fit(Xt_s, yt).score(Xe_s, ye))` },
  ],
  esercizi: [
    { type: 'exercise', difficulty: 'easy', title: 'Es. 7.1 — KNN a mano per un punto',
      prompt: '<p>Dato il training e un nuovo punto <code class="inline">q = [1, 1]</code>, trova i 3 vicini più vicini con distanza euclidea. Salva gli <strong>indici</strong> in <code class="inline">indices</code>.</p>',
      starter: `import numpy as np
X = np.array([[0,0],[2,2],[3,3],[1,2],[5,5],[0.5,0.5]])
q = np.array([1, 1])

# TODO
indices = None
print("Indici:", indices)`,
      check: `import numpy as np
d = np.sqrt(((X - q)**2).sum(1))
_atteso = np.argsort(d)[:3]
if indices is not None and set(list(indices)) == set(list(_atteso)):
    print(f"OK — vicini {list(indices)}")
else:
    print(f"FAIL — atteso {list(_atteso)}")`,
      solution: `d = np.sqrt(((X - q)**2).sum(1))
indices = np.argsort(d)[:3]` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 7.2 — KNN regressore',
      prompt: '<p>Implementa <code class="inline">knn_regress(X, y, x, k)</code> che ritorna la media dei $k$ target dei vicini di <code class="inline">x</code>. Testalo.</p>',
      starter: `import numpy as np
def knn_regress(X, y, x, k):
    # TODO
    return None

Xtr = np.array([[0],[1],[2],[3],[4],[5]])
ytr = np.array([0., 1., 4., 9., 16., 25.])
print(knn_regress(Xtr, ytr, np.array([2.5]), 2))  # atteso (4+9)/2 = 6.5`,
      check: `import numpy as np
Xtr = np.array([[0],[1],[2],[3],[4],[5]])
ytr = np.array([0., 1., 4., 9., 16., 25.])
r1 = knn_regress(Xtr, ytr, np.array([2.5]), 2)
r2 = knn_regress(Xtr, ytr, np.array([0.1]), 3)
if abs(r1 - 6.5) < 1e-6 and abs(r2 - (0+1+4)/3) < 1e-6:
    print("OK")
else:
    print(f"FAIL: r1={r1} (atteso 6.5), r2={r2} (atteso {(0+1+4)/3:.4f})")`,
      solution: `def knn_regress(X, y, x, k):
    d = np.linalg.norm(X - x, axis=1)
    idx = np.argsort(d)[:k]
    return y[idx].mean()` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 7.3 — Trova k ottimale',
      prompt: '<p>Con cross-validation a 5 fold su Iris, trova il $k \\in \\{1,3,5,\\dots,29\\}$ che massimizza l\'accuracy. Salva in <code class="inline">best_k</code>.</p>',
      starter: `from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

X, y = load_iris(return_X_y=True)

# TODO
best_k = None
print("k ottimo:", best_k)`,
      check: `from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
scores = {}
for k in range(1, 30, 2):
    p = Pipeline([('s', StandardScaler()), ('k', KNeighborsClassifier(k))])
    scores[k] = cross_val_score(p, X, y, cv=5).mean()
_atteso_top = sorted(scores, key=scores.get, reverse=True)[:3]
if best_k in _atteso_top:
    print(f"OK — k={best_k} tra i top-3")
else:
    print(f"FAIL — top-3 attesi {_atteso_top}, tuo {best_k}")`,
      solution: `scores = {}
for k in range(1, 30, 2):
    p = Pipeline([('s', StandardScaler()), ('k', KNeighborsClassifier(k))])
    scores[k] = cross_val_score(p, X, y, cv=5).mean()
best_k = max(scores, key=scores.get)` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 7.4 — Distanza pesata',
      prompt: '<p>Implementa <code class="inline">knn_weighted(X_tr, y_tr, X_te, k)</code>: la predizione per ogni test è la media pesata inversamente alla distanza dei $k$ vicini. Attenzione ai zero-division: se un vicino è a distanza 0, usa quello direttamente.</p>',
      starter: `import numpy as np
def knn_weighted(X_tr, y_tr, X_te, k=3):
    # TODO
    return None

# test
X_tr = np.array([[0],[1],[2],[3]], dtype=float)
y_tr = np.array([10., 20., 30., 40.])
X_te = np.array([[0.5],[2.5]])
print(knn_weighted(X_tr, y_tr, X_te, 2))
# per 0.5: vicini 0 e 1, distanze 0.5 e 0.5 -> media (10+20)/2 = 15
# per 2.5: vicini 2 e 3, distanze 0.5 e 0.5 -> media (30+40)/2 = 35`,
      check: `import numpy as np
res = knn_weighted(X_tr, y_tr, X_te, 2)
if res is not None and np.allclose(res, [15., 35.]):
    print("OK")
else:
    print(f"FAIL — atteso [15, 35], tuo {res}")`,
      solution: `def knn_weighted(X_tr, y_tr, X_te, k=3):
    out = []
    for x in X_te:
        d = np.linalg.norm(X_tr - x, axis=1)
        idx = np.argsort(d)[:k]
        d_k = d[idx]
        y_k = y_tr[idx]
        if (d_k == 0).any():
            out.append(y_k[d_k == 0].mean())
        else:
            w = 1 / d_k
            out.append((w * y_k).sum() / w.sum())
    return np.array(out)` },
  ],
  test: [
    { type: 'quiz', questions: [
      { q: 'Con $k=1$, l\'accuracy sul training è:', options: ['0%', 'Sempre 100%', 'Circa 50%', 'Dipende dai dati'], answer: 1, explain: 'Ogni punto è il vicino più vicino di se stesso — memorizzazione perfetta.' },
      { q: 'La maledizione della dimensionalità colpisce KNN perché:', options: ['La distanza euclidea è lenta', 'Le distanze si concentrano — vicini/lontani indistinguibili', 'La memoria non basta', 'Il gradiente svanisce'], answer: 1, explain: 'In alta dimensione tutto è quasi equidistante.' },
      { q: 'KNN è definito "lazy learning" perché:', options: ['Non fa nulla', 'Non ha una fase di training vera e propria', 'È lento a implementarsi', 'Non usa gradienti'], answer: 1, explain: 'Il calcolo è tutto rimandato al momento della predizione.' },
      { q: 'La distanza cosine è più adatta di quella euclidea quando:', options: ['I dati sono numeri interi', 'Interessa la direzione più che la magnitudine (es. testi TF-IDF)', 'Ci sono pochi dati', 'Ci sono outlier'], answer: 1, explain: 'Cosine ignora la norma dei vettori.' },
    ] },
    { type: 'exercise', difficulty: 'hard', title: 'Test pratico — pipeline KNN',
      prompt: '<p>Sul dataset digits (cifre 8x8), costruisci un classificatore KNN con scaling e trova la miglior combinazione di $k \\in \\{3,5,7,9\\}$ e weights $\\in \\{$"uniform","distance"$\\}$ via GridSearchCV. Salva la migliore accuracy test in <code class="inline">acc_test</code> (target &gt; 0.98).</p>',
      starter: `from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split

X, y = load_digits(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# TODO
acc_test = None
print("Accuracy test:", acc_test)`,
      check: `if acc_test is None:
    print("FAIL: None")
elif acc_test > 0.98:
    print(f"OK — {acc_test:.4f}")
else:
    print(f"FAIL — {acc_test:.4f}, target > 0.98")`,
      solution: `pipe = Pipeline([('s', StandardScaler()), ('k', KNeighborsClassifier())])
grid = {'k__n_neighbors':[3,5,7,9], 'k__weights':['uniform','distance']}
gs = GridSearchCV(pipe, grid, cv=5).fit(Xt, yt)
acc_test = gs.score(Xe, ye)` },
  ]
});

// =====================================================================
// LEZIONE 8 — Alberi di decisione e Random Forest
// =====================================================================
LESSONS.push({
  id: 'l08-trees',
  title: 'Alberi di decisione e Random Forest',
  tag: 'Supervisionato',
  summary: 'Modelli non lineari, interpretabili, potenti. La base dei modelli ensemble che vincono le Kaggle.',
  teoria: [
    { type: 'md', content: `
<h3>8.1 Cos'è un albero di decisione</h3>
<p>Una struttura a nodi in cui ogni nodo interno è un test su una feature (es. "$x_3 \\leq 2.5$?"), e le foglie contengono predizioni. Predire significa scendere l'albero seguendo i test.</p>
<p>Un albero binario partiziona $\\mathbb{R}^d$ in regioni rettangolari e assegna una predizione costante ad ogni regione.</p>

<h3>8.2 Come si costruisce (greedy)</h3>
<p>Ad ogni nodo, tra tutte le possibili scelte (feature $j$, soglia $t$), scegli quella che <em>massimizza la riduzione di impurità</em> dei figli:</p>
<p>$$\\Delta = I(\\text{parent}) - \\left(\\frac{n_L}{n} I(L) + \\frac{n_R}{n} I(R)\\right)$$</p>
<p>Poi ricorri su ogni figlio finché non si soddisfa un criterio di stop (profondità max, dimensione minima foglia, purezza raggiunta).</p>

<h3>8.3 Misure di impurità</h3>
<p>Per classificazione (con $p_c$ = frazione di classe $c$):</p>
<ul>
<li><strong>Gini</strong>: $1 - \\sum_c p_c^2$. Zero se puro, massimo con classi uniformi.</li>
<li><strong>Entropia</strong>: $-\\sum_c p_c \\log p_c$.</li>
</ul>
<p>Praticamente equivalenti. Per regressione si usa la varianza (o MSE).</p>

<h3>8.4 Overfitting</h3>
<p>Un albero cresciuto senza limiti memorizza il training set: profondità $= n-1$ nel caso estremo. Contromisure:</p>
<ul>
<li><strong>Pre-pruning</strong>: <code class="inline">max_depth</code>, <code class="inline">min_samples_split</code>, <code class="inline">min_samples_leaf</code></li>
<li><strong>Post-pruning</strong>: cost-complexity pruning con parametro $\\alpha$</li>
</ul>

<h3>8.5 Pro e contro degli alberi singoli</h3>
<p><strong>Pro</strong>: interpretabili, robusti a scala e outlier, gestiscono categoriche, senza bisogno di normalizzazione.</p>
<p><strong>Contro</strong>: alta varianza (piccole modifiche ai dati &rarr; alberi molto diversi), tendono a overfittare, non catturano bene relazioni lineari o additive.</p>

<h3>8.6 Ensemble: bagging e Random Forest</h3>
<p><strong>Bagging</strong> (Bootstrap Aggregating): allena $B$ alberi su $B$ campioni bootstrap del training. Aggrega con media (regressione) o voto (classificazione). Riduce la varianza.</p>
<p><strong>Random Forest</strong>: bagging + a ogni split considera solo un sottoinsieme casuale di feature (tipicamente $\\sqrt{d}$). Questo decorrelazione ulteriore riduce ancora la varianza.</p>
<p>Il beneficio è dimostrabile: se hai $B$ predittori i.i.d. con varianza $\\sigma^2$ e correlazione $\\rho$, la varianza della media è:</p>
<p>$$\\sigma^2_{\\text{avg}} = \\rho \\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2$$</p>
<p>Random Forest cerca di ridurre $\\rho$.</p>

<h3>8.7 Boosting</h3>
<p>Diverso dal bagging: costruisce alberi <em>in sequenza</em>, ognuno concentrato sugli errori del precedente. AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost. Tipicamente il non-plus-ultra per dati tabulari.</p>

<h3>8.8 Feature importance</h3>
<p>Ogni feature ha un\'importanza data dalla riduzione totale di impurità che ha causato attraverso tutti gli split. Utile per interpretare il modello, ma con limiti (biased verso feature con più valori possibili).</p>
` },
    { type: 'callout', variant: 'tip', title: 'Sui dati tabulari', content: 'Nel 2026 come nel 2016, per la maggior parte dei problemi tabulari (righe = esempi, colonne = feature), un gradient boosting ben tarato (XGBoost, LightGBM) batte tutto: reti neurali comprese. Se vuoi vincere una Kaggle su tabulari, parti da lì.' },
  ],
  esempi: [
    { type: 'md', content: '<h3>Esempio 1: albero singolo su Iris</h3>' },
    { type: 'run', content: `from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
Xt, Xe, yt, ye = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target)

tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xt, yt)
print(f"Accuracy: {tree.score(Xe, ye):.4f}")
print("\\nAlbero:")
print(export_text(tree, feature_names=list(iris.feature_names)))` },
    { type: 'md', content: '<h3>Esempio 2: bias/variance con la profondità</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import cross_val_score

X, y = make_moons(n_samples=500, noise=0.3, random_state=0)
print("depth | train | CV mean +/- std")
for d in [1, 2, 5, 10, None]:
    m = DecisionTreeClassifier(max_depth=d, random_state=0)
    m.fit(X, y)
    tr = m.score(X, y)
    cv = cross_val_score(m, X, y, cv=5)
    print(f"{str(d):>5s} | {tr:.3f} | {cv.mean():.3f} +/- {cv.std():.3f}")` },
    { type: 'md', content: '<h3>Esempio 3: Random Forest e feature importance</h3>' },
    { type: 'run', content: `from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np

data = load_wine()
X, y = data.data, data.target
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xt, yt)
print(f"Accuracy: {rf.score(Xe, ye):.4f}")

imp = sorted(zip(rf.feature_importances_, data.feature_names), reverse=True)
print("\\nTop 5 feature:")
for i, (v, n) in enumerate(imp[:5]):
    print(f"  {i+1}. {n:30s} {v:.4f}")` },
    { type: 'md', content: '<h3>Esempio 4: bagging da zero</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from scipy.stats import mode

X, y = make_classification(n_samples=500, n_features=8, random_state=0)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42)

n = len(Xt); B = 50
rng = np.random.default_rng(0)
trees = []
for _ in range(B):
    idx = rng.choice(n, n, replace=True)  # bootstrap
    t = DecisionTreeClassifier(random_state=None).fit(Xt[idx], yt[idx])
    trees.append(t)

# voto di maggioranza
preds = np.array([t.predict(Xe) for t in trees])
final = mode(preds, axis=0, keepdims=False).mode
print(f"Bagging accuracy: {np.mean(final == ye):.4f}")
print(f"Singolo albero:   {DecisionTreeClassifier(random_state=0).fit(Xt, yt).score(Xe, ye):.4f}")` },
  ],
  esercizi: [
    { type: 'exercise', difficulty: 'easy', title: 'Es. 8.1 — Gini a mano',
      prompt: '<p>Implementa <code class="inline">gini(labels)</code>: dato un array di etichette, ritorna l\'impurità di Gini $1 - \\sum p_c^2$.</p>',
      starter: `import numpy as np
def gini(labels):
    # TODO
    return 0.0
print(gini(np.array([0,0,0,1,1])))  # atteso 1 - (0.6^2 + 0.4^2) = 0.48
print(gini(np.array([0,0,0,0])))    # atteso 0
print(gini(np.array([0,1,0,1])))    # atteso 0.5`,
      check: `import numpy as np
if abs(gini(np.array([0,0,0,1,1])) - 0.48) < 1e-6 and \
   gini(np.array([0,0,0,0])) == 0 and \
   abs(gini(np.array([0,1,0,1])) - 0.5) < 1e-6:
    print("OK")
else:
    print("FAIL")`,
      solution: `def gini(labels):
    _, c = np.unique(labels, return_counts=True)
    p = c / len(labels)
    return 1 - (p**2).sum()` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 8.2 — Miglior split',
      prompt: '<p>Data una feature 1-D <code class="inline">x</code> e etichette <code class="inline">y</code>, trova la soglia $t$ (tra i valori unici di $x$) che minimizza il Gini pesato dei due figli. Salva in <code class="inline">best_t</code>.</p>',
      starter: `import numpy as np
def gini(l):
    _,c = np.unique(l, return_counts=True); p=c/len(l); return 1-(p**2).sum()
x = np.array([1,2,3,4,5,6,7,8])
y = np.array([0,0,0,0,1,1,1,1])

# TODO
best_t = None
print("soglia ottima:", best_t)`,
      check: `import numpy as np
def gini(l):
    _,c = np.unique(l, return_counts=True); p=c/len(l); return 1-(p**2).sum()
_thresholds = np.unique(x)
_scores = {}
for t in _thresholds:
    L, R = y[x<=t], y[x>t]
    if len(L)==0 or len(R)==0: continue
    _scores[t] = (len(L)*gini(L) + len(R)*gini(R))/len(y)
_atteso = min(_scores, key=_scores.get)
if best_t == _atteso:
    print(f"OK — soglia {best_t}, gini {_scores[best_t]:.4f}")
else:
    print(f"FAIL — atteso {_atteso}")`,
      solution: `def score(t):
    L,R = y[x<=t], y[x>t]
    if len(L)==0 or len(R)==0: return 1
    return (len(L)*gini(L) + len(R)*gini(R))/len(y)
best_t = min(np.unique(x), key=score)` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 8.3 — RF vs singolo albero',
      prompt: '<p>Su make_moons, confronta accuracy di <code class="inline">DecisionTreeClassifier(max_depth=None)</code> e <code class="inline">RandomForestClassifier(n_estimators=100)</code> su un test set. Verifica che RF batta l\'albero.</p>',
      starter: `import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=1000, noise=0.35, random_state=0)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42)

# TODO
acc_tree = None
acc_rf = None
print(f"Tree: {acc_tree}, RF: {acc_rf}")`,
      check: `if acc_tree is None or acc_rf is None:
    print("FAIL: None")
elif acc_rf > acc_tree:
    print(f"OK — RF {acc_rf:.4f} > Tree {acc_tree:.4f}")
else:
    print(f"WARN — con seed diversi RF batte quasi sempre; qui tree={acc_tree:.4f}, rf={acc_rf:.4f}")`,
      solution: `acc_tree = DecisionTreeClassifier(random_state=0).fit(Xt, yt).score(Xe, ye)
acc_rf = RandomForestClassifier(n_estimators=100, random_state=0).fit(Xt, yt).score(Xe, ye)` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 8.4 — Feature importance',
      prompt: '<p>Su un dataset in cui solo 3 di 10 feature contano davvero (le altre sono rumore), verifica che RF assegni importanza alta alle prime 3. Salva in <code class="inline">top3</code> la lista degli indici delle 3 feature più importanti.</p>',
      starter: `import numpy as np
from sklearn.ensemble import RandomForestRegressor

np.random.seed(0)
n = 1000
X = np.random.randn(n, 10)
y = 3*X[:,0] - 2*X[:,1] + X[:,2]**2 + 0.1*np.random.randn(n)

# TODO: allena RF, estrai importanze, top 3
top3 = None
print("top 3:", top3)`,
      check: `if top3 is not None and set(top3) == {0, 1, 2}:
    print(f"OK — top {top3}")
else:
    print(f"FAIL — atteso {{0,1,2}}, tuo {top3}")`,
      solution: `rf = RandomForestRegressor(n_estimators=200, random_state=0).fit(X, y)
top3 = list(np.argsort(rf.feature_importances_)[-3:])` },
  ],
  test: [
    { type: 'quiz', questions: [
      { q: 'Un albero senza limiti di profondità su training di $n$ esempi tipicamente ha accuracy training:', options: ['~ 0.5', 'Perfetta (100%)', 'Uguale a Random Forest', 'Zero'], answer: 1, explain: 'Memorizza ogni punto in una foglia dedicata.' },
      { q: 'Random Forest riduce principalmente:', options: ['Il bias', 'La varianza', 'Entrambi in ugual misura', 'L\'entropia'], answer: 1, explain: 'Il bagging riduce la varianza degli alberi ad alta varianza.' },
      { q: 'Il "random" in Random Forest deriva da:', options: ['Solo bootstrap dei dati', 'Solo sottoinsiemi di feature', 'Entrambi', 'Random init dei pesi'], answer: 2, explain: 'Bootstrap sui dati + random subset di feature ad ogni split.' },
      { q: 'Il boosting differisce dal bagging perché:', options: ['Non usa alberi', 'Costruisce modelli in sequenza focalizzati sugli errori', 'Non ha regolarizzazione', 'È più veloce'], answer: 1, explain: 'Sequenza vs parallelo. Ogni modello corregge il precedente.' },
    ] },
    { type: 'exercise', difficulty: 'hard', title: 'Test pratico — RF su Wine',
      prompt: '<p>Sul dataset wine, addestra un RandomForest e trova via GridSearch la miglior combinazione di <code class="inline">n_estimators \\in \\{50, 200, 500\\}</code> e <code class="inline">max_depth \\in \\{None, 5, 10\\}</code>. Salva l\'accuracy test in <code class="inline">acc_final</code> (target &gt; 0.96).</p>',
      starter: `from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

X, y = load_wine(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# TODO
acc_final = None`,
      check: `if acc_final is None: print("FAIL: None")
elif acc_final > 0.96: print(f"OK — {acc_final:.4f}")
else: print(f"FAIL — {acc_final:.4f}")`,
      solution: `grid = {'n_estimators':[50,200,500], 'max_depth':[None,5,10]}
gs = GridSearchCV(RandomForestClassifier(random_state=0), grid, cv=5).fit(Xt, yt)
acc_final = gs.score(Xe, ye)` },
  ]
});

// =====================================================================
// LEZIONE 9 — Support Vector Machines
// =====================================================================
LESSONS.push({
  id: 'l09-svm',
  title: 'Support Vector Machines',
  tag: 'Supervisionato',
  summary: 'Massimizzare il margine, il kernel trick, la dualità. Uno dei modelli più belli teoricamente.',
  teoria: [
    { type: 'md', content: `
<h3>9.1 L'intuizione: massimo margine</h3>
<p>Con due classi linearmente separabili in $\\mathbb{R}^d$, ci sono infiniti iperpiani che le dividono. Quale scegliere? L'SVM sceglie quello che <strong>massimizza la distanza dai punti più vicini di ciascuna classe</strong>. Questa distanza è il <em>margine</em>.</p>
<p>Un margine grande è associato a migliore generalizzazione (teoria di Vapnik-Chervonenkis).</p>

<h3>9.2 Formulazione primale (hard margin)</h3>
<p>Con etichette $y_i \\in \\{-1, +1\\}$, l'iperpiano è $w^T x + b = 0$. La distanza dal punto $x_i$ è $|w^T x_i + b|/\\|w\\|$. Se normalizziamo affinché i punti più vicini soddisfino $|w^T x + b| = 1$, il margine è $2/\\|w\\|$.</p>
<p>Massimizzare il margine equivale a minimizzare $\\|w\\|^2$:</p>
<p>$$\\min_{w,b} \\tfrac{1}{2}\\|w\\|^2 \\quad\\text{sotto vincolo}\\quad y_i(w^T x_i + b) \\geq 1\\ \\forall i$$</p>
<p>È un problema di programmazione quadratica convessa.</p>

<h3>9.3 Soft margin</h3>
<p>Se i dati non sono linearmente separabili, introduciamo variabili di slack $\\xi_i \\geq 0$:</p>
<p>$$\\min_{w,b,\\xi} \\tfrac{1}{2}\\|w\\|^2 + C\\sum_i \\xi_i \\quad\\text{s.t.}\\quad y_i(w^T x_i + b) \\geq 1 - \\xi_i$$</p>
<p>Il parametro $C$ controlla il trade-off: $C$ grande &rarr; poca tolleranza agli errori (rischio overfitting); $C$ piccolo &rarr; margine largo, più errori tollerati.</p>

<h3>9.4 Formulazione duale</h3>
<p>Con i moltiplicatori di Lagrange, si arriva alla forma duale in cui i dati appaiono solo tramite prodotti scalari $x_i^T x_j$:</p>
<p>$$\\max_\\alpha \\sum_i \\alpha_i - \\tfrac{1}{2}\\sum_{i,j}\\alpha_i \\alpha_j y_i y_j x_i^T x_j \\quad\\text{s.t.}\\ 0 \\leq \\alpha_i \\leq C,\\ \\sum_i \\alpha_i y_i = 0$$</p>
<p>Solo gli $x_i$ con $\\alpha_i > 0$ contribuiscono: quelli sono i <em>support vector</em>. In pratica, il modello dipende solo da un sottoinsieme (a volte molto piccolo) dei dati.</p>

<h3>9.5 Il kernel trick</h3>
<p>Poiché i dati appaiono solo come prodotti scalari, possiamo <em>sostituire</em> $x^T x'$ con $K(x, x') = \\phi(x)^T \\phi(x')$ dove $\\phi$ mappa in uno spazio ad alta (anche infinita) dimensione. Senza mai calcolare $\\phi$ esplicitamente.</p>
<p>Kernel comuni:</p>
<ul>
<li><strong>Lineare</strong>: $K(x,x') = x^T x'$</li>
<li><strong>Polinomiale</strong>: $K(x,x') = (\\gamma x^T x' + r)^d$</li>
<li><strong>RBF (Gaussiano)</strong>: $K(x,x') = e^{-\\gamma \\|x-x'\\|^2}$ — mappa in uno spazio di dimensione infinita</li>
<li><strong>Sigmoid</strong>: $\\tanh(\\gamma x^T x' + r)$</li>
</ul>
<p>Con RBF, l'SVM può approssimare qualsiasi funzione — è un modello universale.</p>

<h3>9.6 Parametri chiave (in scikit-learn)</h3>
<ul>
<li><code class="inline">C</code>: regolarizzazione (inversa)</li>
<li><code class="inline">gamma</code>: larghezza del kernel RBF. Grande $\\gamma$ &rarr; kernel stretto &rarr; overfitting</li>
<li><code class="inline">kernel</code>: 'linear', 'poly', 'rbf', 'sigmoid'</li>
</ul>
<p>La combinazione (C, gamma) va tarata con GridSearchCV. È la coppia più importante.</p>

<h3>9.7 Regressione: SVR</h3>
<p>Versione regressiva: cerca una funzione $f(x) = w^T x + b$ tale che $|y_i - f(x_i)| \\leq \\varepsilon$ per la maggior parte dei punti. Chiamata $\\varepsilon$-tube regression.</p>
` },
    { type: 'callout', variant: 'note', title: 'Punti forti/deboli', content: 'SVM è potente su dataset di dimensione medio-piccola (fino a decine di migliaia). Su dataset molto grandi il costo O(n² o n³) del training diventa proibitivo — meglio SGD o gradient boosting.' },
  ],
  esempi: [
    { type: 'md', content: '<h3>Esempio 1: SVM lineare</h3>' },
    { type: 'run', content: `from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, random_state=0)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

pipe = Pipeline([('s', StandardScaler()), ('svm', SVC(kernel='linear', C=1.0))]).fit(Xt, yt)
print(f"Accuracy: {pipe.score(Xe, ye):.4f}")
print(f"Numero di support vectors: {pipe.named_steps['svm'].n_support_}")` },
    { type: 'md', content: '<h3>Esempio 2: kernel RBF vs lineare su dati non lineari</h3>' },
    { type: 'run', content: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, kernel in zip(axes, ['linear', 'rbf']):
    m = SVC(kernel=kernel, C=1.0, gamma='scale').fit(X, y)
    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.3, X[:,0].max()+0.3, 200),
                         np.linspace(X[:,1].min()-0.3, X[:,1].max()+0.3, 200))
    Z = m.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=20, cmap='RdBu_r', alpha=0.5)
    ax.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)
    ax.scatter(X[:,0], X[:,1], c=y, cmap='RdBu_r', edgecolor='k')
    ax.set_title(f'kernel={kernel}, acc train={m.score(X,y):.3f}')
plt.tight_layout(); plt.show()` },
    { type: 'md', content: '<h3>Esempio 3: effetto di C e gamma</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
print(f"{'C':>6} {'gamma':>6} {'CV mean':>10}")
for C in [0.1, 1.0, 10.0]:
    for g in [0.1, 1.0, 10.0]:
        p = Pipeline([('s', StandardScaler()), ('svm', SVC(C=C, gamma=g))])
        s = cross_val_score(p, X, y, cv=5).mean()
        print(f"{C:>6.1f} {g:>6.1f} {s:>10.4f}")` },
    { type: 'md', content: '<h3>Esempio 4: kernel trick manuale</h3><p>Costruiamo il kernel polinomiale di grado 2 esplicitamente vs implicitamente.</p>' },
    { type: 'run', content: `import numpy as np
np.random.seed(0)
X = np.random.randn(5, 2)

# esplicito: mappa a phi(x) = (1, sqrt(2)x1, sqrt(2)x2, x1^2, sqrt(2)x1x2, x2^2)
def phi(x):
    return np.array([1, np.sqrt(2)*x[0], np.sqrt(2)*x[1], x[0]**2, np.sqrt(2)*x[0]*x[1], x[1]**2])

Phi = np.array([phi(x) for x in X])
K_explicit = Phi @ Phi.T

# implicito: K(x, x') = (1 + x^T x')^2
K_implicit = (1 + X @ X.T)**2

print("Max differenza:", np.abs(K_explicit - K_implicit).max())
print("=> stesso kernel, ma implicito non richiede di calcolare phi!")` },
  ],
  esercizi: [
    { type: 'exercise', difficulty: 'easy', title: 'Es. 9.1 — SVM lineare vs logistica',
      prompt: '<p>Su breast_cancer, addestra un SVM lineare (<code class="inline">SVC(kernel="linear")</code>) e una LogisticRegression. Confronta accuracy test. Salva in <code class="inline">acc_svm, acc_lr</code>.</p>',
      starter: `from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

acc_svm = None
acc_lr = None
print(acc_svm, acc_lr)`,
      check: `if acc_svm is not None and acc_lr is not None and acc_svm > 0.9 and acc_lr > 0.9:
    print(f"OK — SVM {acc_svm:.4f}, LR {acc_lr:.4f}")
else:
    print("FAIL — devono essere entrambe > 0.9")`,
      solution: `p1 = Pipeline([('s', StandardScaler()), ('c', SVC(kernel='linear'))]).fit(Xt, yt)
p2 = Pipeline([('s', StandardScaler()), ('c', LogisticRegression(max_iter=5000))]).fit(Xt, yt)
acc_svm = p1.score(Xe, ye); acc_lr = p2.score(Xe, ye)` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 9.2 — Kernel gaussiano manuale',
      prompt: '<p>Implementa <code class="inline">rbf_kernel(X1, X2, gamma)</code> senza usare sklearn. Deve restituire una matrice $K \\in \\mathbb{R}^{n_1 \\times n_2}$ con $K_{ij} = e^{-\\gamma \\|x_i - x_j\\|^2}$.</p>',
      starter: `import numpy as np
def rbf_kernel(X1, X2, gamma):
    # TODO
    return None

X1 = np.array([[0,0],[1,1]])
X2 = np.array([[0,0],[2,2],[1,0]])
print(rbf_kernel(X1, X2, gamma=0.5))`,
      check: `import numpy as np
from sklearn.metrics.pairwise import rbf_kernel as sk_rbf
X1 = np.random.randn(4, 3); X2 = np.random.randn(5, 3)
K_mine = rbf_kernel(X1, X2, 0.7)
K_sk = sk_rbf(X1, X2, gamma=0.7)
if K_mine is not None and np.allclose(K_mine, K_sk):
    print("OK — identico a sklearn")
else:
    print("FAIL")`,
      solution: `def rbf_kernel(X1, X2, gamma):
    sq_dist = ((X1[:,None,:] - X2[None,:,:])**2).sum(-1)
    return np.exp(-gamma * sq_dist)` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 9.3 — Grid search SVM RBF',
      prompt: '<p>Su make_moons rumoroso, fai grid search su $C \\in \\{0.1,1,10\\}$ e $\\gamma \\in \\{0.1,1,10\\}$ con CV a 5 fold. Salva la miglior CV score in <code class="inline">best_cv</code>.</p>',
      starter: `import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = make_moons(n_samples=500, noise=0.3, random_state=0)

# TODO
best_cv = None
print(best_cv)`,
      check: `if best_cv is not None and best_cv > 0.85:
    print(f"OK — {best_cv:.4f}")
else:
    print(f"FAIL — {best_cv}")`,
      solution: `pipe = Pipeline([('s', StandardScaler()), ('svm', SVC())])
grid = {'svm__C':[0.1,1,10], 'svm__gamma':[0.1,1,10]}
gs = GridSearchCV(pipe, grid, cv=5).fit(X, y)
best_cv = gs.best_score_` },
  ],
  test: [
    { type: 'quiz', questions: [
      { q: 'Un support vector è:', options: ['Un vettore casuale', 'Un punto del training con $\\alpha_i > 0$, cioè che tocca o viola il margine', 'Un iperpiano', 'Un\'etichetta'], answer: 1, explain: 'Solo i punti che contribuiscono alla soluzione del duale.' },
      { q: 'Il kernel trick permette di:', options: ['Velocizzare gradient descent', 'Calcolare prodotti scalari in spazi ad alta dimensione senza mappare esplicitamente', 'Ridurre la dimensione', 'Regolarizzare'], answer: 1, explain: 'La formula duale usa solo prodotti scalari; li sostituiamo con $K$.' },
      { q: 'In SVM, un $C$ molto grande implica:', options: ['Margine largo, molti errori tollerati', 'Margine stretto, pochi errori tollerati, rischio overfitting', 'Kernel RBF automatico', 'Nessun effetto'], answer: 1, explain: '$C$ pesa la penalità sugli errori: grande = severi.' },
      { q: 'Il kernel RBF mappa in uno spazio di dimensione:', options: ['1', 'Uguale a $d$', 'Uguale a $n$', 'Infinita'], answer: 3, explain: 'Espansione di Taylor di $e^x$ produce infiniti termini.' },
    ] },
    { type: 'exercise', difficulty: 'hard', title: 'Test pratico — digits con SVM',
      prompt: '<p>Sul dataset digits, addestra un SVM con kernel RBF, standardizzazione e grid search su C, gamma. Target accuracy test > 0.98. Salva in <code class="inline">acc</code>.</p>',
      starter: `from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split

X, y = load_digits(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

acc = None`,
      check: `if acc is None: print("FAIL: None")
elif acc > 0.98: print(f"OK — {acc:.4f}")
else: print(f"FAIL — {acc:.4f}")`,
      solution: `pipe = Pipeline([('s', StandardScaler()), ('svm', SVC(kernel='rbf'))])
grid = {'svm__C':[1,10,100], 'svm__gamma':['scale', 0.01, 0.001]}
gs = GridSearchCV(pipe, grid, cv=5).fit(Xt, yt)
acc = gs.score(Xe, ye)` },
  ]
});

// =====================================================================
// LEZIONE 10 — Valutazione, cross-validation, tuning
// =====================================================================
LESSONS.push({
  id: 'l10-eval',
  title: 'Valutazione dei modelli',
  tag: 'Workflow',
  summary: 'Train/val/test, k-fold CV, metriche, learning curves. Fare la valutazione bene è più difficile del training.',
  teoria: [
    { type: 'md', content: `
<h3>10.1 Il pericolo: valutare troppo poco</h3>
<p>La domanda che il ML cerca di rispondere è: <em>come funzionerà su dati mai visti?</em> Se sbagli la valutazione, tutto il resto è inutile: pubblicherai modelli che sembrano ottimi e falliscono in produzione.</p>

<h3>10.2 Train/val/test split</h3>
<p>Il minimo sindacale:</p>
<ul>
<li><strong>Training</strong> (60-80%): per addestrare</li>
<li><strong>Validation</strong> (10-20%): per scegliere iperparametri e modello</li>
<li><strong>Test</strong> (10-20%): un solo utilizzo, alla fine, per stimare la generalizzazione</li>
</ul>
<p>Il test set deve essere sacro. Se lo guardi 100 volte e scegli il modello che fa meglio, hai fatto overfitting sul test.</p>

<h3>10.3 K-fold cross-validation</h3>
<p>Split del training in $K$ fold. Per ciascuno: addestra sui $K-1$ fold, valida sul fold rimanente. Il punteggio è la media. Vantaggi: usa tutti i dati per addestrare e per validare, riduce la varianza della stima.</p>
<p>Varianti:</p>
<ul>
<li><strong>Stratified K-fold</strong>: mantiene la proporzione di classi in ogni fold (obbligatorio con classi sbilanciate)</li>
<li><strong>Leave-one-out</strong>: $K = n$. Massima varianza, minimo bias. Costoso.</li>
<li><strong>Time series split</strong>: per dati temporali, mai fare split casuale. Split cronologico.</li>
<li><strong>Group K-fold</strong>: per dati raggruppati (es. pazienti con più misure) — non contaminare tra train e val</li>
</ul>

<h3>10.4 La regola d'oro del CV</h3>
<p>Ogni operazione che coinvolge $y$ (o statistiche dei dati) deve essere fatta <strong>dentro</strong> ogni fold, non prima. Esempi di leakage:</p>
<ul>
<li>Scaling calcolato su tutti i dati &rarr; leakage</li>
<li>Feature selection basata su correlazione con $y$ prima dello split &rarr; leakage</li>
<li>Imputazione con la media di tutto il dataset &rarr; leakage</li>
</ul>
<p>Usa sempre <code class="inline">Pipeline</code> di scikit-learn: garantisce che ogni step sia refittato dentro il fold.</p>

<h3>10.5 Bias-variance tradeoff</h3>
<p>Per un dato punto $x_0$, decomponendo l'errore atteso:</p>
<p>$$\\mathbb{E}[(y_0 - \\hat f(x_0))^2] = \\underbrace{[\\mathbb{E}\\hat f(x_0) - f(x_0)]^2}_{\\text{bias}^2} + \\underbrace{\\mathrm{Var}(\\hat f(x_0))}_{\\text{variance}} + \\sigma^2$$</p>
<p>Modelli semplici: alto bias, bassa variance (underfit). Modelli complessi: basso bias, alta variance (overfit). L'ottimo è nel mezzo.</p>

<h3>10.6 Learning curves</h3>
<p>Grafico dell'errore in funzione della dimensione del training set. Ti dicono se hai un problema di bias (curve piatte, entrambe alte) o di variance (grande gap tra train e val). Guida per decidere se raccogliere più dati o cambiare modello.</p>

<h3>10.7 Curve ROC e Precision-Recall</h3>
<p>Per classificazione binaria con score continuo:</p>
<ul>
<li><strong>ROC</strong>: TPR (recall) vs FPR al variare della soglia. Area = AUC.</li>
<li><strong>PR</strong>: Precision vs Recall. Preferibile con classi sbilanciate perché non "ricompensa" i negativi facili.</li>
</ul>

<h3>10.8 Regola definitiva</h3>
<p>Il test set si guarda <strong>una volta sola</strong>, alla fine. Se lo consulti durante lo sviluppo, non è più un test: è un secondo validation. Ricomincia da capo con nuovi dati.</p>
` },
  ],
  esempi: [
    { type: 'md', content: '<h3>Esempio 1: pipeline corretta con CV</h3>' },
    { type: 'run', content: `from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
pipe = Pipeline([('s', StandardScaler()), ('lr', LogisticRegression(max_iter=1000))])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
print(f"CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
print(f"Fold: {scores.round(4)}")` },
    { type: 'md', content: '<h3>Esempio 2: learning curve</h3>' },
    { type: 'run', content: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve

X, y = make_classification(n_samples=1000, n_features=20, random_state=0)
sizes = np.linspace(0.1, 1.0, 8)

for ax_title, model in [('Tree depth=None (high var)', DecisionTreeClassifier(random_state=0)),
                         ('Tree depth=2 (high bias)', DecisionTreeClassifier(max_depth=2, random_state=0))]:
    ts, train_s, val_s = learning_curve(model, X, y, train_sizes=sizes, cv=5)
    print(f"\\n{ax_title}")
    for t, tr, va in zip(ts, train_s.mean(1), val_s.mean(1)):
        print(f"  n={t:4d}: train {tr:.3f}, val {va:.3f}")` },
    { type: 'md', content: '<h3>Esempio 3: GridSearchCV con pipeline</h3>' },
    { type: 'run', content: `from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split

X, y = load_wine(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

pipe = Pipeline([('s', StandardScaler()), ('svm', SVC())])
grid = {'svm__C':[0.1,1,10], 'svm__gamma':['scale',0.01,0.1], 'svm__kernel':['rbf','linear']}
gs = GridSearchCV(pipe, grid, cv=5, scoring='accuracy').fit(Xt, yt)

print(f"Migliori parametri: {gs.best_params_}")
print(f"Migliore CV score:  {gs.best_score_:.4f}")
print(f"Score sul test:     {gs.score(Xe, ye):.4f}")` },
    { type: 'md', content: '<h3>Esempio 4: leakage in azione</h3><p>Facciamo un errore di proposito e vediamo cosa succede.</p>' },
    { type: 'run', content: `import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

X, y = load_breast_cancer(return_X_y=True)

# SBAGLIATO: scaling PRIMA di CV
X_scaled = StandardScaler().fit_transform(X)  # usa TUTTI i dati
scores_bad = cross_val_score(LogisticRegression(max_iter=1000), X_scaled, y, cv=5)
print(f"Con leakage (scaling globale): {scores_bad.mean():.4f}")

# CORRETTO: scaling DENTRO la pipeline
from sklearn.pipeline import Pipeline
pipe = Pipeline([('s', StandardScaler()), ('lr', LogisticRegression(max_iter=1000))])
scores_good = cross_val_score(pipe, X, y, cv=5)
print(f"Senza leakage (in pipeline):   {scores_good.mean():.4f}")
print("\\nQui la differenza è piccola, ma con dataset piccoli o feature engineering complesso può essere enorme e ingannevole.")` },
  ],
  esercizi: [
    { type: 'exercise', difficulty: 'easy', title: 'Es. 10.1 — CV manuale',
      prompt: '<p>Implementa una funzione <code class="inline">my_cv(model, X, y, k)</code> che ritorna la lista degli accuracy sui $k$ fold. Non serve stratified, va bene k-fold semplice.</p>',
      starter: `import numpy as np
def my_cv(model, X, y, k=5):
    n = len(X)
    idx = np.arange(n)
    np.random.default_rng(0).shuffle(idx)
    folds = np.array_split(idx, k)
    scores = []
    # TODO: per ogni fold, addestra sul resto e valida sul fold
    return scores

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
print(my_cv(LogisticRegression(max_iter=1000), X, y, k=5))`,
      check: `import numpy as np
scores = my_cv(LogisticRegression(max_iter=1000), X, y, k=5)
if len(scores) == 5 and all(s > 0.8 for s in scores):
    print(f"OK — mean {np.mean(scores):.4f}")
else:
    print(f"FAIL — {scores}")`,
      solution: `for i in range(k):
    val = folds[i]
    train = np.concatenate([folds[j] for j in range(k) if j != i])
    m = model.__class__(**model.get_params()).fit(X[train], y[train])
    scores.append(m.score(X[val], y[val]))
return scores` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 10.2 — Curva ROC manuale',
      prompt: '<p>Data <code class="inline">y_true</code> e <code class="inline">y_score</code>, calcola manualmente TPR e FPR per ogni soglia in <code class="inline">y_score</code>. Salva i due array in <code class="inline">tpr, fpr</code>.</p>',
      starter: `import numpy as np
np.random.seed(0)
y_true = np.array([0,0,1,1,1,0,1,0,1,0])
y_score = np.array([0.1,0.4,0.35,0.8,0.7,0.2,0.6,0.3,0.9,0.15])

thresholds = np.sort(y_score)[::-1]
tpr = []
fpr = []
# TODO
print(list(zip(thresholds, tpr, fpr)))`,
      check: `import numpy as np
from sklearn.metrics import roc_auc_score
# controlla monotonicità e range
if len(tpr) == len(thresholds) and len(fpr) == len(thresholds):
    if all(0 <= x <= 1 for x in tpr+fpr):
        print("OK — arrays coerenti")
    else: print("FAIL — valori fuori [0,1]")
else:
    print("FAIL — lunghezze")`,
      solution: `P = (y_true == 1).sum(); N = (y_true == 0).sum()
for t in thresholds:
    pred = (y_score >= t).astype(int)
    tp = ((pred==1)&(y_true==1)).sum()
    fp = ((pred==1)&(y_true==0)).sum()
    tpr.append(tp/P); fpr.append(fp/N)` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 10.3 — Nested CV',
      prompt: '<p>Implementa una nested cross-validation: outer 5-fold per stimare la performance, inner 3-fold per selezionare l\'iperparametro <code class="inline">C</code> di una logistica tra $\\{0.01, 0.1, 1, 10\\}$. Salva l\'accuracy media in <code class="inline">nested_score</code>.</p>',
      starter: `import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold

X, y = load_iris(return_X_y=True)

# TODO
nested_score = None
print("Nested CV:", nested_score)`,
      check: `if nested_score is not None and nested_score > 0.9:
    print(f"OK — {nested_score:.4f}")
else:
    print(f"FAIL — {nested_score}")`,
      solution: `pipe = Pipeline([('s', StandardScaler()), ('lr', LogisticRegression(max_iter=5000))])
inner = StratifiedKFold(3, shuffle=True, random_state=0)
outer = StratifiedKFold(5, shuffle=True, random_state=0)
gs = GridSearchCV(pipe, {'lr__C':[0.01,0.1,1,10]}, cv=inner)
nested_score = cross_val_score(gs, X, y, cv=outer).mean()` },
  ],
  test: [
    { type: 'quiz', questions: [
      { q: 'Perché non calcolare la media dell\'imputazione su tutto il dataset prima dello split?', options: ['Rallenta', 'Data leakage: il test "vede" statistiche del training e viceversa', 'Non si può in Python', 'Nessun motivo'], answer: 1, explain: 'La media contiene informazioni del test set.' },
      { q: 'La stratified K-fold serve per:', options: ['Rendere il CV più veloce', 'Mantenere la proporzione di classi in ogni fold', 'Ridurre il bias', 'Nessuna delle precedenti'], answer: 1, explain: 'Cruciale con classi sbilanciate.' },
      { q: 'Con learning curve, se train e val hanno errore alto e vicini:', options: ['Overfitting: modello troppo complesso', 'Underfitting: modello troppo semplice', 'Serve più dati', 'Nulla di anomalo'], answer: 1, explain: 'Entrambi alti = alto bias = underfit.' },
      { q: 'Perché usare Pipeline invece di applicare scaling manualmente?', options: ['Estetica', 'Garantisce che tutti i preprocessing siano ripetuti dentro ogni fold, evitando leakage', 'Solo per velocità', 'Solo per Grid Search'], answer: 1, explain: 'Il refit corretto dentro CV.' },
    ] },
    { type: 'exercise', difficulty: 'hard', title: 'Test pratico — valutazione seria',
      prompt: '<p>Sul dataset breast_cancer, costruisci una pipeline con scaling + Logistic + grid search su C. Fai stratified 10-fold CV. Salva la media dell\'accuracy in <code class="inline">cv_acc</code> e la deviazione standard in <code class="inline">cv_std</code>.</p>',
      starter: `from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV

X, y = load_breast_cancer(return_X_y=True)

cv_acc = None
cv_std = None
print(cv_acc, cv_std)`,
      check: `if cv_acc is not None and cv_acc > 0.97:
    print(f"OK — {cv_acc:.4f} +/- {cv_std:.4f}")
else:
    print(f"FAIL — {cv_acc}")`,
      solution: `pipe = Pipeline([('s', StandardScaler()), ('lr', LogisticRegression(max_iter=5000))])
inner = StratifiedKFold(3, shuffle=True, random_state=0)
outer = StratifiedKFold(10, shuffle=True, random_state=0)
gs = GridSearchCV(pipe, {'lr__C':[0.01,0.1,1,10]}, cv=inner)
scores = cross_val_score(gs, X, y, cv=outer)
cv_acc = scores.mean(); cv_std = scores.std()` },
  ]
});

// =====================================================================
// LEZIONE 11 — K-Means e clustering
// =====================================================================
LESSONS.push({
  id: 'l11-kmeans',
  title: 'K-Means e clustering',
  tag: 'Non supervisionato',
  summary: 'Il primo modello non supervisionato: raggruppare senza etichette.',
  teoria: [
    { type: 'md', content: `
<h3>11.1 Il problema del clustering</h3>
<p>Dato $\\{x_i\\}_{i=1}^n$ senza etichette, vogliamo raggrupparli in $K$ cluster tali che punti nello stesso cluster siano "simili" e punti in cluster diversi siano "diversi". La definizione di simile dipende dal problema: è una scelta di modello, non un dato.</p>

<h3>11.2 K-Means: formulazione</h3>
<p>Cerchiamo $K$ centroidi $\\mu_1, \\dots, \\mu_K \\in \\mathbb{R}^d$ e un assegnamento $c : \\{1,\\dots,n\\} \\to \\{1,\\dots,K\\}$ che minimizzano:</p>
<p>$$J(\\mu, c) = \\sum_{i=1}^n \\|x_i - \\mu_{c(i)}\\|^2$$</p>
<p>Cioè la somma delle distanze al quadrato di ogni punto dal proprio centroide (inertia).</p>

<h3>11.3 L'algoritmo (Lloyd)</h3>
<ol>
<li>Inizializza $\\mu_1, \\dots, \\mu_K$ (random o k-means++)</li>
<li><strong>Assign step</strong>: assegna ogni punto al centroide più vicino: $c(i) = \\arg\\min_k \\|x_i - \\mu_k\\|^2$</li>
<li><strong>Update step</strong>: aggiorna ogni centroide come media dei punti assegnati: $\\mu_k = \\frac{1}{|C_k|}\\sum_{i \\in C_k} x_i$</li>
<li>Ripeti finché nessun punto cambia cluster</li>
</ol>
<p>È un algoritmo EM: ogni step riduce $J$, quindi converge in tempo finito (a un minimo locale, non necessariamente globale).</p>

<h3>11.4 Scelta di $K$: metodo del gomito</h3>
<p>Grafica $J$ in funzione di $K$. Aumentando $K$, $J$ decresce sempre (con $K=n$ è zero). Cerca il "gomito": il punto dopo il quale la riduzione rallenta drasticamente.</p>

<h3>11.5 Silhouette score</h3>
<p>Metrica alternativa. Per ogni punto $i$:</p>
<ul>
<li>$a_i$ = distanza media dai punti dello stesso cluster</li>
<li>$b_i$ = distanza media dai punti del cluster più vicino</li>
<li>$s_i = (b_i - a_i) / \\max(a_i, b_i) \\in [-1, 1]$</li>
</ul>
<p>Valori vicini a 1: cluster ben definito; vicini a 0: sul confine; negativi: probabilmente nel cluster sbagliato.</p>

<h3>11.6 Limitazioni</h3>
<p>K-Means assume cluster:</p>
<ul>
<li>Sferici (usa distanza euclidea)</li>
<li>Di dimensione simile</li>
<li>Con densità simile</li>
</ul>
<p>Fallisce su cluster allungati, densità variabile, forme complesse. Per questi casi: DBSCAN (density-based), Gaussian Mixture Models, Spectral Clustering.</p>

<h3>11.7 k-means++</h3>
<p>Inizializzazione intelligente: scegli il primo centroide a caso, poi ogni successivo con probabilità proporzionale alla distanza al quadrato dal centroide più vicino già scelto. Molto meno soggetta a minimi locali cattivi.</p>

<h3>11.8 Altri metodi di clustering</h3>
<ul>
<li><strong>Gerarchico</strong>: costruisce un albero (dendrogramma). Bottom-up (agglomerativo) o top-down (divisivo).</li>
<li><strong>DBSCAN</strong>: raggruppa punti densi. Non serve specificare $K$. Gestisce outlier.</li>
<li><strong>GMM</strong>: fit di una miscela di gaussiane. Cluster ellittici, probabilità soft di appartenenza.</li>
</ul>
` },
  ],
  esempi: [
    { type: 'md', content: '<h3>Esempio 1: K-Means su dati sintetici</h3>' },
    { type: 'run', content: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.7, random_state=0)
km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)

fig, ax = plt.subplots(figsize=(7,5))
ax.scatter(X[:,0], X[:,1], c=km.labels_, cmap='tab10', s=25, edgecolor='k', alpha=0.7)
ax.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], c='red', marker='X', s=200, edgecolor='k', label='centroidi')
ax.legend(); ax.set_title(f'K-Means con K=4, inertia={km.inertia_:.1f}')
plt.show()` },
    { type: 'md', content: '<h3>Esempio 2: metodo del gomito</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, random_state=0)
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    inertias.append(km.inertia_)
    print(f"K={k:2d}: inertia = {km.inertia_:8.2f}")
print("\\nIl 'gomito' è a K=4, coerente con il vero numero di cluster.")` },
    { type: 'md', content: '<h3>Esempio 3: K-Means from scratch</h3>' },
    { type: 'run', content: `import numpy as np
np.random.seed(0)

def kmeans_scratch(X, k, n_iter=100):
    n, d = X.shape
    # init: scegli k punti a caso
    centroids = X[np.random.choice(n, k, replace=False)].copy()
    for it in range(n_iter):
        # assign
        dists = ((X[:,None,:] - centroids[None,:,:])**2).sum(-1)
        labels = np.argmin(dists, axis=1)
        # update
        new_centroids = np.array([X[labels==j].mean(axis=0) if (labels==j).any() else centroids[j] for j in range(k)])
        if np.allclose(new_centroids, centroids):
            print(f"converged in {it} steps")
            break
        centroids = new_centroids
    return centroids, labels

from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=3, random_state=0)
c, l = kmeans_scratch(X, k=3)
print("centroidi:", c.round(2))` },
    { type: 'md', content: '<h3>Esempio 4: quando K-Means fallisce</h3>' },
    { type: 'run', content: `import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=300, noise=0.05, random_state=0)
fig, axes = plt.subplots(1, 2, figsize=(11,4))
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
db = DBSCAN(eps=0.2, min_samples=5).fit(X)
axes[0].scatter(X[:,0], X[:,1], c=km.labels_, cmap='coolwarm', edgecolor='k'); axes[0].set_title('K-Means: fallisce sui half-moon')
axes[1].scatter(X[:,0], X[:,1], c=db.labels_, cmap='coolwarm', edgecolor='k'); axes[1].set_title('DBSCAN: cattura le lune')
plt.tight_layout(); plt.show()` },
  ],
  esercizi: [
    { type: 'exercise', difficulty: 'easy', title: 'Es. 11.1 — Fit e conteggio',
      prompt: '<p>Su dati blobs con 5 centri, fitta un KMeans e conta quanti punti in ciascun cluster. Salva l\'array dei conteggi in <code class="inline">counts</code>.</p>',
      starter: `import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=500, centers=5, random_state=0)

# TODO
counts = None
print(counts)`,
      check: `import numpy as np
if counts is not None and len(counts) == 5 and sum(counts) == 500:
    print(f"OK — {counts}")
else:
    print("FAIL")`,
      solution: `km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(X)
_, counts = np.unique(km.labels_, return_counts=True)` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 11.2 — Silhouette',
      prompt: '<p>Trova il $K \\in \\{2,3,4,5,6,7,8\\}$ che massimizza il silhouette score sul dataset. Salva in <code class="inline">best_k</code>.</p>',
      starter: `import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=300, centers=4, random_state=0)

# TODO
best_k = None
print("k ottimo:", best_k)`,
      check: `import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
scores = {k: silhouette_score(X, KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)) for k in range(2,9)}
_atteso = max(scores, key=scores.get)
if best_k == _atteso:
    print(f"OK — k={best_k}, silhouette={scores[best_k]:.4f}")
else:
    print(f"FAIL — atteso {_atteso}")`,
      solution: `scores = {k: silhouette_score(X, KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)) for k in range(2,9)}
best_k = max(scores, key=scores.get)` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 11.3 — Inerzia manuale',
      prompt: '<p>Data una matrice <code class="inline">X</code>, dei centroidi <code class="inline">centers</code> e le assegnazioni <code class="inline">labels</code>, calcola l\'inerzia (SSE): $\\sum_i \\|x_i - \\mu_{c(i)}\\|^2$.</p>',
      starter: `import numpy as np
np.random.seed(0)
X = np.random.randn(100, 3)
centers = np.random.randn(4, 3)
labels = np.random.randint(0, 4, 100)

# TODO
inertia = None
print(inertia)`,
      check: `import numpy as np
_atteso = sum(((X[labels==k] - centers[k])**2).sum() for k in range(4))
if inertia is not None and abs(inertia - _atteso) < 1e-6:
    print(f"OK — {inertia:.4f}")
else:
    print(f"FAIL — atteso {_atteso:.4f}")`,
      solution: `inertia = sum(((X[labels==k] - centers[k])**2).sum() for k in range(4))
# oppure: inertia = ((X - centers[labels])**2).sum()` },
  ],
  test: [
    { type: 'quiz', questions: [
      { q: 'K-Means converge sempre a:', options: ['Un minimo globale', 'Un minimo locale (dipende dall\'inizializzazione)', 'Un massimo', 'Nessuno dei precedenti'], answer: 1, explain: 'È monotono decrescente ma non convesso — locale, non globale.' },
      { q: 'K-Means su cluster a mezza luna:', options: ['Funziona bene', 'Fallisce (assume forme sferiche)', 'Dipende da K', 'Serve più iterazioni'], answer: 1, explain: 'La distanza euclidea impone confini sferici. DBSCAN è meglio.' },
      { q: 'Il metodo del gomito serve per:', options: ['Rimuovere outlier', 'Scegliere K guardando dove l\'inerzia smette di calare rapidamente', 'Inizializzare centroidi', 'Calcolare silhouette'], answer: 1, explain: 'Cerchi il "cambio di pendenza" nel grafico K vs inertia.' },
      { q: 'k-means++ vs random init:', options: ['Peggio', 'Meglio: distribuisce i centroidi iniziali in modo intelligente riducendo il rischio di minimi locali cattivi', 'Nessuna differenza', 'Solo più veloce'], answer: 1, explain: 'Sceglie centroidi lontani tra loro.' },
    ] },
    { type: 'exercise', difficulty: 'hard', title: 'Test pratico — clustering su Iris (unsupervised)',
      prompt: '<p>Su Iris (senza usare le label), fitta un KMeans con K=3 e calcola quanto ben concorda con le vere classi via <em>adjusted rand index</em>. Salva in <code class="inline">ari</code>. Target > 0.7.</p>',
      starter: `from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y = iris.data, iris.target

ari = None`,
      check: `if ari is None: print("FAIL: None")
elif ari > 0.7: print(f"OK — ARI {ari:.4f}")
else: print(f"FAIL — ARI {ari:.4f}")`,
      solution: `X_s = StandardScaler().fit_transform(X)
labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X_s)
ari = adjusted_rand_score(y, labels)` },
  ]
});

// =====================================================================
// LEZIONE 12 — PCA e riduzione dimensionale
// =====================================================================
LESSONS.push({
  id: 'l12-pca',
  title: 'PCA e riduzione dimensionale',
  tag: 'Non supervisionato',
  summary: 'Trovare le direzioni di massima varianza. Applicazioni: visualizzazione, compressione, denoising.',
  teoria: [
    { type: 'md', content: `
<h3>12.1 Il problema</h3>
<p>Dati $X \\in \\mathbb{R}^{n\\times d}$ ad alta dimensione, vogliamo proiettarli in $\\mathbb{R}^k$ con $k \\ll d$ preservando quanta più informazione possibile. "Informazione" qui significa <em>varianza</em>: le direzioni lungo cui i dati variano di più.</p>

<h3>12.2 Formulazione</h3>
<p>Standardizziamo (media zero, opzionalmente varianza unitaria). Cerchiamo un vettore unitario $v$ tale che la varianza dei punti proiettati $v^T x_i$ sia massima:</p>
<p>$$\\max_{\\|v\\|=1} v^T \\Sigma v \\quad\\text{dove}\\quad \\Sigma = \\tfrac{1}{n}X^T X$$</p>
<p>La soluzione è l'autovettore di $\\Sigma$ associato al più grande autovalore. Il valore del massimo è l'autovalore stesso: quello è la varianza spiegata dalla prima componente principale.</p>
<p>Ripeti nel sottospazio ortogonale &rarr; le prime $k$ componenti sono i primi $k$ autovettori di $\\Sigma$.</p>

<h3>12.3 Via SVD</h3>
<p>Con $X$ centrata, la SVD $X = U\\Sigma V^T$ dà:</p>
<ul>
<li>Le colonne di $V$ sono le direzioni principali (loadings)</li>
<li>$X V$ sono i punti proiettati (scores)</li>
<li>$\\sigma_i^2 / n$ è la varianza spiegata dalla $i$-esima componente</li>
</ul>
<p>SVD è numericamente più stabile del calcolo esplicito di $X^T X$.</p>

<h3>12.4 Varianza spiegata</h3>
<p>Fraction of variance explained by first $k$ components:</p>
<p>$$\\text{VE}(k) = \\frac{\\sum_{i=1}^k \\sigma_i^2}{\\sum_{i=1}^d \\sigma_i^2}$$</p>
<p>Si sceglie $k$ per catturare, per esempio, il 95% della varianza.</p>

<h3>12.5 Interpretazione</h3>
<ul>
<li>PCA è <strong>lineare</strong>: le componenti sono combinazioni lineari delle feature originali</li>
<li>Le componenti sono <strong>ortogonali</strong></li>
<li>La <strong>ricostruzione</strong> approssimata è $\\hat X = XV_k V_k^T$: proiezione poi anti-proiezione, l'errore quadratico medio è minimo tra tutte le proiezioni di rango $k$</li>
</ul>

<h3>12.6 Cosa fa PCA <strong>bene</strong> e cosa NO</h3>
<p>Bene: visualizzazione 2D/3D, denoising, decorrelazione delle feature, preprocessing.</p>
<p>Non bene: se le direzioni di massima varianza <em>non</em> coincidono con quelle utili per la classificazione. PCA non guarda $y$. Per un\'alternativa supervisionata: Linear Discriminant Analysis (LDA).</p>

<h3>12.7 Standardizzazione: obbligatoria</h3>
<p>Se le feature hanno scale diverse, quelle con varianza grande dominano PCA anche se non sono le più informative. <strong>Sempre</strong> standardizza prima.</p>

<h3>12.8 Alternative non lineari</h3>
<ul>
<li><strong>t-SNE</strong>: preserva similarità locali. Ottimo per visualizzazione, non per feature engineering.</li>
<li><strong>UMAP</strong>: come t-SNE ma più veloce e preserva anche struttura globale.</li>
<li><strong>Autoencoder</strong>: rete neurale che comprime e decomprime. Non lineare, flessibile.</li>
</ul>
` },
    { type: 'callout', variant: 'note', title: 'Legame con la fisica', content: 'La matrice di covarianza è simmetrica e semidefinita positiva: diagonalizzabile con base ortonormale, autovalori non negativi. Esattamente come il tensore d\'inerzia in meccanica classica. Le componenti principali sono gli "assi principali di inerzia" della nuvola di punti.' },
  ],
  esempi: [
    { type: 'md', content: '<h3>Esempio 1: PCA su Iris (visualizzazione 2D)</h3>' },
    { type: 'run', content: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)

pca = PCA(n_components=2).fit(X)
Z = pca.transform(X)
print(f"Varianza spiegata: {pca.explained_variance_ratio_}")
print(f"Totale: {pca.explained_variance_ratio_.sum():.3f}")

fig, ax = plt.subplots(figsize=(7,5))
for i, name in enumerate(iris.target_names):
    ax.scatter(Z[iris.target==i, 0], Z[iris.target==i, 1], label=name, s=40, edgecolor='k', alpha=0.7)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(); ax.set_title('Iris in 2D via PCA')
plt.show()` },
    { type: 'md', content: '<h3>Esempio 2: quante componenti tenere?</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = load_digits().data  # 64 feature (immagini 8x8)
X = StandardScaler().fit_transform(X)

pca = PCA().fit(X)
cum = np.cumsum(pca.explained_variance_ratio_)
for thresh in [0.80, 0.90, 0.95, 0.99]:
    k = np.searchsorted(cum, thresh) + 1
    print(f"Per {thresh*100:.0f}% varianza servono {k} componenti (su 64)")` },
    { type: 'md', content: '<h3>Esempio 3: PCA from scratch via SVD</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X = StandardScaler().fit_transform(load_iris().data)
n = len(X)

U, S, Vt = np.linalg.svd(X, full_matrices=False)
print(f"Valori singolari: {S.round(3)}")
print(f"Varianza spiegata: {(S**2 / (n-1) / (S**2/(n-1)).sum()).round(4)}")

# proiezione sulle prime 2 componenti
Z_mio = X @ Vt[:2].T
Z_sk = PCA(n_components=2).fit_transform(X)
# le colonne coincidono a meno di segno
print(f"\\nMax diff (in valore assoluto): {min(np.abs(np.abs(Z_mio) - np.abs(Z_sk)).max(), 1):.2e}")` },
    { type: 'md', content: '<h3>Esempio 4: PCA come denoiser</h3>' },
    { type: 'run', content: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

digits = load_digits()
X = digits.data
np.random.seed(0)
X_noisy = X + 4 * np.random.randn(*X.shape)

pca = PCA(n_components=25).fit(X)
X_denoised = pca.inverse_transform(pca.transform(X_noisy))

fig, axes = plt.subplots(3, 6, figsize=(10, 5))
for i in range(6):
    axes[0,i].imshow(X[i].reshape(8,8), cmap='gray_r'); axes[0,i].axis('off')
    axes[1,i].imshow(X_noisy[i].reshape(8,8), cmap='gray_r'); axes[1,i].axis('off')
    axes[2,i].imshow(X_denoised[i].reshape(8,8), cmap='gray_r'); axes[2,i].axis('off')
for r, t in enumerate(['originale', 'rumoroso', 'denoised (PCA 25 comp)']):
    axes[r,0].set_title(t, loc='left', fontsize=9)
plt.tight_layout(); plt.show()` },
  ],
  esercizi: [
    { type: 'exercise', difficulty: 'easy', title: 'Es. 12.1 — Varianza spiegata',
      prompt: '<p>Su breast_cancer standardizzato, calcola quante componenti servono per catturare almeno il 90% della varianza. Salva in <code class="inline">k</code>.</p>',
      starter: `import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(load_breast_cancer().data)

# TODO
k = None
print("k:", k)`,
      check: `import numpy as np
pca = PCA().fit(X)
cum = np.cumsum(pca.explained_variance_ratio_)
_atteso = int(np.searchsorted(cum, 0.90) + 1)
if k == _atteso:
    print(f"OK — {k} componenti per 90%")
else:
    print(f"FAIL — atteso {_atteso}")`,
      solution: `pca = PCA().fit(X)
cum = np.cumsum(pca.explained_variance_ratio_)
k = int(np.searchsorted(cum, 0.90) + 1)` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 12.2 — PCA manuale',
      prompt: '<p>Implementa PCA a mano: centra i dati, calcola $\\Sigma = X^T X / (n-1)$, trova autovettori, proietta sui primi 2. Salva la matrice risultato in <code class="inline">Z</code>.</p>',
      starter: `import numpy as np
np.random.seed(0)
X = np.random.randn(100, 5) @ np.random.randn(5, 5)

# TODO: Z shape (100, 2)
Z = None
print(Z.shape if Z is not None else "None")`,
      check: `import numpy as np
from sklearn.decomposition import PCA
Z_sk = PCA(n_components=2).fit_transform(X - X.mean(0))
# confronto a meno di segno
if Z is None: print("FAIL: None")
elif Z.shape == (100, 2):
    diff = min(np.abs(np.abs(Z) - np.abs(Z_sk)).max(), 1e10)
    if diff < 1e-6: print("OK")
    else: print(f"FAIL — max diff (abs) {diff}")
else:
    print(f"FAIL — shape {Z.shape}")`,
      solution: `Xc = X - X.mean(0)
Sigma = Xc.T @ Xc / (len(X)-1)
vals, vecs = np.linalg.eigh(Sigma)
# eigh ritorna in ordine crescente, prendo gli ultimi 2
V = vecs[:, [-1, -2]]
Z = Xc @ V` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 12.3 — Ricostruzione ed errore',
      prompt: '<p>Su digits, per $k \\in \\{5, 10, 20, 40\\}$ calcola l\'errore di ricostruzione MSE. Salva un dizionario <code class="inline">errs</code> con chiave $k$.</p>',
      starter: `import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

X = load_digits().data

errs = {}
# TODO
print(errs)`,
      check: `import numpy as np
_atteso = {}
for k in [5,10,20,40]:
    p = PCA(n_components=k).fit(X)
    Xr = p.inverse_transform(p.transform(X))
    _atteso[k] = float(((Xr - X)**2).mean())
ok = all(abs(errs.get(k,0) - _atteso[k]) < 1e-6 for k in [5,10,20,40])
print("OK" if ok else f"FAIL: atteso {_atteso}, tuo {errs}")`,
      solution: `for k in [5, 10, 20, 40]:
    p = PCA(n_components=k).fit(X)
    Xr = p.inverse_transform(p.transform(X))
    errs[k] = float(((Xr - X)**2).mean())` },
  ],
  test: [
    { type: 'quiz', questions: [
      { q: 'La prima componente principale è la direzione:', options: ['Con media massima', 'Con varianza massima dei dati proiettati', 'Ortogonale a $y$', 'Con densità massima'], answer: 1, explain: 'Per definizione: massimizza la varianza.' },
      { q: 'PCA è supervisionato?', options: ['Sì, usa le label', 'No, non guarda $y$', 'Dipende', 'Solo per classificazione'], answer: 1, explain: 'Solo la struttura di $X$.' },
      { q: 'Non standardizzare prima di PCA:', options: ['Non ha effetto', 'Le feature con scala grande dominano le componenti principali', 'Rende PCA più veloce', 'Rende PCA più accurato'], answer: 1, explain: 'La varianza dipende dall\'unità di misura.' },
      { q: 'Se una componente ha varianza spiegata 0.65, significa:', options: ['Predice il 65% dei valori', 'Contiene il 65% della varianza totale dei dati', 'Ha 65% di accuracy', 'È il 65-esimo autovettore'], answer: 1, explain: '$\\sigma^2 / \\sum \\sigma^2$.' },
    ] },
    { type: 'exercise', difficulty: 'hard', title: 'Test pratico — PCA + classificatore',
      prompt: '<p>Su digits, applica PCA per ridurre a $k$ tale che si catturi il 95% di varianza, poi addestra una LogisticRegression. Salva il numero di componenti in <code class="inline">k95</code> e l\'accuracy test in <code class="inline">acc</code> (target > 0.94).</p>',
      starter: `import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

k95 = None
acc = None`,
      check: `if k95 is None or acc is None: print("FAIL: None")
elif acc > 0.94: print(f"OK — {k95} componenti, acc {acc:.4f}")
else: print(f"FAIL — {k95} comp, acc {acc:.4f}")`,
      solution: `sc = StandardScaler().fit(Xt)
Xt_s = sc.transform(Xt); Xe_s = sc.transform(Xe)
p = PCA(n_components=0.95).fit(Xt_s)
k95 = p.n_components_
lr = LogisticRegression(max_iter=5000).fit(p.transform(Xt_s), yt)
acc = lr.score(p.transform(Xe_s), ye)` },
  ]
});

// =====================================================================
// LEZIONE 13 — Reti neurali
// =====================================================================
LESSONS.push({
  id: 'l13-neural-nets',
  title: 'Introduzione alle reti neurali',
  tag: 'Deep Learning',
  summary: 'Percettrone, MLP, backpropagation. La transizione al deep learning.',
  teoria: [
    { type: 'md', content: `
<h3>13.1 Dal lineare al non lineare</h3>
<p>La regressione logistica è un neurone singolo. Se componiamo tanti neuroni in strati otteniamo un <em>perceptron multistrato</em> (MLP), capace di approssimare qualsiasi funzione continua (teorema di approssimazione universale).</p>

<h3>13.2 Un neurone artificiale</h3>
<p>Un neurone calcola:</p>
<p>$$a = \\phi(w^T x + b)$$</p>
<p>dove $\\phi$ è la <em>funzione di attivazione</em> non lineare. Senza $\\phi$, comporre neuroni sarebbe equivalente a una singola trasformazione lineare — inutile.</p>

<h3>13.3 Attivazioni</h3>
<ul>
<li><strong>Sigmoide</strong>: $\\sigma(z) = 1/(1+e^{-z})$. Ora poco usata: gradienti che svaniscono.</li>
<li><strong>Tanh</strong>: $\\tanh(z)$. Centrata su zero, meglio della sigmoide.</li>
<li><strong>ReLU</strong>: $\\max(0, z)$. Standard moderno. Semplice, veloce, evita gradient vanishing.</li>
<li><strong>Leaky ReLU, ELU, GELU</strong>: varianti che risolvono limiti di ReLU.</li>
<li><strong>Softmax</strong>: per l\'output di classificazione multi-classe.</li>
</ul>

<h3>13.4 Un MLP</h3>
<p>Strato per strato, con notazione compatta:</p>
<p>$$h^{(l+1)} = \\phi(W^{(l+1)} h^{(l)} + b^{(l+1)})$$</p>
<p>con $h^{(0)} = x$ e output $\\hat y = h^{(L)}$. I parametri sono tutti i $W^{(l)}, b^{(l)}$.</p>
<p>Il numero di neuroni per strato e il numero di strati sono <em>iperparametri architetturali</em>.</p>

<h3>13.5 Loss e training</h3>
<p>Per regressione: MSE. Per classificazione binaria: BCE. Per multi-classe con softmax: cross-entropy. Il training minimizza la loss con gradient descent (o Adam, RMSprop...).</p>

<h3>13.6 Backpropagation</h3>
<p>Il gradiente della loss rispetto a ogni parametro si calcola con la <em>regola della catena</em> applicata all\'indietro. Concettualmente semplice, tecnicamente delicato solo per la contabilità delle dimensioni.</p>
<p>Per uno strato $l$:</p>
<p>$$\\delta^{(l)} = (W^{(l+1)T} \\delta^{(l+1)}) \\odot \\phi'(z^{(l)})$$</p>
<p>$$\\frac{\\partial \\mathcal{L}}{\\partial W^{(l)}} = \\delta^{(l)} (h^{(l-1)})^T$$</p>
<p>I framework moderni (PyTorch, TensorFlow) fanno tutto automaticamente via <em>autograd</em>: tu scrivi il forward, loro calcolano il backward.</p>

<h3>13.7 Regolarizzazione</h3>
<ul>
<li><strong>L2 weight decay</strong>: penalità $\\lambda \\|W\\|^2$</li>
<li><strong>Dropout</strong>: azzera casualmente frazione dei neuroni durante training</li>
<li><strong>Batch normalization</strong>: normalizza le attivazioni per stabilizzare</li>
<li><strong>Early stopping</strong>: fermati quando val loss smette di scendere</li>
</ul>

<h3>13.8 Ottimizzatori</h3>
<ul>
<li><strong>SGD</strong>: base, spesso con momentum</li>
<li><strong>Adam</strong>: adatta il learning rate per parametro. Default ragionevole.</li>
<li><strong>RMSprop, AdaGrad</strong>: varianti</li>
</ul>

<h3>13.9 Oltre l'MLP</h3>
<ul>
<li><strong>CNN</strong>: per immagini, sfruttano la struttura spaziale</li>
<li><strong>RNN, LSTM</strong>: per sequenze</li>
<li><strong>Transformer</strong>: attualmente dominanti per testo, immagini, quasi tutto</li>
</ul>
<p>Sono la lezione successiva di un corso, ma tutto si costruisce sulle stesse basi.</p>
` },
  ],
  esempi: [
    { type: 'md', content: '<h3>Esempio 1: MLP con scikit-learn</h3>' },
    { type: 'run', content: `from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = load_digits(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

pipe = Pipeline([('s', StandardScaler()),
                 ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32),
                                       activation='relu',
                                       max_iter=200,
                                       random_state=0))])
pipe.fit(Xt, yt)
print(f"Train acc: {pipe.score(Xt, yt):.4f}")
print(f"Test acc:  {pipe.score(Xe, ye):.4f}")
print(f"Num layer: {len(pipe.named_steps['mlp'].coefs_)}")
for i, W in enumerate(pipe.named_steps['mlp'].coefs_):
    print(f"  layer {i}: shape {W.shape}")` },
    { type: 'md', content: '<h3>Esempio 2: MLP from scratch (2 layer)</h3>' },
    { type: 'run', content: `import numpy as np
np.random.seed(0)

# dati XOR — non separabile linearmente
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

# architettura: 2 -> 4 -> 1
W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros(4)
W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros(1)

def sig(z): return 1/(1+np.exp(-z))

lr = 0.5
for step in range(5000):
    # forward
    z1 = X @ W1 + b1; a1 = sig(z1)
    z2 = a1 @ W2 + b2; y_hat = sig(z2)
    # loss (BCE)
    loss = -np.mean(y*np.log(y_hat+1e-9) + (1-y)*np.log(1-y_hat+1e-9))
    # backward
    dz2 = (y_hat - y) / len(X)
    dW2 = a1.T @ dz2; db2 = dz2.sum(0)
    da1 = dz2 @ W2.T
    dz1 = da1 * a1 * (1-a1)
    dW1 = X.T @ dz1; db1 = dz1.sum(0)
    # update
    W1 -= lr*dW1; b1 -= lr*db1; W2 -= lr*dW2; b2 -= lr*db2
    if step % 1000 == 0:
        print(f"step {step:4d}: loss = {loss:.5f}")

print("\\nPredizioni:", y_hat.round(3).ravel(), "vere:", y.ravel())` },
    { type: 'md', content: '<h3>Esempio 3: effetto della profondità</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

X, y = make_moons(n_samples=500, noise=0.3, random_state=0)
print(f"{'arch':>20} {'CV':>10}")
for arch in [(2,), (10,), (10,10), (50,), (50,50,50), (200,200)]:
    p = Pipeline([('s', StandardScaler()),
                  ('m', MLPClassifier(hidden_layer_sizes=arch, max_iter=500, random_state=0))])
    s = cross_val_score(p, X, y, cv=5).mean()
    print(f"{str(arch):>20} {s:>10.4f}")` },
  ],
  esercizi: [
    { type: 'exercise', difficulty: 'medium', title: 'Es. 13.1 — Forward pass manuale',
      prompt: '<p>Implementa il forward di un MLP a 2 layer: input dim 3, hidden 4 con ReLU, output 1 con sigmoide. Data <code class="inline">x</code>, <code class="inline">W1, b1, W2, b2</code>, salva l\'output in <code class="inline">y_hat</code>.</p>',
      starter: `import numpy as np
np.random.seed(0)
x = np.array([1.0, -0.5, 2.0])
W1 = np.random.randn(3, 4); b1 = np.zeros(4)
W2 = np.random.randn(4, 1); b2 = np.zeros(1)

# TODO
y_hat = None
print(y_hat)`,
      check: `import numpy as np
_z1 = x @ W1 + b1; _a1 = np.maximum(0, _z1)
_z2 = _a1 @ W2 + b2; _yh = 1/(1+np.exp(-_z2))
if y_hat is not None and np.allclose(y_hat, _yh):
    print("OK")
else:
    print(f"FAIL — atteso {_yh}")`,
      solution: `z1 = x @ W1 + b1
a1 = np.maximum(0, z1)  # ReLU
z2 = a1 @ W2 + b2
y_hat = 1/(1+np.exp(-z2))  # sigmoide` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 13.2 — MLP su moons',
      prompt: '<p>Su make_moons, addestra un MLP con architettura a scelta. Target: test accuracy > 0.9.</p>',
      starter: `import numpy as np
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=1000, noise=0.25, random_state=0)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42)

acc = None`,
      check: `if acc is None: print("FAIL: None")
elif acc > 0.9: print(f"OK — {acc:.4f}")
else: print(f"FAIL — {acc:.4f}")`,
      solution: `p = Pipeline([('s', StandardScaler()),
              ('m', MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=500, random_state=0))])
p.fit(Xt, yt)
acc = p.score(Xe, ye)` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 13.3 — ReLU e derivata',
      prompt: '<p>Implementa ReLU e la sua derivata. La derivata è 1 per $z > 0$, 0 altrove.</p>',
      starter: `import numpy as np
def relu(z):
    return None
def relu_grad(z):
    return None

z = np.array([-2., 0., 1., 3.])
print(relu(z), relu_grad(z))`,
      check: `import numpy as np
if np.allclose(relu(z), [0,0,1,3]) and np.allclose(relu_grad(z), [0,0,1,1]):
    print("OK")
else:
    print("FAIL")`,
      solution: `def relu(z): return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(float)` },
  ],
  test: [
    { type: 'quiz', questions: [
      { q: 'Senza attivazioni non lineari, un MLP profondo è equivalente a:', options: ['Un modello lineare', 'Una foresta', 'Un SVM', 'Niente'], answer: 0, explain: 'Composizione di lineari = lineare.' },
      { q: 'ReLU è preferita alla sigmoide perché:', options: ['È più facile da scrivere', 'Ha gradiente non-nullo per $z>0$ senza saturare', 'È non lineare', 'È simmetrica'], answer: 1, explain: 'Sigmoide satura per $|z|$ grande, gradient vanishing.' },
      { q: 'Backpropagation applica:', options: ['Il gradient descent', 'La regola della catena all\'indietro nel grafo di calcolo', 'La trasformata di Fourier', 'L\'ottimizzazione lineare'], answer: 1, explain: 'Calcola gradienti composti.' },
      { q: 'Il dropout serve per:', options: ['Velocizzare training', 'Regolarizzare: azzera casualmente neuroni per prevenire co-adaptation', 'Ridurre memoria', 'Aumentare accuracy training'], answer: 1, explain: 'Riduce overfitting.' },
    ] },
    { type: 'exercise', difficulty: 'hard', title: 'Test pratico — MLP su digits',
      prompt: '<p>Costruisci un MLP che raggiunga accuracy > 0.97 su digits. Puoi scegliere qualsiasi architettura, regolarizzazione, ottimizzatore. Salva in <code class="inline">acc</code>.</p>',
      starter: `from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

acc = None`,
      check: `if acc is None: print("FAIL: None")
elif acc > 0.97: print(f"OK — {acc:.4f}")
else: print(f"FAIL — {acc:.4f}")`,
      solution: `p = Pipeline([('s', StandardScaler()),
              ('m', MLPClassifier(hidden_layer_sizes=(128, 64),
                                  activation='relu',
                                  alpha=1e-4,
                                  max_iter=500,
                                  random_state=0))])
p.fit(Xt, yt)
acc = p.score(Xe, ye)` },
  ]
});

// =====================================================================
// LEZIONE 14 — Progetto finale
// =====================================================================
LESSONS.push({
  id: 'l14-project',
  title: 'Progetto finale: un modello end-to-end',
  tag: 'Progetto',
  summary: 'Mettere tutto insieme: EDA, feature engineering, modelli, tuning, valutazione.',
  teoria: [
    { type: 'md', content: `
<h3>14.1 Il workflow di un progetto ML</h3>
<p>Ora che conosci i pezzi, ecco l'ordine in cui li combini per un progetto reale.</p>
<ol>
<li><strong>Definisci il problema</strong>: cosa vuoi predire? Regressione o classificazione? Che metrica ti interessa davvero? (Accuracy non è sempre la risposta giusta)</li>
<li><strong>Raccogli e ispeziona i dati</strong>: EDA, statistiche descrittive, grafici, valori mancanti, outlier.</li>
<li><strong>Split train/val/test</strong>: <em>subito</em>, prima di guardare i dati troppo a lungo. Isola il test.</li>
<li><strong>Baseline</strong>: modello stupido (majority class, media, regressione lineare). Ti dice il minimo accettabile.</li>
<li><strong>Preprocessing e feature engineering</strong>: dentro Pipeline. Standardizza, encoding categoriche, feature derivate.</li>
<li><strong>Prova più modelli</strong>: lineare, tree, ensemble, magari MLP. Con CV per stimare performance.</li>
<li><strong>Tuning</strong>: grid search o random search sui migliori 1-2 modelli.</li>
<li><strong>Valutazione finale</strong>: sul test set, una volta sola. Riporta con intervallo di confidenza.</li>
<li><strong>Interpretazione</strong>: feature importance, esempi predetti male, casi limite. Un modello che non capisci è pericoloso.</li>
</ol>

<h3>14.2 Errori comuni</h3>
<ul>
<li><strong>Data leakage</strong>: statistiche calcolate su tutto il dataset, uso di feature che nascondono $y$</li>
<li><strong>Guardare il test durante lo sviluppo</strong>: overfitting sul test</li>
<li><strong>Ignorare l\'imbalance</strong>: accuracy 99% con classe rara è fake</li>
<li><strong>Trascurare i baseline</strong>: se un modello mediocre batte il tuo modello complesso, hai un bug</li>
<li><strong>Cercare complessità prima di robustezza</strong>: prima fai funzionare la lineare, poi passi ai transformer</li>
</ul>

<h3>14.3 Prossimi passi</h3>
<p>Fatta questa base, per continuare:</p>
<ul>
<li>Kaggle: 3-5 competizioni tabulari per consolidare. Poi immagini/testo.</li>
<li>Corso di deep learning di Andrew Ng o fast.ai.</li>
<li>Libro "Hands-On Machine Learning" di Géron dalla copertina alla copertina.</li>
<li>Un progetto personale con dati che ti interessano davvero. La motivazione è il carburante.</li>
</ul>
` },
    { type: 'callout', variant: 'tip', title: 'Ultima raccomandazione', content: 'Il ML si impara costruendo modelli, non leggendo di modelli. Ogni ora di libro deve essere seguita da due ore di codice. E non aver paura di fallire: un modello che crolla ti insegna più di dieci che funzionano subito.' },
  ],
  esempi: [
    { type: 'md', content: '<h3>Progetto completo su breast_cancer</h3><p>Un template end-to-end che puoi copiare e adattare a qualsiasi problema tabulare.</p>' },
    { type: 'run', content: `import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# 1) Carica e ispeziona
data = load_breast_cancer()
X, y = data.data, data.target
print(f"Shape: {X.shape}, classi: {np.bincount(y)}")

# 2) Split
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3) Baseline
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy='most_frequent').fit(Xt, yt)
print(f"Baseline (majority): {dummy.score(Xe, ye):.4f}")

# 4) Prova tre modelli con CV
models = {
    'Logistic':  Pipeline([('s', StandardScaler()), ('m', LogisticRegression(max_iter=5000))]),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=0),
    'SVM-RBF':   Pipeline([('s', StandardScaler()), ('m', SVC(probability=True))]),
}
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for name, m in models.items():
    s = cross_val_score(m, Xt, yt, cv=cv, scoring='roc_auc')
    print(f"{name:15s} CV AUC: {s.mean():.4f} +/- {s.std():.4f}")

# 5) Tuning del migliore
grid = {'m__C':[0.01, 0.1, 1, 10, 100]}
gs = GridSearchCV(models['Logistic'], grid, cv=cv, scoring='roc_auc').fit(Xt, yt)
print(f"\\nMigliori parametri: {gs.best_params_}")

# 6) Valutazione finale
final = gs.best_estimator_
yp = final.predict(Xe); yprob = final.predict_proba(Xe)[:,1]
print(f"\\nTest AUC:      {roc_auc_score(ye, yprob):.4f}")
print(f"Test Accuracy: {(yp==ye).mean():.4f}")
print("\\nMatrice di confusione:\\n", confusion_matrix(ye, yp))
print("\\nReport:\\n", classification_report(ye, yp, digits=3))` },
  ],
  esercizi: [
    { type: 'exercise', difficulty: 'hard', title: 'Progetto — California housing',
      prompt: '<p>Task di regressione. Prevedi il target sulla base di 8 feature. Devi ottenere <strong>R² test > 0.75</strong>. Sei libero di scegliere modello, preprocessing, tuning.</p>',
      starter: `import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

try:
    d = fetch_california_housing()
    X, y = d.data, d.target
except Exception as e:
    # fallback se il download non è possibile in Pyodide
    np.random.seed(0)
    X = np.random.randn(2000, 8)
    y = (2*X[:,0] - X[:,1] + 0.5*X[:,0]*X[:,2] + np.random.randn(2000)*0.5)

Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=42)

r2 = None
print("R²:", r2)`,
      check: `if r2 is None: print("FAIL: None")
elif r2 > 0.75: print(f"OK — R² = {r2:.4f}")
else: print(f"FAIL — R² = {r2:.4f}")`,
      solution: `m = GradientBoostingRegressor(n_estimators=300, max_depth=4, random_state=0).fit(Xt, yt)
r2 = m.score(Xe, ye)
# Random Forest funziona quasi identico:
# m = RandomForestRegressor(n_estimators=200, random_state=0).fit(Xt, yt)` },

    { type: 'exercise', difficulty: 'hard', title: 'Progetto — pipeline completa da zero',
      prompt: '<p>Sul dataset wine, costruisci una pipeline che comprenda: standardizzazione, PCA (mantieni il 95% di varianza), e un classificatore a tua scelta. Fai tuning con GridSearchCV. Target: accuracy test > 0.95. Salva in <code class="inline">acc</code>.</p>',
      starter: `from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

X, y = load_wine(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

acc = None`,
      check: `if acc is None: print("FAIL: None")
elif acc > 0.95: print(f"OK — {acc:.4f}")
else: print(f"FAIL — {acc:.4f}")`,
      solution: `pipe = Pipeline([('s', StandardScaler()),
                 ('pca', PCA(n_components=0.95)),
                 ('c', LogisticRegression(max_iter=5000))])
gs = GridSearchCV(pipe, {'c__C':[0.1, 1, 10]}, cv=5).fit(Xt, yt)
acc = gs.score(Xe, ye)` },
  ],
  test: [
    { type: 'md', content: '<h3>Test finale — riflessione</h3><p>Non c\'è codice: rispondi alle domande. Sono le più importanti che ti farai iniziando un progetto reale.</p>' },
    { type: 'quiz', questions: [
      { q: 'Un modello ha CV accuracy 0.95 ma sul test cala a 0.85. La spiegazione più probabile è:', options: ['Test set piccolo (varianza alta)', 'Data leakage durante training', 'Hai fatto tuning contro il test set — hai overfittato sul test', 'Tutte plausibili, andrebbero verificate una per una'], answer: 3, explain: 'Diagnosticare senza dati aggiuntivi richiede investigazione, non conclusioni affrettate.' },
      { q: 'Il primo passo di un progetto è:', options: ['Scegliere il modello', 'Definire chiaramente cosa vuoi predire e con che metrica', 'Cercare dati', 'Installare le librerie'], answer: 1, explain: 'Senza definizione del problema, tutto il resto è casuale.' },
      { q: 'Un baseline stupido è utile perché:', options: ['Fa perdere tempo', 'Stabilisce il floor che qualsiasi modello serio deve battere', 'È divertente', 'Non lo è'], answer: 1, explain: 'Se il tuo Deep Learning fa uguale alla media, hai un bug o un problema di segnale.' },
      { q: 'Il test set va guardato:', options: ['Ogni volta che iteri', 'Solo alla fine, una sola volta', 'Ogni epoch', 'Solo se train val hanno grandi gap'], answer: 1, explain: 'Guardare il test durante lo sviluppo lo trasforma in un secondo val set.' },
    ] },
    { type: 'md', content: '<h3>Congratulazioni</h3><p>Se sei arrivato qui e hai completato gli esercizi, hai una base solida di machine learning classico. I prossimi passi naturali sono: (1) fare 3-5 progetti Kaggle su dati tabulari, (2) imparare gradient boosting avanzato (XGBoost, LightGBM), (3) iniziare deep learning con PyTorch, (4) leggere il libro di Géron per approfondire.</p><p><strong>Il ML si impara facendo. Adesso inizia a farlo.</strong></p>' },
  ]
});

