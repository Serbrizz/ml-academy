/* Lezione: Introduzione alle reti neurali (approfondita) */
window.LESSONS.push({
  id: 'l13-neural-nets',
  title: 'Reti neurali: fondamenta profonde',
  tag: 'Deep Learning',
  summary: 'Percettrone, MLP, backpropagation da zero. Ottimizzatori moderni, regolarizzazione, scelte architetturali. Base indispensabile per capire tutto il deep learning.',
  teoria: [
    { type: 'md', content: `
<h3>13.1 Dal lineare al non lineare</h3>
<p>La regressione logistica è un neurone singolo. Se componiamo tanti neuroni in strati con non-linearità intermedie, otteniamo un <em>Multi-Layer Perceptron</em> (MLP). Il teorema di approssimazione universale (Cybenko 1989, Hornik 1991) garantisce che un MLP con anche solo un hidden layer sufficientemente largo può approssimare qualsiasi funzione continua su un compatto.</p>
<p>Il teorema però non dice quanto largo o quanti dati servano. Nella pratica reti profonde (molti layer) sono più efficienti in termini di parametri rispetto a reti molto larghe con un solo strato: catturano gerarchie di rappresentazioni.</p>

<h3>13.2 Un neurone artificiale</h3>
<p>Un singolo neurone calcola:</p>
<p>$$a = \\phi(w^T x + b)$$</p>
<p>dove $\\phi$ è la <em>funzione di attivazione</em> non lineare. Se rimuovi $\\phi$, la composizione di più neuroni è ancora una singola trasformazione lineare — inutile aggiungere layer.</p>

<h3>13.3 Funzioni di attivazione</h3>
<p>Le scelte principali, con pro e contro:</p>
<ul>
<li><strong>Sigmoide</strong>: $\\sigma(z) = 1/(1+e^{-z}) \\in (0,1)$. Storicamente popolare, oggi <em>quasi mai</em> nei hidden layer: satura per $|z|$ grande, gradienti che svaniscono.</li>
<li><strong>Tanh</strong>: $\\tanh(z) \\in (-1,1)$. Centrata su zero, meglio della sigmoide, ma satura ancora.</li>
<li><strong>ReLU</strong>: $\\max(0, z)$. Standard moderno. Semplice, veloce, non satura per $z>0$. Problema: "dying ReLU" (neuroni bloccati a zero per sempre).</li>
<li><strong>Leaky ReLU</strong>: $z$ se $z>0$, altrimenti $\\alpha z$ con $\\alpha=0.01$. Risolve il dying ReLU.</li>
<li><strong>ELU, SELU, GELU</strong>: varianti smooth. GELU è quella usata nei Transformer.</li>
<li><strong>Softmax</strong>: normalizza un vettore in una distribuzione di probabilità. Usato solo in <em>output</em> per classificazione multi-classe.</li>
</ul>

<h3>13.4 Un MLP formalmente</h3>
<p>Layer per layer, in notazione compatta:</p>
<p>$$h^{(l)} = \\phi\\!\\left(W^{(l)} h^{(l-1)} + b^{(l)}\\right)$$</p>
<p>con $h^{(0)} = x$ (input) e $\\hat y = h^{(L)}$ (output). I parametri sono $\\theta = \\{W^{(l)}, b^{(l)}\\}_{l=1}^L$.</p>
<p>Il numero di neuroni per layer e il numero di layer sono <em>iperparametri architetturali</em>. Regole empiriche:</p>
<ul>
<li>Layer larghi al centro, più stretti alle estremità o costanti.</li>
<li>Per problemi tabulari: 2-3 hidden layer da 64-256 neuroni funzionano quasi sempre.</li>
<li>Aggiungere layer dà rendimenti decrescenti oltre una certa profondità (7-10) senza tecniche come residual connections.</li>
</ul>

<h3>13.5 Loss e output layer per problema</h3>
<table style="border-collapse:collapse;">
<tr style="border-bottom:1px solid var(--border);"><th style="padding:6px 12px;">Problema</th><th style="padding:6px 12px;">Attivazione output</th><th style="padding:6px 12px;">Loss</th></tr>
<tr><td style="padding:4px 12px;">Regressione</td><td style="padding:4px 12px;">Lineare</td><td style="padding:4px 12px;">MSE (o MAE, Huber)</td></tr>
<tr><td style="padding:4px 12px;">Classificazione binaria</td><td style="padding:4px 12px;">Sigmoide</td><td style="padding:4px 12px;">Binary Cross-Entropy</td></tr>
<tr><td style="padding:4px 12px;">Classificazione multi-classe</td><td style="padding:4px 12px;">Softmax</td><td style="padding:4px 12px;">Categorical Cross-Entropy</td></tr>
<tr><td style="padding:4px 12px;">Multi-label</td><td style="padding:4px 12px;">Sigmoide per classe</td><td style="padding:4px 12px;">BCE per classe (sommate)</td></tr>
</table>

<h3>13.6 Backpropagation, spiegato bene</h3>
<p>Il training minimizza la loss con gradient descent. Il calcolo del gradiente rispetto a ogni parametro si fa applicando la regola della catena all'indietro attraverso il grafo di calcolo.</p>
<p>Notazione: per un MLP con L layer, l'errore al layer $l$ è:</p>
<p>$$\\delta^{(l)} = \\left(W^{(l+1)T} \\delta^{(l+1)}\\right) \\odot \\phi'\\!\\left(z^{(l)}\\right)$$</p>
<p>dove $z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}$ è la pre-attivazione e $\\odot$ è il prodotto element-wise.</p>
<p>I gradienti sono:</p>
<p>$$\\frac{\\partial \\mathcal{L}}{\\partial W^{(l)}} = \\delta^{(l)} (h^{(l-1)})^T, \\quad \\frac{\\partial \\mathcal{L}}{\\partial b^{(l)}} = \\delta^{(l)}$$</p>
<p>Nei framework moderni (PyTorch, TensorFlow, JAX) tutto questo è calcolato automaticamente da <em>autograd</em>: tu scrivi solo il forward, il backward è gratis.</p>

<h3>13.7 Il problema del gradient vanishing/exploding</h3>
<p>In reti profonde con sigmoid/tanh, i gradienti moltiplicati per L layer possono <strong>svanire esponenzialmente</strong> (se sono $<1$) o <strong>esplodere</strong> (se sono $>1$). Soluzioni:</p>
<ul>
<li><strong>ReLU</strong>: gradiente 1 per $z>0$, non satura.</li>
<li><strong>Inizializzazione intelligente</strong>: Glorot (Xavier), He initialization. Scala i pesi in modo che le varianze delle attivazioni si mantengano attraverso i layer.</li>
<li><strong>Batch normalization</strong>: normalizza le attivazioni di ogni layer, stabilizza il training.</li>
<li><strong>Residual connections</strong> (ResNet): permettono al gradiente di "saltare" i layer, cruciali per reti >20 layer.</li>
</ul>

<h3>13.8 Ottimizzatori moderni</h3>
<p>SGD è il minimo. Nella pratica si usano varianti più sofisticate:</p>
<ul>
<li><strong>SGD con momentum</strong>: aggiunge inerzia. Update: $v_t = \\gamma v_{t-1} + \\eta \\nabla \\mathcal{L}$, poi $\\theta_t = \\theta_{t-1} - v_t$. $\\gamma \\approx 0.9$.</li>
<li><strong>Nesterov momentum</strong>: variante che "guarda avanti" prima di calcolare il gradiente.</li>
<li><strong>RMSprop</strong>: divide il gradiente per una media mobile della sua magnitudine.</li>
<li><strong>Adam</strong>: combina momentum + RMSprop + bias correction. Il default ragionevole per la maggior parte dei casi. lr tipico: 1e-3.</li>
<li><strong>AdamW</strong>: Adam con weight decay corretto. Preferito nei Transformer.</li>
</ul>

<h3>13.9 Regolarizzazione</h3>
<ul>
<li><strong>L2 weight decay</strong>: aggiunge $\\lambda \\|W\\|^2$ alla loss. Equivalente a shrinkage.</li>
<li><strong>L1 weight decay</strong>: penalità $\\lambda \\|W\\|_1$. Produce sparsità (raramente usato in DL).</li>
<li><strong>Dropout</strong>: durante il training azzera casualmente una frazione $p$ (tipicamente 0.2-0.5) dei neuroni. Al test, tutti i neuroni sono attivi ma scalati. Impedisce ai neuroni di "collaborare troppo" (co-adaptation).</li>
<li><strong>Data augmentation</strong>: aumentare artificialmente il training set con trasformazioni (crop, rotazioni per immagini). Regolarizzatore potentissimo.</li>
<li><strong>Early stopping</strong>: fermarsi quando la val loss smette di scendere.</li>
</ul>

<h3>13.10 Tuning delle reti</h3>
<p>Priorità in ordine di impatto tipico:</p>
<ol>
<li><strong>Learning rate</strong>: il singolo iperparametro più importante. Prova 1e-1, 1e-2, 1e-3, 1e-4, 1e-5 su scala logaritmica. Con Adam, 1e-3 è ragionevole default.</li>
<li><strong>Batch size</strong>: 32-256 tipico. Batch grandi = training stabile ma generalizzazione a volte peggiore.</li>
<li><strong>Architettura</strong> (numero layer, neuroni per layer).</li>
<li><strong>Regolarizzazione</strong> (dropout, weight decay).</li>
<li><strong>Learning rate scheduling</strong>: ridurre lr durante il training (cosine annealing, step decay).</li>
</ol>

<h3>13.11 Diagnosi problemi di training</h3>
<ul>
<li><strong>Loss diverge o è NaN</strong>: learning rate troppo alto, o gradient explosion. Riduci lr o clippa i gradienti.</li>
<li><strong>Loss stagna (non scende)</strong>: lr troppo basso, o inizializzazione sbagliata, o dying ReLU.</li>
<li><strong>Train loss scende, val loss no</strong>: overfitting. Aggiungi regolarizzazione, dati, dropout.</li>
<li><strong>Train loss non scende quasi mai</strong>: underfitting. Aumenta capacità (layer/neuroni), o problema di dati.</li>
</ul>

<h3>13.12 Perché scikit-learn MLP non è "vero" deep learning</h3>
<p>Il MLPClassifier di sklearn è didatticamente ok ma limitato: non ha GPU, tantissime opzioni moderne (batch norm, layer complessi, dropout controllabile) sono mancanti. Per fare deep learning vero si usa <strong>PyTorch</strong> (o TensorFlow/Keras). PyTorch non è installato in Pyodide, quindi in questa lezione useremo NumPy per capire i concetti e sklearn per esempi rapidi. Nella lezione 17 vediamo il codice PyTorch che poi eseguirai in locale.</p>
` },
    { type: 'callout', variant: 'tip', title: 'Consiglio pratico', content: 'Per la maggior parte dei problemi tabulari, un HistGradientBoosting ben tarato batte una rete neurale a meno di non avere milioni di righe. Le reti neurali dominano dove ci sono strutture (immagini, testo, audio, grafi) che i tree-based non catturano naturalmente.' },
  ],
  esempi: [
    { type: 'md', content: '<h3>Esempio 1: MLP con sklearn</h3>' },
    { type: 'run', content: `from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = load_digits(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

pipe = Pipeline([('s', StandardScaler()),
                 ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64),
                                       activation='relu',
                                       solver='adam',
                                       learning_rate_init=1e-3,
                                       alpha=1e-4,
                                       max_iter=300,
                                       random_state=0))])
pipe.fit(Xt, yt)
print(f"Train acc: {pipe.score(Xt, yt):.4f}")
print(f"Test  acc: {pipe.score(Xe, ye):.4f}")
print(f"\\nArchitettura: input({X.shape[1]}) -> 128 -> 64 -> {len(set(y))} classi")
for i, W in enumerate(pipe.named_steps['mlp'].coefs_):
    print(f"  layer {i}: pesi shape {W.shape}, bias shape {pipe.named_steps['mlp'].intercepts_[i].shape}")
print(f"Numero totale parametri: {sum(W.size for W in pipe.named_steps['mlp'].coefs_) + sum(b.size for b in pipe.named_steps['mlp'].intercepts_)}")` },
    { type: 'md', content: '<h3>Esempio 2: MLP + backprop from scratch su XOR</h3><p>XOR è non separabile linearmente. Un neurone singolo non ce la fa; una rete piccola sì. Implementiamo tutto a mano.</p>' },
    { type: 'run', content: `import numpy as np
np.random.seed(0)

# XOR: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

# Architettura: 2 -> 8 (relu) -> 1 (sigmoid)
W1 = np.random.randn(2, 8) * 0.5
b1 = np.zeros(8)
W2 = np.random.randn(8, 1) * 0.5
b2 = np.zeros(1)

def sig(z): return 1/(1+np.exp(-np.clip(z, -50, 50)))
def relu(z): return np.maximum(0, z)

lr = 0.5
for step in range(3000):
    # Forward
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    y_hat = sig(z2)
    # Loss (BCE)
    loss = -np.mean(y*np.log(y_hat+1e-9) + (1-y)*np.log(1-y_hat+1e-9))
    # Backward
    dz2 = (y_hat - y) / len(X)
    dW2 = a1.T @ dz2
    db2 = dz2.sum(0)
    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0)     # derivata di ReLU
    dW1 = X.T @ dz1
    db1 = dz1.sum(0)
    # Update
    W1 -= lr*dW1; b1 -= lr*db1
    W2 -= lr*dW2; b2 -= lr*db2
    if step % 500 == 0:
        print(f"step {step:4d}: loss = {loss:.5f}")

print("\\nPredizioni finali:", y_hat.ravel().round(4))
print("Vere:               ", y.ravel())` },
    { type: 'md', content: '<h3>Esempio 3: confronto ottimizzatori</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = load_digits(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

results = {}
for solver in ['sgd', 'adam', 'lbfgs']:
    p = Pipeline([('s', StandardScaler()),
                   ('m', MLPClassifier(hidden_layer_sizes=(64,),
                                       solver=solver,
                                       max_iter=200,
                                       random_state=0))])
    p.fit(Xt, yt)
    results[solver] = p.score(Xe, ye)
    print(f"{solver:6s}: test acc = {results[solver]:.4f}, n_iter = {p.named_steps['m'].n_iter_}")` },
    { type: 'md', content: '<h3>Esempio 4: effetto della profondità</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

X, y = make_moons(n_samples=1000, noise=0.3, random_state=0)

print(f"{'architettura':>25} {'CV acc':>8}")
for arch in [(2,), (16,), (16, 16), (64, 32), (128, 64, 32), (256, 128, 64, 32)]:
    p = Pipeline([('s', StandardScaler()),
                   ('m', MLPClassifier(hidden_layer_sizes=arch, max_iter=500,
                                       random_state=0, alpha=1e-4))])
    s = cross_val_score(p, X, y, cv=5).mean()
    print(f"{str(arch):>25} {s:>8.4f}")` },
    { type: 'md', content: '<h3>Esempio 5: dying ReLU e come diagnosticarlo</h3>' },
    { type: 'run', content: `import numpy as np
np.random.seed(0)

# Simuliamo un layer che riceve input negativo per pesi mal inizializzati
X = np.random.randn(1000, 20)
W = np.random.randn(20, 50) * 5  # varianza ENORME -> molte attivazioni negative
z = X @ W
a = np.maximum(0, z)  # ReLU

frac_zero_per_neuron = (a == 0).mean(axis=0)
dead_neurons = (frac_zero_per_neuron > 0.99).sum()
print(f"Neuroni con >99% output zero (dead): {dead_neurons}/{a.shape[1]}")
print(f"Percentuale media di attivazioni zero: {frac_zero_per_neuron.mean()*100:.1f}%")

# Con inizializzazione He (corretta per ReLU): std = sqrt(2/fan_in)
W_he = np.random.randn(20, 50) * np.sqrt(2 / 20)
a_he = np.maximum(0, X @ W_he)
dead_he = ((a_he == 0).mean(axis=0) > 0.99).sum()
print(f"\\nCon inizializzazione He: dead neurons = {dead_he}/{a_he.shape[1]}")
print(f"Percentuale zero: {(a_he == 0).mean()*100:.1f}%")` },
    { type: 'md', content: '<h3>Esempio 6: dropout in NumPy</h3>' },
    { type: 'run', content: `import numpy as np
np.random.seed(0)

def forward_with_dropout(x, W, b, p_drop=0.3, training=True):
    z = x @ W + b
    a = np.maximum(0, z)
    if training and p_drop > 0:
        mask = (np.random.rand(*a.shape) > p_drop).astype(float)
        a = a * mask / (1 - p_drop)   # inverted dropout: scala già al training
    return a

x = np.random.randn(4, 10)
W = np.random.randn(10, 5) * 0.5
b = np.zeros(5)

print("Con dropout (training):")
print(forward_with_dropout(x, W, b, p_drop=0.3, training=True).round(3))
print("\\nSenza dropout (inference):")
print(forward_with_dropout(x, W, b, p_drop=0.3, training=False).round(3))
print("\\nNota: nel training, alcuni output sono zero (dropped) e gli altri sono amplificati.")
print("Al test, tutti i neuroni sono attivi con output originale.")` },
    { type: 'md', content: '<h3>Esempio 7: training curve, diagnosi visiva</h3>' },
    { type: 'run', content: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
Xt_s = StandardScaler().fit(Xt).transform(Xt)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, lr in zip(axes, [1e-4, 1e-2, 1e-1]):
    m = MLPClassifier(hidden_layer_sizes=(64,), max_iter=100,
                      solver='sgd', learning_rate_init=lr,
                      random_state=0).fit(Xt_s, yt)
    ax.plot(m.loss_curve_)
    ax.set_title(f'lr = {lr}')
    ax.set_xlabel('Iterazione'); ax.set_ylabel('Loss')
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
print("lr troppo basso: convergenza lenta")
print("lr giusto: convergenza rapida e stabile")
print("lr troppo alto: oscillazione o divergenza")` },
  ],
  esercizi: [
    { type: 'exercise', difficulty: 'easy', title: 'Es. 13.1 — Forward pass a mano',
      prompt: '<p>Implementa il forward pass di una rete 3 -> 5 -> 2 con ReLU nel primo layer e softmax nell\'output. Salva l\'output (probabilità che somma a 1) in <code class="inline">probs</code>.</p>',
      starter: `import numpy as np
np.random.seed(0)
x = np.array([1.0, -0.5, 2.0])
W1 = np.random.randn(3, 5); b1 = np.zeros(5)
W2 = np.random.randn(5, 2); b2 = np.zeros(2)

# TODO
probs = None
print(probs, "somma:", probs.sum() if probs is not None else None)`,
      check: `import numpy as np
_z1 = x @ W1 + b1
_a1 = np.maximum(0, _z1)
_z2 = _a1 @ W2 + b2
_e = np.exp(_z2 - _z2.max())
_expected = _e / _e.sum()
if probs is not None and np.allclose(probs, _expected, atol=1e-6):
    print("OK — probs somma a", probs.sum())
else:
    print(f"FAIL — atteso {_expected}")`,
      solution: `z1 = x @ W1 + b1
a1 = np.maximum(0, z1)
z2 = a1 @ W2 + b2
# softmax numericamente stabile
e = np.exp(z2 - z2.max())
probs = e / e.sum()` },

    { type: 'exercise', difficulty: 'easy', title: 'Es. 13.2 — Derivate delle attivazioni',
      prompt: '<p>Implementa le derivate di ReLU, sigmoide, tanh. Ogni funzione prende un array e ritorna un array con la derivata elemento per elemento.</p>',
      starter: `import numpy as np
def d_relu(z):
    return None
def d_sigmoid(z):
    return None
def d_tanh(z):
    return None

z = np.array([-2., -0.5, 0., 0.5, 2.])
print("d_relu   :", d_relu(z))
print("d_sigmoid:", d_sigmoid(z))
print("d_tanh   :", d_tanh(z))`,
      check: `import numpy as np
z = np.array([-2., -0.5, 0., 0.5, 2.])
_relu = (z > 0).astype(float)
_sig_val = 1/(1+np.exp(-z)); _sig = _sig_val*(1-_sig_val)
_tanh = 1 - np.tanh(z)**2
if np.allclose(d_relu(z), _relu) and np.allclose(d_sigmoid(z), _sig) and np.allclose(d_tanh(z), _tanh):
    print("OK")
else:
    print("FAIL")`,
      solution: `def d_relu(z): return (z > 0).astype(float)
def d_sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s * (1 - s)
def d_tanh(z): return 1 - np.tanh(z)**2` },

    { type: 'exercise', difficulty: 'easy', title: 'Es. 13.3 — Cross-entropy multi-classe',
      prompt: '<p>Data <code class="inline">y_true</code> (interi, indici di classe) e <code class="inline">probs</code> (matrice n×K di probabilità), calcola cross-entropy media. Salva in <code class="inline">ce</code>.</p>',
      starter: `import numpy as np
np.random.seed(0)
n, K = 100, 3
y_true = np.random.randint(0, K, n)
probs = np.random.dirichlet(np.ones(K), n)

ce = None
print(ce)`,
      check: `import numpy as np
from sklearn.metrics import log_loss
_expected = log_loss(y_true, probs, labels=list(range(K)))
if ce is not None and abs(ce - _expected) < 1e-6:
    print(f"OK — CE {ce:.4f}")
else:
    print(f"FAIL — atteso {_expected:.4f}")`,
      solution: `eps = 1e-9
ce = -np.mean(np.log(probs[np.arange(n), y_true] + eps))` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 13.4 — Backprop di un layer',
      prompt: '<p>Dato un layer $a = \\phi(Wx + b)$ con $\\phi$ = ReLU, dato l\'errore sopra $\\delta_{out}$, ritorna la tupla <code class="inline">(dW, db, dx)</code> del backward.</p>',
      starter: `import numpy as np
def layer_backward(x, W, b, delta_out, activation='relu'):
    z = x @ W + b
    a = np.maximum(0, z)
    # TODO: calcola dW, db, dx
    dW = None
    db = None
    dx = None
    return dW, db, dx

np.random.seed(0)
x = np.random.randn(4, 3)   # batch di 4
W = np.random.randn(3, 5)
b = np.zeros(5)
delta_out = np.random.randn(4, 5)   # errore sopra il layer

dW, db, dx = layer_backward(x, W, b, delta_out)
print("dW shape:", dW.shape if dW is not None else None)
print("db shape:", db.shape if db is not None else None)
print("dx shape:", dx.shape if dx is not None else None)`,
      check: `import numpy as np
_z = x @ W + b
_dz = delta_out * (_z > 0)
_dW = x.T @ _dz
_db = _dz.sum(0)
_dx = _dz @ W.T
if (np.allclose(dW, _dW) and np.allclose(db, _db) and np.allclose(dx, _dx)):
    print("OK")
else:
    print("FAIL")`,
      solution: `z = x @ W + b
dz = delta_out * (z > 0)
dW = x.T @ dz
db = dz.sum(0)
dx = dz @ W.T
return dW, db, dx` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 13.5 — MLP su moons',
      prompt: '<p>Su make_moons con noise=0.25, addestra un MLP che raggiunge test accuracy > 0.94. Standardizza. Salva in <code class="inline">acc</code>.</p>',
      starter: `import numpy as np
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=1000, noise=0.25, random_state=0)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.25, random_state=42)
acc = None`,
      check: `if acc is not None and acc > 0.94:
    print(f"OK — {acc:.4f}")
else:
    print(f"FAIL — {acc}")`,
      solution: `p = Pipeline([('s', StandardScaler()),
              ('m', MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=1000,
                                   alpha=1e-4, random_state=0))]).fit(Xt, yt)
acc = p.score(Xe, ye)` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 13.6 — He initialization',
      prompt: '<p>Implementa l\'inizializzazione He per un layer con fan_in neuroni in ingresso e fan_out in uscita: pesi $\\sim \\mathcal{N}(0, 2/fan\\_in)$. Verifica che la varianza delle attivazioni post-ReLU rimanga stabile.</p>',
      starter: `import numpy as np
def he_init(fan_in, fan_out):
    return None

np.random.seed(0)
x = np.random.randn(1000, 100)
W = he_init(100, 100)
a = np.maximum(0, x @ W)
print(f"Var input: {x.var():.3f}")
print(f"Var post-ReLU: {a.var():.3f}")
print("(dovrebbero essere confrontabili)")`,
      check: `import numpy as np
np.random.seed(0)
_x = np.random.randn(1000, 100)
_W = he_init(100, 100)
_a = np.maximum(0, _x @ _W)
if _W is not None and abs(_a.var() - _x.var()) / _x.var() < 0.5:
    print(f"OK — var stabile: input {_x.var():.3f}, output {_a.var():.3f}")
else:
    print(f"FAIL — var input {_x.var()}, output {_a.var()}")`,
      solution: `def he_init(fan_in, fan_out):
    return np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 13.7 — MLP 2-layer completo da zero',
      prompt: '<p>Implementa un MLP 2-layer (input -> hidden ReLU -> output softmax) con training completo su Iris. Target: test accuracy > 0.9. Salva la funzione <code class="inline">train_mlp(X, y, hidden_dim, epochs, lr)</code> che ritorna (W1, b1, W2, b2).</p>',
      starter: `import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def train_mlp(X, y, hidden_dim=16, epochs=500, lr=0.1):
    # TODO
    return None, None, None, None

W1, b1, W2, b2 = train_mlp(Xt, yt, hidden_dim=16, epochs=500, lr=0.1)

# valuta
def predict(X, W1, b1, W2, b2):
    a1 = np.maximum(0, X @ W1 + b1)
    z2 = a1 @ W2 + b2
    return np.argmax(z2, axis=1)

acc = np.mean(predict(Xe, W1, b1, W2, b2) == ye) if W1 is not None else 0
print("Test acc:", acc)`,
      check: `import numpy as np
acc = np.mean(predict(Xe, W1, b1, W2, b2) == ye)
if acc > 0.9:
    print(f"OK — {acc:.4f}")
else:
    print(f"FAIL — {acc:.4f}")`,
      solution: `def train_mlp(X, y, hidden_dim=16, epochs=500, lr=0.1):
    np.random.seed(0)
    n, d = X.shape
    K = y.max() + 1
    W1 = np.random.randn(d, hidden_dim) * np.sqrt(2/d)
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(hidden_dim, K) * np.sqrt(2/hidden_dim)
    b2 = np.zeros(K)
    y_onehot = np.eye(K)[y]
    for _ in range(epochs):
        z1 = X @ W1 + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2 + b2
        e = np.exp(z2 - z2.max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
        # backward
        dz2 = (p - y_onehot) / n
        dW2 = a1.T @ dz2
        db2 = dz2.sum(0)
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(0)
        W1 -= lr*dW1; b1 -= lr*db1
        W2 -= lr*dW2; b2 -= lr*db2
    return W1, b1, W2, b2` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 13.8 — SGD con momentum',
      prompt: '<p>Implementa un ottimizzatore SGD con momentum. Formula: $v_t = \\gamma v_{t-1} + \\eta g$, $\\theta_t = \\theta_{t-1} - v_t$. Applica per minimizzare $f(x, y) = 10x^2 + y^2$ partendo da $(1, 1)$, con $\\eta=0.1$ e $\\gamma=0.9$, per 50 iterazioni. Salva il punto finale in <code class="inline">xy_final</code>.</p>',
      starter: `import numpy as np
def grad_f(v):
    return np.array([20*v[0], 2*v[1]])

v = np.array([1.0, 1.0])
velocity = np.zeros(2)
lr = 0.1
momentum = 0.9

# TODO: 50 iterazioni con momentum
xy_final = None`,
      check: `import numpy as np
_v = np.array([1.0, 1.0])
_vel = np.zeros(2)
for _ in range(50):
    _g = np.array([20*_v[0], 2*_v[1]])
    _vel = 0.9*_vel + 0.1*_g
    _v = _v - _vel
if xy_final is not None and np.allclose(xy_final, _v, atol=1e-6):
    print(f"OK — punto finale {xy_final}")
else:
    print(f"FAIL — atteso {_v}")`,
      solution: `for _ in range(50):
    g = grad_f(v)
    velocity = momentum * velocity + lr * g
    v = v - velocity
xy_final = v` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 13.9 — Confronto ottimizzatori',
      prompt: '<p>Su digits, confronta MLPClassifier con solver=sgd, adam, lbfgs. Salva la loro test accuracy in un dizionario <code class="inline">scores</code>.</p>',
      starter: `from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scores = {}
# TODO`,
      check: `if scores and set(scores.keys()) == {'sgd','adam','lbfgs'} and all(v > 0.9 for v in scores.values()):
    print(f"OK — {scores}")
else:
    print(f"FAIL — {scores}")`,
      solution: `for s in ['sgd','adam','lbfgs']:
    p = Pipeline([('sc', StandardScaler()),
                  ('m', MLPClassifier(hidden_layer_sizes=(64,), solver=s, max_iter=300, random_state=0))]).fit(Xt, yt)
    scores[s] = p.score(Xe, ye)` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 13.10 — Regolarizzazione',
      prompt: '<p>Su make_moons molto rumoroso (noise=0.4), confronta MLP con alpha=0 (nessuna regolarizzazione) e alpha=0.1 (forte L2). Salva <code class="inline">acc_no_reg, acc_reg</code>. Verifica che quello regolarizzato generalizzi meglio.</p>',
      starter: `import numpy as np
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=300, noise=0.4, random_state=0)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=0)

acc_no_reg = None
acc_reg = None`,
      check: `if acc_no_reg is not None and acc_reg is not None:
    if acc_reg >= acc_no_reg - 0.02:
        print(f"OK — senza reg {acc_no_reg:.4f}, con reg {acc_reg:.4f}")
    else:
        print(f"WARN — regolarizzato non migliore: {acc_no_reg:.4f} vs {acc_reg:.4f}")`,
      solution: `def make(alpha):
    return Pipeline([('s', StandardScaler()),
                     ('m', MLPClassifier(hidden_layer_sizes=(128, 64), alpha=alpha,
                                          max_iter=1000, random_state=0))])
acc_no_reg = make(0).fit(Xt, yt).score(Xe, ye)
acc_reg = make(0.1).fit(Xt, yt).score(Xe, ye)` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 13.11 — Dropout inverted da zero',
      prompt: '<p>Implementa <code class="inline">inverted_dropout(a, p, training)</code>: se training, applica dropout con probabilità p di azzerare, scalando gli attivi per $1/(1-p)$. Se non training, ritorna a invariato.</p>',
      starter: `import numpy as np
def inverted_dropout(a, p=0.3, training=True):
    return None

np.random.seed(0)
a = np.random.randn(4, 6)
print("Training:", inverted_dropout(a, 0.3, True).round(3))
print("\\nInference:", inverted_dropout(a, 0.3, False).round(3))`,
      check: `import numpy as np
# training deve variare, inference deve essere identico all'input
np.random.seed(0)
_a = np.random.randn(1000, 100)
_a_train = inverted_dropout(_a, 0.3, True)
_a_infer = inverted_dropout(_a, 0.3, False)
if _a_train is not None and _a_infer is not None:
    ok = np.allclose(_a_infer, _a) and 0.25 < (_a_train == 0).mean() < 0.35
    print("OK" if ok else "FAIL")`,
      solution: `def inverted_dropout(a, p=0.3, training=True):
    if not training or p == 0:
        return a
    mask = (np.random.rand(*a.shape) > p).astype(float)
    return a * mask / (1 - p)` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 13.12 — Diagnosi training issues',
      prompt: '<p>Data la loss curve di un training, decidi se il modello è: <code class="inline">"overfit"</code>, <code class="inline">"underfit"</code>, <code class="inline">"lr_too_high"</code>, <code class="inline">"good"</code>. Salva la stringa in <code class="inline">diagnosis</code>.</p>',
      starter: `import numpy as np
# Loss di training e validation registrate per 100 epoch
# Train scende bene, val scende poi risale
train_loss = np.exp(-np.linspace(0, 3, 100)) + 0.1
val_loss = np.concatenate([np.exp(-np.linspace(0, 2, 50)) + 0.2, 0.2 + 0.01*np.arange(50)])

diagnosis = None`,
      check: `if diagnosis == "overfit":
    print("OK — riconosciuto overfit: train scende, val risale")
else:
    print(f"FAIL — atteso 'overfit', tuo '{diagnosis}'")`,
      solution: `diagnosis = "overfit"` },
  ],
  test: [
    { type: 'quiz', questions: [
      { q: 'Perché senza attivazioni non lineari un MLP profondo si riduce a un modello lineare?', options: ['Non è vero', 'La composizione di trasformazioni lineari è ancora una trasformazione lineare', 'I gradienti si annullano', 'La softmax rimuove la non-linearità'], answer: 1, explain: 'Se $\\phi$ è identità, $h^{(l)} = W^{(l)}W^{(l-1)}...W^{(1)}x$ è ancora lineare in $x$.' },
      { q: 'Il dying ReLU si risolve con:', options: ['Sigmoid ovunque', 'Inizializzazione He + Leaky ReLU', 'Learning rate zero', 'Dropout'], answer: 1, explain: 'Inizializzazione corretta previene attivazioni negative massive; Leaky ReLU lascia sempre passare un piccolo gradiente.' },
      { q: 'Adam rispetto a SGD:', options: ['È sempre migliore', 'Combina momentum + normalizzazione adattiva del gradiente, spesso converge più velocemente', 'È più semplice', 'Non serve tuning'], answer: 1, explain: 'Adam adatta il lr per parametro; buon default ma non sempre supera SGD ben tarato in generalizzazione.' },
      { q: 'Il dropout al momento dell\'inference:', options: ['Azzera come al training', 'È spento (con inverted dropout gli attivi non sono scalati al training, non serve compensare)', 'Aumenta il lr', 'Si applica solo all\'output'], answer: 1, explain: 'Al test, tutti i neuroni sono attivi. L\'inverted dropout scala già durante il training, quindi inference è identica a modello normale.' },
      { q: 'Loss di training scende, val loss stagna o sale. Diagnosi:', options: ['Underfit', 'Overfit: aggiungi regolarizzazione/data/dropout', 'lr troppo basso', 'Bug nel codice'], answer: 1, explain: 'Gap crescente train/val = varianza alta = overfit.' },
    ] },
    { type: 'exercise', difficulty: 'hard', title: 'Test finale — MLP su digits',
      prompt: '<p>Costruisci un MLP che raggiunge test accuracy > 0.98 su digits. Puoi scegliere architettura, ottimizzatore, regolarizzazione. Standardizza. Salva in <code class="inline">acc</code>.</p>',
      starter: `from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

acc = None`,
      check: `if acc is None: print("FAIL: None")
elif acc > 0.98: print(f"OK — {acc:.4f}")
else: print(f"FAIL — {acc:.4f}")`,
      solution: `p = Pipeline([('s', StandardScaler()),
              ('m', MLPClassifier(hidden_layer_sizes=(256, 128, 64),
                                  activation='relu',
                                  solver='adam',
                                  learning_rate_init=1e-3,
                                  alpha=1e-4,
                                  max_iter=500,
                                  random_state=0))]).fit(Xt, yt)
acc = p.score(Xe, ye)` },

    { type: 'exercise', difficulty: 'hard', title: 'Bonus expert — MLP da zero completo',
      prompt: '<p><strong>Sfida</strong>. Implementa un MLP completo da zero (NO sklearn per il modello): 2 hidden layer con ReLU, output softmax, training con SGD + momentum, weight decay L2, mini-batch. Ottieni test accuracy > 0.95 su digits. Salva in <code class="inline">acc</code>.</p>',
      starter: `import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X = StandardScaler().fit_transform(X)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# TODO: MLP from scratch con batch, momentum, weight decay
acc = None`,
      check: `if acc is None: print("FAIL: None")
elif acc > 0.95: print(f"OK — {acc:.4f}")
else: print(f"FAIL — {acc:.4f}")`,
      solution: `np.random.seed(0)
K = 10
n, d = Xt.shape
h1, h2 = 128, 64
def he(fin, fout): return np.random.randn(fin, fout) * np.sqrt(2/fin)
W1, b1 = he(d, h1), np.zeros(h1)
W2, b2 = he(h1, h2), np.zeros(h2)
W3, b3 = he(h2, K), np.zeros(K)
vW1 = np.zeros_like(W1); vW2 = np.zeros_like(W2); vW3 = np.zeros_like(W3)
vb1 = np.zeros_like(b1); vb2 = np.zeros_like(b2); vb3 = np.zeros_like(b3)
lr, mom, wd, bs, epochs = 0.05, 0.9, 1e-4, 64, 40
y_onehot = np.eye(K)[yt]
for _ in range(epochs):
    idx = np.random.permutation(n)
    for start in range(0, n, bs):
        bi = idx[start:start+bs]
        xb = Xt[bi]; yb = y_onehot[bi]
        z1 = xb @ W1 + b1; a1 = np.maximum(0, z1)
        z2 = a1 @ W2 + b2; a2 = np.maximum(0, z2)
        z3 = a2 @ W3 + b3
        e = np.exp(z3 - z3.max(1, keepdims=True)); p = e / e.sum(1, keepdims=True)
        dz3 = (p - yb) / len(bi)
        dW3 = a2.T @ dz3 + wd*W3; db3 = dz3.sum(0)
        da2 = dz3 @ W3.T; dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2 + wd*W2; db2 = dz2.sum(0)
        da1 = dz2 @ W2.T; dz1 = da1 * (z1 > 0)
        dW1 = xb.T @ dz1 + wd*W1; db1 = dz1.sum(0)
        vW1 = mom*vW1 + lr*dW1; W1 -= vW1; vb1 = mom*vb1 + lr*db1; b1 -= vb1
        vW2 = mom*vW2 + lr*dW2; W2 -= vW2; vb2 = mom*vb2 + lr*db2; b2 -= vb2
        vW3 = mom*vW3 + lr*dW3; W3 -= vW3; vb3 = mom*vb3 + lr*db3; b3 -= vb3
z1 = np.maximum(0, Xe @ W1 + b1)
z2 = np.maximum(0, z1 @ W2 + b2)
z3 = z2 @ W3 + b3
pred = np.argmax(z3, axis=1)
acc = np.mean(pred == ye)` },
  ]
});
