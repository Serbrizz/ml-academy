/* Lezione: Regressione lineare (approfondita) */
window.LESSONS.push({
  id: 'l05-linear-regression',
  title: 'Regressione lineare',
  tag: 'Supervisionato',
  summary: 'Il modello più importante da capire davvero: fondamento matematico di tutto il ML, dalle equazioni normali alla regolarizzazione al ragionamento pratico.',
  teoria: [
    { type: 'md', content: `
<h3>5.1 Il modello e cosa significa "lineare"</h3>
<p>Dato $X \\in \\mathbb{R}^{n\\times d}$ (n esempi, d feature) e $y \\in \\mathbb{R}^n$, cerchiamo:</p>
<p>$$\\hat y_i = w_1 x_{i1} + w_2 x_{i2} + \\dots + w_d x_{id} + b = w^T x_i + b$$</p>
<p><strong>La linearità è nei parametri $w$, non nelle feature</strong>. Puoi mettere $x_i^2$, $\\log(x_i)$, $x_i x_j$ come nuove feature — il modello resta "lineare" nel senso ottimizzativo, ma può fittare curve. Questo è un punto che confonde molti principianti.</p>

<h3>5.2 La loss e la sua derivata</h3>
<p>Assorbiamo $b$ dentro $w$ aggiungendo una colonna di 1 a $X$. La loss quadratica è:</p>
<p>$$\\mathcal{L}(w) = \\frac{1}{n}\\|Xw - y\\|^2 = \\frac{1}{n}\\sum_{i=1}^n (w^T x_i - y_i)^2$$</p>
<p>Espandiamo: $\\mathcal{L}(w) = \\frac{1}{n}(Xw - y)^T(Xw - y) = \\frac{1}{n}(w^T X^T X w - 2 w^T X^T y + y^T y)$.</p>
<p>Il gradiente (fisica: come una forza che spinge verso il minimo):</p>
<p>$$\\nabla_w \\mathcal{L} = \\frac{2}{n}\\left(X^T X w - X^T y\\right) = \\frac{2}{n} X^T(Xw - y)$$</p>
<p>Ponendo $\\nabla_w \\mathcal{L} = 0$ otteniamo le famose <em>normal equations</em>:</p>
<p>$$X^T X \\hat w = X^T y \\quad\\Longrightarrow\\quad \\hat w = (X^T X)^{-1} X^T y$$</p>

<h3>5.3 Quando la formula chiusa fallisce</h3>
<p>$(X^T X)^{-1}$ esiste se e solo se $X$ ha colonne linearmente indipendenti. Quando fallisce:</p>
<ul>
<li><strong>Multicollinearità</strong>: due feature sono combinazione lineare di altre (es. "peso in kg" e "peso in libbre"). $X^T X$ è singolare.</li>
<li><strong>$n < d$</strong>: hai più feature che esempi. Il sistema è sottodeterminato, infinite soluzioni.</li>
<li><strong>Feature quasi collineari</strong>: $X^T X$ è mal condizionata (numero di condizione grande), soluzione numericamente instabile.</li>
</ul>
<p>Soluzione: usare la <strong>pseudoinversa</strong> $\\hat w = X^+ y$ calcolata via SVD, oppure regolarizzare (Ridge).</p>

<h3>5.4 Interpretazione probabilistica (perché MSE?)</h3>
<p>Assumiamo $y_i = w^T x_i + \\varepsilon_i$ con $\\varepsilon_i \\sim \\mathcal{N}(0, \\sigma^2)$ i.i.d. La verosimiglianza di un esempio è:</p>
<p>$$p(y_i | x_i, w) = \\frac{1}{\\sqrt{2\\pi}\\sigma} \\exp\\left(-\\frac{(y_i - w^T x_i)^2}{2\\sigma^2}\\right)$$</p>
<p>La log-likelihood totale è $-\\frac{1}{2\\sigma^2}\\sum_i (y_i - w^T x_i)^2 + \\text{cost}$. Massimizzarla equivale a minimizzare MSE.</p>
<p><strong>Conclusione</strong>: least squares è MLE sotto ipotesi di rumore gaussiano. Non è "solo" una scelta arbitraria: assume che gli errori siano gaussiani i.i.d.</p>

<h3>5.5 Statistica inferenziale sui coefficienti</h3>
<p>Sotto le ipotesi classiche (linearità, rumore normale, omoschedasticità, indipendenza), il vettore $\\hat w$ è a sua volta gaussiano:</p>
<p>$$\\hat w \\sim \\mathcal{N}(w^*, \\sigma^2 (X^T X)^{-1})$$</p>
<p>Da qui puoi calcolare:</p>
<ul>
<li><strong>Standard errors</strong>: $\\text{SE}(\\hat w_j) = \\hat\\sigma \\sqrt{[(X^T X)^{-1}]_{jj}}$</li>
<li><strong>t-statistic</strong>: $t_j = \\hat w_j / \\text{SE}(\\hat w_j)$ per testare $H_0 : w_j = 0$</li>
<li><strong>p-value</strong>: $P(|T_{n-d-1}| > |t_j|)$ dove $T$ è una t di Student</li>
<li><strong>Intervalli di confidenza</strong>: $\\hat w_j \\pm t_{\\alpha/2, n-d-1} \\cdot \\text{SE}(\\hat w_j)$</li>
</ul>
<p>Questa è la <em>parte non-ML</em> della regressione lineare, quella che statisti usano da un secolo. È preziosa per capire quali feature sono "significative".</p>

<h3>5.6 Ipotesi di Gauss-Markov e cosa succede se falliscono</h3>
<ol>
<li><strong>Linearità</strong>: la vera relazione $E[y|x]$ è lineare in $x$. Se non lo è, il modello è mal specificato (bias non-nullo).</li>
<li><strong>Media zero</strong>: $E[\\varepsilon | X] = 0$. Se violata, i coefficienti sono biased.</li>
<li><strong>Omoschedasticità</strong>: $\\mathrm{Var}(\\varepsilon | X) = \\sigma^2$ costante. Se non lo è (eteroschedasticità), gli standard errors sono sbagliati, ma i coefficienti restano non-biased. Rimedio: robust standard errors o Weighted Least Squares.</li>
<li><strong>Incorrelazione</strong>: $\\text{Cov}(\\varepsilon_i, \\varepsilon_j) = 0$ per $i \\neq j$. Violata nelle time series (correlazione temporale) e dati clustered.</li>
<li><strong>Rango pieno</strong>: $X$ ha rango $d$. Vedi §5.3.</li>
</ol>
<p><strong>Teorema di Gauss-Markov</strong>: sotto queste ipotesi, OLS è il Best Linear Unbiased Estimator (BLUE). Vale la pena ricordarlo: OLS è ottimo tra i lineari unbiased. Modelli non-lineari o biased possono batterlo (es. Ridge trade-off bias-varianza).</p>

<h3>5.7 Diagnostica dei residui</h3>
<p>I residui $e_i = y_i - \\hat y_i$ sono la fonte primaria di diagnostica. Guarda sempre:</p>
<ul>
<li><strong>Residuals vs Fitted</strong>: deve essere una nuvola casuale attorno a zero. Se vedi pattern &rarr; non-linearità.</li>
<li><strong>Residual normality (Q-Q plot)</strong>: se i residui non sono gaussiani, gli intervalli di confidenza sono inaffidabili.</li>
<li><strong>Scale-Location</strong>: verifica omoschedasticità.</li>
<li><strong>Leverage e distanza di Cook</strong>: identifica punti influenti che dominano il fit.</li>
</ul>

<h3>5.8 Metriche e cosa dicono davvero</h3>
<p><strong>MSE / RMSE</strong>: unità di $y$ (o quadrate). Facile da ottimizzare (differenziabile), sensibile agli outlier.</p>
<p><strong>MAE</strong>: robusta agli outlier. Non differenziabile in zero (attenzione con solver gradient-based).</p>
<p><strong>$R^2 = 1 - \\text{SSR}/\\text{SST}$</strong>: frazione di varianza spiegata. Attenzione:</p>
<ul>
<li>Non è correlazione al quadrato in generale (lo è solo se il modello ha intercetta ed è OLS).</li>
<li>Aumenta sempre aggiungendo feature (anche rumorose). Usa $R^2$ aggiustato: $1 - (1-R^2)\\frac{n-1}{n-d-1}$.</li>
<li>Può essere negativo se il modello è peggiore di una costante.</li>
</ul>
<p><strong>MAPE</strong>: errore percentuale medio $\\frac{1}{n}\\sum |e_i/y_i|$. Interpretabile ma esplode se $y_i \\approx 0$.</p>

<h3>5.9 Ridge, Lasso, Elastic Net</h3>
<p><strong>Ridge (L2)</strong>: minimizza $\\|Xw - y\\|^2 + \\lambda \\|w\\|_2^2$. Soluzione:</p>
<p>$$\\hat w_{\\text{Ridge}} = (X^T X + \\lambda I)^{-1} X^T y$$</p>
<p>$\\lambda > 0$ garantisce invertibilità anche con multicollinearità. Riduce tutti i coefficienti verso zero ma non li azzera. Bias aumenta, varianza cala.</p>
<p><strong>Lasso (L1)</strong>: minimizza $\\|Xw - y\\|^2 + \\lambda \\|w\\|_1$. Nessuna forma chiusa. La forma a rombo della palla $\\ell_1$ produce soluzioni <em>sparse</em>: alcuni coefficienti diventano esattamente zero &rarr; selezione automatica delle feature.</p>
<p><strong>Elastic Net</strong>: combina L1 + L2. Utile quando hai gruppi di feature correlate: L1 sceglie tra loro, L2 stabilizza.</p>

<h3>5.10 Come scegliere $\\lambda$</h3>
<p>Con cross-validation. Scikit-learn ha classi <code class="inline">RidgeCV</code> e <code class="inline">LassoCV</code> che fanno tutto in automatico. In pratica prova una griglia logaritmica ($\\lambda \\in [10^{-4}, 10^4]$) e prendi quello con minor MSE su CV.</p>

<h3>5.11 Quando NON usare regressione lineare</h3>
<p>Anti-pattern che vedrai spesso:</p>
<ul>
<li><strong>Relazione fortemente non lineare</strong> senza feature engineering. Fitting una retta a $y = \\sin(x)$ è inutile.</li>
<li><strong>Feature categoriche non encodate</strong>. "Rosso, Verde, Blu" come 1, 2, 3 impone un ordinamento fittizio.</li>
<li><strong>Outlier estremi</strong>. OLS è dominato dal quadrato; un outlier può stravolgere il fit. Usa Huber loss o rimuovi outlier con criterio.</li>
<li><strong>$n \\ll d$</strong>. Con più feature che esempi serve regolarizzazione forte o riduzione dimensionale prima.</li>
<li><strong>Variabile target vincolata</strong> (probabilità, conteggi, ecc.). Usa logistic (per probabilità) o Poisson regression (per conteggi).</li>
<li><strong>Rumore non gaussiano con code pesanti</strong>. Considera regressioni robuste o quantile regression.</li>
</ul>
` },
    { type: 'callout', variant: 'note', title: 'Analogia con la fisica', content: 'Il regolarizzatore L2 è formalmente identico all\'aggiunta di una massa/energia potenziale nella funzione di costo: penalizza deviazioni grandi dall\'origine. Il Lasso è come un potenziale con "spigoli" nell\'origine, che produce soluzioni ferme sull\'asse (sparsità).' },
  ],
  esempi: [
    { type: 'md', content: '<h3>Esempio 1: OLS da zero e confronto sklearn</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(0)
n, d = 200, 4
X = np.random.randn(n, d)
w_true = np.array([1.5, -2.0, 0.5, 3.0])
b_true = 1.2
y = X @ w_true + b_true + 0.3 * np.random.randn(n)

# Da zero
X_aug = np.hstack([X, np.ones((n, 1))])  # bias in ultima col
w_hat = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y)
print("A mano  :", w_hat.round(4))

# Sklearn
sk = LinearRegression().fit(X, y)
print("Sklearn :", np.append(sk.coef_, sk.intercept_).round(4))
print("Vero    :", np.append(w_true, b_true))` },
    { type: 'md', content: '<h3>Esempio 2: statistica inferenziale sui coefficienti</h3>' },
    { type: 'run', content: `import numpy as np
from scipy.stats import t

np.random.seed(0)
n, d = 100, 3
X = np.random.randn(n, d)
X_aug = np.hstack([X, np.ones((n, 1))])
y = X @ np.array([2.0, 0.0, -1.5]) + 0.5 + np.random.randn(n) * 0.5

# fit
w_hat = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y)
y_pred = X_aug @ w_hat
resid = y - y_pred

# stima di sigma^2 (varianza dei residui, con correzione dei gradi di libertà)
sigma2 = (resid @ resid) / (n - d - 1)
cov_w = sigma2 * np.linalg.inv(X_aug.T @ X_aug)
se = np.sqrt(np.diag(cov_w))
t_stat = w_hat / se
p_val = 2 * (1 - t.cdf(np.abs(t_stat), df=n - d - 1))

names = ['x1', 'x2', 'x3', 'intercept']
print(f"{'feature':>10} {'coef':>8} {'SE':>8} {'t':>8} {'p':>10}")
for i, name in enumerate(names):
    print(f"{name:>10} {w_hat[i]:>8.4f} {se[i]:>8.4f} {t_stat[i]:>8.3f} {p_val[i]:>10.4g}")
print("\\nx2 ha coefficiente vero 0: dovrebbe uscire p-value alto, non significativo")` },
    { type: 'md', content: '<h3>Esempio 3: diagnostica dei residui</h3>' },
    { type: 'run', content: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(0)
x = np.linspace(-3, 3, 200)
# Rumore eteroschedastico: cresce con |x|
y = 2 * x + 1 + np.random.randn(200) * (0.5 + 0.6 * np.abs(x))

X = x.reshape(-1, 1)
m = LinearRegression().fit(X, y)
resid = y - m.predict(X)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# Residuals vs Fitted
axes[0].scatter(m.predict(X), resid, alpha=0.5)
axes[0].axhline(0, color='r')
axes[0].set_title('Residuals vs Fitted (eteroschedasticità evidente)')
axes[0].set_xlabel('Fitted'); axes[0].set_ylabel('Residual')

# Q-Q plot manuale
sorted_resid = np.sort(resid)
theoretical_q = np.random.randn(1000)
theoretical_q.sort()
axes[1].scatter(np.quantile(theoretical_q, np.linspace(0.01, 0.99, len(sorted_resid))),
                sorted_resid, alpha=0.5)
axes[1].plot([-3, 3], [-3 * resid.std(), 3 * resid.std()], 'r--')
axes[1].set_title('Q-Q plot (normalità residui)')

# Histogram
axes[2].hist(resid, bins=30, edgecolor='k', alpha=0.7)
axes[2].set_title(f'Distribuzione residui\\nmedia={resid.mean():.3f}, std={resid.std():.3f}')

plt.tight_layout()
plt.show()` },
    { type: 'md', content: '<h3>Esempio 4: multicollinearità in azione</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

np.random.seed(0)
n = 100
x1 = np.random.randn(n)
x2 = x1 + 0.001 * np.random.randn(n)   # quasi identica a x1
x3 = np.random.randn(n)
X = np.column_stack([x1, x2, x3])
y = 2 * x1 + 1.5 * x3 + 0.2 * np.random.randn(n)

# OLS: coefficienti impazziscono
print("Condizione X^T X:", np.linalg.cond(X.T @ X))
print("OLS   coefs:", LinearRegression().fit(X, y).coef_.round(3))
print("Ridge coefs (alpha=1):", Ridge(alpha=1.0).fit(X, y).coef_.round(3))
print()
print("OLS assegna pesi enormi con segni opposti a x1 e x2 per compensarsi.")
print("Ridge distribuisce il peso equamente tra x1 e x2.")` },
    { type: 'md', content: '<h3>Esempio 5: regolarizzazione path (Lasso)</h3>' },
    { type: 'run', content: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True)
X = StandardScaler().fit_transform(X)

alphas = np.logspace(-3, 1, 30)
coefs = []
for a in alphas:
    m = Lasso(alpha=a, max_iter=10000).fit(X, y)
    coefs.append(m.coef_)
coefs = np.array(coefs)

fig, ax = plt.subplots(figsize=(9, 5))
for i in range(X.shape[1]):
    ax.plot(alphas, coefs[:, i], marker='o', markersize=3, label=f'x{i+1}')
ax.set_xscale('log')
ax.set_xlabel('alpha (regolarizzazione)')
ax.set_ylabel('coefficiente')
ax.set_title('Lasso path: coefficienti si azzerano gradualmente all\\'aumentare di alpha')
ax.legend(fontsize=8, ncol=2, loc='upper right')
ax.axhline(0, color='k', linewidth=0.5)
plt.tight_layout()
plt.show()` },
    { type: 'md', content: '<h3>Esempio 6: CV per scegliere alpha</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

X, y = load_diabetes(return_X_y=True)

# LassoCV
pipe_lasso = Pipeline([('s', StandardScaler()),
                       ('m', LassoCV(alphas=np.logspace(-3, 1, 20), cv=5, max_iter=10000))]).fit(X, y)
alpha_lasso = pipe_lasso.named_steps['m'].alpha_
nonzero = int((pipe_lasso.named_steps['m'].coef_ != 0).sum())
print(f"LassoCV alpha ottimo: {alpha_lasso:.4f}, feature non-zero: {nonzero}/{X.shape[1]}")

# RidgeCV
pipe_ridge = Pipeline([('s', StandardScaler()),
                       ('m', RidgeCV(alphas=np.logspace(-3, 3, 30)))]).fit(X, y)
print(f"RidgeCV alpha ottimo: {pipe_ridge.named_steps['m'].alpha_:.4f}")

# Confronto CV score
for name, pipe in [('OLS baseline', Pipeline([('s', StandardScaler()),
                                              ('m', LassoCV(alphas=[1e-8], cv=5))])),
                    ('Ridge', pipe_ridge),
                    ('Lasso', pipe_lasso)]:
    scores = cross_val_score(pipe, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"{name:15s}  CV MSE: {-scores.mean():.2f} +/- {scores.std():.2f}")` },
    { type: 'md', content: '<h3>Esempio 7: feature polinomiali e overfitting</h3>' },
    { type: 'run', content: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(0)
x_true = np.linspace(0, 1, 100)
y_true = np.sin(2 * np.pi * x_true)
x_tr = np.random.uniform(0, 1, 15)
y_tr = np.sin(2 * np.pi * x_tr) + 0.15 * np.random.randn(15)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, deg in zip(axes, [1, 4, 15]):
    m = make_pipeline(PolynomialFeatures(deg), LinearRegression()).fit(x_tr.reshape(-1,1), y_tr)
    y_pred_true = m.predict(x_true.reshape(-1, 1))
    ax.plot(x_true, y_true, 'g-', label='verità', linewidth=2)
    ax.plot(x_true, y_pred_true, 'b-', label=f'fit grado {deg}')
    ax.scatter(x_tr, y_tr, c='red', s=40, edgecolor='k', zorder=5, label='training')
    ax.set_ylim(-2, 2)
    ax.set_title(f'Grado {deg}')
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
print("Grado 1: underfit. Grado 4: quasi perfetto. Grado 15: overfitting su 15 punti.")` },
  ],
  esercizi: [
    { type: 'exercise', difficulty: 'easy', title: 'Es. 5.1 — OLS a mano',
      prompt: '<p>Implementa da zero <code class="inline">ols_fit(X, y)</code> che ritorna la tupla <code class="inline">(w, b)</code> senza usare sklearn né np.linalg.lstsq. Aggiungi tu la colonna bias.</p>',
      starter: `import numpy as np
def ols_fit(X, y):
    # TODO
    return None, None

X = np.random.randn(50, 3)
y = X @ np.array([1.0, -2.0, 0.5]) + 1.5 + 0.1*np.random.randn(50)
w, b = ols_fit(X, y)
print("w:", w, "b:", b)`,
      check: `import numpy as np
from sklearn.linear_model import LinearRegression
sk = LinearRegression().fit(X, y)
if w is not None and np.allclose(w, sk.coef_, atol=1e-6) and abs(b - sk.intercept_) < 1e-6:
    print("OK — identico a sklearn")
else:
    print(f"FAIL — sklearn w={sk.coef_}, b={sk.intercept_}")`,
      solution: `def ols_fit(X, y):
    n = len(X)
    Xa = np.hstack([X, np.ones((n, 1))])
    theta = np.linalg.solve(Xa.T @ Xa, Xa.T @ y)
    return theta[:-1], theta[-1]` },

    { type: 'exercise', difficulty: 'easy', title: 'Es. 5.2 — Diagnostica R²',
      prompt: '<p>Data <code class="inline">y_true</code> e <code class="inline">y_pred</code>, calcola R² a mano usando la formula $R^2 = 1 - SS_{res}/SS_{tot}$. Salvalo in <code class="inline">r2</code>.</p>',
      starter: `import numpy as np
y_true = np.array([3.0, 5.0, 7.0, 9.0, 11.0])
y_pred = np.array([3.2, 4.9, 6.8, 9.1, 11.0])
r2 = None
print(r2)`,
      check: `import numpy as np
from sklearn.metrics import r2_score
if r2 is not None and abs(r2 - r2_score(y_true, y_pred)) < 1e-9:
    print(f"OK — R2 = {r2:.6f}")
else:
    print(f"FAIL — atteso {r2_score(y_true, y_pred):.6f}")`,
      solution: `ss_res = np.sum((y_true - y_pred)**2)
ss_tot = np.sum((y_true - y_true.mean())**2)
r2 = 1 - ss_res / ss_tot` },

    { type: 'exercise', difficulty: 'easy', title: 'Es. 5.3 — R² aggiustato',
      prompt: '<p>Implementa R² aggiustato: $R^2_{adj} = 1 - (1-R^2)\\frac{n-1}{n-d-1}$. Salvalo in <code class="inline">r2_adj</code>.</p>',
      starter: `n = 50
d = 5
r2 = 0.85
r2_adj = None
print(r2_adj)`,
      check: `_atteso = 1 - (1 - r2) * (n - 1) / (n - d - 1)
if r2_adj is not None and abs(r2_adj - _atteso) < 1e-9:
    print(f"OK — R2_adj = {r2_adj:.6f}")
else:
    print(f"FAIL — atteso {_atteso:.6f}")`,
      solution: `r2_adj = 1 - (1 - r2) * (n - 1) / (n - d - 1)` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 5.4 — Standard errors',
      prompt: '<p>Data la fit di regressione lineare e i residui, calcola gli standard error dei coefficienti (bias incluso). Salvali in <code class="inline">se</code> come array.</p>',
      starter: `import numpy as np
np.random.seed(0)
n, d = 100, 3
X = np.random.randn(n, d)
y = X @ np.array([1.0, 2.0, -0.5]) + 0.5 + np.random.randn(n) * 0.3
X_aug = np.hstack([X, np.ones((n, 1))])
w_hat = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y)
resid = y - X_aug @ w_hat

# TODO: calcola SE per ogni coefficiente
se = None
print(se)`,
      check: `import numpy as np
_sigma2 = (resid @ resid) / (n - d - 1)
_se = np.sqrt(_sigma2 * np.diag(np.linalg.inv(X_aug.T @ X_aug)))
if se is not None and np.allclose(se, _se, atol=1e-6):
    print("OK — SE:", se.round(5))
else:
    print(f"FAIL — atteso {_se}")`,
      solution: `sigma2 = (resid @ resid) / (n - d - 1)
se = np.sqrt(sigma2 * np.diag(np.linalg.inv(X_aug.T @ X_aug)))` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 5.5 — Ridge da zero',
      prompt: '<p>Implementa <code class="inline">ridge_fit(X, y, lam)</code> che risolve $(X^T X + \\lambda I)^{-1} X^T y$ (senza bias, per semplicità). Confronta con sklearn Ridge.</p>',
      starter: `import numpy as np
from sklearn.linear_model import Ridge

def ridge_fit(X, y, lam):
    # TODO
    return None

np.random.seed(0)
X = np.random.randn(50, 4)
y = X @ np.array([1, -1, 2, 0.5]) + np.random.randn(50) * 0.2
lam = 3.0
w = ridge_fit(X, y, lam)
print("Mio    :", w)
print("Sklearn:", Ridge(alpha=lam, fit_intercept=False).fit(X, y).coef_)`,
      check: `import numpy as np
sk = Ridge(alpha=lam, fit_intercept=False).fit(X, y).coef_
if w is not None and np.allclose(w, sk, atol=1e-6):
    print("OK")
else:
    print("FAIL")`,
      solution: `def ridge_fit(X, y, lam):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 5.6 — Lasso path manuale',
      prompt: '<p>Su dataset diabetes standardizzato, per ogni $\\lambda \\in [0.01, 0.1, 1, 10]$ addestra un Lasso e conta quante feature NON sono zero. Salva un dizionario <code class="inline">nonzero_per_alpha</code>.</p>',
      starter: `import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True)
X = StandardScaler().fit_transform(X)

nonzero_per_alpha = {}
# TODO
print(nonzero_per_alpha)`,
      check: `_atteso = {}
for a in [0.01, 0.1, 1, 10]:
    _atteso[a] = int((Lasso(alpha=a, max_iter=10000).fit(X, y).coef_ != 0).sum())
if nonzero_per_alpha == _atteso:
    print(f"OK — {nonzero_per_alpha}")
else:
    print(f"FAIL — atteso {_atteso}, tuo {nonzero_per_alpha}")`,
      solution: `for a in [0.01, 0.1, 1, 10]:
    m = Lasso(alpha=a, max_iter=10000).fit(X, y)
    nonzero_per_alpha[a] = int((m.coef_ != 0).sum())` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 5.7 — Robust regression Huber',
      prompt: '<p>Su dati con outlier, confronta LinearRegression vs HuberRegressor. Calcola la differenza (assoluta) di RMSE. Salva in <code class="inline">delta_rmse</code>.</p>',
      starter: `import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor

np.random.seed(0)
n = 100
X = np.random.randn(n, 1)
y = 2 * X.ravel() + 1 + 0.3 * np.random.randn(n)
# Aggiungo 5 outlier estremi
y[::20] += 20

delta_rmse = None
# TODO: fitta entrambi, calcola RMSE su X, y, e la differenza`,
      check: `import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor
m1 = LinearRegression().fit(X, y)
m2 = HuberRegressor().fit(X, y.ravel())
r1 = np.sqrt(((m1.predict(X) - y)**2).mean())
r2 = np.sqrt(((m2.predict(X) - y)**2).mean())
if delta_rmse is not None and abs(delta_rmse - abs(r1 - r2)) < 1e-3:
    print(f"OK — LR RMSE {r1:.3f}, Huber RMSE {r2:.3f}, delta {delta_rmse:.3f}")
else:
    print(f"FAIL — atteso {abs(r1-r2):.3f}")`,
      solution: `m1 = LinearRegression().fit(X, y)
m2 = HuberRegressor().fit(X, y.ravel())
r1 = np.sqrt(((m1.predict(X) - y)**2).mean())
r2 = np.sqrt(((m2.predict(X) - y)**2).mean())
delta_rmse = abs(r1 - r2)` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 5.8 — Interazione tra feature',
      prompt: '<p>Data <code class="inline">y = 2*x1 + 3*x2 - 1.5*x1*x2 + noise</code>, dimostra che una regressione lineare senza interazione ha $R^2$ nettamente peggiore di una con interazione. Salva i due $R^2$ in <code class="inline">r2_no_int, r2_int</code>.</p>',
      starter: `import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(0)
n = 300
x1 = np.random.randn(n)
x2 = np.random.randn(n)
y = 2*x1 + 3*x2 - 1.5*x1*x2 + 0.1*np.random.randn(n)

X_no_int = np.column_stack([x1, x2])
X_int = np.column_stack([x1, x2, x1*x2])

r2_no_int = None
r2_int = None
print(r2_no_int, r2_int)`,
      check: `import numpy as np
if r2_no_int is not None and r2_int is not None:
    if r2_int > r2_no_int + 0.3:
        print(f"OK — R² senza {r2_no_int:.3f} vs con {r2_int:.3f}")
    else:
        print(f"FAIL — R² sono troppo vicini {r2_no_int:.3f} vs {r2_int:.3f}")`,
      solution: `r2_no_int = LinearRegression().fit(X_no_int, y).score(X_no_int, y)
r2_int = LinearRegression().fit(X_int, y).score(X_int, y)` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 5.9 — Distanza di Cook',
      prompt: '<p>Implementa la distanza di Cook per identificare punti influenti. La formula: $D_i = \\frac{e_i^2}{d \\cdot \\hat\\sigma^2} \\frac{h_{ii}}{(1-h_{ii})^2}$ dove $h_{ii}$ è il leverage. Trova l\'indice del punto con la distanza massima. Salvalo in <code class="inline">worst_idx</code>.</p>',
      starter: `import numpy as np
np.random.seed(0)
X = np.random.randn(50, 2)
y = X @ np.array([1.5, -2.0]) + 0.5 + 0.2 * np.random.randn(50)
# Inserisco un outlier grave in posizione 20
y[20] += 15

X_aug = np.hstack([X, np.ones((50, 1))])
w = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y)
resid = y - X_aug @ w
sigma2 = (resid @ resid) / (50 - 3)
H = X_aug @ np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T
h = np.diag(H)

# TODO: calcola distanze di Cook
worst_idx = None
print("Punto piu influente:", worst_idx)`,
      check: `import numpy as np
_cook = (resid**2 / (3 * sigma2)) * h / (1 - h)**2
_atteso = int(np.argmax(_cook))
if worst_idx == _atteso:
    print(f"OK — punto {worst_idx} (Cook = {_cook[worst_idx]:.3f})")
else:
    print(f"FAIL — atteso {_atteso}")`,
      solution: `cook = (resid**2 / (3 * sigma2)) * h / (1 - h)**2
worst_idx = int(np.argmax(cook))` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 5.10 — Grid search Elastic Net',
      prompt: '<p>Su diabetes, fai grid search di ElasticNet su $\\alpha \\in [0.01, 0.1, 1]$ e $l1\\_ratio \\in [0.1, 0.5, 0.9]$ con 5-fold CV. Salva $R^2$ test in <code class="inline">r2_best</code>.</p>',
      starter: `import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split

X, y = load_diabetes(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.25, random_state=42)

r2_best = None`,
      check: `if r2_best is None: print("FAIL: None")
elif r2_best > 0.4: print(f"OK — R² {r2_best:.4f}")
else: print(f"FAIL — R² {r2_best:.4f}")`,
      solution: `pipe = Pipeline([('s', StandardScaler()), ('m', ElasticNet(max_iter=10000))])
grid = {'m__alpha':[0.01, 0.1, 1], 'm__l1_ratio':[0.1, 0.5, 0.9]}
gs = GridSearchCV(pipe, grid, cv=5).fit(Xt, yt)
r2_best = gs.score(Xe, ye)` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 5.11 — Feature engineering intelligente',
      prompt: '<p>Sul dataset <code class="inline">y = 3 sin(x1) + 2 log(1 + x2^2) + noise</code>, crea le feature giuste per fare $R^2 > 0.95$ con LinearRegression. Salva l\'array delle predizioni test in <code class="inline">y_pred</code>.</p>',
      starter: `import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(0)
n = 500
x1 = np.random.uniform(-3, 3, n)
x2 = np.random.uniform(-2, 2, n)
y = 3*np.sin(x1) + 2*np.log(1 + x2**2) + 0.2*np.random.randn(n)

X_raw = np.column_stack([x1, x2])
Xt, Xe, yt, ye = train_test_split(X_raw, y, test_size=0.3, random_state=0)

# TODO: crea feature trasformate, addestra, predici
y_pred = None`,
      check: `import numpy as np
if y_pred is not None:
    r2 = 1 - np.sum((y_pred - ye)**2) / np.sum((ye - ye.mean())**2)
    if r2 > 0.95:
        print(f"OK — R² {r2:.4f}")
    else:
        print(f"FAIL — R² {r2:.4f}")`,
      solution: `def features(X):
    return np.column_stack([np.sin(X[:,0]), np.log(1 + X[:,1]**2)])
Xt_f = features(Xt); Xe_f = features(Xe)
m = LinearRegression().fit(Xt_f, yt)
y_pred = m.predict(Xe_f)` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 5.12 — Weighted Least Squares',
      prompt: '<p>Con rumore eteroschedastico (varianza crescente con x), OLS è subottimale. Implementa WLS con pesi $w_i = 1/\\sigma_i^2$ dove $\\sigma_i = 1 + |x_i|$. Salva il coefficiente stimato in <code class="inline">w_wls</code>.</p>',
      starter: `import numpy as np
np.random.seed(0)
n = 200
x = np.random.uniform(-3, 3, n)
sigma = 1 + np.abs(x)
y = 2.5 * x + 1 + np.random.randn(n) * sigma  # eterosch. crescente

# TODO: WLS. Hint: WLS = OLS su (sqrt(w)*X, sqrt(w)*y)
w_wls = None
print("Coeff stimato:", w_wls, "(vero 2.5)")`,
      check: `if w_wls is not None and abs(w_wls - 2.5) < 0.15:
    print(f"OK — WLS {w_wls:.3f} vs vero 2.5")
else:
    print(f"FAIL — {w_wls}")`,
      solution: `X = x.reshape(-1, 1)
X_aug = np.hstack([X, np.ones((n, 1))])
weights = 1 / sigma**2
Xw = X_aug * np.sqrt(weights).reshape(-1, 1)
yw = y * np.sqrt(weights)
theta = np.linalg.solve(Xw.T @ Xw, Xw.T @ yw)
w_wls = theta[0]` },
  ],
  test: [
    { type: 'quiz', questions: [
      { q: 'Le normal equations $X^T X w = X^T y$ hanno soluzione unica se:', options: ['$n > d$ sempre', '$X$ ha colonne linearmente indipendenti', '$y$ è nel range di $X$', 'Sempre'], answer: 1, explain: 'Serve rango pieno di X, altrimenti $X^T X$ è singolare.' },
      { q: 'Il teorema di Gauss-Markov dice che OLS è:', options: ['Il miglior stimatore possibile', 'Il miglior stimatore lineare unbiased sotto le ipotesi classiche', 'Sempre biased', 'Non-parametrico'], answer: 1, explain: 'BLUE = Best Linear Unbiased Estimator. Modelli biased possono batterlo (es. Ridge).' },
      { q: 'R² può essere negativo?', options: ['No, mai', 'Sì, se il modello è peggiore della media', 'Solo con overfitting', 'Solo con Ridge'], answer: 1, explain: 'R² < 0 significa peggiore di una costante.' },
      { q: 'Ridge (L2) rispetto a OLS su feature molto correlate:', options: ['Peggiora', 'Stabilizza i coefficienti riducendone la varianza', 'Rimuove tutte le feature', 'Non ha effetto'], answer: 1, explain: 'Ridge aggiunge $\\lambda I$ che stabilizza numericamente.' },
      { q: 'Un p-value di 0.03 per un coefficiente significa:', options: ['Il coefficiente è 0.03', 'Sotto $H_0: w=0$, la probabilità di osservare un $|t|$ così grande è 3%', 'Il modello ha 97% accuracy', 'Il coefficiente vale zero'], answer: 1, explain: 'Interpretazione classica: se il vero coefficiente fosse zero, sarebbe raro (3%) osservare questi dati.' },
      { q: 'Perché standardizzare prima di Ridge/Lasso è importante?', options: ['Per velocità', 'Perché la penalità è la stessa per ogni coefficiente, quindi feature con scala grande vengono penalizzate meno relativamente', 'Non è importante', 'Solo per Lasso'], answer: 1, explain: 'Senza standardizzazione la penalità è "ingiusta".' },
    ] },
    { type: 'exercise', difficulty: 'hard', title: 'Test finale — modeling professionale',
      prompt: '<p>Sul dataset diabetes, produci una pipeline che:</p><ol><li>Standardizza le feature</li><li>Aggiunge feature polinomiali fino a grado 2</li><li>Applica ElasticNet con CV per scegliere alpha e l1_ratio</li><li>Riporta R² test</li></ol><p>Target: R² test > 0.42. Salvalo in <code class="inline">r2_test</code>.</p>',
      starter: `import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.25, random_state=42)

r2_test = None`,
      check: `if r2_test is None: print("FAIL: None")
elif r2_test > 0.42: print(f"OK — R² {r2_test:.4f}")
else: print(f"FAIL — R² {r2_test:.4f}")`,
      solution: `pipe = Pipeline([('s', StandardScaler()),
                 ('p', PolynomialFeatures(degree=2, interaction_only=False)),
                 ('m', ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-3, 1, 20), cv=5, max_iter=10000))]).fit(Xt, yt)
r2_test = pipe.score(Xe, ye)` },

    { type: 'exercise', difficulty: 'hard', title: 'Bonus expert — sfida Kaggle-level',
      prompt: '<p><strong>Simulazione di un problema reale.</strong> Il target ha un rumore eteroschedastico, presenza di outlier, e alcune feature sono altamente correlate. Devi produrre un modello che:</p><ol><li>Robusto agli outlier</li><li>Con regolarizzazione appropriata</li><li>Con feature engineering per la non-linearità</li></ol><p>Target: <strong>RMSE test &lt; 3.0</strong>. Salvalo in <code class="inline">rmse</code>.</p>',
      starter: `import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import HuberRegressor, ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

np.random.seed(42)
n = 800
x1 = np.random.randn(n)
x2 = x1 + 0.05 * np.random.randn(n)  # correlata con x1
x3 = np.random.randn(n)
x4 = np.random.randn(n)
x5 = np.random.uniform(-2, 2, n)

# Target non-lineare + rumore eteroschedastico + outlier
noise = np.random.randn(n) * (1 + 0.5*np.abs(x3))
y = 2*x1 + 3*np.sin(x5) - 0.5*x4**2 + noise
y[::40] += 15  # outlier occasionali

X = np.column_stack([x1, x2, x3, x4, x5])
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.25, random_state=0)

# TODO: costruisci pipeline robusta
rmse = None`,
      check: `import numpy as np
if rmse is None: print("FAIL: None")
elif rmse < 3.0: print(f"OK — RMSE {rmse:.4f}")
else: print(f"FAIL — RMSE {rmse:.4f}, target < 3.0")`,
      solution: `# Feature engineering: aggiungi non-linearità visibili + robusto
def eng(X):
    return np.column_stack([X, np.sin(X[:, 4:5]), X[:, 3:4]**2])
Xt_e = eng(Xt); Xe_e = eng(Xe)
pipe = Pipeline([('s', StandardScaler()), ('m', HuberRegressor(alpha=0.01))]).fit(Xt_e, yt)
rmse = np.sqrt(np.mean((pipe.predict(Xe_e) - ye)**2))` },
  ]
});
