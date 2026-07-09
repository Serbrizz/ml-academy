/* Lezione: Gradient Boosting */
window.LESSONS.push({
  id: 'l15-gradient-boosting',
  title: 'Gradient Boosting: il re dei dati tabulari',
  tag: 'Supervisionato avanzato',
  summary: 'XGBoost, LightGBM, CatBoost. Il modello che vince l\'80% delle Kaggle su dati tabulari. Algoritmo, tuning serio, feature importance.',
  teoria: [
    { type: 'md', content: `
<h3>15.1 Perché è importante</h3>
<p>Sui dati tabulari (righe = esempi, colonne = feature), il <strong>gradient boosting</strong> è oggi il modello dominante. XGBoost, LightGBM e CatBoost vincono la maggior parte delle competizioni Kaggle e in industria sono lo standard per problemi come credit scoring, churn prediction, click-through rate. Anche con l'avvento del deep learning, sui tabulari il boosting resta imbattuto.</p>
<p>Se devi imparare <em>un solo</em> algoritmo avanzato dopo la regressione lineare, imparaci questo. Vale il 30-40% del valore pratico del tuo bagaglio ML.</p>

<h3>15.2 L'idea: apprendere dagli errori</h3>
<p>Boosting significa "potenziamento". Costruiamo un ensemble di modelli deboli (tipicamente alberi shallow, "stump" o profondità 3-8), ognuno che <strong>corregge gli errori</strong> del precedente. Non in parallelo come Random Forest, ma <strong>in sequenza</strong>.</p>
<p>Formalmente, l'ensemble finale è:</p>
<p>$$F_M(x) = \\sum_{m=1}^M \\eta \\cdot f_m(x)$$</p>
<p>dove $\\eta$ è il <em>learning rate</em> (piccolo, es. 0.05-0.1) e $f_m$ è l'$m$-esimo albero.</p>

<h3>15.3 Il gradient</h3>
<p>Ecco l'insight geniale di Friedman (2001): l'$m$-esimo albero non è addestrato per predire $y$, ma il <strong>gradiente negativo della loss</strong> rispetto alle predizioni correnti.</p>
<p>Per la loss quadratica $\\mathcal{L}(y, F) = \\frac{1}{2}(y - F)^2$:</p>
<p>$$-\\frac{\\partial \\mathcal{L}}{\\partial F} = y - F$$</p>
<p>Che è esattamente il <em>residuo</em>. Quindi in regressione l'$m$-esimo albero impara i residui del precedente. In classificazione con log-loss il gradiente è più complesso ma il principio è identico.</p>
<p>Perché "gradiente"? Perché stiamo facendo <em>gradient descent nello spazio delle funzioni</em>: ogni albero aggiunto è un passo che sposta $F$ nella direzione che riduce la loss.</p>

<h3>15.4 Algoritmo passo-passo (regressione)</h3>
<ol>
<li>Inizializza $F_0(x) = \\bar y$ (media del target).</li>
<li>Per $m = 1, \\dots, M$:
<ol type="a">
<li>Calcola i residui $r_i = y_i - F_{m-1}(x_i)$.</li>
<li>Fitta un albero $f_m$ sui residui: $f_m(x) \\approx r$.</li>
<li>Aggiorna $F_m(x) = F_{m-1}(x) + \\eta f_m(x)$.</li>
</ol>
</li>
<li>Ritorna $F_M$.</li>
</ol>

<h3>15.5 XGBoost, LightGBM, CatBoost — cosa cambia</h3>
<p>Tutti implementano l'idea di base, ma con diverse ottimizzazioni:</p>
<ul>
<li><strong>XGBoost</strong>: regolarizzazione L1/L2 nell'albero, second-order approximation della loss (usa gradiente + hessiana), splitting parallelo. Ha vinto tantissime Kaggle nel 2015-2018.</li>
<li><strong>LightGBM</strong>: usa <em>histogram-based splitting</em> (aggrega feature in bin), <em>leaf-wise growth</em> (invece di level-wise). Molto più veloce di XGBoost su dataset grandi. Default moderno per production.</li>
<li><strong>CatBoost</strong>: ottimo per feature categoriche, gestione automatica, riduce l'overfitting con <em>ordered boosting</em>. Meno tuning richiesto.</li>
</ul>
<p>In sklearn c'è <code class="inline">HistGradientBoostingClassifier/Regressor</code>, che è ispirato a LightGBM ed è già installato di default. In Pyodide useremo quello.</p>

<h3>15.6 Iperparametri chiave</h3>
<p>Il tuning è dove si gioca la partita. In ordine di importanza:</p>
<ul>
<li><strong>learning_rate ($\\eta$)</strong>: passo di ogni albero. Piccolo (0.01-0.1) = più stabile ma richiede più alberi. Il compromesso classico: $\\eta = 0.05$ con 500-2000 alberi.</li>
<li><strong>n_estimators / max_iter</strong>: numero di alberi. Con early stopping, mettine tanti (2000-5000) e lascia che si fermi.</li>
<li><strong>max_depth</strong>: profondità albero. Shallow (3-8) va bene per la maggior parte dei casi. Deep (>10) rischia overfitting.</li>
<li><strong>min_samples_leaf / min_child_weight</strong>: minimo campioni per foglia. Regolarizza contro l'overfitting.</li>
<li><strong>subsample</strong>: frazione di righe campionate per albero. Come bagging, aggiunge randomness e riduce varianza. 0.7-0.9 tipico.</li>
<li><strong>colsample_bytree</strong>: frazione di colonne campionate per albero. Come Random Forest. 0.7-1.0.</li>
<li><strong>reg_alpha (L1), reg_lambda (L2)</strong>: regolarizzazione dell'albero.</li>
</ul>

<h3>15.7 Early stopping</h3>
<p>La tecnica più importante di tutte. Monitori la loss di validazione a ogni iterazione e ti fermi quando smette di scendere per N iterazioni consecutive. Vantaggi:</p>
<ul>
<li>Non devi tunare precisamente <code class="inline">n_estimators</code>: mettine tantissimi.</li>
<li>Trovi automaticamente il "sweet spot" del compromesso bias-variance.</li>
<li>Grazie ad early stopping, i default di learning_rate e depth sono spesso già ottimi.</li>
</ul>

<h3>15.8 Feature importance</h3>
<p>Diversi tipi:</p>
<ul>
<li><strong>Gain</strong>: totale della riduzione di loss dovuta agli split su quella feature. Il più usato.</li>
<li><strong>Cover</strong>: numero di campioni influenzati.</li>
<li><strong>Frequency</strong>: quante volte la feature è usata negli split.</li>
</ul>
<p>Attenzione: sono <em>biased</em> verso feature con più valori distinti (soprattutto Gain e Frequency). Alternative migliori:</p>
<ul>
<li><strong>Permutation importance</strong>: mescoli una feature e misuri quanto peggiora la performance. Model-agnostic, meno biased.</li>
<li><strong>SHAP values</strong>: theoretically grounded (game theory), fornisce contributi per singola predizione. Il gold standard oggi.</li>
</ul>

<h3>15.9 Categoriche</h3>
<p>Nei tree-based si può gestire categoriche in vari modi:</p>
<ul>
<li><strong>One-hot</strong>: crea colonne binarie. Funziona ma esplode se hai molte categorie.</li>
<li><strong>Target encoding</strong>: sostituisci la categoria con la media di $y$ per quella categoria (calcolata in CV per evitare leakage).</li>
<li><strong>Native support</strong>: LightGBM e CatBoost gestiscono nativamente categoriche senza encoding.</li>
</ul>

<h3>15.10 Quando NON usare gradient boosting</h3>
<ul>
<li><strong>Immagini, testo, audio</strong>: deep learning è meglio. GBT è per dati tabulari.</li>
<li><strong>Dataset piccolissimo (n < 100)</strong>: modelli più semplici (lineari) generalizzano meglio.</li>
<li><strong>Latenza estrema (< 1ms)</strong>: ensemble di 1000 alberi non è veloce. Considera modelli lineari o distillazione.</li>
<li><strong>Interpretabilità totale richiesta</strong>: un albero singolo è più interpretabile, o un modello lineare regolarizzato.</li>
</ul>
` },
    { type: 'callout', variant: 'tip', title: 'Regola pratica', content: 'Per un nuovo problema tabulare, il baseline è: standardizzare, LogisticRegression (o Ridge per regressione). Poi HistGradientBoosting con default. Se il boosting non batte la lineare del 5%+, hai un problema (o è un problema facile). Poi tuning serio del boosting.' },
  ],
  esempi: [
    { type: 'md', content: '<h3>Esempio 1: GBM vs Random Forest vs modello lineare</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score

# Fallback a diabetes se California non caricabile in Pyodide
X, y = load_diabetes(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.25, random_state=42)

models = {
    'Ridge': Pipeline([('s', StandardScaler()), ('m', Ridge(alpha=1.0))]),
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=0),
    'GradientBoosting': HistGradientBoostingRegressor(max_iter=500, random_state=0),
}
for name, m in models.items():
    m.fit(Xt, yt)
    r2 = m.score(Xe, ye)
    rmse = np.sqrt(((m.predict(Xe) - ye)**2).mean())
    print(f"{name:20s} R2={r2:.4f}   RMSE={rmse:.2f}")` },
    { type: 'md', content: '<h3>Esempio 2: gradient boosting from scratch (regressione)</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.25, random_state=42)

# GBM from scratch
def gbm_regressor(X_train, y_train, X_test, n_estimators=200, learning_rate=0.05, max_depth=3):
    F_train = np.full(len(y_train), y_train.mean())  # initial constant
    F_test = np.full(len(X_test), y_train.mean())
    trees = []
    for m in range(n_estimators):
        residual = y_train - F_train
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=m).fit(X_train, residual)
        F_train += learning_rate * tree.predict(X_train)
        F_test += learning_rate * tree.predict(X_test)
        trees.append(tree)
    return F_test

pred = gbm_regressor(Xt, yt, Xe, n_estimators=300, learning_rate=0.05, max_depth=3)
r2 = 1 - ((pred - ye)**2).sum() / ((ye - ye.mean())**2).sum()
print(f"GBM from scratch R2 = {r2:.4f}")

from sklearn.ensemble import HistGradientBoostingRegressor
m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3, random_state=0).fit(Xt, yt)
print(f"HistGBR sklearn R2 = {m.score(Xe, ye):.4f}")` },
    { type: 'md', content: '<h3>Esempio 3: effetto del learning rate</h3>' },
    { type: 'run', content: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.25, random_state=42)

fig, ax = plt.subplots(figsize=(9, 5))
for lr in [0.01, 0.05, 0.1, 0.3]:
    m = HistGradientBoostingRegressor(learning_rate=lr, max_iter=500,
                                       early_stopping=False, random_state=0).fit(Xt, yt)
    losses = -m.validation_score_ if m.validation_score_ is not None else None
    # Se early_stopping è False non abbiamo validation_score; usiamo train_score come proxy
    tr_scores = m.train_score_
    ax.plot(tr_scores, label=f'lr={lr}')
ax.set_xlabel('Iterazione (albero)')
ax.set_ylabel('Loss training (negativa)')
ax.set_title('Effetto del learning rate: piccoli lr convergono più lentamente ma più stabili')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()` },
    { type: 'md', content: '<h3>Esempio 4: early stopping</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.25, random_state=42)

# Configuro tantissime iterazioni e lascio che early stopping decida
m = HistGradientBoostingRegressor(
    max_iter=2000,
    learning_rate=0.05,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=30,
    random_state=0,
).fit(Xt, yt)

print(f"Numero di iterazioni effettive: {m.n_iter_}")
print(f"Massimo consentito:              {m.max_iter}")
print(f"R2 test:                         {m.score(Xe, ye):.4f}")
print("\\nEarly stopping ha automaticamente trovato il numero ottimale di alberi.")` },
    { type: 'md', content: '<h3>Esempio 5: tuning avanzato con GridSearch</h3>' },
    { type: 'run', content: `from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV, train_test_split

X, y = load_diabetes(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.25, random_state=42)

param_grid = {
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 6, None],
    'min_samples_leaf': [10, 30],
}
m = HistGradientBoostingRegressor(max_iter=500, early_stopping=True, random_state=0)
gs = GridSearchCV(m, param_grid, cv=5, scoring='r2', n_jobs=1).fit(Xt, yt)

print(f"Migliori parametri: {gs.best_params_}")
print(f"CV R2:              {gs.best_score_:.4f}")
print(f"Test R2:            {gs.score(Xe, ye):.4f}")` },
    { type: 'md', content: '<h3>Esempio 6: feature importance vs permutation importance</h3>' },
    { type: 'run', content: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.datasets import load_diabetes

data = load_diabetes()
X, y = data.data, data.target

m = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X, y)
perm = permutation_importance(m, X, y, n_repeats=10, random_state=0)

fig, ax = plt.subplots(figsize=(9, 5))
order = np.argsort(perm.importances_mean)
ax.barh(range(len(order)), perm.importances_mean[order], xerr=perm.importances_std[order])
ax.set_yticks(range(len(order)))
ax.set_yticklabels([data.feature_names[i] for i in order])
ax.set_xlabel('Riduzione R² per permutazione')
ax.set_title('Permutation importance (con std)')
plt.tight_layout()
plt.show()` },
    { type: 'md', content: '<h3>Esempio 7: classificazione + gestione sbilanciamento</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

X, y = make_classification(n_samples=3000, n_features=10, weights=[0.92, 0.08],
                           n_informative=6, random_state=0)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Sample weights per gestire lo sbilanciamento
w = np.where(yt == 1, (yt == 0).mean() / (yt == 1).mean(), 1.0)

m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(Xt, yt, sample_weight=w)
prob = m.predict_proba(Xe)[:, 1]

print(f"ROC-AUC:       {roc_auc_score(ye, prob):.4f}")
print(f"PR-AUC:        {average_precision_score(ye, prob):.4f}")
print(classification_report(ye, m.predict(Xe), digits=3))` },
  ],
  esercizi: [
    { type: 'exercise', difficulty: 'easy', title: 'Es. 15.1 — GBM base',
      prompt: '<p>Su diabetes, fitta un HistGradientBoostingRegressor con default. Salva l\'R² test in <code class="inline">r2</code>.</p>',
      starter: `import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.25, random_state=42)
r2 = None`,
      check: `if r2 is not None and r2 > 0.35:
    print(f"OK — R² {r2:.4f}")
else:
    print(f"FAIL — {r2}")`,
      solution: `r2 = HistGradientBoostingRegressor(random_state=0).fit(Xt, yt).score(Xe, ye)` },

    { type: 'exercise', difficulty: 'easy', title: 'Es. 15.2 — Residui = gradiente',
      prompt: '<p>Verifica numericamente: per la loss $\\mathcal{L} = \\frac{1}{2}(y-F)^2$, il gradiente rispetto a $F$ è $-(y-F)$. Salva l\'array dei gradienti (con segno) in <code class="inline">grads</code> per $y = [3,5,2]$ e $F=[2,6,3]$.</p>',
      starter: `import numpy as np
y = np.array([3.0, 5.0, 2.0])
F = np.array([2.0, 6.0, 3.0])
grads = None  # gradiente della loss rispetto a F
print(grads)`,
      check: `import numpy as np
_atteso = -(y - F)
if grads is not None and np.allclose(grads, _atteso):
    print(f"OK — {grads}")
else:
    print(f"FAIL — atteso {_atteso}")`,
      solution: `grads = -(y - F)  # o equivalentemente F - y` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 15.3 — GBM manuale 3 alberi',
      prompt: '<p>Costruisci un GBM manualmente con 3 alberi profondi 2 su diabetes. Learning rate 0.1. Salva l\'R² test in <code class="inline">r2_manual</code>.</p>',
      starter: `import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.25, random_state=42)

# TODO
r2_manual = None`,
      check: `import numpy as np
F_tr = np.full(len(yt), yt.mean())
F_te = np.full(len(Xe), yt.mean())
for _ in range(3):
    resid = yt - F_tr
    t = DecisionTreeRegressor(max_depth=2, random_state=0).fit(Xt, resid)
    F_tr += 0.1 * t.predict(Xt)
    F_te += 0.1 * t.predict(Xe)
_r2 = 1 - ((F_te - ye)**2).sum() / ((ye - ye.mean())**2).sum()
if r2_manual is not None and abs(r2_manual - _r2) < 0.02:
    print(f"OK — R² {r2_manual:.4f}")
else:
    print(f"FAIL — atteso ~{_r2:.4f}, tuo {r2_manual}")`,
      solution: `F_tr = np.full(len(yt), yt.mean())
F_te = np.full(len(Xe), yt.mean())
for _ in range(3):
    resid = yt - F_tr
    t = DecisionTreeRegressor(max_depth=2, random_state=0).fit(Xt, resid)
    F_tr += 0.1 * t.predict(Xt)
    F_te += 0.1 * t.predict(Xe)
r2_manual = 1 - ((F_te - ye)**2).sum() / ((ye - ye.mean())**2).sum()` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 15.4 — Learning rate vs n_estimators',
      prompt: '<p>Su diabetes, dimostra che con learning_rate=0.01 servono più alberi di learning_rate=0.1 per stessa performance. Trova l\'n_estimators necessario per raggiungere R² &gt; 0.4 con lr=0.01. Salvalo in <code class="inline">n_needed</code>.</p>',
      starter: `import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.25, random_state=42)

n_needed = None`,
      check: `if n_needed is not None and n_needed > 100:
    m = HistGradientBoostingRegressor(learning_rate=0.01, max_iter=n_needed,
                                        early_stopping=False, random_state=0).fit(Xt, yt)
    r2 = m.score(Xe, ye)
    if r2 > 0.4:
        print(f"OK — con n={n_needed} raggiungi R²={r2:.4f}")
    else:
        print(f"FAIL — n={n_needed} dà R²={r2:.4f}")`,
      solution: `for n in [200, 500, 1000, 2000]:
    r2 = HistGradientBoostingRegressor(learning_rate=0.01, max_iter=n, early_stopping=False, random_state=0).fit(Xt, yt).score(Xe, ye)
    if r2 > 0.4:
        n_needed = n
        break` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 15.5 — Early stopping',
      prompt: '<p>Su diabetes, addestra un GBM con max_iter=2000, early_stopping=True, n_iter_no_change=20. Salva il numero di iterazioni EFFETTIVE in <code class="inline">n_iter</code>.</p>',
      starter: `from sklearn.datasets import load_diabetes
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.25, random_state=42)
n_iter = None`,
      check: `if n_iter is not None and n_iter > 20 and n_iter < 2000:
    print(f"OK — early stopping a {n_iter} iterazioni")
else:
    print(f"FAIL — n_iter {n_iter}")`,
      solution: `m = HistGradientBoostingRegressor(max_iter=2000, early_stopping=True,
                                    validation_fraction=0.15, n_iter_no_change=20,
                                    random_state=0).fit(Xt, yt)
n_iter = m.n_iter_` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 15.6 — Permutation importance',
      prompt: '<p>Su diabetes, dopo aver addestrato un GBM, trova le 3 feature più importanti secondo permutation importance. Salva la lista dei loro NOMI (in ordine da più a meno importante) in <code class="inline">top3</code>.</p>',
      starter: `import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

data = load_diabetes()
X, y = data.data, data.target

top3 = None`,
      check: `import numpy as np
m = HistGradientBoostingRegressor(random_state=0).fit(X, y)
_perm = permutation_importance(m, X, y, n_repeats=10, random_state=0)
_top = [data.feature_names[i] for i in np.argsort(_perm.importances_mean)[::-1][:3]]
if top3 == _top:
    print(f"OK — {top3}")
else:
    print(f"FAIL — atteso {_top}, tuo {top3}")`,
      solution: `m = HistGradientBoostingRegressor(random_state=0).fit(X, y)
perm = permutation_importance(m, X, y, n_repeats=10, random_state=0)
top3 = [data.feature_names[i] for i in np.argsort(perm.importances_mean)[::-1][:3]]` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 15.7 — GBM vs RF',
      prompt: '<p>Confronta HistGradientBoosting e RandomForest su diabetes. Ottieni R² test di entrambi in <code class="inline">r2_gbm, r2_rf</code>. Verifica che GBM batta RF (differenza > 0.02).</p>',
      starter: `from sklearn.datasets import load_diabetes
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.25, random_state=42)

r2_gbm = None
r2_rf = None`,
      check: `if r2_gbm and r2_rf:
    if r2_gbm > r2_rf:
        print(f"OK — GBM {r2_gbm:.4f} > RF {r2_rf:.4f}")
    else:
        print(f"WARN — RF ha battuto GBM ({r2_rf:.4f} vs {r2_gbm:.4f})")`,
      solution: `r2_gbm = HistGradientBoostingRegressor(random_state=0).fit(Xt, yt).score(Xe, ye)
r2_rf = RandomForestRegressor(n_estimators=200, random_state=0).fit(Xt, yt).score(Xe, ye)` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 15.8 — Classificazione sbilanciata',
      prompt: '<p>Su make_classification con weights=[0.95,0.05], addestra un GBM binario. Ottieni PR-AUC test in <code class="inline">pr_auc</code>. Usa sample_weight per bilanciare. Target &gt; 0.4.</p>',
      starter: `import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

X, y = make_classification(n_samples=2000, n_features=10, weights=[0.95, 0.05],
                           n_informative=5, random_state=0)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

pr_auc = None`,
      check: `if pr_auc is not None and pr_auc > 0.4:
    print(f"OK — PR-AUC {pr_auc:.4f}")
else:
    print(f"FAIL — PR-AUC {pr_auc}")`,
      solution: `import numpy as np
w = np.where(yt==1, (yt==0).mean()/(yt==1).mean(), 1.0)
m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(Xt, yt, sample_weight=w)
pr_auc = average_precision_score(ye, m.predict_proba(Xe)[:, 1])` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 15.9 — Curva di validazione',
      prompt: '<p>Su diabetes, varia max_depth in [2, 3, 4, 6, 10, None] e per ciascuno addestra un GBM (max_iter=300). Trova il max_depth con miglior R² CV a 5 fold. Salvalo in <code class="inline">best_depth</code>.</p>',
      starter: `import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score

X, y = load_diabetes(return_X_y=True)
best_depth = None`,
      check: `import numpy as np
_scores = {}
for d in [2, 3, 4, 6, 10, None]:
    m = HistGradientBoostingRegressor(max_iter=300, max_depth=d, random_state=0)
    _scores[d if d is not None else 'None'] = cross_val_score(m, X, y, cv=5, scoring='r2').mean()
_atteso = max(_scores, key=_scores.get)
if str(best_depth if best_depth is not None else 'None') == str(_atteso):
    print(f"OK — best depth = {best_depth}")
else:
    print(f"FAIL — atteso {_atteso}, tuo {best_depth}")`,
      solution: `scores = {}
for d in [2, 3, 4, 6, 10, None]:
    m = HistGradientBoostingRegressor(max_iter=300, max_depth=d, random_state=0)
    scores[d] = cross_val_score(m, X, y, cv=5, scoring='r2').mean()
best_depth = max(scores, key=scores.get)` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 15.10 — GridSearch aggressivo',
      prompt: '<p>Su diabetes, fai grid search estesa: <code class="inline">learning_rate ∈ {0.03, 0.05, 0.1}</code>, <code class="inline">max_depth ∈ {3, 6, None}</code>, <code class="inline">min_samples_leaf ∈ {10, 30}</code>. Salva la migliore combinazione e l\'R² test in <code class="inline">r2_test</code>. Target > 0.42.</p>',
      starter: `from sklearn.datasets import load_diabetes
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

X, y = load_diabetes(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.25, random_state=42)

r2_test = None`,
      check: `if r2_test is not None and r2_test > 0.42:
    print(f"OK — R² test {r2_test:.4f}")
else:
    print(f"FAIL — R² {r2_test}")`,
      solution: `grid = {'learning_rate':[0.03,0.05,0.1], 'max_depth':[3,6,None], 'min_samples_leaf':[10,30]}
gs = GridSearchCV(HistGradientBoostingRegressor(max_iter=500, random_state=0), grid, cv=5, scoring='r2').fit(Xt, yt)
r2_test = gs.score(Xe, ye)` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 15.11 — Interazione feature',
      prompt: '<p>Il GBM cattura naturalmente interazioni. Verificalo: crea $y = 3 x_1 + 2 x_2 - 4 x_1 x_2 + noise$ con $x_1, x_2, ..., x_{10}$ tutti random. Confronta R² test di Ridge vs GBM. Salva <code class="inline">r2_ridge, r2_gbm</code>. Verifica che GBM sia decisamente meglio.</p>',
      starter: `import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

np.random.seed(0)
n, d = 2000, 10
X = np.random.randn(n, d)
y = 3*X[:,0] + 2*X[:,1] - 4*X[:,0]*X[:,1] + 0.5*np.random.randn(n)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.25, random_state=42)

r2_ridge = None
r2_gbm = None`,
      check: `if r2_ridge and r2_gbm and r2_gbm > r2_ridge + 0.2:
    print(f"OK — Ridge {r2_ridge:.4f}, GBM {r2_gbm:.4f}")
else:
    print(f"FAIL — Ridge {r2_ridge}, GBM {r2_gbm}")`,
      solution: `r2_ridge = Ridge().fit(Xt, yt).score(Xe, ye)
r2_gbm = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Xt, yt).score(Xe, ye)` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 15.12 — Baseline vs boosting sofisticato',
      prompt: '<p>Su breast_cancer confronta: (1) LogisticRegression con Ridge tuning, (2) HistGradientBoosting con tuning. Salva le due CV ROC-AUC medie (5-fold) in <code class="inline">auc_lr, auc_gbm</code>.</p>',
      starter: `from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score

X, y = load_breast_cancer(return_X_y=True)

auc_lr = None
auc_gbm = None`,
      check: `if auc_lr and auc_gbm and auc_lr > 0.99 and auc_gbm > 0.99:
    print(f"OK — LR {auc_lr:.4f}, GBM {auc_gbm:.4f}")
else:
    print(f"FAIL — LR {auc_lr}, GBM {auc_gbm}")`,
      solution: `p1 = Pipeline([('s', StandardScaler()), ('lr', LogisticRegression(max_iter=5000))])
p2 = HistGradientBoostingClassifier(max_iter=300, random_state=0)
gs = GridSearchCV(p1, {'lr__C':[0.01,0.1,1,10]}, cv=5, scoring='roc_auc').fit(X, y)
auc_lr = gs.best_score_
auc_gbm = cross_val_score(p2, X, y, cv=5, scoring='roc_auc').mean()` },
  ],
  test: [
    { type: 'quiz', questions: [
      { q: 'Nel gradient boosting, l\'m-esimo albero è addestrato per predire:', options: ['La classe direttamente', 'Il gradiente negativo della loss (residui in caso di MSE)', 'La differenza tra classi', 'Un valore casuale'], answer: 1, explain: 'È la chiave dell\'algoritmo: ogni albero fa un passo di gradient descent nello spazio delle funzioni.' },
      { q: 'Riduci il learning rate a metà. Cosa devi fare?', options: ['Ridurre n_estimators', 'Aumentare n_estimators (tipicamente raddoppiare)', 'Nulla', 'Aumentare la profondità'], answer: 1, explain: 'lr piccolo = passi più piccoli = servono più alberi per stessa performance.' },
      { q: 'Early stopping in gradient boosting:', options: ['Fa perdere accuracy', 'Trova automaticamente il numero ottimo di alberi monitorando validation', 'Non è supportato', 'Serve solo con XGBoost'], answer: 1, explain: 'Best practice: max_iter grande + early stopping.' },
      { q: 'La feature importance "gain" può essere biased:', options: ['No, è imparziale', 'Sì, verso feature con più valori distinti; permutation importance è più affidabile', 'Sì, verso feature categoriche', 'Sì, verso il target'], answer: 1, explain: 'Le feature con più valori hanno più opportunità di split.' },
      { q: 'Su dati tabulari, il gradient boosting rispetto al deep learning:', options: ['È peggio nel 90% dei casi', 'È tipicamente migliore o pari, con molto meno tuning', 'È equivalente', 'Serve solo per dati piccoli'], answer: 1, explain: 'GBT domina sui tabulari. Deep learning vince su immagini/testo/audio.' },
    ] },
    { type: 'exercise', difficulty: 'hard', title: 'Test finale — pipeline completa',
      prompt: '<p>Su diabetes, costruisci una pipeline con StandardScaler + HistGradientBoosting, tuning con RandomizedSearchCV di learning_rate, max_depth, min_samples_leaf, max_iter. Salva R² test in <code class="inline">r2</code>. Target > 0.43.</p>',
      starter: `import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split

X, y = load_diabetes(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.25, random_state=42)

r2 = None`,
      check: `if r2 is None: print("FAIL: None")
elif r2 > 0.43: print(f"OK — R² {r2:.4f}")
else: print(f"FAIL — R² {r2:.4f}")`,
      solution: `pipe = Pipeline([('s', StandardScaler()), ('m', HistGradientBoostingRegressor(random_state=0))])
dist = {
    'm__learning_rate':[0.03, 0.05, 0.08, 0.1],
    'm__max_depth':[3, 4, 6, 8, None],
    'm__min_samples_leaf':[10, 20, 30, 50],
    'm__max_iter':[200, 400, 800],
}
rs = RandomizedSearchCV(pipe, dist, n_iter=15, cv=5, scoring='r2', random_state=0).fit(Xt, yt)
r2 = rs.score(Xe, ye)` },

    { type: 'exercise', difficulty: 'hard', title: 'Bonus expert — mini competizione',
      prompt: '<p><strong>Simulazione Kaggle</strong>. Dataset sintetico con: 15 feature (5 informative, 10 rumorose), 3 interazioni non lineari, feature ridondanti, rumore eteroschedastico. Devi raggiungere <strong>RMSE test < 8.0</strong>. Sei libero di usare feature engineering, ensemble, tuning.</p>',
      starter: `import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

np.random.seed(42)
n = 4000
X = np.random.randn(n, 15)
# 3 interazioni non lineari
inter = 5 * np.sin(X[:,0]) + 3 * X[:,1] * X[:,2] + 2 * np.exp(X[:,3]/2) - X[:,4]**2
y = inter + np.random.randn(n) * (2 + 0.5*np.abs(X[:,3]))

Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.25, random_state=0)

rmse = None`,
      check: `import numpy as np
if rmse is None: print("FAIL: None")
elif rmse < 8.0: print(f"OK — RMSE {rmse:.4f}")
else: print(f"FAIL — RMSE {rmse:.4f}, target < 8.0")`,
      solution: `m = HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.05,
                                    max_depth=6, min_samples_leaf=20,
                                    early_stopping=True, random_state=0).fit(Xt, yt)
rmse = np.sqrt(((m.predict(Xe) - ye)**2).mean())` },
  ]
});
