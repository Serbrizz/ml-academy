/* Lezione: Valutazione modelli (approfondita) */
window.LESSONS.push({
  id: 'l10-eval',
  title: 'Valutazione dei modelli',
  tag: 'Workflow',
  summary: 'Il capitolo più importante e più bistrattato del ML. Cross-validation, metriche giuste, gestione dello sbilanciamento, calibrazione, model selection senza illusioni.',
  teoria: [
    { type: 'md', content: `
<h3>10.1 Il vero problema</h3>
<p>La domanda a cui il ML cerca di rispondere è: <em>come funzionerà su dati mai visti?</em>. Se sbagli questa stima, tutto quello che fai dopo è teatro. Il 90% dei modelli che "sembrano ottimi ma falliscono in produzione" ha problemi di valutazione, non di modeling.</p>

<h3>10.2 Split base: train/val/test</h3>
<ul>
<li><strong>Training set (60-80%)</strong>: adatti i parametri del modello.</li>
<li><strong>Validation set (10-20%)</strong>: scegli iperparametri, confronti modelli, decidi feature.</li>
<li><strong>Test set (10-20%)</strong>: <strong>UNA volta sola</strong>, alla fine, per stimare la generalizzazione.</li>
</ul>
<p>Se guardi il test set 50 volte durante lo sviluppo e scegli il modello che va meglio, hai fatto <em>overfitting sul test</em>. Ora il tuo "test set" è di fatto un secondo validation, e la stima di generalizzazione è ottimista.</p>

<h3>10.3 Cross-validation: la struttura</h3>
<p><strong>K-fold CV</strong>:</p>
<ol>
<li>Dividi il training in K partizioni (fold).</li>
<li>Per ogni fold: addestra sui $K-1$ fold rimanenti, valida sul fold escluso.</li>
<li>Media i K punteggi.</li>
</ol>
<p><strong>Varianti importanti:</strong></p>
<ul>
<li><strong>Stratified K-fold</strong>: mantiene la proporzione di classi in ogni fold. <strong>Obbligatorio con classi sbilanciate.</strong></li>
<li><strong>Leave-One-Out (LOO)</strong>: $K = n$. Minimo bias, massima varianza. Costoso.</li>
<li><strong>Repeated K-fold</strong>: ripeti K-fold R volte con seed diversi. Media più stabile.</li>
<li><strong>GroupKFold</strong>: quando hai dati raggruppati (es. più misure per paziente). Un gruppo intero sta o in train o in val, mai spezzato.</li>
<li><strong>TimeSeriesSplit / Walk-forward</strong>: split cronologico. MAI shuffle nelle time series.</li>
</ul>

<h3>10.4 Nested CV: la valutazione onesta</h3>
<p>Se fai grid search del miglior iperparametro con CV, il best score che ottieni è <em>ottimista</em>: hai selezionato tra 100 configurazioni, ci sono buone probabilità che una vada bene per caso. Per stimare la vera generalizzazione:</p>
<ol>
<li><strong>Outer CV</strong>: dividi in K fold "esterni".</li>
<li>Per ogni fold esterno: applica grid search con <strong>inner CV</strong> sul solo training, poi valuta sul fold esterno.</li>
<li>La media dei K punteggi esterni è la stima onesta.</li>
</ol>
<p>Nested CV è pesante (K_out × K_in × n_configurazioni fit) ma è l'unico modo di stimare correttamente la performance di una pipeline con tuning.</p>

<h3>10.5 Data leakage: gli errori mortali</h3>
<p>Ogni operazione che coinvolge $y$ (o statistiche del dataset intero) deve stare <strong>dentro</strong> il fold, non prima. Errori tipici:</p>
<ul>
<li><strong>Scaling globale</strong>: calcoli $\\mu, \\sigma$ su tutto il dataset, poi splitti. Il test "vede" la media del training.</li>
<li><strong>Imputazione globale</strong>: idem con la mediana.</li>
<li><strong>Feature selection basata su correlazione con $y$</strong> prima dello split. Selezioni feature "aiutate" dal test.</li>
<li><strong>Target encoding</strong> senza fold: calcoli media di $y$ per categoria usando tutto il dataset.</li>
<li><strong>SMOTE prima dello split</strong>: gli esempi sintetici del training contaminano il test.</li>
<li><strong>Feature che "conoscono il futuro"</strong>: es. prevedere fraud usando "customer_status" che è stato aggiornato dopo l'evento.</li>
</ul>
<p><strong>Regola d'oro</strong>: usa sempre <code class="inline">Pipeline</code> di scikit-learn. Ogni step è refittato dentro ogni fold automaticamente.</p>

<h3>10.6 Metriche di regressione</h3>
<ul>
<li><strong>MSE / RMSE</strong>: sensibile agli outlier. Loss standard per training.</li>
<li><strong>MAE</strong>: robusta agli outlier. Interpretabile ("errore medio in unità di y").</li>
<li><strong>MAPE</strong>: errore percentuale medio. Esplode se $y_i \\approx 0$. Attenzione anche a $|y|$ molto piccolo: pochi punti dominano il totale.</li>
<li><strong>$R^2$</strong>: varianza spiegata. Utile per confronto tra modelli sullo stesso dataset.</li>
<li><strong>Median AE / Quantile loss</strong>: robuste. Median AE = errore mediano.</li>
</ul>

<h3>10.7 Metriche di classificazione — la trappola dell'accuracy</h3>
<p>Con classi bilanciate, l'accuracy va bene. Con classi sbilanciate diventa fuorviante: predire sempre la classe maggioritaria in un dataset 99/1 dà 99% accuracy, ma il modello è inutile.</p>
<h4>Matrice di confusione binaria</h4>
<pre class="code">              Predetto negativo | Predetto positivo
Vero negativo        TN         |        FP
Vero positivo        FN         |        TP</pre>
<ul>
<li><strong>Precision</strong> $= \\frac{TP}{TP+FP}$: di quelli che ho detto positivi, quanti lo erano davvero?</li>
<li><strong>Recall (sensitivity, TPR)</strong> $= \\frac{TP}{TP+FN}$: dei positivi veri, quanti ho trovato?</li>
<li><strong>Specificity (TNR)</strong> $= \\frac{TN}{TN+FP}$: dei negativi veri, quanti ho classificato bene?</li>
<li><strong>F1</strong> $= \\frac{2 PR}{P+R}$: media armonica di precision e recall.</li>
<li><strong>MCC</strong> $= \\frac{TP \\cdot TN - FP \\cdot FN}{\\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$: coefficiente di correlazione di Matthews. Robusto allo sbilanciamento. -1..1, 0 = casuale.</li>
</ul>

<h3>10.8 ROC-AUC vs Precision-Recall AUC</h3>
<p><strong>ROC</strong>: TPR vs FPR al variare della soglia. AUC = probabilità che uno positivo random abbia score più alto di un negativo random. Insensibile al bilanciamento delle classi (a livello di ranking).</p>
<p><strong>PR curve</strong>: Precision vs Recall al variare della soglia. AUC dipende dal bilanciamento (baseline = frazione di positivi).</p>
<p><strong>Quando usare cosa</strong>:</p>
<ul>
<li>Classi bilanciate o interesse simmetrico per entrambe: ROC-AUC.</li>
<li>Classi molto sbilanciate e interesse per la classe rara (fraud, medical): <strong>PR-AUC</strong>. Le curve ROC possono "sembrare buone" anche quando il modello è mediocre sui rari.</li>
</ul>

<h3>10.9 Calibrazione delle probabilità</h3>
<p>Non basta ordinare bene: se dici "probabilità 0.9", vuoi che sia davvero il 90% dei casi positivo. Diverse tecniche di calibrazione:</p>
<ul>
<li><strong>Platt scaling</strong>: fitting logistico sopra gli score.</li>
<li><strong>Isotonic regression</strong>: non parametrica.</li>
</ul>
<p>Sklearn: <code class="inline">CalibratedClassifierCV</code>. Diagnostica: <em>reliability diagram</em> (predetta vs osservata).</p>

<h3>10.10 Model comparison: test statistici</h3>
<p>Quando confronti due modelli con CV, non basta guardare le medie: la differenza potrebbe non essere significativa. Test utili:</p>
<ul>
<li><strong>Paired t-test</strong> sui K punteggi fold-by-fold. Assume normalità.</li>
<li><strong>Wilcoxon signed-rank</strong>: non parametrico, robusto.</li>
<li><strong>5x2 CV F-test</strong> di Dietterich: la versione classica per confronto di modelli ML.</li>
<li><strong>McNemar's test</strong>: per confrontare classificatori sugli stessi test set (test di significatività).</li>
</ul>

<h3>10.11 Learning curves e diagnosi</h3>
<p>Grafica loss di train e val al variare della dimensione del training set. Ti dice:</p>
<ul>
<li>Curve alte e vicine &rarr; <strong>bias alto</strong> (underfitting). Serve modello più complesso.</li>
<li>Curve con gap grande &rarr; <strong>varianza alta</strong> (overfitting). Serve regolarizzazione o più dati.</li>
<li>Curva val ancora in discesa a $n$ finale &rarr; con più dati miglioreresti.</li>
</ul>

<h3>10.12 Time series: la validation walk-forward</h3>
<p>Con dati temporali <strong>MAI shuffle</strong>: la CV standard usa dati "futuri" per addestrare e "passati" per validare, invertendo il tempo. Usa <code class="inline">TimeSeriesSplit</code>:</p>
<pre class="code">fold 1: train [t=1..100],    val [101..120]
fold 2: train [t=1..120],    val [121..140]
fold 3: train [t=1..140],    val [141..160]</pre>
<p>La finestra di training cresce (o è a lunghezza fissa). La val è sempre nel futuro.</p>
` },
    { type: 'callout', variant: 'warn', title: 'Errore classico', content: 'Applicare SMOTE (oversampling della classe minoritaria) prima dello split è LA fonte di data leakage più comune nei tutorial online. Fallo dentro pipeline con imblearn.pipeline.Pipeline (non sklearn) che gestisce il resampling solo nel training di ogni fold.' },
  ],
  esempi: [
    { type: 'md', content: '<h3>Esempio 1: pipeline corretta con CV stratified</h3>' },
    { type: 'run', content: `from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

X, y = load_breast_cancer(return_X_y=True)
pipe = Pipeline([('s', StandardScaler()),
                  ('lr', LogisticRegression(max_iter=1000))])
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
print(f"ROC-AUC: {scores.mean():.4f} +/- {scores.std():.4f}")
print(f"Fold: {scores.round(4)}")
print(f"Intervallo di confidenza 95%: [{scores.mean() - 1.96*scores.std()/(10**0.5):.4f}, {scores.mean() + 1.96*scores.std()/(10**0.5):.4f}]")` },
    { type: 'md', content: '<h3>Esempio 2: leakage in azione</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

X, y = load_breast_cancer(return_X_y=True)

# SBAGLIATO
X_scaled = StandardScaler().fit_transform(X)   # usa tutto
scores_bad = cross_val_score(LogisticRegression(max_iter=1000), X_scaled, y, cv=10, scoring='roc_auc')

# CORRETTO
pipe = Pipeline([('s', StandardScaler()), ('m', LogisticRegression(max_iter=1000))])
scores_good = cross_val_score(pipe, X, y, cv=10, scoring='roc_auc')

print(f"Con leakage:  {scores_bad.mean():.5f}")
print(f"Senza:        {scores_good.mean():.5f}")
print(f"Delta:        {(scores_bad.mean() - scores_good.mean())*1000:.3f} punti per mille")
print("\\nQui la differenza è piccola perché breast_cancer è pulito e le feature ben-behaved.")
print("In dataset piccoli o con outlier, la differenza può essere del 5-10%.")` },
    { type: 'md', content: '<h3>Esempio 3: nested CV completa</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold

X, y = load_breast_cancer(return_X_y=True)

pipe = Pipeline([('s', StandardScaler()), ('svm', SVC())])
grid = {'svm__C': [0.1, 1, 10], 'svm__gamma': ['scale', 0.01]}

inner_cv = StratifiedKFold(3, shuffle=True, random_state=0)
outer_cv = StratifiedKFold(5, shuffle=True, random_state=0)

gs = GridSearchCV(pipe, grid, cv=inner_cv, scoring='roc_auc')
# NESTED CV: stima onesta della performance della PIPELINE con tuning
nested_scores = cross_val_score(gs, X, y, cv=outer_cv, scoring='roc_auc')

# CV NON nested: rischio di over-ottimismo
gs.fit(X, y)
non_nested_best = gs.best_score_

print(f"Nested CV score:      {nested_scores.mean():.4f} +/- {nested_scores.std():.4f}")
print(f"Non-nested (best):    {non_nested_best:.4f}")
print(f"Differenza:           {(non_nested_best - nested_scores.mean())*100:.2f}%")
print("\\nIl non-nested è quasi sempre più ottimista — non usarlo per riportare la performance finale.")` },
    { type: 'md', content: '<h3>Esempio 4: sbilanciamento e metriche appropriate</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, average_precision_score, matthews_corrcoef,
                              confusion_matrix)

X, y = make_classification(n_samples=2000, weights=[0.98, 0.02], random_state=0)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Classe positiva nel test: {ye.mean()*100:.1f}%")

# Modello "dummy": predice sempre 0
class Dummy:
    def fit(self, X, y): return self
    def predict(self, X): return np.zeros(len(X))
    def predict_proba(self, X): return np.column_stack([np.ones(len(X))*0.98, np.ones(len(X))*0.02])

models = {'Dummy (sempre 0)': Dummy().fit(Xt, yt),
          'Logistic': LogisticRegression(max_iter=1000).fit(Xt, yt)}

print(f"\\n{'Modello':<20} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'ROC-AUC':>8} {'PR-AUC':>7} {'MCC':>7}")
for name, m in models.items():
    yp = m.predict(Xe)
    yprob = m.predict_proba(Xe)[:, 1] if hasattr(m, 'predict_proba') else yp
    print(f"{name:<20} {accuracy_score(ye, yp):>7.4f} "
          f"{precision_score(ye, yp, zero_division=0):>7.4f} "
          f"{recall_score(ye, yp, zero_division=0):>7.4f} "
          f"{f1_score(ye, yp, zero_division=0):>7.4f} "
          f"{roc_auc_score(ye, yprob):>8.4f} "
          f"{average_precision_score(ye, yprob):>7.4f} "
          f"{matthews_corrcoef(ye, yp):>7.4f}")
print("\\nAccuracy 0.98 sembra ottima ma recall/PR-AUC rivelano che il dummy è inutile.")` },
    { type: 'md', content: '<h3>Esempio 5: ROC vs PR curve</h3>' },
    { type: 'run', content: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc

X, y = make_classification(n_samples=2000, weights=[0.95, 0.05], random_state=0)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
m = LogisticRegression(max_iter=1000).fit(Xt, yt)
prob = m.predict_proba(Xe)[:, 1]

fpr, tpr, _ = roc_curve(ye, prob)
prec, rec, _ = precision_recall_curve(ye, prob)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(fpr, tpr, label=f'AUC={auc(fpr,tpr):.3f}')
axes[0].plot([0, 1], [0, 1], '--', color='gray')
axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR'); axes[0].set_title('ROC curve')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(rec, prec, color='green', label=f'AUC={auc(rec,prec):.3f}')
axes[1].axhline(ye.mean(), linestyle='--', color='gray', label=f'baseline={ye.mean():.3f}')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision'); axes[1].set_title('Precision-Recall curve')
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()
print("Con classi 95/5, la PR curve è più informativa: baseline già a 0.05 sull'asse y")` },
    { type: 'md', content: '<h3>Esempio 6: calibrazione delle probabilità</h3>' },
    { type: 'run', content: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=3000, random_state=0)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# RF spesso mal calibrato
rf = RandomForestClassifier(n_estimators=50, random_state=0).fit(Xt, yt)
# Calibrato con Platt scaling
rf_cal = CalibratedClassifierCV(RandomForestClassifier(n_estimators=50, random_state=0), method='sigmoid', cv=5).fit(Xt, yt)
lr = LogisticRegression(max_iter=1000).fit(Xt, yt)

fig, ax = plt.subplots(figsize=(7, 6))
for name, model in [('RF grezzo', rf), ('RF calibrato', rf_cal), ('Logistic', lr)]:
    prob = model.predict_proba(Xe)[:, 1]
    frac_pos, mean_pred = calibration_curve(ye, prob, n_bins=10)
    ax.plot(mean_pred, frac_pos, marker='o', label=name)
ax.plot([0, 1], [0, 1], '--', color='gray', label='perfetta')
ax.set_xlabel('Probabilità predetta media')
ax.set_ylabel('Frazione positivi osservata')
ax.set_title('Reliability diagram')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()` },
    { type: 'md', content: '<h3>Esempio 7: confronto tra modelli con paired t-test</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.stats import ttest_rel, wilcoxon

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(10, shuffle=True, random_state=42)

pipe_lr = Pipeline([('s', StandardScaler()), ('m', LogisticRegression(max_iter=5000))])
pipe_rf = RandomForestClassifier(n_estimators=100, random_state=0)

scores_lr = cross_val_score(pipe_lr, X, y, cv=cv, scoring='roc_auc')
scores_rf = cross_val_score(pipe_rf, X, y, cv=cv, scoring='roc_auc')

print(f"LR : {scores_lr.mean():.4f} +/- {scores_lr.std():.4f}")
print(f"RF : {scores_rf.mean():.4f} +/- {scores_rf.std():.4f}")
print(f"Delta medio: {scores_lr.mean() - scores_rf.mean():.4f}")

t_stat, p_val = ttest_rel(scores_lr, scores_rf)
print(f"\\nPaired t-test: t = {t_stat:.3f}, p = {p_val:.4f}")
w_stat, p_val_w = wilcoxon(scores_lr, scores_rf)
print(f"Wilcoxon:     W = {w_stat:.3f}, p = {p_val_w:.4f}")
print("\\np-value alto (>0.05) = differenza non statisticamente significativa")` },
    { type: 'md', content: '<h3>Esempio 8: time series validation walk-forward</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# Time series artificiale
np.random.seed(0)
n = 500
t = np.arange(n)
X = np.column_stack([np.sin(t/50), np.cos(t/50), np.random.randn(n)])
y = 2*X[:,0] + X[:,1] + 0.5*np.random.randn(n) + 0.001*t  # trend + rumore

tscv = TimeSeriesSplit(n_splits=5)
for i, (tr_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"Fold {i+1}: train indices {tr_idx.min()}..{tr_idx.max()}, val {val_idx.min()}..{val_idx.max()}")

scores = cross_val_score(Ridge(), X, y, cv=tscv, scoring='r2')
print(f"\\nR² per fold: {scores.round(3)}")
print(f"R² medio:  {scores.mean():.3f}")
print("\\nNota che i primi fold hanno meno dati -> spesso R² peggiore. Normale.")` },
  ],
  esercizi: [
    { type: 'exercise', difficulty: 'easy', title: 'Es. 10.1 — CV manuale',
      prompt: '<p>Implementa <code class="inline">my_cv(estimator, X, y, k)</code> che ritorna una lista degli accuracy sui $k$ fold (fai clone dell\'estimator per ogni fit).</p>',
      starter: `import numpy as np
from sklearn.base import clone

def my_cv(estimator, X, y, k=5):
    n = len(X)
    idx = np.arange(n)
    np.random.default_rng(0).shuffle(idx)
    folds = np.array_split(idx, k)
    scores = []
    # TODO
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
    m = clone(estimator).fit(X[train], y[train])
    scores.append(m.score(X[val], y[val]))
return scores` },

    { type: 'exercise', difficulty: 'easy', title: 'Es. 10.2 — Confusion matrix a mano',
      prompt: '<p>Data <code class="inline">y_true</code> e <code class="inline">y_pred</code>, calcola TP, FP, FN, TN (binaria con label 0/1). Salvali nel dizionario <code class="inline">cm</code>.</p>',
      starter: `import numpy as np
y_true = np.array([0,1,1,0,1,0,1,1,0,1])
y_pred = np.array([0,1,0,0,1,1,1,1,0,0])
cm = {'TP': None, 'FP': None, 'FN': None, 'TN': None}
print(cm)`,
      check: `import numpy as np
_TP = int(((y_true==1)&(y_pred==1)).sum())
_FP = int(((y_true==0)&(y_pred==1)).sum())
_FN = int(((y_true==1)&(y_pred==0)).sum())
_TN = int(((y_true==0)&(y_pred==0)).sum())
if cm == {'TP':_TP,'FP':_FP,'FN':_FN,'TN':_TN}:
    print(f"OK — {cm}")
else:
    print(f"FAIL — atteso TP={_TP},FP={_FP},FN={_FN},TN={_TN}")`,
      solution: `cm = {
    'TP': int(((y_true==1)&(y_pred==1)).sum()),
    'FP': int(((y_true==0)&(y_pred==1)).sum()),
    'FN': int(((y_true==1)&(y_pred==0)).sum()),
    'TN': int(((y_true==0)&(y_pred==0)).sum()),
}` },

    { type: 'exercise', difficulty: 'easy', title: 'Es. 10.3 — F-beta score',
      prompt: '<p>Implementa F_beta: $F_\\beta = (1+\\beta^2)\\frac{PR}{\\beta^2 P + R}$. Con $\\beta > 1$ pesa più il recall, con $\\beta < 1$ pesa più la precision.</p>',
      starter: `def fbeta(precision, recall, beta):
    # TODO
    return None
print(fbeta(0.8, 0.6, 1))    # F1 = 0.6857
print(fbeta(0.8, 0.6, 2))    # F2 pesa di più recall
print(fbeta(0.8, 0.6, 0.5))  # F0.5 pesa di più precision`,
      check: `if abs(fbeta(0.8, 0.6, 1) - 0.685714) < 1e-4 and \
   abs(fbeta(0.8, 0.6, 2) - 0.631579) < 1e-4:
    print("OK")
else:
    print("FAIL")`,
      solution: `def fbeta(precision, recall, beta):
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 10.4 — ROC curve a mano',
      prompt: '<p>Data <code class="inline">y_true</code> e <code class="inline">y_score</code>, per ogni soglia in np.linspace(0,1,21) calcola TPR e FPR. Salva i due array <code class="inline">tpr, fpr</code>.</p>',
      starter: `import numpy as np
np.random.seed(0)
y_true = np.random.randint(0, 2, size=200)
y_score = np.clip(0.5 + 0.35*(2*y_true-1) + 0.3*np.random.randn(200), 0, 1)

thresholds = np.linspace(0, 1, 21)
tpr = []
fpr = []
# TODO`,
      check: `import numpy as np
P = (y_true==1).sum(); N = (y_true==0).sum()
for t in thresholds:
    pred = (y_score >= t).astype(int)
    tpr_t = ((pred==1)&(y_true==1)).sum() / P if P>0 else 0
    fpr_t = ((pred==1)&(y_true==0)).sum() / N if N>0 else 0
if len(tpr)==21 and len(fpr)==21:
    from sklearn.metrics import auc
    order = np.argsort(fpr)
    _auc = auc(np.array(fpr)[order], np.array(tpr)[order])
    print(f"OK — AUC approx {_auc:.4f}")
else:
    print("FAIL — length")`,
      solution: `P = (y_true==1).sum(); N = (y_true==0).sum()
for t in thresholds:
    pred = (y_score >= t).astype(int)
    tp = ((pred==1)&(y_true==1)).sum()
    fp = ((pred==1)&(y_true==0)).sum()
    tpr.append(tp/P); fpr.append(fp/N)` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 10.5 — Threshold ottimale',
      prompt: '<p>Su una serie di score, trova la soglia in np.linspace(0,1,101) che massimizza <strong>MCC</strong>. Salva in <code class="inline">best_t</code>.</p>',
      starter: `import numpy as np
from sklearn.metrics import matthews_corrcoef

np.random.seed(0)
y_true = np.random.randint(0, 2, 500)
y_score = np.clip(0.5 + 0.3*(2*y_true-1) + 0.35*np.random.randn(500), 0, 1)

best_t = None
print(best_t)`,
      check: `import numpy as np
from sklearn.metrics import matthews_corrcoef
_scores = {t: matthews_corrcoef(y_true, (y_score>=t).astype(int)) for t in np.linspace(0,1,101)}
_atteso = max(_scores, key=_scores.get)
if best_t is not None and abs(best_t - _atteso) < 0.03:
    print(f"OK — soglia {best_t:.2f}, MCC {_scores[_atteso]:.4f}")
else:
    print(f"FAIL — atteso {_atteso:.2f}")`,
      solution: `from sklearn.metrics import matthews_corrcoef
best_t = max(np.linspace(0,1,101),
             key=lambda t: matthews_corrcoef(y_true, (y_score>=t).astype(int)))` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 10.6 — Nested CV',
      prompt: '<p>Implementa nested CV: outer 5-fold, inner 3-fold per grid search di LogisticRegression(C) su C in {0.01, 0.1, 1, 10, 100}. Salva media dei 5 outer score in <code class="inline">nested_mean</code>.</p>',
      starter: `import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold

X, y = load_iris(return_X_y=True)
nested_mean = None
print(nested_mean)`,
      check: `if nested_mean is not None and nested_mean > 0.9:
    print(f"OK — nested mean {nested_mean:.4f}")
else:
    print(f"FAIL — {nested_mean}")`,
      solution: `pipe = Pipeline([('s', StandardScaler()), ('lr', LogisticRegression(max_iter=5000))])
inner = StratifiedKFold(3, shuffle=True, random_state=0)
outer = StratifiedKFold(5, shuffle=True, random_state=0)
gs = GridSearchCV(pipe, {'lr__C':[0.01,0.1,1,10,100]}, cv=inner)
nested_mean = cross_val_score(gs, X, y, cv=outer).mean()` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 10.7 — Detect leakage',
      prompt: '<p>Confronta CV accuracy con scaling corretto (Pipeline) vs errato (fit_transform globale). Con dati generati puliti la differenza è piccola; usa <code class="inline">breast_cancer</code> e mostra i due valori. Salva in <code class="inline">acc_ok, acc_leak</code>.</p>',
      starter: `import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

X, y = load_breast_cancer(return_X_y=True)

acc_ok = None
acc_leak = None`,
      check: `if acc_ok is not None and acc_leak is not None:
    if abs(acc_ok - acc_leak) < 0.02 and acc_ok > 0.95:
        print(f"OK — corretto {acc_ok:.5f} vs leakage {acc_leak:.5f}")
    else:
        print("FAIL")`,
      solution: `pipe = Pipeline([('s', StandardScaler()), ('lr', LogisticRegression(max_iter=5000))])
acc_ok = cross_val_score(pipe, X, y, cv=10).mean()

X_scaled = StandardScaler().fit_transform(X)  # leakage
acc_leak = cross_val_score(LogisticRegression(max_iter=5000), X_scaled, y, cv=10).mean()` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 10.8 — Curve di apprendimento',
      prompt: '<p>Usa <code class="inline">learning_curve</code> di sklearn per diagnosticare bias/variance di un albero senza max_depth vs con max_depth=2 sullo stesso dataset. Trova quale dei due ha gap train-val più grande. Salva la stringa <code class="inline">problem</code> con "underfit" o "overfit" per l\'albero senza limiti.</p>',
      starter: `import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=0)
sizes = np.linspace(0.1, 1.0, 8)

# TODO: usa learning_curve per entrambi i modelli
problem = None
print("Diagnosi albero senza limiti:", problem)`,
      check: `if problem == "overfit":
    print("OK — albero senza limiti overfitta (gap train-val grande)")
else:
    print(f"FAIL — atteso 'overfit', tuo '{problem}'")`,
      solution: `# Albero senza limiti: train ~100%, val 70-80% -> overfit
_, tr, va = learning_curve(DecisionTreeClassifier(random_state=0), X, y, train_sizes=sizes, cv=5)
gap = tr.mean(1)[-1] - va.mean(1)[-1]
problem = "overfit" if gap > 0.1 else "underfit"` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 10.9 — Calibrazione delle probabilità',
      prompt: '<p>Su make_classification, allena un RandomForest e la sua versione calibrata (Platt). Calcola il Brier score di entrambi (più basso = meglio). Salva <code class="inline">brier_raw, brier_cal</code>.</p>',
      starter: `import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

X, y = make_classification(n_samples=3000, random_state=0)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

brier_raw = None
brier_cal = None`,
      check: `if brier_raw is not None and brier_cal is not None:
    if brier_cal < brier_raw:
        print(f"OK — calibrato migliore ({brier_cal:.4f} < {brier_raw:.4f})")
    else:
        print(f"WARN — calibrato non migliore in questo caso ({brier_cal:.4f} vs {brier_raw:.4f})")`,
      solution: `rf = RandomForestClassifier(n_estimators=50, random_state=0).fit(Xt, yt)
rf_cal = CalibratedClassifierCV(RandomForestClassifier(n_estimators=50, random_state=0), method='sigmoid', cv=3).fit(Xt, yt)
brier_raw = brier_score_loss(ye, rf.predict_proba(Xe)[:,1])
brier_cal = brier_score_loss(ye, rf_cal.predict_proba(Xe)[:,1])` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 10.10 — Time series walk-forward',
      prompt: '<p>Con TimeSeriesSplit a 5 fold, valuta un modello Ridge su una serie sintetica. Salva l\'array R² dei 5 fold in <code class="inline">scores</code>.</p>',
      starter: `import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

np.random.seed(0)
n = 400
t = np.arange(n)
X = np.column_stack([np.sin(t/30), np.cos(t/30), t/100])
y = 2*X[:,0] + X[:,2] + 0.4*np.random.randn(n)

scores = None`,
      check: `if scores is not None and len(scores) == 5 and scores.mean() > 0.5:
    print(f"OK — R² per fold {scores.round(3)}, media {scores.mean():.3f}")
else:
    print(f"FAIL — {scores}")`,
      solution: `scores = cross_val_score(Ridge(), X, y, cv=TimeSeriesSplit(5), scoring='r2')` },
  ],
  test: [
    { type: 'quiz', questions: [
      { q: 'Con classi 99% vs 1%, la migliore metrica è:', options: ['Accuracy', 'F1 o PR-AUC o MCC', 'ROC-AUC', 'Precision'], answer: 1, explain: 'Con sbilanciamento estremo servono metriche che non ricompensino il predire sempre la maggioranza.' },
      { q: 'Il pericolo del non-nested CV è:', options: ['Troppo lento', 'Ottimismo: il best score da GridSearch è biased in alto perché hai selezionato tra molte configurazioni', 'Non funziona con classi sbilanciate', 'Nessuno'], answer: 1, explain: 'Selezionare tra molte config artificialmente gonfia il punteggio.' },
      { q: 'Applicare SMOTE PRIMA di train_test_split è:', options: ['Best practice', 'Data leakage grave', 'Ininfluente', 'Solo un problema di performance'], answer: 1, explain: 'Esempi sintetici del training possono finire nel test set e viceversa.' },
      { q: 'Nel time series validation, si può fare shuffle prima di CV?', options: ['Sì, sempre', 'No, mai. Serve TimeSeriesSplit o walk-forward', 'Solo con Random Forest', 'Solo se il dataset è grande'], answer: 1, explain: 'Lo shuffle rompe l\'ordine temporale, contamina il training con dati "futuri".' },
      { q: 'Un modello con AUC ROC = 0.95 ma reliability diagram lontano dalla diagonale è:', options: ['Ottimo', 'Ben calibrato', 'Ranking-buono ma probabilità mal calibrate', 'Non-informativo'], answer: 2, explain: 'AUC misura ordine, non calibrazione. Sono cose distinte.' },
    ] },
    { type: 'exercise', difficulty: 'hard', title: 'Test finale — valutazione professionale',
      prompt: '<p>Sul breast_cancer, produci una valutazione seria di una LogisticRegression con Ridge (C tuning). Devi produrre:</p><ol><li><code class="inline">cv_mean</code>: media di ROC-AUC via nested CV (outer 10-fold, inner 3-fold)</li><li><code class="inline">cv_std</code>: std dei 10 outer scores</li></ol><p>Target cv_mean > 0.99.</p>',
      starter: `from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold

X, y = load_breast_cancer(return_X_y=True)
cv_mean = None
cv_std = None`,
      check: `if cv_mean is None: print("FAIL: None")
elif cv_mean > 0.99: print(f"OK — {cv_mean:.4f} +/- {cv_std:.4f}")
else: print(f"FAIL — {cv_mean:.4f}")`,
      solution: `pipe = Pipeline([('s', StandardScaler()), ('lr', LogisticRegression(max_iter=5000))])
inner = StratifiedKFold(3, shuffle=True, random_state=0)
outer = StratifiedKFold(10, shuffle=True, random_state=0)
gs = GridSearchCV(pipe, {'lr__C':[0.01,0.1,1,10,100]}, cv=inner, scoring='roc_auc')
scores = cross_val_score(gs, X, y, cv=outer, scoring='roc_auc')
cv_mean = scores.mean(); cv_std = scores.std()` },

    { type: 'exercise', difficulty: 'hard', title: 'Bonus expert — pipeline anti-leakage completa',
      prompt: '<p><strong>Scenario realistico</strong>. Ti passo un dataset con: valori mancanti, feature con scale enormemente diverse, classi sbilanciate 90/10. Costruisci una pipeline che gestisce tutto correttamente e raggiunge PR-AUC test > 0.75. Attenzione al leakage.</p>',
      starter: `import numpy as np
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

np.random.seed(42)
X, y = make_classification(n_samples=3000, n_features=15, weights=[0.9, 0.1],
                           n_informative=8, random_state=42)
X[:, 5] *= 1000
X[:, 8] *= 0.001
# inserisco NaN casuali
mask = np.random.rand(*X.shape) < 0.05
X[mask] = np.nan

Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

pr_auc = None`,
      check: `if pr_auc is None: print("FAIL: None")
elif pr_auc > 0.75: print(f"OK — PR-AUC {pr_auc:.4f}")
else: print(f"FAIL — PR-AUC {pr_auc:.4f}")`,
      solution: `pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('s', StandardScaler()),
    ('lr', LogisticRegression(class_weight='balanced', C=1.0, max_iter=5000))
]).fit(Xt, yt)
prob = pipe.predict_proba(Xe)[:, 1]
pr_auc = average_precision_score(ye, prob)` },
  ]
});
