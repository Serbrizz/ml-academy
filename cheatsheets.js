/* Cheat sheets — riepilogo stampabile per lezione */
window.CHEATSHEETS = [
  {
    title: 'Fondamenti ML',
    body: `
<h4>Tipologie</h4>
<ul>
<li><b>Supervisionato</b>: hai coppie $(x, y)$. Classificazione se $y$ discreto, regressione se continuo.</li>
<li><b>Non supervisionato</b>: solo $x$. Clustering (K-Means), riduzione dimensionale (PCA).</li>
<li><b>Reinforcement</b>: agente + ambiente + ricompense.</li>
</ul>
<h4>Rischio empirico</h4>
<div class="formula">$\\hat{R}(f) = \\frac{1}{n}\\sum_i \\ell(f(x_i), y_i)$</div>
<h4>Overfitting vs underfitting</h4>
<ul>
<li>Training alta + test bassa &rarr; overfitting (varianza).</li>
<li>Entrambe basse &rarr; underfitting (bias).</li>
</ul>`
  },
  {
    title: 'Algebra lineare e calcolo utili',
    body: `
<h4>Formule chiave</h4>
<div class="formula">$\\nabla_x (a^T x) = a$</div>
<div class="formula">$\\nabla_x (x^T A x) = (A + A^T) x = 2Ax$ se A simmetrica</div>
<div class="formula">$\\nabla_x \\|Xx - y\\|^2 = 2 X^T (Xx - y)$</div>
<h4>Gradient descent</h4>
<div class="formula">$\\theta_{t+1} = \\theta_t - \\eta \\nabla \\mathcal{L}$</div>
<h4>Bayes</h4>
<div class="formula">$p(y|x) = \\frac{p(x|y)p(y)}{p(x)}$</div>`
  },
  {
    title: 'NumPy essentials',
    body: `
<pre class="code">import numpy as np
X = np.random.randn(100, 3)      # (n, d)
X.mean(axis=0)                    # media per colonna
X.std(axis=0)
X @ w                             # prodotto matrice-vettore
X.T @ X                           # (d, d) covarianza non centrata
np.linalg.solve(A, b)            # meglio di inv(A) @ b
np.linalg.eigh(A_simm)           # autovalori/vettori
np.linalg.svd(X)                  # SVD (base di PCA)
# Broadcasting
(X - X.mean(0)) / X.std(0)       # standardizza</pre>`
  },
  {
    title: 'Pandas essentials',
    body: `
<pre class="code">import pandas as pd
df = pd.read_csv('file.csv')
df.head(); df.info(); df.describe()
df['col'].value_counts()
df.groupby('key')['val'].mean()
df.pivot(index=, columns=, values=)
df.merge(df2, on='key')
df.dropna() / df.fillna(df.median())
df[df['col'] &gt; 5]                 # boolean mask
df['col'].apply(func)              # per casi non vettorializzabili</pre>`
  },
  {
    title: 'Regressione lineare',
    body: `
<h4>Modello</h4>
<div class="formula">$\\hat y = w^T x + b$, minimizza $\\|Xw-y\\|^2$</div>
<h4>Closed-form</h4>
<div class="formula">$\\hat w = (X^T X)^{-1} X^T y$</div>
<h4>Ridge / Lasso</h4>
<ul>
<li>Ridge (L2): $\\hat w = (X^T X + \\lambda I)^{-1} X^T y$. Riduce coefficienti, li mantiene tutti.</li>
<li>Lasso (L1): niente forma chiusa, produce coefficienti sparsi.</li>
</ul>
<h4>Metriche</h4>
MSE, RMSE, MAE, $R^2 = 1 - \\frac{SS_{res}}{SS_{tot}}$
<pre class="code">from sklearn.linear_model import LinearRegression, Ridge, Lasso
m = Ridge(alpha=1.0).fit(X_tr, y_tr)</pre>`
  },
  {
    title: 'Regressione logistica',
    body: `
<h4>Modello</h4>
<div class="formula">$P(y=1|x) = \\sigma(w^T x + b) = \\frac{1}{1+e^{-(w^T x + b)}}$</div>
<h4>Cross-entropy</h4>
<div class="formula">$\\mathcal{L} = -\\frac{1}{n}\\sum [y_i \\log \\hat p_i + (1-y_i)\\log(1-\\hat p_i)]$</div>
<h4>Metriche di classificazione</h4>
Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = media armonica, ROC-AUC.
<pre class="code">from sklearn.linear_model import LogisticRegression
m = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
prob = m.predict_proba(X)[:, 1]</pre>`
  },
  {
    title: 'KNN — quando usarlo',
    body: `
<h4>Come funziona</h4>
Predice usando i K punti di training più vicini (voto o media).
<h4>Regole d\'oro</h4>
<ul>
<li><b>Sempre standardizzare</b> prima (feature con scala grande dominano).</li>
<li>$k$ piccolo = overfitting, grande = underfitting. Cerca con CV.</li>
<li>Male in alta dimensione (curse of dimensionality). $d < 20$ tipicamente ok.</li>
</ul>
<pre class="code">from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
p = Pipeline([('s', StandardScaler()), ('k', KNeighborsClassifier(n_neighbors=5))])</pre>`
  },
  {
    title: 'Alberi e Random Forest',
    body: `
<h4>Albero singolo</h4>
Split greedy che massimizza riduzione di Gini/entropia. Fino a convergenza.
Pro: interpretabile, non serve scaling. Contro: alta varianza, overfitting.
<h4>Random Forest</h4>
Bagging (bootstrap sui dati) + random subset di feature ad ogni split.
<h4>Gradient Boosting</h4>
Alberi in sequenza, ognuno impara i residui del precedente. XGBoost, LightGBM.
<pre class="code">from sklearn.ensemble import RandomForestClassifier
m = RandomForestClassifier(n_estimators=200, max_depth=None).fit(X, y)
imp = m.feature_importances_</pre>`
  },
  {
    title: 'SVM — parametri chiave',
    body: `
<h4>Idea</h4>
Massimizza il margine tra classi. Con kernel non lineare (RBF) modello universale.
<h4>Iperparametri</h4>
<ul>
<li><code>C</code>: inverso della regolarizzazione. Grande = poco tollerante agli errori.</li>
<li><code>gamma</code>: larghezza del kernel RBF. Grande = kernel stretto, rischio overfitting.</li>
</ul>
<h4>Kernel</h4>
<ul><li>linear: fast, feature molte</li><li>rbf: default, non lineare</li><li>poly: raramente meglio di rbf</li></ul>
<pre class="code">from sklearn.svm import SVC
m = SVC(kernel='rbf', C=1.0, gamma='scale').fit(X, y)</pre>`
  },
  {
    title: 'Valutazione e CV',
    body: `
<h4>Split</h4>
Train (60-80%) / Val (10-20%) / Test (10-20%). Test si guarda UNA volta sola.
<h4>K-fold CV</h4>
Media su K addestramenti. StratifiedKFold per classificazione.
<h4>Regola aurea</h4>
Tutto ciò che tocca $y$ o statistiche dei dati va DENTRO la Pipeline, MAI prima dello split.
<pre class="code">from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
cv = StratifiedKFold(5, shuffle=True, random_state=0)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')</pre>
<h4>Grid search</h4>
<pre class="code">from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(pipe, {'clf__C':[0.1,1,10]}, cv=5).fit(X, y)</pre>`
  },
  {
    title: 'K-Means clustering',
    body: `
<h4>Algoritmo</h4>
1. Inizializza K centroidi (k-means++)
2. Assign: ogni punto al centroide più vicino
3. Update: centroide = media dei suoi punti
4. Ripeti fino a convergenza
<h4>Scelta di K</h4>
Metodo del gomito (inertia vs K) o silhouette score.
<h4>Limiti</h4>
Assume cluster sferici, di dimensione simile. Per forme complesse: DBSCAN o GMM.
<pre class="code">from sklearn.cluster import KMeans
km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)</pre>`
  },
  {
    title: 'PCA',
    body: `
<h4>Idea</h4>
Proietta i dati sulle direzioni di massima varianza. Componenti = autovettori della matrice di covarianza.
<h4>Standardizza SEMPRE</h4>
Altrimenti feature con scala grande dominano.
<h4>Quanti componenti?</h4>
Sceglie K per catturare 90-95% della varianza.
<pre class="code">from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
X_s = StandardScaler().fit_transform(X)
pca = PCA(n_components=0.95).fit(X_s)     # 95% variance
Z = pca.transform(X_s)
print(pca.explained_variance_ratio_)</pre>`
  },
  {
    title: 'Reti neurali',
    body: `
<h4>MLP</h4>
<div class="formula">$h^{(l+1)} = \\phi(W^{(l+1)} h^{(l)} + b^{(l+1)})$</div>
<h4>Attivazioni</h4>
<ul>
<li>ReLU: default per hidden layer</li>
<li>Sigmoid: output classificazione binaria</li>
<li>Softmax: output classificazione multi-classe</li>
</ul>
<h4>Loss</h4>
<ul>
<li>Regressione: MSE</li>
<li>Classificazione binaria: BCE</li>
<li>Multi-classe: categorical cross-entropy</li>
</ul>
<h4>Regolarizzazione</h4>
Dropout, weight decay L2, batch norm, early stopping.
<pre class="code">from sklearn.neural_network import MLPClassifier
m = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                   alpha=1e-4, max_iter=500).fit(X, y)</pre>`
  },
  {
    title: 'Workflow di un progetto',
    body: `
<ol>
<li>Definisci problema e metrica</li>
<li>Raccogli dati, ispeziona</li>
<li>Split train/val/test IMMEDIATAMENTE</li>
<li>Baseline (media, majority, lineare)</li>
<li>Preprocessing dentro Pipeline</li>
<li>Prova più modelli con CV</li>
<li>Tuning con GridSearch / Optuna</li>
<li>Valuta UNA volta sul test</li>
<li>Interpreta, documenta, deploya</li>
</ol>
<h4>Errori tipici</h4>
Leakage, ignorare imbalance, non testare baseline, guardare test 100 volte.`
  },
  {
    title: 'Gradient Boosting',
    body: `
<h4>L’idea</h4>
Ensemble sequenziale: ogni albero impara i residui del precedente. Gradient descent nello spazio delle funzioni.
<div class="formula">$F_M(x) = \sum_{m=1}^M \eta \cdot f_m(x)$</div>
<h4>Iperparametri chiave (ordine di importanza)</h4>
<ol>
<li><b>learning_rate</b>: 0.03-0.1 tipico</li>
<li><b>n_estimators / max_iter</b>: tanti (500-2000) + early stopping</li>
<li><b>max_depth</b>: 3-8, di solito shallow</li>
<li><b>min_samples_leaf</b>: 10-50</li>
<li><b>subsample</b>: 0.7-1.0</li>
</ol>
<h4>Codice sklearn</h4>
<pre class="code">from sklearn.ensemble import HistGradientBoostingRegressor
m = HistGradientBoostingRegressor(
    max_iter=1000, learning_rate=0.05,
    max_depth=6, min_samples_leaf=20,
    early_stopping=True, n_iter_no_change=30,
    validation_fraction=0.15,
    random_state=0
).fit(X_tr, y_tr)</pre>
<h4>Feature importance</h4>
Sempre usare permutation_importance o SHAP, non le importanze native (biased).`
  },
  {
    title: 'Feature Engineering',
    body: `
<h4>Encoding categoriche</h4>
<ul>
<li><b>One-hot</b>: pochi valori distinti, no ordine implicito</li>
<li><b>Ordinal</b>: solo se c’è ordine naturale (small/medium/large)</li>
<li><b>Target encoding</b>: alta cardinalità. USA out-of-fold + smoothing per evitare leakage</li>
<li><b>Frequency</b>: quando la frequenza porta info</li>
</ul>
<h4>Time features</h4>
Componenti (mese, ora, weekday), cicliche (sin/cos), lag, rolling stats, distanze temporali.
<h4>Interazioni</h4>
Modelli lineari non le catturano: creale esplicite. Trees le catturano ma beneficiano di espliciti.
<h4>Missing values</h4>
<ol>
<li>Rimozione se pochi</li>
<li>Imputazione mediana/moda</li>
<li><b>+ Missing indicator</b>: preserva l’info che il dato mancava</li>
<li>Model-based (KNN, iterativo) se sofisticato</li>
</ol>
<h4>Regola d’oro anti-leakage</h4>
Tutto dentro Pipeline. Sempre.
<pre class="code">from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('s', StandardScaler())])
ct = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
])</pre>`
  },
  {
    title: 'Reti neurali — parametri e training',
    body: `
<h4>Struttura MLP</h4>
<div class="formula">$h^{(l)} = \phi(W^{(l)} h^{(l-1)} + b^{(l)})$</div>
<h4>Attivazioni</h4>
<ul>
<li>ReLU: default hidden</li>
<li>Sigmoid: output binaria</li>
<li>Softmax: output multi-classe</li>
</ul>
<h4>Ottimizzatori</h4>
<ul>
<li>Adam(lr=1e-3): default</li>
<li>SGD + momentum: quando serve generalizzare bene</li>
<li>AdamW: Adam con weight decay corretto (Transformer)</li>
</ul>
<h4>Diagnosi problemi</h4>
<ul>
<li>Loss NaN/diverge: lr troppo alto</li>
<li>Loss stagna: lr troppo basso, dead ReLU, inizializzazione</li>
<li>Train ok, val no: overfit -> aggiungi dropout, weight decay, data augmentation</li>
<li>Nessuna scende: capacità insufficiente, o bug</li>
</ul>
<pre class="code">from sklearn.neural_network import MLPClassifier
m = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu', solver='adam',
    learning_rate_init=1e-3, alpha=1e-4,
    early_stopping=True, validation_fraction=0.1,
    max_iter=500, random_state=0
)</pre>`
  },
  {
    title: 'CNN — cheat sheet',
    body: `
<h4>Convolution output size</h4>
<div class="formula">$\lfloor (H + 2p - k)/s \rfloor + 1$</div>
<p>con H=input, k=kernel, p=padding, s=stride.</p>
<h4>Architettura tipica</h4>
<pre class="code">Conv 3x3 -> BatchNorm -> ReLU -> [Conv 3x3 -> BatchNorm -> ReLU]xN -> MaxPool 2x2
(ripeti N volte con canali crescenti)
Flatten -> Dropout -> Linear -> Output</pre>
<h4>Transfer learning</h4>
<pre class="code">import torchvision.models as models
model = models.resnet18(weights='DEFAULT')
model.fc = nn.Linear(model.fc.in_features, num_classes)
# Fine-tune con lr=1e-4 per feature layer, lr=1e-3 per fc</pre>
<h4>Data augmentation</h4>
RandomHorizontalFlip, RandomCrop, RandomRotation, ColorJitter, MixUp, CutMix.`
  },
  {
    title: 'MLOps essentials',
    body: `
<h4>Experiment tracking (MLflow)</h4>
<pre class="code">import mlflow
with mlflow.start_run():
    mlflow.log_param('lr', 0.05)
    mlflow.log_metric('roc_auc', auc)
    mlflow.sklearn.log_model(model, 'model')</pre>
<h4>Serving (FastAPI)</h4>
<pre class="code">from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()
class Input(BaseModel):
    features: list[float]
@app.post('/predict')
def predict(inp: Input):
    return {'prob': float(model.predict_proba([inp.features])[0, 1])}</pre>
<h4>Docker</h4>
<pre class="code">FROM python:3.11-slim
COPY requirements.txt app.py model.joblib ./
RUN pip install -r requirements.txt
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]</pre>
<h4>Drift monitoring</h4>
<ul>
<li>KS test per feature: p < 0.05 = drift significativo</li>
<li>PSI: < 0.1 stabile, 0.1-0.2 lieve, > 0.2 severo</li>
<li>Distribuzione dei score del modello nel tempo</li>
</ul>`
  },
  {
    title: 'Imbalanced classification — checklist',
    body: `
<h4>Metriche</h4>
<ul>
<li><b>PR-AUC</b> (average_precision): la più informativa con sbilanciamento severo</li>
<li>Recall @ precision fissa: risposta business diretta</li>
<li>Cost-based total = FN*C_FN + FP*C_FP</li>
<li>MCC: robusto allo sbilanciamento, tra -1 e 1</li>
</ul>
<h4>Threshold ottimale cost-sensitive</h4>
<div class="formula">$t^* = C_{FP}/(C_{FP} + C_{FN})$</div>
<h4>Tecniche</h4>
<ol>
<li><b>class_weight="balanced"</b> o sample_weight calibrati -> quasi sempre sufficiente</li>
<li>SMOTE (dentro Pipeline!): solo se class_weight non basta</li>
<li>Threshold tuning: cruciale per costo asimmetrico</li>
<li>Ensemble bilanciati (BalancedRandomForest da imblearn)</li>
</ol>
<h4>Non fare mai</h4>
<ul>
<li>Bilanciare il test set (falsa la stima)</li>
<li>SMOTE prima dello split</li>
<li>Usare threshold 0.5 di default</li>
<li>Guardare solo accuracy</li>
</ul>`
  }


];
