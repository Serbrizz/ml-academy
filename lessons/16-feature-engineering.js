/* Lezione: Feature Engineering avanzato */
window.LESSONS.push({
  id: 'l16-feature-engineering',
  title: 'Feature Engineering avanzato',
  tag: 'Workflow',
  summary: 'La differenza tra un modello mediocre e uno vincente. Encoding categoriche, interazioni, time features, target encoding, gestione missing values, detect leakage.',
  teoria: [
    { type: 'md', content: `
<h3>16.1 Perché il feature engineering vince</h3>
<p>Un vecchio detto Kaggle: "prima le feature, poi il modello". La verità operativa è che <strong>feature engineering di qualità batte quasi sempre l\'algoritmo più sofisticato</strong>. Un HistGradientBoosting con feature engineering brillante batte XGBoost con feature grezze.</p>
<p>Feature engineering non è "aggiungere colonne a caso". È tradurre <em>conoscenza del dominio</em> in numeri che aiutino il modello a distinguere pattern che altrimenti dovrebbe inferire faticosamente dai dati.</p>

<h3>16.2 Encoding delle variabili categoriche</h3>
<p>Un modello ML lavora su numeri. Le categorie vanno trasformate. I metodi principali:</p>

<h4>One-Hot Encoding</h4>
<p>Per ogni categoria crei una colonna binaria. "Città = [Roma, Milano, Torino]" diventa 3 colonne. Semplice, senza assunzioni sull\'ordine.</p>
<p><strong>Pro</strong>: nessuna ipotesi implicita di ordinamento.</p>
<p><strong>Contro</strong>: esplode con alta cardinalità (10000 categorie → 10000 colonne). Perde info sulla frequenza.</p>

<h4>Ordinal Encoding</h4>
<p>Assegni un intero per ogni categoria. Usalo <strong>SOLO</strong> se c\'è un ordine naturale ("basso", "medio", "alto").</p>
<p><strong>Trappola</strong>: se mappi "Rosso=1, Verde=2, Blu=3" a random, imponi al modello un ordinamento fittizio che degrada le performance dei modelli lineari.</p>

<h4>Target Encoding (Mean Encoding)</h4>
<p>Sostituisci ogni categoria con la <strong>media di $y$</strong> per quella categoria. Es. per city="Roma", metti la media dei prezzi degli immobili a Roma.</p>
<p><strong>Pro</strong>: potentissimo per alta cardinalità. Comprime info in una sola colonna.</p>
<p><strong>Contro</strong>: <strong>data leakage</strong> se non fatto bene. La categoria "vede" il proprio $y$ nel calcolo della media. Contromisure:</p>
<ul>
<li><strong>Out-of-fold encoding</strong>: per ogni fold di CV, calcola la media usando solo gli altri fold.</li>
<li><strong>Smoothing</strong>: media pesata con la media globale, riduce l\'overfitting di categorie rare: $\\hat y_c = \\frac{n_c \\bar y_c + \\alpha \\bar y}{n_c + \\alpha}$.</li>
<li><strong>Noise injection</strong>: aggiungi rumore gaussiano al target durante l\'encoding.</li>
</ul>

<h4>Frequency / Count Encoding</h4>
<p>Sostituisci ogni categoria con la sua frequenza. Utile quando la frequenza porta informazione (es. utenti che comprano tanto sono diversi da chi compra poco).</p>

<h4>Embedding (per neural networks)</h4>
<p>Le reti neurali possono imparare rappresentazioni dense per categoriche. Utile con alta cardinalità (es. user_id, product_id).</p>

<h3>16.3 Interazioni tra feature</h3>
<p>Modelli lineari non catturano interazioni automaticamente. Devi crearle esplicitamente:</p>
<ul>
<li><strong>Prodotto</strong>: $x_1 \\times x_2$ (es. "prezzo" x "quantità" = "revenue").</li>
<li><strong>Rapporto</strong>: $x_1 / (x_2 + \\epsilon)$ (es. "debiti/reddito").</li>
<li><strong>Somma/differenza</strong>: $x_1 - x_2$ (es. differenze temporali).</li>
</ul>
<p>Trees e GBM catturano interazioni implicitamente, ma anche loro beneficiano di interazioni esplicite ben pensate.</p>

<h3>16.4 Binning / Discretizzazione</h3>
<p>Trasformare feature numeriche in categoriche. Strategie:</p>
<ul>
<li><strong>Equal-width</strong>: intervalli di larghezza uguale.</li>
<li><strong>Equal-frequency (quantiles)</strong>: ogni bin ha lo stesso numero di sample.</li>
<li><strong>Model-based (KBinsDiscretizer)</strong>: sklearn ha strategie sofisticate come K-means binning.</li>
</ul>
<p>Utile quando la relazione è fortemente non-lineare e vuoi che un modello lineare la catturi, o per gestire outlier (i valori estremi finiscono in bin ai bordi).</p>

<h3>16.5 Time features</h3>
<p>Da un timestamp puoi derivare tantissime feature:</p>
<ul>
<li><strong>Componenti</strong>: anno, mese, giorno, ora, giorno della settimana, giorno dell\'anno, settimana dell\'anno.</li>
<li><strong>Cicliche</strong>: per catturare periodicità, usa sin/cos dell\'angolo: $\\sin(2\\pi h/24)$ e $\\cos(2\\pi h/24)$ per l\'ora (evita il "salto" 23→0).</li>
<li><strong>Distanze temporali</strong>: giorni dall\'ultimo evento, giorni al prossimo evento noto.</li>
<li><strong>Flag</strong>: weekend, festivo, orario lavorativo.</li>
<li><strong>Lag features</strong>: valore di $t-1$, $t-7$, $t-30$ come feature per prevedere $t$.</li>
<li><strong>Rolling stats</strong>: media/std/max degli ultimi $k$ periodi.</li>
</ul>

<h3>16.6 Aggregation features</h3>
<p>Quando hai dati transazionali (utente con N transazioni), aggreghi per creare feature per l\'entità principale:</p>
<ul>
<li>Numero totale di transazioni</li>
<li>Media/somma/std/min/max degli importi</li>
<li>Ultimo importo, prima transazione</li>
<li>Numero di categorie distinte</li>
<li>Trend (regressione lineare dei valori nel tempo)</li>
</ul>
<p>Queste sono spesso le feature più predittive in problemi B2C.</p>

<h3>16.7 Gestione dei missing values</h3>
<p>Strategie in ordine di sofisticazione:</p>
<ol>
<li><strong>Rimozione</strong>: elimini righe/colonne. Solo se pochi missing.</li>
<li><strong>Imputazione semplice</strong>: media, mediana (numeriche), moda (categoriche). Robusta ma perde info.</li>
<li><strong>Missing indicator</strong>: aggiungi colonna binaria "era missing?". Preserva l\'informazione che il dato mancava, che a volte è predittiva.</li>
<li><strong>Model-based imputation</strong>: KNNImputer, IterativeImputer di sklearn.</li>
<li><strong>Missing as category</strong>: per categoriche, tratta "missing" come categoria a sé.</li>
</ol>
<p><strong>Regola d\'oro</strong>: fai imputazione DENTRO la Pipeline. Calcolare la mediana globale prima dello split è leakage.</p>

<h3>16.8 Scaling e trasformazioni</h3>
<ul>
<li><strong>StandardScaler</strong>: media 0, std 1. Assume approx-gaussianità. Sensibile a outlier.</li>
<li><strong>MinMaxScaler</strong>: in [0, 1]. Sensibile a outlier.</li>
<li><strong>RobustScaler</strong>: usa mediana e IQR. Robusto agli outlier.</li>
<li><strong>PowerTransformer (Yeo-Johnson, Box-Cox)</strong>: rende distribuzioni più gaussiane.</li>
<li><strong>QuantileTransformer</strong>: mappa a distribuzione uniforme o normale via quantili. Molto robusto.</li>
<li><strong>Log/sqrt</strong>: per feature con code lunghe (redditi, prezzi).</li>
</ul>

<h3>16.9 Target leakage — la trappola mortale</h3>
<p>Target leakage significa che le feature "sanno" del target in modo che non sarà disponibile in produzione. Esempi reali visti:</p>
<ul>
<li>Predire "cliente convertirà?" ma includere "data prossima call" nelle feature — la call è programmata solo per chi ha già mostrato interesse.</li>
<li>Predire "malattia grave?" con "n_visite_specialista" — se hai la malattia già hai visitato specialisti.</li>
<li>Predire "prezzo di vendita casa" con "prezzo_richiesto" — troppo vicino al target.</li>
</ul>
<p>Sintomi tipici del leakage:</p>
<ul>
<li>Accuracy sospettosamente alta (99%+ in problema difficile).</li>
<li>Una feature ha importance dominante.</li>
<li>La performance in produzione crolla rispetto ai test.</li>
</ul>
<p><strong>Contromisura</strong>: chiediti sempre "questa feature sarà davvero disponibile <em>al momento della predizione</em>?"</p>

<h3>16.10 Feature selection</h3>
<p>Anche il feature engineering ha un limite: troppi feature può degradare le performance.</p>
<ul>
<li><strong>Variance threshold</strong>: rimuovi feature con varianza vicino a zero.</li>
<li><strong>Correlation filter</strong>: rimuovi feature altamente correlate tra loro.</li>
<li><strong>Univariate (SelectKBest)</strong>: seleziona le $k$ migliori secondo un test statistico.</li>
<li><strong>Model-based (SelectFromModel)</strong>: usa importanze di Random Forest o coefficienti Lasso.</li>
<li><strong>RFE (Recursive Feature Elimination)</strong>: elimina iterativamente le meno importanti.</li>
</ul>

<h3>16.11 Target encoding con smoothing</h3>
<p>Il target encoding standard stima la media di $y$ per categoria $c$: con pochi campioni la stima è instabile. Lo <strong>smoothing</strong> mescola la media locale con la media globale con peso proporzionale alla dimensione del gruppo:</p>
<p>$$\\text{enc}(c) = \\lambda(c) \\cdot \\bar{y}_c + (1-\\lambda(c)) \\cdot \\bar{y}$$</p>
<p>dove $\\lambda(c) = \\frac{n_c}{n_c + m}$ e $m$ è l\'iperparametro di smoothing (tipico $m = 300$).</p>
<ul>
<li>Se $n_c$ è grande: $\\lambda \\to 1$ → usa la media locale (stima affidabile).</li>
<li>Se $n_c$ è piccolo: $\\lambda \\to 0$ → usa la media globale (shrinkage verso prior).</li>
</ul>
<p>Motivazione: evita l\'overfitting su categorie rare che potrebbero avere medie di $y$ molto rumorose. In sklearn: <code class="inline">TargetEncoder(smooth=\'auto\')</code> stima $m$ empiricamente con leave-one-out CV.</p>

<h3>16.12 Out-of-fold encoding (OOF)</h3>
<p>Nel target encoding naive, ogni campione contribuisce al calcolo della propria media di categoria → <strong>leakage</strong>: il modello "ricorda" $y$ tramite l\'encoding. La soluzione è l\'<strong>out-of-fold encoding</strong>:</p>
<ol>
<li>Dividi il training in $K$ fold.</li>
<li>Per ogni fold $k$, calcola la media di $y$ per categoria usando <em>solo</em> i $K-1$ fold rimanenti.</li>
<li>Applica quell\'encoding al fold $k$ escluso: ogni campione viene encodato senza vedere il proprio $y$.</li>
<li><strong>Test set</strong>: calcola l\'encoding su <em>tutto</em> il training (non fold-by-fold), poi applica al test.</li>
</ol>
<p>Il risultato è un encoding leakage-free che mantiene tutta la potenza informativa del target encoding. In pratica si combina sempre con lo smoothing (sezione 16.11) per gestire le categorie rare.</p>

<h3>16.13 Encoding circolare per feature temporali</h3>
<p>Ora del giorno, giorno della settimana e mese sono <strong>variabili circolari</strong>: le 23:00 è vicina alle 0:00, domenica è vicina a lunedì. Un ordinale intero rompe questa struttura (la distanza 23→0 appare enorme al modello). Soluzione: <strong>codifica sin/cos</strong>:</p>
<p>$$h_{\\sin} = \\sin\\!\\left(\\frac{2\\pi \\cdot x}{T}\\right), \\quad h_{\\cos} = \\cos\\!\\left(\\frac{2\\pi \\cdot x}{T}\\right)$$</p>
<p>dove $T$ è il periodo ($T=24$ per le ore, $T=7$ per i giorni della settimana, $T=12$ per i mesi). Le due feature prodotte preservano la geometria circolare: punti vicini nel ciclo restano vicini nello spazio delle feature. Esempio: ora 23 → $\\sin(2\\pi \\cdot 23/24) \\approx \\sin(2\\pi \\cdot 0/24)$, quindi le coordinate sono quasi identiche all\'ora 0.</p>

<h3>16.14 Alta cardinalità</h3>
<p>Categoriche con migliaia di valori distinti (codici postali, user_id, product_id) rendono il one-hot encoding impraticabile: la dimensionalità esplode e la matrice è sparsissima. Strategie per l\'alta cardinalità:</p>
<ul>
<li><strong>Target encoding + smoothing</strong>: comprime in una singola colonna continua informativa (vedi 16.11). Ottimo per modelli tree-based.</li>
<li><strong>Frequency encoding</strong>: sostituisce ogni categoria con il suo conteggio o frequenza relativa. Cattura la popolarità della categoria.</li>
<li><strong>Hashing trick</strong>: mappa la categoria via hash function in $n$ bucket fissi (<code class="inline">sklearn.feature_extraction.FeatureHasher</code>). Scalabile a cardinalità arbitraria, possibili collisioni tra categorie diverse.</li>
<li><strong>Embeddings</strong>: in reti neurali, mappa ogni categoria a un vettore denso di dimensione $d$ appreso end-to-end. Cattura similarità semantiche (prodotti simili → vettori vicini). Richiede dati abbondanti.</li>
</ul>

<h3>16.15 Feature crossing</h3>
<p>Un modello lineare non cattura il fatto che "reddito alto + età giovane" ha un effetto <em>diverso</em> dai due contributi separati. La soluzione è creare <strong>interazioni esplicite</strong> come nuove feature:</p>
<ul>
<li>Prodotto continuo: $x_1 \\cdot x_2$ (es. età × reddito).</li>
<li>Prodotto con soglie (dummy crossing): $\\mathbb{1}[x_1 > s_1] \\cdot \\mathbb{1}[x_2 > s_2]$.</li>
<li>Automatico: <code class="inline">PolynomialFeatures(degree=2, interaction_only=True)</code> genera tutte le interazioni di ordine 2.</li>
</ul>
<p>Attenzione: con $d$ feature le interazioni di ordine 2 sono $O(d^2)$. Con $d=100$ si generano ~5000 nuove colonne. Usa domain knowledge o feature importance per selezionare solo quelle informative ed evitare overfitting e curse of dimensionality.</p>

<h3>16.16 Population Stability Index (PSI)</h3>
<p>In produzione le distribuzioni cambiano nel tempo (<strong>data drift</strong>). Il PSI misura quanto la distribuzione di una feature si è spostata rispetto al training, binning sia training che produzione in $n$ intervalli e confrontando le proporzioni:</p>
<p>$$\\text{PSI} = \\sum_{i=1}^{n} (A_i - E_i) \\cdot \\ln\\frac{A_i}{E_i}$$</p>
<p>dove $E_i$ è la proporzione attesa (training) e $A_i$ quella attuale (produzione) nel bin $i$. Soglie pratiche ampiamente usate in ambito bancario e assicurativo:</p>
<ul>
<li><strong>PSI &lt; 0.1</strong>: distribuzione stabile, nessuna azione.</li>
<li><strong>0.1 ≤ PSI &lt; 0.2</strong>: leggero shift, monitorare attentamente.</li>
<li><strong>PSI ≥ 0.2</strong>: shift significativo → rivalutare il modello o riallena.</li>
</ul>
<p>Calcola PSI su ogni feature ad alto impatto con cadenza regolare (settimanale o mensile). Il PSI segnala il drift <em>prima</em> che le metriche di business degradino visibilmente, permettendo un retraining proattivo.</p>
` },
    { type: 'callout', variant: 'tip', title: 'Regola pratica', content: 'Se dopo un\'ora di feature engineering non hai migliorato la validation di almeno un 1-2%, probabilmente stai lavorando sulle feature sbagliate. Fai un passo indietro, riguarda l\'EDA, e chiediti che pattern dovrebbe imparare il modello.' },
    { type: 'callout', variant: 'warn', title: 'OOF encoding: attenzione al test set', content: 'Nell\'out-of-fold encoding il test set NON va trattato fold-by-fold: va encodato con le statistiche calcolate sull\'intero training set. Errore comune: applicare la logica OOF anche al test introducendo varianza artificiosa. Il test non ha fold propri — usa l\'encoding globale del training.' },
    { type: 'callout', variant: 'tip', title: 'PSI in produzione', content: 'Automatizza un alert quando PSI > 0.2 su feature critiche del modello. Il PSI segnala il data drift prima che la business metric (AUC, RMSE) cali visibilmente: ti permette di riallena in modo proattivo invece di reagire a un degrado già avvenuto in produzione.' },
  ],
  esempi: [
    { type: 'md', content: '<h3>Esempio 1: encoding categoriche</h3>' },
    { type: 'run', content: `import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

df = pd.DataFrame({
    'city': ['Roma', 'Milano', 'Torino', 'Roma', 'Milano', 'Napoli'],
    'size': ['small', 'medium', 'large', 'medium', 'small', 'large'],
})

# One-hot
ohe = OneHotEncoder(sparse_output=False).fit(df[['city']])
print("One-hot city:")
print(pd.DataFrame(ohe.transform(df[['city']]), columns=ohe.get_feature_names_out()))

# Ordinal (ok per size perché ha ordine intrinseco)
oe = OrdinalEncoder(categories=[['small', 'medium', 'large']]).fit(df[['size']])
print("\\nOrdinal size:")
print(oe.transform(df[['size']]).ravel())

# Frequency encoding
freq = df['city'].value_counts(normalize=True).to_dict()
print("\\nFrequency city:")
print(df['city'].map(freq).values)` },
    { type: 'md', content: '<h3>Esempio 2: target encoding con smoothing e leakage-safe</h3>' },
    { type: 'run', content: `import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def target_encode_kfold(X_col, y, n_splits=5, smoothing=10):
    """Target encoding con out-of-fold + smoothing per evitare leakage."""
    global_mean = y.mean()
    encoded = np.zeros(len(X_col), dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    for tr_idx, val_idx in kf.split(X_col):
        # media per categoria calcolata SOLO su tr_idx
        stats = pd.DataFrame({'x': X_col.iloc[tr_idx], 'y': y[tr_idx]}).groupby('x')['y'].agg(['mean', 'count'])
        # smoothing
        smooth = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)
        # applica a val_idx
        encoded[val_idx] = X_col.iloc[val_idx].map(smooth).fillna(global_mean).values
    return encoded

# Esempio
np.random.seed(0)
n = 500
cities = np.random.choice(['Roma', 'Milano', 'Torino', 'Napoli'], n)
# target correlato con città
means = {'Roma': 200, 'Milano': 300, 'Torino': 180, 'Napoli': 150}
y = np.array([means[c] for c in cities]) + np.random.randn(n) * 30

df = pd.DataFrame({'city': cities, 'y': y})
df['city_target'] = target_encode_kfold(df['city'], df['y'].values)
print(df.head(10))
print("\\nMedie encoding per città:")
print(df.groupby('city')['city_target'].mean().round(1))
print("\\nMedie vere:")
print(df.groupby('city')['y'].mean().round(1))` },
    { type: 'md', content: '<h3>Esempio 3: feature time-based con encoding ciclico</h3>' },
    { type: 'run', content: `import numpy as np
import pandas as pd

dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
df = pd.DataFrame({'date': dates})

# Feature base
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek  # 0=lunedi
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

# Encoding ciclico (importante per non far "saltare" 23->0)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

print(df.head())
print(df.tail(3))` },
    { type: 'md', content: '<h3>Esempio 4: interazioni e feature engineering visibili</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

np.random.seed(0)
n = 3000
X = np.random.randn(n, 5)
# Target con forte interazione
y = 2*X[:,0]*X[:,1] + 3*np.sin(X[:,2]) - X[:,3]**2 + 0.3*np.random.randn(n)

Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.25, random_state=42)

# 1) Grezzo con Ridge
r2_grezzo = Ridge().fit(Xt, yt).score(Xe, ye)

# 2) Grezzo con GBM
r2_gbm = HistGradientBoostingRegressor(random_state=0).fit(Xt, yt).score(Xe, ye)

# 3) Ridge con feature engineering "smart"
def eng(X):
    return np.column_stack([
        X[:,0], X[:,1], X[:,2], X[:,3], X[:,4],
        X[:,0] * X[:,1],           # interazione x0*x1
        np.sin(X[:,2]),            # non-linearità x2
        X[:,3]**2,                 # non-linearità x3
    ])
r2_ridge_fe = Ridge().fit(eng(Xt), yt).score(eng(Xe), ye)

print(f"Ridge grezzo:        R² = {r2_grezzo:.4f}")
print(f"GBM grezzo:          R² = {r2_gbm:.4f}")
print(f"Ridge + FE smart:    R² = {r2_ridge_fe:.4f}")
print("\\nCon feature engineering, la Ridge batte il GBM sui dati grezzi!")` },
    { type: 'md', content: '<h3>Esempio 5: missing values — strategie a confronto</h3>' },
    { type: 'run', content: `import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

np.random.seed(0)
n, d = 1000, 5
X = np.random.randn(n, d)
y = X @ np.array([2, -1, 0.5, 1.5, -0.5]) + np.random.randn(n) * 0.3
# Introduco NaN in colonna 0 e 2
X_nan = X.copy()
mask0 = np.random.rand(n) < 0.15
mask2 = np.random.rand(n) < 0.20
X_nan[mask0, 0] = np.nan
X_nan[mask2, 2] = np.nan

Xt, Xe, yt, ye = train_test_split(X_nan, y, test_size=0.25, random_state=42)

# Strategia 1: Median imputer
imp1 = SimpleImputer(strategy='median').fit(Xt)
r2_1 = HistGradientBoostingRegressor(random_state=0).fit(imp1.transform(Xt), yt).score(imp1.transform(Xe), ye)

# Strategia 2: KNN imputer
imp2 = KNNImputer(n_neighbors=5).fit(Xt)
r2_2 = HistGradientBoostingRegressor(random_state=0).fit(imp2.transform(Xt), yt).score(imp2.transform(Xe), ye)

# Strategia 3: Median + missing indicator
from sklearn.impute import MissingIndicator
mi = MissingIndicator(features='all').fit(Xt)
def with_indicator(X_):
    return np.hstack([SimpleImputer(strategy='median').fit(Xt).transform(X_), mi.transform(X_)])
Xt_i = with_indicator(Xt); Xe_i = with_indicator(Xe)
r2_3 = HistGradientBoostingRegressor(random_state=0).fit(Xt_i, yt).score(Xe_i, ye)

# Strategia 4: HistGB gestisce NaN nativamente
r2_4 = HistGradientBoostingRegressor(random_state=0).fit(Xt, yt).score(Xe, ye)

print(f"Median simple:       R² = {r2_1:.4f}")
print(f"KNN imputation:      R² = {r2_2:.4f}")
print(f"Median + indicator:  R² = {r2_3:.4f}")
print(f"HistGB native NaN:   R² = {r2_4:.4f}")` },
    { type: 'md', content: '<h3>Esempio 6: aggregazione per entità</h3>' },
    { type: 'run', content: `import pandas as pd
import numpy as np

# Simulo transazioni utente
np.random.seed(0)
trans = pd.DataFrame({
    'user_id': np.random.choice(['A','B','C','D'], 200),
    'date': pd.date_range('2024-01-01', periods=200, freq='D')[np.random.permutation(200)],
    'amount': np.random.exponential(50, 200),
    'category': np.random.choice(['food','tech','clothes','other'], 200),
}).sort_values(['user_id', 'date']).reset_index(drop=True)

# Feature per utente
features = trans.groupby('user_id').agg(
    n_trans=('amount', 'count'),
    amount_sum=('amount', 'sum'),
    amount_mean=('amount', 'mean'),
    amount_std=('amount', 'std'),
    amount_max=('amount', 'max'),
    n_categories=('category', 'nunique'),
    first_date=('date', 'min'),
    last_date=('date', 'max'),
).reset_index()
features['days_span'] = (features['last_date'] - features['first_date']).dt.days
features['trans_per_day'] = features['n_trans'] / (features['days_span'] + 1)

print("Feature aggregate per utente:")
print(features)` },
    { type: 'md', content: '<h3>Esempio 7: rilevamento leakage con importanza dominante</h3>' },
    { type: 'run', content: `import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

np.random.seed(0)
n = 1000
x1 = np.random.randn(n)
x2 = np.random.randn(n)
y = (x1 + 0.5*x2 + np.random.randn(n) > 0).astype(int)
# Introduco feature con leakage
x_leak = y + 0.05*np.random.randn(n)  # essenzialmente rivela y

X = np.column_stack([x1, x2, x_leak])
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.25, random_state=42)

m = HistGradientBoostingClassifier(random_state=0).fit(Xt, yt)
acc = m.score(Xe, ye)
print(f"Accuracy: {acc:.4f}  <-- sospetto se problema è difficile!")

perm = permutation_importance(m, Xe, ye, n_repeats=10, random_state=0)
print(f"\\nPermutation importance:")
for i, name in enumerate(['x1','x2','x_leak']):
    print(f"  {name}: {perm.importances_mean[i]:.4f}")
print("\\nx_leak domina completamente -> segnale che è correlato al target in modo sospetto.")` },
  ],
  esercizi: [
    { type: 'exercise', difficulty: 'easy', title: 'Es. 16.1 — One-hot manuale',
      prompt: '<p>Implementa <code class="inline">one_hot(series)</code> che prende una Series di stringhe e ritorna un DataFrame con colonne one-hot.</p>',
      starter: `import pandas as pd
def one_hot(s):
    # TODO
    return None

s = pd.Series(['a','b','a','c','b','a'])
print(one_hot(s))`,
      check: `import pandas as pd
result = one_hot(s)
_atteso = pd.get_dummies(s).astype(int) if pd.get_dummies(s).dtypes[0] == bool else pd.get_dummies(s)
if result is not None and result.shape == (6, 3):
    print(f"OK — shape {result.shape}, colonne {list(result.columns)}")
else:
    print("FAIL")`,
      solution: `def one_hot(s):
    return pd.get_dummies(s).astype(int)` },

    { type: 'exercise', difficulty: 'easy', title: 'Es. 16.2 — Encoding ciclico',
      prompt: '<p>Data una Series di ore in $[0, 23]$, crea le due colonne cicliche $\\sin(2\\pi h/24)$ e $\\cos(2\\pi h/24)$. Salvale in <code class="inline">df_cyc</code> come DataFrame di due colonne.</p>',
      starter: `import numpy as np
import pandas as pd
hours = pd.Series([0, 6, 12, 18, 23])
df_cyc = None
print(df_cyc)`,
      check: `import numpy as np
if df_cyc is not None:
    _sin = np.sin(2*np.pi*hours/24).values
    _cos = np.cos(2*np.pi*hours/24).values
    if np.allclose(df_cyc.iloc[:, 0], _sin) and np.allclose(df_cyc.iloc[:, 1], _cos):
        print("OK")
    else:
        print("FAIL")`,
      solution: `df_cyc = pd.DataFrame({
    'hour_sin': np.sin(2*np.pi*hours/24),
    'hour_cos': np.cos(2*np.pi*hours/24),
})` },

    { type: 'exercise', difficulty: 'easy', title: 'Es. 16.3 — Frequency encoding',
      prompt: '<p>Data una Series di categorie, applica frequency encoding (mappi ogni categoria alla sua frequenza normalizzata). Salva in <code class="inline">freq_encoded</code> come Series.</p>',
      starter: `import pandas as pd
s = pd.Series(['a','b','a','a','c','b','c','c','c','a'])
freq_encoded = None
print(freq_encoded)`,
      check: `import pandas as pd
_atteso = s.map(s.value_counts(normalize=True))
if freq_encoded is not None and (freq_encoded == _atteso).all():
    print("OK")
else:
    print("FAIL")`,
      solution: `freq_encoded = s.map(s.value_counts(normalize=True))` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 16.4 — Target encoding con smoothing',
      prompt: '<p>Implementa <code class="inline">target_encode_smooth(x, y, smoothing=10)</code>: media di y per categoria, smussata con la media globale usando la formula $(n_c \\bar y_c + \\alpha \\bar y) / (n_c + \\alpha)$.</p>',
      starter: `import numpy as np
import pandas as pd
def target_encode_smooth(x, y, smoothing=10):
    # TODO: ritorna dict {categoria -> valore encodato}
    return None

x = pd.Series(['a','a','a','b','b','c'])
y = np.array([10, 20, 30, 5, 15, 100])
print(target_encode_smooth(x, y, smoothing=2))`,
      check: `import numpy as np, pandas as pd
_global = y.mean()
_stats = pd.DataFrame({'x': x, 'y': y}).groupby('x')['y'].agg(['mean','count'])
_expected = ((_stats['mean'] * _stats['count'] + _global * 2) / (_stats['count'] + 2)).to_dict()
result = target_encode_smooth(x, y, smoothing=2)
if result is not None and all(abs(result.get(k, -1) - v) < 1e-6 for k, v in _expected.items()):
    print(f"OK — {result}")
else:
    print(f"FAIL — atteso {_expected}")`,
      solution: `def target_encode_smooth(x, y, smoothing=10):
    global_mean = np.mean(y)
    stats = pd.DataFrame({'x': x, 'y': y}).groupby('x')['y'].agg(['mean','count'])
    smooth = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)
    return smooth.to_dict()` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 16.5 — Binning',
      prompt: '<p>Usa <code class="inline">KBinsDiscretizer</code> per binnare una feature continua in 5 bin equal-frequency (quantile). Salva l\'array binned in <code class="inline">binned</code>.</p>',
      starter: `import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

np.random.seed(0)
x = np.random.randn(200).reshape(-1, 1)
binned = None`,
      check: `import numpy as np
if binned is not None and binned.shape == (200, 1):
    unique = np.unique(binned)
    if len(unique) == 5:
        counts = np.array([np.sum(binned==u) for u in unique])
        if counts.max() - counts.min() <= 2:
            print(f"OK — 5 bin bilanciati")
        else:
            print(f"FAIL — bins sbilanciati {counts}")
    else:
        print(f"FAIL — atteso 5 bin, {len(unique)}")
else:
    print("FAIL")`,
      solution: `binned = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile').fit_transform(x)` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 16.6 — Feature interaction',
      prompt: '<p>Crea la feature <code class="inline">x1 * x2</code> come nuova colonna in un DataFrame. Verifica poi che una LinearRegression con questa nuova feature abbia R² molto migliore rispetto a senza.</p>',
      starter: `import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(0)
n = 500
x1 = np.random.randn(n)
x2 = np.random.randn(n)
y = 3*x1*x2 + 0.2*np.random.randn(n)

df_no = pd.DataFrame({'x1': x1, 'x2': x2})
df_with = df_no.copy()
# TODO: aggiungi feature x1_x2 al df_with

r2_no = None
r2_with = None`,
      check: `if r2_no is not None and r2_with is not None:
    if r2_with > r2_no + 0.5:
        print(f"OK — senza {r2_no:.4f}, con {r2_with:.4f}")
    else:
        print(f"FAIL — {r2_no:.4f} vs {r2_with:.4f}")`,
      solution: `df_with['x1_x2'] = df_with['x1'] * df_with['x2']
Xt, Xe, yt, ye = train_test_split(df_no, y, test_size=0.3, random_state=0)
r2_no = LinearRegression().fit(Xt, yt).score(Xe, ye)
Xt, Xe, yt, ye = train_test_split(df_with, y, test_size=0.3, random_state=0)
r2_with = LinearRegression().fit(Xt, yt).score(Xe, ye)` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 16.7 — Missing indicator',
      prompt: '<p>Data una matrice X con NaN, crea la matrice combinata (imputazione mediana + colonne indicatrici). Salva in <code class="inline">X_combined</code>. Se X ha $d$ colonne, X_combined deve averne $2d$.</p>',
      starter: `import numpy as np
from sklearn.impute import SimpleImputer, MissingIndicator

np.random.seed(0)
X = np.random.randn(50, 3)
X[np.random.rand(50, 3) < 0.15] = np.nan

X_combined = None`,
      check: `import numpy as np
if X_combined is not None and X_combined.shape == (50, 6):
    if not np.isnan(X_combined).any():
        print(f"OK — shape {X_combined.shape}, senza NaN")
    else:
        print("FAIL — ci sono ancora NaN")
else:
    print("FAIL")`,
      solution: `imp = SimpleImputer(strategy='median').fit(X)
mi = MissingIndicator(features='all').fit(X)
X_combined = np.hstack([imp.transform(X), mi.transform(X).astype(float)])` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 16.8 — Robust scaling',
      prompt: '<p>Su dati con outlier, confronta StandardScaler e RobustScaler in termini di quanto vengono "distorti" dai outlier. Calcola la mediana dopo scaling con entrambi. Salva in <code class="inline">med_std, med_rob</code>.</p>',
      starter: `import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

np.random.seed(0)
x = np.concatenate([np.random.randn(100), [50, 60, 70]])  # 3 outlier estremi
x = x.reshape(-1, 1)

med_std = None
med_rob = None`,
      check: `if med_std is not None and med_rob is not None:
    if abs(med_rob) < abs(med_std) and abs(med_rob) < 0.5:
        print(f"OK — StandardScaler mediana {med_std:.3f}, RobustScaler {med_rob:.3f}")
        print(f"RobustScaler è meno sensibile agli outlier.")
    else:
        print(f"FAIL — {med_std}, {med_rob}")`,
      solution: `import numpy as np
med_std = float(np.median(StandardScaler().fit_transform(x)))
med_rob = float(np.median(RobustScaler().fit_transform(x)))` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 16.9 — Feature aggregation',
      prompt: '<p>Dato un DataFrame di transazioni (user_id, amount), crea un DataFrame per utente con: n_trans, mean_amount, std_amount, max_amount, ratio_max_mean. Salvalo in <code class="inline">user_features</code>.</p>',
      starter: `import pandas as pd
import numpy as np
np.random.seed(0)
trans = pd.DataFrame({
    'user_id': np.random.choice(['A','B','C'], 300),
    'amount': np.random.exponential(50, 300),
})

user_features = None
print(user_features)`,
      check: `import pandas as pd
if user_features is not None:
    cols = user_features.columns if hasattr(user_features, 'columns') else []
    required = {'n_trans','mean_amount','std_amount','max_amount','ratio_max_mean'}
    if required.issubset(set(cols)) and len(user_features) == 3:
        print(f"OK — {list(cols)}")
    else:
        print(f"FAIL — colonne {list(cols)}")`,
      solution: `agg = trans.groupby('user_id')['amount'].agg(
    n_trans='count', mean_amount='mean', std_amount='std', max_amount='max'
).reset_index()
agg['ratio_max_mean'] = agg['max_amount'] / agg['mean_amount']
user_features = agg` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 16.10 — Detect leakage',
      prompt: '<p>Data una feature <code class="inline">x_susp</code> potenzialmente leaky, verifica con permutation importance se domina troppo. Salva la sua importanza in <code class="inline">imp_susp</code>. Se > 0.4 sospetta leakage.</p>',
      starter: `import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

np.random.seed(0)
n = 800
x1 = np.random.randn(n)
x2 = np.random.randn(n)
y = (x1 + 0.5*x2 + np.random.randn(n) > 0).astype(int)
x_susp = y + 0.03*np.random.randn(n)   # leaky

X = np.column_stack([x1, x2, x_susp])
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.25, random_state=42)

imp_susp = None`,
      check: `if imp_susp is not None:
    if imp_susp > 0.4:
        print(f"OK — x_susp ha importanza {imp_susp:.3f}, LEAKAGE probabile!")
    else:
        print(f"OK ma importanza {imp_susp:.3f} < 0.4 (dipende dal seed)")`,
      solution: `m = HistGradientBoostingClassifier(random_state=0).fit(Xt, yt)
perm = permutation_importance(m, Xe, ye, n_repeats=10, random_state=0)
imp_susp = perm.importances_mean[2]` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 16.11 — Log transform su y skew',
      prompt: '<p>Il target ha coda lunga (distribuzione esponenziale). Confronta R² di un Ridge su y grezzo vs su log(y). Salva <code class="inline">r2_raw, r2_log</code>. Nota: valuta entrambi nell\'unità originale di y (converti log-pred con expm1).</p>',
      starter: `import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

np.random.seed(0)
n = 800
X = np.random.randn(n, 3)
y = np.exp(0.5 * X[:,0] + 0.3 * X[:,1] + 0.2*np.random.randn(n))
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.25, random_state=0)

r2_raw = None
r2_log = None`,
      check: `import numpy as np
if r2_raw is not None and r2_log is not None:
    if r2_log > r2_raw + 0.05:
        print(f"OK — raw {r2_raw:.4f}, log {r2_log:.4f}")
    else:
        print(f"FAIL — raw {r2_raw:.4f}, log {r2_log:.4f}")`,
      solution: `r2_raw = Ridge().fit(Xt, yt).score(Xe, ye)
m = Ridge().fit(Xt, np.log(yt))
y_pred = np.exp(m.predict(Xe))
r2_log = 1 - ((y_pred - ye)**2).sum() / ((ye - ye.mean())**2).sum()` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 16.12 — Pipeline FE completa',
      prompt: '<p>Dataset con: numeriche, categoriche, NaN. Costruisci una pipeline che gestisce tutto (imputazione + scaling per numeriche, one-hot per categoriche) usando ColumnTransformer. Fit + score. Salva R² test in <code class="inline">r2</code>.</p>',
      starter: `import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

np.random.seed(0)
n = 500
df = pd.DataFrame({
    'x1': np.random.randn(n),
    'x2': np.random.randn(n),
    'cat': np.random.choice(['a','b','c'], n),
})
# Inserisco NaN
df.loc[np.random.rand(n) < 0.1, 'x1'] = np.nan
y = 2*df['x1'].fillna(0) + df['x2'] + df['cat'].map({'a':1,'b':-1,'c':2}) + 0.3*np.random.randn(n)

Xt, Xe, yt, ye = train_test_split(df, y, test_size=0.25, random_state=42)

r2 = None`,
      check: `if r2 is not None and r2 > 0.7:
    print(f"OK — R² {r2:.4f}")
else:
    print(f"FAIL — {r2}")`,
      solution: `num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('s', StandardScaler())])
ct = ColumnTransformer([
    ('num', num_pipe, ['x1', 'x2']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['cat']),
])
pipe = Pipeline([('pre', ct), ('m', Ridge(alpha=1.0))]).fit(Xt, yt)
r2 = pipe.score(Xe, ye)` },
  ],
  test: [
    { type: 'quiz', questions: [
      { q: 'Il target encoding senza precauzioni causa:', options: ['Nulla di grave', 'Data leakage: la categoria "vede" il proprio y nella media', 'Overfitting sul test', 'È sempre sbagliato'], answer: 1, explain: 'Se calcoli la media di y usando anche la riga stessa, il modello "ricorda" y. Usa out-of-fold + smoothing.' },
      { q: 'Encoding ciclico per l\'ora del giorno usa:', options: ['Ordinal 0-23', 'One-hot 24 colonne', 'sin(2πh/24) e cos(2πh/24)', 'sqrt(h)'], answer: 2, explain: 'Sin/cos evita il "salto" 23→0 mantenendo la distanza corretta.' },
      { q: 'Missing indicator + median imputation:', options: ['È peggio di median soltanto', 'Preserva l\'info che il dato mancava, che può essere predittiva', 'Serve solo per feature categoriche', 'Non è mai utile'], answer: 1, explain: 'A volte "il fatto che manca" è un pattern (es. dati non compilati da certi tipi di utenti).' },
      { q: 'Feature "prezzo_richiesto" per predire "prezzo_venduto" è:', options: ['Ottima feature', 'Probabile leakage: quasi identica al target', 'Non-informativa', 'Utile solo con Ridge'], answer: 1, explain: 'Se la feature è troppo vicina al target, il modello impara la relazione banale.' },
      { q: 'RobustScaler è preferibile a StandardScaler quando:', options: ['Ci sono outlier', 'I dati sono gaussiani', 'Sempre', 'Con dataset piccoli'], answer: 0, explain: 'Usa mediana e IQR, non sensibili agli outlier.' },
    ] },
    { type: 'exercise', difficulty: 'hard', title: 'Test finale — dataset misto',
      prompt: '<p>Dataset simulato con numeriche, categoriche, NaN, un\'interazione forte. Devi costruire una pipeline completa (encoding + imputazione + scaling) e ottenere R² test > 0.7.</p>',
      starter: `import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 1500
df = pd.DataFrame({
    'age': np.random.randint(20, 80, n),
    'income': np.random.exponential(30000, n),
    'city': np.random.choice(['Roma','Milano','Torino','Napoli','Bologna'], n),
    'segment': np.random.choice(['A','B','C'], n),
})
df.loc[np.random.rand(n) < 0.1, 'income'] = np.nan
y = 0.05 * df['income'].fillna(30000) + \
    2 * df['age'] + \
    df['city'].map({'Roma':10,'Milano':30,'Torino':5,'Napoli':-5,'Bologna':15}) + \
    df['segment'].map({'A':100,'B':50,'C':0}) + \
    (df['age'] * df['segment'].map({'A':2,'B':1,'C':-1})) + \
    np.random.randn(n) * 20

Xt, Xe, yt, ye = train_test_split(df, y, test_size=0.25, random_state=0)

r2 = None`,
      check: `if r2 is not None and r2 > 0.7:
    print(f"OK — R² {r2:.4f}")
else:
    print(f"FAIL — {r2}")`,
      solution: `num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('s', StandardScaler())])
ct = ColumnTransformer([
    ('num', num_pipe, ['age', 'income']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['city', 'segment']),
])
pipe = Pipeline([('pre', ct), ('m', HistGradientBoostingRegressor(max_iter=300, random_state=0))]).fit(Xt, yt)
r2 = pipe.score(Xe, ye)` },

    { type: 'exercise', difficulty: 'hard', title: 'Bonus expert — feature engineering domain-aware',
      prompt: '<p><strong>Dataset immobiliare simulato.</strong> Prezzo dipende da: superficie (mq), numero stanze, età, zona, e un\'interazione superficie/stanze. Vuoi predire il log-prezzo. Target R² > 0.90 sul log-prezzo. Sfida: crea feature engineering intelligenti (es. mq per stanza, log di superficie, etc).</p>',
      starter: `import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

np.random.seed(0)
n = 3000
df = pd.DataFrame({
    'sqm': np.random.uniform(30, 300, n),
    'rooms': np.random.randint(1, 8, n),
    'age': np.random.randint(0, 100, n),
    'zone': np.random.choice(['centro','semi','periferia'], n),
})
zone_mult = df['zone'].map({'centro':3.0,'semi':1.5,'periferia':0.8})
y_price = df['sqm']**0.9 * (1 + df['rooms']/10) * zone_mult * np.exp(-df['age']/200) * 100
y_price *= np.exp(np.random.randn(n) * 0.15)  # rumore multiplicativo
y_log = np.log(y_price)

Xt, Xe, yt, ye = train_test_split(df, y_log, test_size=0.25, random_state=0)

r2 = None`,
      check: `if r2 is not None and r2 > 0.85:
    print(f"OK — R² {r2:.4f}")
else:
    print(f"FAIL — R² {r2}")`,
      solution: `def engineer(X):
    Xf = X.copy()
    Xf['sqm_per_room'] = Xf['sqm'] / Xf['rooms']
    Xf['log_sqm'] = np.log(Xf['sqm'])
    Xf['age2'] = Xf['age']**2
    return pd.get_dummies(Xf, columns=['zone'], drop_first=False)
Xt_e = engineer(Xt); Xe_e = engineer(Xe)
r2 = Ridge(alpha=0.1).fit(Xt_e, yt).score(Xe_e, ye)` },
  ]
});
