/* Lezione: Progetto End-to-End Fraud Detection */
window.LESSONS.push({
  id: 'l19-fraud-project',
  title: 'Progetto: Fraud Detection End-to-End',
  tag: 'Progetto',
  summary: 'Un progetto completo di classificazione fortemente sbilanciata: EDA, feature engineering, tuning, threshold cost-sensitive, interpretabilità, monitoring simulato.',
  teoria: [
    { type: 'md', content: `
<h3>19.1 Il problema</h3>
<p>Le transazioni fraudolente sono rare (0.1-1% del totale) ma molto costose: una frode media di 500€, con migliaia di transazioni al giorno, sono milioni all\'anno. Un modello che riduce il tasso di frode del 10% è già un caso ROI clamoroso.</p>
<p>Sfide specifiche:</p>
<ul>
<li><strong>Class imbalance estremo</strong>: 0.1-1% classe positiva.</li>
<li><strong>Costo asimmetrico</strong>: falso negativo (frode non rilevata) costa 500€, falso positivo (cliente onesto bloccato) costa 5€ + insoddisfazione.</li>
<li><strong>Feedback ritardato</strong>: la vera etichetta di una frode può arrivare giorni/settimane dopo (chargeback).</li>
<li><strong>Adversarial</strong>: i frodatori imparano e adattano tecniche.</li>
<li><strong>Concept drift veloce</strong>: pattern di frode cambiano continuamente.</li>
</ul>

<h3>19.2 Le metriche giuste</h3>
<p>Con classi 999:1, accuracy è inutile (99.9% predicendo sempre "non frode"). Le metriche che contano:</p>
<ul>
<li><strong>Precision-Recall AUC</strong>: ideale per sbilanciamento.</li>
<li><strong>Recall@precision fixed</strong>: es. "quale recall a precision 90%?" — cioè "se accettiamo max 10% di falsi positivi, quante frodi catturiamo?"</li>
<li><strong>Cost-based metric</strong>: la vera metrica business è la somma dei costi.</li>
</ul>
<p>La metrica cost-based è definita come:</p>
<p>$$\\text{Cost} = C_{FN} \\cdot FN + C_{FP} \\cdot FP$$</p>
<p>Il modello ottimo minimizza il costo totale atteso, non necessariamente l\'accuracy.</p>

<h3>19.3 Threshold ottimale sensitive to cost</h3>
<p>Uscita del modello: probabilità $p$ di frode. Soglia default: 0.5. Ma con costi asimmetrici, la soglia ottima è:</p>
<p>$$t^* = \\frac{C_{FP}}{C_{FP} + C_{FN}}$$</p>
<p>Se $C_{FN} = 500, C_{FP} = 5$: $t^* = 5/505 = 0.0099$. Cioè: <em>anche una probabilità dell\'1% giustifica bloccare la transazione</em>.</p>

<h3>19.4 Il workflow completo</h3>
<ol>
<li><strong>EDA</strong>: distribuzione classi, feature importance preliminari, correlazioni con la classe rara.</li>
<li><strong>Split stratificato</strong>: fondamentale con sbilanciamento estremo.</li>
<li><strong>Feature engineering</strong>: aggregate per utente (transazioni ultime 24h, media importi), velocità (tempo tra transazioni), z-score dell\'importo rispetto alla storia del cliente.</li>
<li><strong>Modello</strong>: gradient boosting è il default. Con class_weight="balanced" o sample_weight per gestire lo sbilanciamento.</li>
<li><strong>Threshold tuning</strong>: cerchiamo la soglia che minimizza il costo atteso.</li>
<li><strong>Interpretabilità</strong>: permutation importance, SHAP-like per capire quali feature dominano.</li>
<li><strong>Monitoring</strong>: distribuzione dei score nel tempo, PSI, alert su drift.</li>
</ol>

<h3>19.5 Strategie per lo sbilanciamento</h3>
<ul>
<li><strong>Non fare nulla</strong>: molti modelli moderni (GBM) gestiscono bene lo sbilanciamento se lo tuni.</li>
<li><strong>class_weight="balanced"</strong>: sklearn pesa automaticamente le classi.</li>
<li><strong>sample_weight custom</strong>: pesi calibrati ai costi business.</li>
<li><strong>Oversampling della classe minoritaria</strong>: duplicare (rischio overfitting) o SMOTE (sintesi di esempi nuovi via interpolazione).</li>
<li><strong>Undersampling della maggioranza</strong>: butta esempi (perdita informazione ma velocità).</li>
<li><strong>Ensemble techniques</strong>: EasyEnsemble, BalancedRandomForest (imblearn).</li>
</ul>
<p>Nella mia esperienza pratica: <em>class_weight su GBM + threshold tuning</em> vince quasi sempre. SMOTE aggiunge complessità senza reale beneficio.</p>

<h3>19.6 Pattern comuni di frode</h3>
<p>Feature engineering domain-aware (esempi reali che vedresti in un\'azienda):</p>
<ul>
<li>Importo z-score rispetto alla storia dell\'utente</li>
<li>Numero transazioni ultima ora / 24h / 7 giorni</li>
<li>Tempo dalla transazione precedente</li>
<li>Distanza geografica dalle transazioni recenti (impossibile teletrasportarsi)</li>
<li>Merchant nuovo per il cliente</li>
<li>Ora del giorno atipica per il cliente</li>
<li>Device fingerprint nuovo</li>
<li>Velocità di typing (biometric)</li>
</ul>

<h3>19.7 Anti-pattern classici</h3>
<ul>
<li><strong>Bilanciare test set</strong>: se rimuovi negativi dal test, non stimi più la performance reale.</li>
<li><strong>SMOTE prima dello split</strong>: contamina il test con esempi sintetici derivati dal training.</li>
<li><strong>Solo accuracy</strong>: già detto.</li>
<li><strong>Threshold 0.5 sempre</strong>: quasi sempre subottimale con costi asimmetrici.</li>
<li><strong>Non monitorare</strong>: i pattern cambiano, il modello degrada.</li>
</ul>
` },
    { type: 'callout', variant: 'tip', title: 'Il progetto da portfolio', content: 'Un fraud detection ben fatto (con EDA, feature engineering, threshold cost-sensitive, interpretabilità, dashboard di monitoring) è UNO dei migliori progetti da mostrare in un colloquio ML entry-level. Copre praticamente tutte le competenze principali in un dominio con impatto business chiaro.' },
    { type: 'md', content: `
<h3>19.8 SMOTE: algoritmo dettagliato</h3>
<p>SMOTE (Synthetic Minority Over-sampling Technique) genera esempi sintetici interpolando tra punti esistenti della classe rara. Per ogni punto $x_i$:</p>
<ol>
<li>Trova i $k$ vicini più vicini nella classe rara (default $k=5$).</li>
<li>Scegli casualmente un vicino $x_{\\text{neighbor}}$.</li>
<li>Genera: $x_{\\text{new}} = x_i + \\lambda \\cdot (x_{\\text{neighbor}} - x_i)$, con $\\lambda \\sim U(0,1)$.</li>
</ol>
<p>Il punto sintetico cade sul segmento tra $x_i$ e il vicino. L'interpolazione avviene nello spazio delle feature, non nel dominio fisico dei valori: le feature numeriche vengono interpolate linearmente, producendo valori verosimili ma mai osservati nel dataset reale.</p>
<p><strong>Regola critica: applica SMOTE solo sul training set, MAI prima dello split.</strong> Se SMOTE viene applicato sull'intero dataset, i punti sintetici — derivati da osservazioni reali — finiscono sia nel train che nel test. Il modello ha già visto varianti di quei punti in training: le performance in validazione saranno sovrastimate (data leakage). Con <code class="inline">imblearn.pipeline.Pipeline</code> SMOTE viene automaticamente applicato solo al fold di training durante cross-validation.</p>
<p>Varianti principali (libreria <code class="inline">imbalanced-learn</code>):</p>
<ul>
<li><strong>ADASYN</strong>: genera più punti sintetici nelle zone dove il classificatore sbaglia maggiormente, aumentando la densità della classe rara dove è più difficile separarla dalla maggioranza.</li>
<li><strong>Borderline-SMOTE</strong>: genera solo esempi vicini al confine decisionale, dove la classificazione è più incerta e gli esempi sintetici sono più utili.</li>
<li><strong>SMOTE-NC</strong>: gestisce feature miste (numeriche + categoriche), interpolando le numeriche e usando la moda dei vicini per le categoriche.</li>
</ul>

<h3>19.9 Costo ottimale e threshold business: derivazione completa</h3>
<p>Il costo totale del sistema come funzione della soglia $\\tau$:</p>
<p>$$\\text{Costo}(\\tau) = FP(\\tau) \\cdot C_{FP} + FN(\\tau) \\cdot C_{FN}$$</p>
<p>La soglia ottimale minimizza il costo atteso:</p>
<p>$$\\tau^* = \\arg\\min_{\\tau \\in [0,1]} \\left[ FP(\\tau) \\cdot C_{FP} + FN(\\tau) \\cdot C_{FN} \\right]$$</p>
<p>Con $C_{FN}/C_{FP} = 100$: ogni frode non rilevata pesa 100 falsi allarmi. Abbassare $\\tau$ riduce FN (costosi) ma aumenta FP (economici). Il punto ottimo si trova empiricamente molto sotto 0.5 — spesso intorno al 5-15% — perché conviene bloccare anche a bassa probabilità di frode pur di non perdere quelle reali.</p>
<p><strong>Cost-sensitive learning tramite class_weight</strong>: <code class="inline">class_weight={0: 1, 1: 100}</code> amplifica il gradiente per gli esempi positivi 100x durante il training. Il modello ottimizza già tenendo conto del rapporto di costi: la frontiera decisionale si sposta implicitamente verso il recall elevato. Questo è complementare al threshold tuning — class_weight agisce sul training, il threshold sull'inferenza. Usarli entrambi dà risultati migliori di ognuno da solo.</p>

<h3>19.10 Temporal leakage nei dati finanziari</h3>
<p>Il temporal leakage è il rischio più insidioso nei dati finanziari: non genera errori, non produce warning, ma causa performance eccellenti in validazione e pessime in produzione. È il motivo più comune di "il modello funzionava benissimo in sviluppo e non funziona in produzione".</p>
<p>Avviene quando una feature usata al tempo $T$ è calcolata con informazioni disponibili solo dopo $T$. Esempi classici:</p>
<ul>
<li><strong>Saldo aggiornato post-transazione</strong>: il saldo del conto riflette già l'esito della transazione fraudolenta, non lo stato al momento della decisione.</li>
<li><strong>Flag "cliente segnalato per frode"</strong>: deriva da chargeback che arrivano settimane dopo la transazione originale. Al momento T, questa informazione non esiste ancora.</li>
<li><strong>Aggregati con window mal definita</strong>: "media importi ultimi 7 giorni" calcolata includendo la transazione corrente o transazioni successive a T.</li>
<li><strong>Statistiche globali su tutto il dataset</strong>: z-score rispetto alla media del merchant calcolata su tutto il periodo storico, incluso il futuro rispetto alla transazione che si sta classificando.</li>
</ul>
<p><strong>Regola aurea</strong>: ogni feature alla transazione $T$ deve usare esclusivamente dati con timestamp strettamente precedente a $T$:</p>
<p><em>feature(transazione T) = aggregato(eventi dello stesso utente con timestamp &lt; T)</em></p>
<p><strong>Split temporale corretto</strong>: ordina sempre per timestamp prima dello split, non fare shuffle. Usa il periodo più recente come test (hold-out temporale). Il random split su dati temporali introduce leakage diretto: il modello "vede" pattern dal futuro durante il training e li memorizza come segnali predittivi.</p>

<h3>19.11 Calcolo dei class weight: formula e pratica</h3>
<p>Con <code class="inline">class_weight="balanced"</code>, sklearn calcola automaticamente:</p>
<p>$$w_i = \\frac{n_{\\text{samples}}}{n_{\\text{classes}} \\cdot n_i}$$</p>
<p>Esempio: 9900 negativi e 100 positivi (1% fraud rate) su 10000 campioni:</p>
<ul>
<li>$w_0 = 10000 / (2 \\cdot 9900) \\approx 0.505$</li>
<li>$w_1 = 10000 / (2 \\cdot 100) = 50$</li>
<li>Rapporto $w_1 / w_0 = 99$: ogni positivo contribuisce al gradiente quanto 99 negativi.</li>
</ul>
<p>Equivalente con <code class="inline">sample_weight</code>:</p>
<p><code class="inline">w = np.where(y == 1, n_neg / n_pos, 1.0)</code></p>
<p>Per allineare al costo business invece del bilanciamento statistico, usa <code class="inline">ratio = C_FN / C_FP</code> come peso della classe positiva: se la frode non rilevata costa 100x il falso allarme, usa ratio=100. Questo incorpora il costo direttamente nel training — non solo nella soglia di decisione al momento dell'inferenza.</p>
<p><strong>Nota pratica</strong>: il bilanciamento perfetto (ratio = n_neg/n_pos) massimizza il recall ma può sacrificare troppa precision. Il ratio ottimale si trova di solito tra il bilanciamento statistico e il rapporto dei costi. Cerca con cross-validation sul validation set.</p>

<h3>19.12 Population Stability Index (PSI) per il monitoring</h3>
<p>Il PSI misura lo shift di distribuzione tra training (riferimento) e produzione (attuale), dividendo i valori in $N$ bin (tipicamente 10-20):</p>
<p>$$\\text{PSI} = \\sum_{i=1}^{N} (\\text{Act}_i - \\text{Exp}_i) \\cdot \\ln\\!\\left(\\frac{\\text{Act}_i}{\\text{Exp}_i}\\right)$$</p>
<p>$\\text{Exp}_i$ = proporzione nel training set per il bin $i$; $\\text{Act}_i$ = proporzione in produzione. Il PSI è simmetrico rispetto alla KL-divergence e penalizza sia aumenti che diminuzioni di densità in ogni bin.</p>
<p>Soglie di interpretazione standard:</p>
<ul>
<li><strong>PSI &lt; 0.1</strong>: distribuzione stabile, nessuna azione necessaria.</li>
<li><strong>0.1 ≤ PSI &lt; 0.25</strong>: cambiamento moderato, aumentare la frequenza di monitoraggio.</li>
<li><strong>PSI ≥ 0.25</strong>: shift significativo, valutare retraining del modello.</li>
</ul>
<p>Calcola PSI su due livelli:</p>
<ul>
<li><strong>Score del modello</strong>: segnale diretto di degradazione — se la distribuzione degli score di probabilità cambia, il modello ha cambiato comportamento globale.</li>
<li><strong>Feature chiave</strong> (importo, frequenza transazioni, ora del giorno): identifica quale input sta guidando il drift e permette di diagnosticare la causa.</li>
</ul>
<p>In produzione: calcola PSI su finestre rolling settimanali rispetto alla distribuzione di training. Imposta alert automatici su PSI ≥ 0.25 per score o feature critiche. Il PSI delle feature è utile anche per capire <em>perché</em> il modello degrada, non solo <em>che</em> degrada.</p>
` },
    { type: 'callout', variant: 'warn', title: 'Temporal leakage: il bug silenzioso', content: 'Il temporal leakage produce modelli con performance eccellenti in validazione e pessime in produzione. Non genera errori: il modello "sembra" funzionare perché usa informazioni dal futuro. Audit sistematico di tutte le feature con timestamp è obbligatorio prima del deploy su dati finanziari.' },
    { type: 'callout', variant: 'tip', title: 'SMOTE: quando usarlo davvero', content: 'SMOTE raramente supera class_weight su GBM moderni. Usalo quando: il modello non supporta sample_weight, il training set è molto piccolo (meno di 500 esempi positivi), o vuoi aumentare la varianza della classe rara. Usa sempre imblearn.pipeline.Pipeline per garantire che SMOTE venga applicato solo al fold di training durante cross-validation.' },
  ],
  esempi: [
    { type: 'md', content: '<h3>Esempio 1: dataset simulato di fraud detection</h3>' },
    { type: 'run', content: `import numpy as np
import pandas as pd

np.random.seed(42)
n = 20000
fraud_rate = 0.02  # 2% di frodi

# Simula pattern realistici
# Feature: amount, hour, days_since_signup, num_transactions_24h, avg_amount_history, is_new_merchant
amounts = np.concatenate([np.random.exponential(50, int(n * (1 - fraud_rate))),
                            np.random.exponential(300, int(n * fraud_rate))])  # frodi tendono importi alti
hours = np.concatenate([np.random.normal(14, 4, int(n * (1 - fraud_rate))),
                          np.random.normal(3, 4, int(n * fraud_rate))])  # frodi di notte
days_signup = np.concatenate([np.random.exponential(500, int(n * (1 - fraud_rate))),
                                np.random.exponential(30, int(n * fraud_rate))])
num_trans_24h = np.random.poisson(2, n)
avg_hist = np.random.exponential(50, n)
is_new_merch = np.random.rand(n) < np.concatenate([np.full(int(n * (1 - fraud_rate)), 0.1),
                                                     np.full(int(n * fraud_rate), 0.6)])

y = np.concatenate([np.zeros(int(n * (1 - fraud_rate))), np.ones(int(n * fraud_rate))]).astype(int)

# Shuffle
idx = np.random.permutation(n)
df = pd.DataFrame({
    'amount': amounts[idx],
    'hour': np.clip(hours[idx], 0, 23),
    'days_since_signup': days_signup[idx],
    'num_transactions_24h': num_trans_24h[idx],
    'avg_amount_history': avg_hist[idx],
    'is_new_merchant': is_new_merch[idx].astype(int),
    'fraud': y[idx],
})

print("Shape:", df.shape)
print("Fraud rate:", df['fraud'].mean())
print("\\nStatistiche per classe:")
print(df.groupby('fraud')[['amount', 'hour', 'days_since_signup']].mean().round(2))` },
    { type: 'md', content: '<h3>Esempio 2: EDA — distribuzione feature per classe</h3>' },
    { type: 'run', content: `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
n = 20000
fraud_rate = 0.02
n_f = int(n * fraud_rate)
n_nf = n - n_f
amount = np.concatenate([np.random.exponential(50, n_nf), np.random.exponential(300, n_f)])
hour = np.concatenate([np.random.normal(14, 4, n_nf), np.random.normal(3, 4, n_f)])
y = np.concatenate([np.zeros(n_nf), np.ones(n_f)])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for cls, color in [(0, 'blue'), (1, 'red')]:
    axes[0].hist(amount[y == cls], bins=50, alpha=0.5, color=color,
                  label=f'fraud={int(cls)}', density=True)
    axes[1].hist(hour[y == cls] % 24, bins=24, alpha=0.5, color=color,
                  label=f'fraud={int(cls)}', density=True)
axes[0].set_xlabel('Amount'); axes[0].set_title('Distribuzione importi')
axes[0].legend(); axes[0].set_xlim(0, 500)
axes[1].set_xlabel('Hour'); axes[1].set_title('Distribuzione ore')
axes[1].legend()
plt.tight_layout(); plt.show()
print("Frodi tendono ad avere importi maggiori e ad essere di notte.")` },
    { type: 'md', content: '<h3>Esempio 3: baseline vs modello sbilanciato-aware</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score)

# Riprendiamo il dataset
np.random.seed(42)
n = 20000; fr = 0.02
n_f = int(n * fr); n_nf = n - n_f
X = np.column_stack([
    np.concatenate([np.random.exponential(50, n_nf), np.random.exponential(300, n_f)]),
    np.concatenate([np.random.normal(14, 4, n_nf), np.random.normal(3, 4, n_f)]),
    np.random.exponential(500, n),
    np.random.poisson(2, n),
])
y = np.concatenate([np.zeros(n_nf), np.ones(n_f)])
idx = np.random.permutation(n); X = X[idx]; y = y[idx]
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Fraud in train: {yt.mean()*100:.2f}%, in test: {ye.mean()*100:.2f}%")
print()

def evaluate(name, y_true, y_pred, y_prob=None):
    print(f"=== {name} ===")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1:        {f1_score(y_true, y_pred, zero_division=0):.4f}")
    if y_prob is not None:
        print(f"  ROC-AUC:   {roc_auc_score(y_true, y_prob):.4f}")
        print(f"  PR-AUC:    {average_precision_score(y_true, y_prob):.4f}")

# 1) Baseline: sempre negativo
dummy = DummyClassifier(strategy='most_frequent').fit(Xt, yt)
evaluate('Dummy (majority)', ye, dummy.predict(Xe))

# 2) GBM senza sbilanciamento management
m1 = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xt, yt)
evaluate('GBM default', ye, m1.predict(Xe), m1.predict_proba(Xe)[:, 1])

# 3) GBM con sample_weight
w = np.where(yt == 1, (yt == 0).mean() / (yt == 1).mean(), 1.0)
m2 = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xt, yt, sample_weight=w)
evaluate('GBM sample_weight', ye, m2.predict(Xe), m2.predict_proba(Xe)[:, 1])` },
    { type: 'md', content: '<h3>Esempio 4: threshold ottimo cost-sensitive</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

np.random.seed(42)
n = 20000; fr = 0.02
n_f = int(n * fr); n_nf = n - n_f
X = np.column_stack([
    np.concatenate([np.random.exponential(50, n_nf), np.random.exponential(300, n_f)]),
    np.concatenate([np.random.normal(14, 4, n_nf), np.random.normal(3, 4, n_f)]),
])
y = np.concatenate([np.zeros(n_nf), np.ones(n_f)])
idx = np.random.permutation(n); X = X[idx]; y = y[idx]
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
m = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xt, yt)
prob = m.predict_proba(Xe)[:, 1]

# Costi: FN (frode non rilevata) = 500€, FP (cliente onesto bloccato) = 5€
C_FN = 500
C_FP = 5

print(f"{'threshold':>10} {'FN':>6} {'FP':>6} {'total_cost':>12} {'recall':>8}")
for t in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
    pred = (prob >= t).astype(int)
    fn = ((pred == 0) & (ye == 1)).sum()
    fp = ((pred == 1) & (ye == 0)).sum()
    total = fn * C_FN + fp * C_FP
    recall = ((pred == 1) & (ye == 1)).sum() / (ye == 1).sum() if (ye == 1).sum() else 0
    print(f"{t:>10.2f} {fn:>6d} {fp:>6d} {total:>12,d} {recall:>8.3f}")

# Threshold ottimo cost-based
best_t = None
best_cost = float('inf')
for t in np.linspace(0.001, 0.999, 500):
    pred = (prob >= t).astype(int)
    fn = ((pred == 0) & (ye == 1)).sum()
    fp = ((pred == 1) & (ye == 0)).sum()
    cost = fn * C_FN + fp * C_FP
    if cost < best_cost:
        best_cost = cost; best_t = t

print(f"\\nThreshold ottimo cost-based: {best_t:.4f}, costo totale {best_cost:,}€")
print(f"Formula teorica t* = C_FP/(C_FP+C_FN) = {C_FP/(C_FP+C_FN):.4f}")` },
    { type: 'md', content: '<h3>Esempio 5: recall @ precision fixed</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.metrics import precision_recall_curve

# Riprendiamo dagli esempi precedenti
np.random.seed(42)
n = 5000
prob = np.clip(np.random.beta(2, 8, n) + 0.4*np.random.randn(n)*0.3, 0.001, 0.999)
y_true = (prob + 0.15*np.random.randn(n) > 0.35).astype(int)

precision, recall, thresholds = precision_recall_curve(y_true, prob)

# Recall a diversi livelli di precision fissa
target_precisions = [0.5, 0.7, 0.8, 0.9, 0.95]
print(f"{'target_precision':>16} {'recall_at':>10} {'threshold':>10}")
for tp in target_precisions:
    idx = np.searchsorted(precision, tp)
    idx = min(idx, len(precision) - 1)
    if precision[idx] >= tp and idx < len(thresholds):
        print(f"{tp:>16.2f} {recall[idx]:>10.4f} {thresholds[idx]:>10.4f}")
    else:
        print(f"{tp:>16.2f} {'N/A':>10} {'N/A':>10}")` },
    { type: 'md', content: '<h3>Esempio 6: feature importance via permutation</h3>' },
    { type: 'run', content: `import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 15000; fr = 0.02
n_f = int(n * fr); n_nf = n - n_f
feat_names = ['amount', 'hour', 'days_signup', 'num_trans_24h', 'is_new_merchant']
X = np.column_stack([
    np.concatenate([np.random.exponential(50, n_nf), np.random.exponential(300, n_f)]),
    np.concatenate([np.random.normal(14, 4, n_nf), np.random.normal(3, 4, n_f)]),
    np.concatenate([np.random.exponential(500, n_nf), np.random.exponential(30, n_f)]),
    np.random.poisson(2, n),
    np.random.rand(n) < np.concatenate([np.full(n_nf, 0.1), np.full(n_f, 0.6)]),
])
X = X.astype(float)
y = np.concatenate([np.zeros(n_nf), np.ones(n_f)])
idx = np.random.permutation(n); X = X[idx]; y = y[idx]
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

m = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(Xt, yt)
perm = permutation_importance(m, Xe, ye, n_repeats=10, random_state=0, scoring='average_precision')

print(f"{'feature':>20} {'importance':>12} {'std':>10}")
for i in np.argsort(perm.importances_mean)[::-1]:
    print(f"{feat_names[i]:>20} {perm.importances_mean[i]:>12.4f} {perm.importances_std[i]:>10.4f}")
print("\\nDelle 5 feature, quelle informative dominano correttamente (amount, hour, days_signup, is_new_merchant).")` },
  ],
  esercizi: [
    { type: 'exercise', difficulty: 'easy', title: 'Es. 19.1 — Fraud rate',
      prompt: '<p>Calcola il fraud rate in train e test set dopo uno split stratificato. Verificati che siano molto vicini. Salva in <code class="inline">rate_train, rate_test</code>.</p>',
      starter: `import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(0)
n = 10000
y = np.zeros(n)
y[:200] = 1   # 2% frodi
np.random.shuffle(y)
X = np.random.randn(n, 3)

Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

rate_train = None
rate_test = None`,
      check: `if rate_train is not None and rate_test is not None:
    if abs(rate_train - rate_test) < 0.003:
        print(f"OK — train {rate_train*100:.3f}%, test {rate_test*100:.3f}%")
    else:
        print(f"FAIL — differiscono troppo")`,
      solution: `rate_train = float(yt.mean())
rate_test = float(ye.mean())` },

    { type: 'exercise', difficulty: 'easy', title: 'Es. 19.2 — Cost calculation',
      prompt: '<p>Data una confusion matrix (TP, FP, FN, TN) e i costi $C_{FN}=500, C_{FP}=5$, calcola il costo totale. Salvalo in <code class="inline">total_cost</code>.</p>',
      starter: `TP = 100
FP = 50
FN = 20
TN = 5000
C_FN = 500
C_FP = 5

total_cost = None
print(total_cost)`,
      check: `_expected = 20 * 500 + 50 * 5
if total_cost == _expected:
    print(f"OK — {total_cost}")
else:
    print(f"FAIL — atteso {_expected}")`,
      solution: `total_cost = FN * C_FN + FP * C_FP` },

    { type: 'exercise', difficulty: 'easy', title: 'Es. 19.3 — Threshold teorico',
      prompt: '<p>Calcola il threshold cost-optimal teorico: $t^* = C_{FP}/(C_{FP}+C_{FN})$ per i costi dati. Salvalo in <code class="inline">t_opt</code>.</p>',
      starter: `C_FP = 5
C_FN = 500

t_opt = None
print(t_opt)`,
      check: `if abs(t_opt - 5/505) < 1e-9:
    print(f"OK — t* = {t_opt:.5f}")
else:
    print(f"FAIL — atteso {5/505:.5f}")`,
      solution: `t_opt = C_FP / (C_FP + C_FN)` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 19.4 — Sample weights bilanciati',
      prompt: '<p>Calcola sample_weight per bilanciare le classi: peso 1 per la maggioranza, peso (n_maj/n_min) per la minoranza. Salva l\'array in <code class="inline">weights</code>.</p>',
      starter: `import numpy as np
np.random.seed(0)
y = np.zeros(10000)
y[:200] = 1
np.random.shuffle(y)

weights = None
print("Peso classe 1:", weights[y==1][0] if weights is not None else None)`,
      check: `import numpy as np
_ratio = (y == 0).sum() / (y == 1).sum()
if weights is not None and abs(weights[y == 1][0] - _ratio) < 1e-6 and (weights[y == 0] == 1).all():
    print(f"OK — ratio {_ratio:.2f}")
else:
    print("FAIL")`,
      solution: `weights = np.where(y == 1, (y == 0).sum() / (y == 1).sum(), 1.0)` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 19.5 — GBM su fraud simulato',
      prompt: '<p>Su un dataset simulato di fraud (2% classe positiva), fitta un GBM con sample_weight bilanciato. Salva PR-AUC test in <code class="inline">pr_auc</code>. Target > 0.7.</p>',
      starter: `import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

np.random.seed(42)
n = 15000; fr = 0.02
n_f = int(n * fr); n_nf = n - n_f
X = np.column_stack([
    np.concatenate([np.random.exponential(50, n_nf), np.random.exponential(300, n_f)]),
    np.concatenate([np.random.normal(14, 4, n_nf), np.random.normal(3, 4, n_f)]),
    np.concatenate([np.random.exponential(500, n_nf), np.random.exponential(30, n_f)]),
    (np.random.rand(n) < np.concatenate([np.full(n_nf, 0.1), np.full(n_f, 0.6)])).astype(float),
])
y = np.concatenate([np.zeros(n_nf), np.ones(n_f)])
idx = np.random.permutation(n); X = X[idx]; y = y[idx]
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

pr_auc = None`,
      check: `if pr_auc is not None and pr_auc > 0.7:
    print(f"OK — PR-AUC {pr_auc:.4f}")
else:
    print(f"FAIL — PR-AUC {pr_auc}")`,
      solution: `w = np.where(yt == 1, (yt == 0).mean() / (yt == 1).mean(), 1.0)
m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(Xt, yt, sample_weight=w)
pr_auc = average_precision_score(ye, m.predict_proba(Xe)[:, 1])` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 19.6 — Threshold cost-optimal empirico',
      prompt: '<p>Data una serie di probabilità e verità, con $C_{FN}=500, C_{FP}=5$, trova il threshold che minimizza il costo totale su una griglia di 100 valori. Salvalo in <code class="inline">best_t</code>.</p>',
      starter: `import numpy as np
np.random.seed(42)
n = 3000
y_true = np.random.rand(n) < 0.02  # 2% frodi
prob = np.clip(0.02 + 0.5 * y_true + 0.15 * np.random.randn(n), 0.001, 0.999)

C_FN = 500; C_FP = 5

best_t = None`,
      check: `import numpy as np
_costs = {}
for t in np.linspace(0.001, 0.999, 100):
    pred = (prob >= t).astype(int)
    fn = ((pred == 0) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    _costs[t] = fn * 500 + fp * 5
_expected = min(_costs, key=_costs.get)
if best_t is not None and abs(best_t - _expected) < 0.02:
    print(f"OK — t* = {best_t:.4f}")
else:
    print(f"FAIL — atteso {_expected:.4f}")`,
      solution: `import numpy as np
best_cost = float('inf'); best_t = None
for t in np.linspace(0.001, 0.999, 100):
    pred = (prob >= t).astype(int)
    fn = ((pred == 0) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    c = fn * C_FN + fp * C_FP
    if c < best_cost:
        best_cost = c; best_t = t` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 19.7 — Recall @ precision',
      prompt: '<p>Con la precision_recall_curve, trova il recall al primo threshold in cui precision >= 0.9. Salva in <code class="inline">recall_at_prec_09</code>.</p>',
      starter: `import numpy as np
from sklearn.metrics import precision_recall_curve

np.random.seed(0)
n = 5000
y_true = (np.random.rand(n) < 0.05).astype(int)
prob = np.clip(0.05 + 0.4 * y_true + 0.2 * np.random.randn(n), 0.001, 0.999)

recall_at_prec_09 = None`,
      check: `import numpy as np
from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(y_true, prob)
mask = precision >= 0.9
_expected = float(recall[mask].max()) if mask.any() else 0
if recall_at_prec_09 is not None and abs(recall_at_prec_09 - _expected) < 0.02:
    print(f"OK — recall @ prec>=0.9: {recall_at_prec_09:.4f}")
else:
    print(f"FAIL — atteso {_expected:.4f}")`,
      solution: `precision, recall, _ = precision_recall_curve(y_true, prob)
mask = precision >= 0.9
recall_at_prec_09 = float(recall[mask].max()) if mask.any() else 0` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 19.8 — Feature engineering domain',
      prompt: '<p>Dati transazioni per utente, crea 3 feature aggregate per ogni utente: <code class="inline">num_trans_last_24h</code>, <code class="inline">amount_zscore</code> (rispetto alla media/std dell\'utente), <code class="inline">time_since_prev</code> (secondi dalla transazione precedente). Salva il DataFrame in <code class="inline">df_eng</code>.</p>',
      starter: `import pandas as pd
import numpy as np
np.random.seed(0)
df = pd.DataFrame({
    'user_id': np.random.choice(['u1','u2','u3'], 100),
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='2h'),
    'amount': np.random.exponential(50, 100),
}).sort_values(['user_id', 'timestamp']).reset_index(drop=True)

df_eng = None
print(df_eng)`,
      check: `import pandas as pd
if df_eng is not None:
    required = {'num_trans_last_24h', 'amount_zscore', 'time_since_prev'}
    if required.issubset(df_eng.columns):
        print(f"OK — colonne {list(df_eng.columns)}")
    else:
        print(f"FAIL — colonne mancanti: {required - set(df_eng.columns)}")`,
      solution: `df_eng = df.copy()
df_eng['time_since_prev'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
# Rolling count 24h: contiamo transazioni per utente nelle 24h precedenti
df_eng['num_trans_last_24h'] = df.groupby('user_id')['timestamp'].transform(
    lambda s: s.apply(lambda t: ((s > t - pd.Timedelta('24h')) & (s <= t)).sum()))
# z-score dell'importo per utente
grp = df.groupby('user_id')['amount']
df_eng['amount_zscore'] = (df['amount'] - grp.transform('mean')) / (grp.transform('std') + 1e-9)` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 19.9 — Pipeline completa',
      prompt: '<p>Assembla la pipeline: scaling, GBM con class_weight bilanciato, poi trova threshold cost-optimal ($C_{FN}=500, C_{FP}=5$). Restituisci in <code class="inline">total_cost_test</code> il costo totale sul test.</p>',
      starter: `import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

np.random.seed(42)
n = 15000; fr = 0.02
n_f = int(n * fr); n_nf = n - n_f
X = np.column_stack([
    np.concatenate([np.random.exponential(50, n_nf), np.random.exponential(300, n_f)]),
    np.concatenate([np.random.normal(14, 4, n_nf), np.random.normal(3, 4, n_f)]),
])
y = np.concatenate([np.zeros(n_nf), np.ones(n_f)])
idx = np.random.permutation(n); X = X[idx]; y = y[idx]
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

C_FN = 500; C_FP = 5
total_cost_test = None`,
      check: `if total_cost_test is not None and total_cost_test < 15000:
    print(f"OK — costo totale test {total_cost_test:,.0f}€")
else:
    print(f"FAIL — costo {total_cost_test}")`,
      solution: `w = np.where(yt == 1, (yt == 0).mean() / (yt == 1).mean(), 1.0)
m = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(Xt, yt, sample_weight=w)
prob = m.predict_proba(Xe)[:, 1]
best_cost = float('inf')
for t in np.linspace(0.001, 0.999, 200):
    pred = (prob >= t).astype(int)
    fn = ((pred == 0) & (ye == 1)).sum()
    fp = ((pred == 1) & (ye == 0)).sum()
    c = fn * C_FN + fp * C_FP
    if c < best_cost: best_cost = c
total_cost_test = best_cost` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 19.10 — Monitor drift dei score',
      prompt: '<p>Simula il monitoring: hai i score del modello per il mese di training e per il mese 6. Calcola PSI dei score e determina se serve retraining (PSI > 0.2). Salva <code class="inline">needs_retrain</code> boolean.</p>',
      starter: `import numpy as np
np.random.seed(0)
scores_train = np.random.beta(2, 15, 5000)     # distribuzione training
scores_month6 = np.random.beta(2, 8, 500)      # shift verso più alti = drift

def psi(reference, current, n_bins=10):
    bp = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bp[0] = -np.inf; bp[-1] = np.inf
    ref_freq, _ = np.histogram(reference, bins=bp)
    cur_freq, _ = np.histogram(current, bins=bp)
    ref_pct = np.clip(ref_freq / len(reference), 1e-6, None)
    cur_pct = np.clip(cur_freq / len(current), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

needs_retrain = None`,
      check: `if needs_retrain is True:
    print(f"OK — drift severo, serve retraining")
else:
    print(f"FAIL — {needs_retrain}")`,
      solution: `p = psi(scores_train, scores_month6)
needs_retrain = bool(p > 0.2)
print(f"PSI = {p:.4f}")` },
  ],
  test: [
    { type: 'quiz', questions: [
      { q: 'Con fraud rate 1%, la metrica principale da guardare è:', options: ['Accuracy', 'PR-AUC o cost-based metric', 'Precision', 'ROC-AUC'], answer: 1, explain: 'Accuracy fuorviante, PR-AUC sensibile al rare positive, cost-based riflette il vero valore business.' },
      { q: 'Il threshold ottimale con $C_{FN}=500, C_{FP}=5$ è:', options: ['0.5', '$C_{FP}/(C_{FP}+C_{FN}) \\approx 0.01$', '$C_{FN}/C_{FP} = 100$', '0.1'], answer: 1, explain: 'Con costo FN molto più alto, meglio bloccare anche a bassa probabilità.' },
      { q: 'Fare SMOTE prima dello split:', options: ['È best practice', 'Data leakage grave — esempi sintetici finiscono in train e test', 'Aumenta velocità', 'Nulla di male'], answer: 1, explain: 'SMOTE genera esempi correlati; se stanno anche nel test, il modello li vede due volte.' },
      { q: 'Recall @ precision 90% è utile per:', options: ['Nulla di particolare', 'Rispondere alla domanda business "se accettiamo max 10% falsi positivi, che frazione di frodi cattureremo?"', 'Confronto tra modelli', 'Solo con classi bilanciate'], answer: 1, explain: 'Metrica direttamente interpretabile a operations.' },
      { q: 'Un fraud model degrada nel tempo per:', options: ['I dati aumentano', 'Concept drift + i frodatori si adattano (adversarial)', 'Solo per bug', 'La rete rallenta'], answer: 1, explain: 'Il fraud è un problema adversarial: pattern cambiano perché i criminali evolvono.' },
    ] },
    { type: 'exercise', difficulty: 'hard', title: 'Test finale — end-to-end pipeline',
      prompt: '<p>Costruisci end-to-end una pipeline di fraud detection: 1) fit GBM con sample_weight, 2) trova threshold cost-optimal, 3) valuta sul test con PR-AUC e costo totale. Salva <code class="inline">pr_auc</code>, <code class="inline">total_cost</code>, <code class="inline">best_threshold</code>. Target: pr_auc > 0.7, total_cost < 5000.</p>',
      starter: `import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

np.random.seed(42)
n = 20000; fr = 0.02
n_f = int(n * fr); n_nf = n - n_f
X = np.column_stack([
    np.concatenate([np.random.exponential(50, n_nf), np.random.exponential(300, n_f)]),
    np.concatenate([np.random.normal(14, 4, n_nf), np.random.normal(3, 4, n_f)]),
    np.concatenate([np.random.exponential(500, n_nf), np.random.exponential(30, n_f)]),
    (np.random.rand(n) < np.concatenate([np.full(n_nf, 0.1), np.full(n_f, 0.6)])).astype(float),
])
y = np.concatenate([np.zeros(n_nf), np.ones(n_f)])
idx = np.random.permutation(n); X = X[idx]; y = y[idx]
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

C_FN = 500; C_FP = 5
pr_auc = None
total_cost = None
best_threshold = None`,
      check: `if pr_auc is None or total_cost is None:
    print("FAIL: None")
elif pr_auc > 0.7 and total_cost < 5000:
    print(f"OK — PR-AUC {pr_auc:.4f}, cost {total_cost:,}€, threshold {best_threshold:.4f}")
else:
    print(f"FAIL — PR-AUC {pr_auc:.4f}, cost {total_cost}")`,
      solution: `w = np.where(yt == 1, (yt == 0).mean() / (yt == 1).mean(), 1.0)
m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=0).fit(Xt, yt, sample_weight=w)
prob = m.predict_proba(Xe)[:, 1]
pr_auc = average_precision_score(ye, prob)
best_cost = float('inf'); best_threshold = None
for t in np.linspace(0.001, 0.999, 200):
    pred = (prob >= t).astype(int)
    fn = ((pred == 0) & (ye == 1)).sum()
    fp = ((pred == 1) & (ye == 0)).sum()
    c = fn * C_FN + fp * C_FP
    if c < best_cost:
        best_cost = c; best_threshold = t
total_cost = best_cost` },

    { type: 'exercise', difficulty: 'hard', title: 'Bonus expert — ROI del modello',
      prompt: '<p><strong>Sfida business</strong>. Confronta il "sistema attuale" (che blocca solo transazioni > 500€) con il tuo modello ML. Calcola il RISPARMIO in €. Il modello serve un\'azienda con 100k transazioni/mese al fraud rate del 2%. Salva <code class="inline">savings_per_month</code> in €.</p>',
      starter: `import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

np.random.seed(42)
n = 20000; fr = 0.02
n_f = int(n * fr); n_nf = n - n_f
X = np.column_stack([
    np.concatenate([np.random.exponential(50, n_nf), np.random.exponential(300, n_f)]),
    np.concatenate([np.random.normal(14, 4, n_nf), np.random.normal(3, 4, n_f)]),
])
y = np.concatenate([np.zeros(n_nf), np.ones(n_f)])
amounts = X[:, 0].copy()
idx = np.random.permutation(n); X = X[idx]; y = y[idx]; amounts = amounts[idx]
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
_, amts_e = train_test_split(amounts, test_size=0.3, random_state=42, stratify=y)

# Sistema attuale: blocca se amount > 500
system_pred = (amts_e > 500).astype(int)

C_FN = 500; C_FP = 5

savings_per_month = None`,
      check: `if savings_per_month is not None and savings_per_month > 0:
    print(f"OK — risparmio stimato {savings_per_month:,.0f}€/mese")
else:
    print(f"FAIL — {savings_per_month}")`,
      solution: `import numpy as np
w = np.where(yt == 1, (yt == 0).mean() / (yt == 1).mean(), 1.0)
m = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(Xt, yt, sample_weight=w)
prob = m.predict_proba(Xe)[:, 1]

# Cost del sistema attuale
fn_sys = ((system_pred == 0) & (ye == 1)).sum()
fp_sys = ((system_pred == 1) & (ye == 0)).sum()
cost_sys = fn_sys * C_FN + fp_sys * C_FP

# Cost del ML con threshold ottimo
best_cost_ml = float('inf')
for t in np.linspace(0.001, 0.999, 100):
    pred = (prob >= t).astype(int)
    fn = ((pred == 0) & (ye == 1)).sum()
    fp = ((pred == 1) & (ye == 0)).sum()
    c = fn * C_FN + fp * C_FP
    if c < best_cost_ml: best_cost_ml = c

# Scaling: dal test set (~6000) a 100000/mese
scaling = 100000 / len(ye)
savings_per_month = (cost_sys - best_cost_ml) * scaling` },
  ]
});
