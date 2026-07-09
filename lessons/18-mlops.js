/* Lezione: MLOps e Deployment */
window.LESSONS.push({
  id: 'l18-mlops',
  title: 'MLOps e Deployment',
  tag: 'Produzione',
  summary: 'Come portare un modello in produzione. MLflow, FastAPI, Docker, monitoring, data drift. La differenza tra "notebook che gira" e "sistema affidabile".',
  teoria: [
    { type: 'md', content: `
<h3>18.1 Perché MLOps esiste</h3>
<p>Un modello che gira su un notebook Jupyter è a metà del lavoro. Per essere davvero utile a un\'azienda deve:</p>
<ul>
<li>Essere <strong>riproducibile</strong>: chi lo eredita deve poter rigenerare gli stessi risultati.</li>
<li>Essere <strong>servito</strong>: risponde a richieste HTTP con predizioni in millisecondi.</li>
<li>Essere <strong>monitorato</strong>: sappiamo se sta funzionando bene o degradando.</li>
<li>Essere <strong>aggiornabile</strong>: retrain automatico o su richiesta, senza downtime.</li>
<li>Essere <strong>affidabile</strong>: se crasha, si riavvia. Se dà errori, li logga.</li>
</ul>
<p>MLOps è l\'insieme di tecniche, tool e best practice per rispondere a questi problemi. È dove si separa il "data scientist junior" dal "ML engineer" — la differenza vale letteralmente il 50% del compenso.</p>

<h3>18.2 Il ciclo di vita di un modello in produzione</h3>
<ol>
<li><strong>Data collection</strong>: raccolta dati, versioning, quality checks.</li>
<li><strong>EDA e feature engineering</strong>: notebook esplorativi.</li>
<li><strong>Sperimentazione</strong>: prova modelli, iperparametri. Tracciata con MLflow o WandB.</li>
<li><strong>Training pipeline</strong>: automatizzato, riproducibile. Airflow, Kubeflow, Prefect.</li>
<li><strong>Model registry</strong>: dove versionare i modelli con metadata.</li>
<li><strong>Serving</strong>: API REST/gRPC per predizioni. FastAPI, TorchServe, BentoML, Sagemaker.</li>
<li><strong>Monitoring</strong>: metriche di sistema (latenza, errori) e di modello (accuracy, drift).</li>
<li><strong>Retraining</strong>: automatico se drift o degradation, o su calendario.</li>
</ol>

<h3>18.3 Experiment tracking con MLflow</h3>
<p>MLflow è lo standard open-source per tracciare esperimenti. Ti permette di loggare:</p>
<ul>
<li><strong>Parameters</strong>: iperparametri usati (lr, depth, ecc.)</li>
<li><strong>Metrics</strong>: risultati (accuracy, AUC, loss)</li>
<li><strong>Artifacts</strong>: modelli serializzati, grafici, dataset</li>
<li><strong>Code version</strong>: commit git di quando l\'esperimento è girato</li>
</ul>
<pre class="code">import mlflow
import mlflow.sklearn

mlflow.set_experiment('customer_churn')
with mlflow.start_run(run_name='xgb_v3'):
    mlflow.log_param('learning_rate', 0.05)
    mlflow.log_param('max_depth', 6)
    mlflow.log_param('n_estimators', 500)
    # ... training ...
    mlflow.log_metric('roc_auc', 0.87)
    mlflow.log_metric('pr_auc', 0.72)
    mlflow.sklearn.log_model(model, 'model')</pre>
<p>UI locale con <code class="inline">mlflow ui</code> (default sulla porta 5000): confronti fianco a fianco tutti gli esperimenti.</p>

<h3>18.4 Model serialization</h3>
<p>Salvare il modello per riusarlo. Formati principali:</p>
<ul>
<li><strong>joblib</strong> (default per sklearn): efficiente per numpy arrays.</li>
<li><strong>pickle</strong>: standard Python ma non sicuro (arbitrary code execution).</li>
<li><strong>ONNX</strong>: formato aperto, cross-framework (sklearn, PyTorch, TensorFlow), ottimizzato per inference.</li>
<li><strong>PyTorch .pt</strong>: state_dict o modello completo per PyTorch.</li>
<li><strong>SafeTensors</strong>: alternativa sicura al pickle per Transformers.</li>
</ul>
<pre class="code">import joblib
joblib.dump(pipeline, 'model.joblib')
loaded = joblib.load('model.joblib')

# ONNX per portabilità
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
onnx = convert_sklearn(model, initial_types=[('input', FloatTensorType([None, n_features]))])
with open('model.onnx', 'wb') as f: f.write(onnx.SerializeToString())</pre>

<h3>18.5 Serving con FastAPI</h3>
<p>FastAPI è il framework Python più moderno per API. Async, validazione automatica dei tipi con Pydantic, documentazione OpenAPI generata automaticamente.</p>
<pre class="code"># app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title='ML Model Server')
model = joblib.load('model.joblib')

class Input(BaseModel):
    features: list[float]

class Prediction(BaseModel):
    probability: float
    prediction: int

@app.post('/predict', response_model=Prediction)
def predict(inp: Input):
    x = np.array(inp.features).reshape(1, -1)
    prob = model.predict_proba(x)[0, 1]
    return Prediction(probability=float(prob), prediction=int(prob > 0.5))

@app.get('/health')
def health():
    return {'status': 'ok', 'model_version': '1.2.3'}

# Lancio: uvicorn app:app --host 0.0.0.0 --port 8000
# Docs interattive: http://localhost:8000/docs</pre>

<h3>18.6 Containerizzazione con Docker</h3>
<p>Docker impacchetta l\'app + dipendenze + sistema in un container isolato. Girerà identico ovunque (tuo laptop, server AWS, cluster Kubernetes).</p>
<pre class="code"># Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py model.joblib ./

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]</pre>
<pre class="code"># requirements.txt
fastapi==0.115.0
uvicorn==0.32.0
scikit-learn==1.5.2
joblib==1.4.2
numpy==1.26.4
pydantic==2.9.2</pre>
<pre class="code"># Comandi
docker build -t ml-app .
docker run -p 8000:8000 ml-app</pre>

<h3>18.7 Deployment cloud</h3>
<p>Opzioni comuni (in ordine di semplicità → sofisticazione):</p>
<ul>
<li><strong>Hugging Face Spaces</strong>: gratis, un click per deployare app Gradio/Streamlit/Docker.</li>
<li><strong>Render, Fly.io, Railway</strong>: piattaforme PaaS semplici. Piani free per esperimenti.</li>
<li><strong>Google Cloud Run / AWS App Runner</strong>: serverless containers, scala automaticamente a zero.</li>
<li><strong>Kubernetes (AWS EKS, GKE, AKS)</strong>: massima flessibilità, tanto lavoro DevOps.</li>
<li><strong>Sagemaker, Vertex AI, Azure ML</strong>: platform-specific ML deployment.</li>
</ul>

<h3>18.8 CI/CD per modelli ML</h3>
<p>Come CI/CD per software normale, ma con specificità ML:</p>
<ul>
<li>Test unitari del codice (pytest).</li>
<li>Test di validazione dei dati (Great Expectations, Pandera).</li>
<li>Test di regressione del modello (sanity checks: accuratezza minima su un test set fisso).</li>
<li>Build container.</li>
<li>Deploy staging + smoke test.</li>
<li>Deploy production con blue/green o canary rollout.</li>
</ul>
<pre class="code"># .github/workflows/deploy.yml
name: Deploy Model
on:
  push:
    branches: [main]
jobs:
  test-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -r requirements.txt
      - run: pytest tests/
      - run: python train.py    # (o pulls artefact da model registry)
      - run: docker build -t ml-app .
      - run: docker push ...    # a un registry
      - run: kubectl apply -f k8s/</pre>

<h3>18.9 Monitoring in produzione</h3>
<p>Un modello degrada nel tempo per <em>data drift</em> (distribuzione degli input cambia) o <em>concept drift</em> (relazione input-output cambia). Non accorgersene è pericolosissimo.</p>
<p>Cosa monitorare:</p>
<ul>
<li><strong>Metriche di sistema</strong>: latency (p50, p95, p99), throughput, error rate, memory/CPU.</li>
<li><strong>Metriche di modello</strong>: distribuzione delle predizioni, confidence, tasso di edge cases.</li>
<li><strong>Metriche di dati</strong>: distribuzione delle feature (KS test, PSI - Population Stability Index).</li>
<li><strong>Metriche di outcome (se disponibili)</strong>: accuracy vs truth quando arriva il feedback (a volte con ritardo).</li>
</ul>
<p>Tool: Prometheus + Grafana per sistema, Evidently AI o WhyLabs per drift, Sentry per errori.</p>

<h3>18.10 A/B testing e canary deployment</h3>
<p>Non deployare un nuovo modello sostituendo istantaneamente il vecchio: rischi troppo alti se qualcosa va male.</p>
<ul>
<li><strong>Canary</strong>: nuovo modello serve il 5% del traffico, monitori, poi 20%, 50%, 100%.</li>
<li><strong>A/B test</strong>: metà utenti vecchio modello, metà nuovo. Confronto rigoroso via metriche di business.</li>
<li><strong>Shadow deployment</strong>: nuovo modello riceve richieste ma le sue predizioni non tornano all\'utente, servono solo per confronto.</li>
</ul>

<h3>18.11 Feature stores</h3>
<p>Problema: training e inference devono usare le STESSE feature calcolate nello stesso modo. Feature stores (Feast, Tecton, Databricks) centralizzano definizioni e computazione, prevenendo il "training-serving skew".</p>
` },
    { type: 'callout', variant: 'note', title: 'Realismo pratico', content: 'Il codice di questa lezione richiede installazione locale di FastAPI, MLflow, Docker. Non gira in Pyodide. Ma è codice reale che userai davvero in produzione: salvalo come reference, prova gli esempi in locale quando puoi. Anche solo scrivere un Dockerfile per il tuo modello sklearn è un enorme step avanti per un CV junior ML.' },
  ],
  esempi: [
    { type: 'md', content: '<h3>Esempio 1: model persistence con joblib (Pyodide OK)</h3>' },
    { type: 'run', content: `import joblib
import io
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=50, random_state=0).fit(Xt, yt)

# Serializziamo in memoria (in produzione useresti un file)
buf = io.BytesIO()
joblib.dump(model, buf)
buf.seek(0)

# Caricamento
model_loaded = joblib.load(buf)

# Verifica: predizioni identiche
import numpy as np
p1 = model.predict(Xe)
p2 = model_loaded.predict(Xe)
print(f"Predizioni identiche: {np.array_equal(p1, p2)}")
print(f"Accuracy: {model_loaded.score(Xe, ye):.4f}")
print(f"Dimensione modello serializzato: {len(buf.getvalue())} bytes")` },
    { type: 'md', content: '<h3>Esempio 2: simulazione tracking manuale (senza MLflow installato)</h3>' },
    { type: 'run', content: `# Simuliamo il tracciamento di esperimenti come farebbe MLflow
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

X, y = load_breast_cancer(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

experiments = []

# Esperimento 1: Logistic
p = Pipeline([('s', StandardScaler()), ('m', LogisticRegression(C=1.0, max_iter=5000))]).fit(Xt, yt)
experiments.append({'name': 'logistic_C1', 'params': {'C': 1.0}, 
                    'metrics': {'test_acc': p.score(Xe, ye)}})

# Esperimento 2: Logistic con C diverso
p = Pipeline([('s', StandardScaler()), ('m', LogisticRegression(C=0.1, max_iter=5000))]).fit(Xt, yt)
experiments.append({'name': 'logistic_C01', 'params': {'C': 0.1},
                    'metrics': {'test_acc': p.score(Xe, ye)}})

# Esperimento 3: Random Forest
p = RandomForestClassifier(n_estimators=100, random_state=0).fit(Xt, yt)
experiments.append({'name': 'random_forest', 'params': {'n_estimators': 100},
                    'metrics': {'test_acc': p.score(Xe, ye)}})

# Esperimento 4: GBM
p = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(Xt, yt)
experiments.append({'name': 'histgb', 'params': {'max_iter': 300},
                    'metrics': {'test_acc': p.score(Xe, ye)}})

# Report
print(f"{'nome':<20} {'params':<30} {'test_acc':>10}")
print('-' * 65)
for e in experiments:
    print(f"{e['name']:<20} {str(e['params']):<30} {e['metrics']['test_acc']:>10.4f}")

best = max(experiments, key=lambda e: e['metrics']['test_acc'])
print(f"\\nMigliore: {best['name']} con accuracy {best['metrics']['test_acc']:.4f}")` },
    { type: 'md', content: '<h3>Esempio 3: MLflow codice reale (da eseguire in locale)</h3>' },
    { type: 'code', content: `# Esegui in locale: pip install mlflow
# Poi avvia UI: mlflow ui
# Naviga a http://localhost:5000

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

mlflow.set_experiment('breast_cancer')

for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    with mlflow.start_run(run_name=f'logistic_C{C}'):
        pipe = Pipeline([('s', StandardScaler()),
                         ('lr', LogisticRegression(C=C, max_iter=5000))]).fit(Xt, yt)
        
        # Log iperparametri
        mlflow.log_param('C', C)
        mlflow.log_param('solver', 'lbfgs')
        mlflow.log_param('model_type', 'LogisticRegression')
        
        # Log metriche
        train_acc = pipe.score(Xt, yt)
        test_acc = pipe.score(Xe, ye)
        cv_scores = cross_val_score(pipe, Xt, yt, cv=5, scoring='roc_auc')
        mlflow.log_metric('train_acc', train_acc)
        mlflow.log_metric('test_acc', test_acc)
        mlflow.log_metric('cv_roc_auc_mean', cv_scores.mean())
        mlflow.log_metric('cv_roc_auc_std', cv_scores.std())
        
        # Log del modello (registrato con schema di input)
        mlflow.sklearn.log_model(pipe, 'model')
        
        print(f"C={C}: test_acc={test_acc:.4f}")

# Nella UI puoi confrontare visualmente tutti i run, filtrare, ordinare, ecc.
` },
    { type: 'md', content: '<h3>Esempio 4: server FastAPI (da eseguire in locale)</h3>' },
    { type: 'code', content: `# app.py
# Esegui in locale: pip install fastapi uvicorn scikit-learn joblib
# Lancio: uvicorn app:app --reload --host 0.0.0.0 --port 8000
# Documentazione interattiva: http://localhost:8000/docs

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import joblib
import numpy as np
from typing import List

app = FastAPI(
    title='Breast Cancer Classifier',
    description='API di classificazione per il dataset breast cancer',
    version='1.0.0',
)

# Carica il modello all'avvio (una sola volta)
try:
    model = joblib.load('model.joblib')
    print('Model loaded successfully')
except Exception as e:
    print(f'Errore caricamento modello: {e}')
    model = None

class PredictionInput(BaseModel):
    features: List[float] = Field(..., min_length=30, max_length=30,
                                    description='30 feature del dataset breast cancer')
    
    @field_validator('features')
    def check_no_nan(cls, v):
        if any(np.isnan(x) or np.isinf(x) for x in v):
            raise ValueError('Le feature non possono contenere NaN o infiniti')
        return v

class PredictionOutput(BaseModel):
    probability_malignant: float = Field(..., ge=0, le=1)
    prediction: str  # 'benign' | 'malignant'
    confidence: str  # 'low' | 'medium' | 'high'

@app.get('/')
def root():
    return {'message': 'ML API', 'docs': '/docs'}

@app.get('/health')
def health():
    return {
        'status': 'ok' if model is not None else 'degraded',
        'model_loaded': model is not None,
        'model_type': type(model).__name__ if model else None,
    }

@app.post('/predict', response_model=PredictionOutput)
def predict(inp: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    try:
        x = np.array(inp.features).reshape(1, -1)
        prob = float(model.predict_proba(x)[0, 1])
        pred = 'malignant' if prob > 0.5 else 'benign'
        # Confidence based on distance from 0.5
        distance = abs(prob - 0.5)
        confidence = 'high' if distance > 0.4 else ('medium' if distance > 0.2 else 'low')
        return PredictionOutput(
            probability_malignant=prob,
            prediction=pred,
            confidence=confidence,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/batch_predict')
def batch_predict(inputs: List[PredictionInput]):
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    X = np.array([inp.features for inp in inputs])
    probs = model.predict_proba(X)[:, 1]
    return {'predictions': [{'idx': i, 'prob': float(p)} for i, p in enumerate(probs)]}
` },
    { type: 'md', content: '<h3>Esempio 5: Dockerfile completo per il servizio</h3>' },
    { type: 'code', content: `# Dockerfile
# Multi-stage build per ridurre dimensione finale

# Stage 1: build dependencies
FROM python:3.11-slim as builder

WORKDIR /build
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Stage 2: runtime
FROM python:3.11-slim

RUN groupadd -r mlapp && useradd -r -g mlapp mlapp

WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl \\
    && rm -rf /wheels

COPY --chown=mlapp:mlapp app.py model.joblib ./

USER mlapp

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# ---
# requirements.txt
# fastapi==0.115.0
# uvicorn[standard]==0.32.0
# scikit-learn==1.5.2
# joblib==1.4.2
# numpy==1.26.4
# pydantic==2.9.2

# Build e run:
# docker build -t breast-cancer-api .
# docker run -p 8000:8000 breast-cancer-api
` },
    { type: 'md', content: '<h3>Esempio 6: detect data drift con KS test (Pyodide OK)</h3>' },
    { type: 'run', content: `import numpy as np
from scipy.stats import ks_2samp

# Simula training data
np.random.seed(0)
X_train = np.random.normal(loc=50, scale=10, size=(1000, 5))

# Simula production data — mese 1 (no drift)
X_prod_m1 = np.random.normal(loc=50, scale=10, size=(500, 5))

# Simula production data — mese 3 (drift: media shifted)
X_prod_m3 = np.random.normal(loc=55, scale=10, size=(500, 5))  # +5 nella media

# Simula production data — mese 6 (drift severo)
X_prod_m6 = np.random.normal(loc=60, scale=15, size=(500, 5))

def detect_drift(reference, current, alpha=0.05):
    drifts = []
    for i in range(reference.shape[1]):
        stat, pval = ks_2samp(reference[:, i], current[:, i])
        drifts.append({'feature': i, 'ks_stat': stat, 'p_value': pval, 'drift': pval < alpha})
    return drifts

print("=== Mese 1 (no drift) ===")
for d in detect_drift(X_train, X_prod_m1):
    print(f"  Feature {d['feature']}: KS={d['ks_stat']:.3f}, p={d['p_value']:.4f}, drift={d['drift']}")

print("\\n=== Mese 3 (drift lieve) ===")
for d in detect_drift(X_train, X_prod_m3):
    print(f"  Feature {d['feature']}: KS={d['ks_stat']:.3f}, p={d['p_value']:.4f}, drift={d['drift']}")

print("\\n=== Mese 6 (drift severo) ===")
n_drifted = sum(1 for d in detect_drift(X_train, X_prod_m6) if d['drift'])
print(f"  {n_drifted}/5 feature con drift significativo → serve retraining")` },
    { type: 'md', content: '<h3>Esempio 7: Population Stability Index (PSI)</h3>' },
    { type: 'run', content: `import numpy as np

def psi(reference, current, n_bins=10):
    """Population Stability Index: <0.1 stabile, 0.1-0.2 lieve, >0.2 significativo."""
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    ref_freq, _ = np.histogram(reference, bins=breakpoints)
    cur_freq, _ = np.histogram(current, bins=breakpoints)
    ref_pct = np.clip(ref_freq / len(reference), 1e-6, None)
    cur_pct = np.clip(cur_freq / len(current), 1e-6, None)
    return np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

np.random.seed(0)
ref = np.random.normal(50, 10, 5000)
for name, cur in [('same', np.random.normal(50, 10, 1000)),
                    ('slight shift', np.random.normal(53, 10, 1000)),
                    ('major shift', np.random.normal(60, 15, 1000))]:
    p = psi(ref, cur)
    verdict = 'stabile' if p < 0.1 else ('lieve' if p < 0.2 else 'significativo')
    print(f"{name:20s}: PSI={p:.4f} ({verdict})")` },
  ],
  esercizi: [
    { type: 'exercise', difficulty: 'easy', title: 'Es. 18.1 — Serialization roundtrip',
      prompt: '<p>Addestra un modello, salvalo con joblib in un BytesIO buffer, caricalo, verifica che le predizioni siano identiche. Salva in <code class="inline">predictions_match</code> un boolean.</p>',
      starter: `import io
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)

# TODO: serializza -> deserializza -> confronta
predictions_match = None`,
      check: `if predictions_match is True:
    print("OK — modello serializzato e caricato correttamente")
else:
    print(f"FAIL — {predictions_match}")`,
      solution: `buf = io.BytesIO()
joblib.dump(model, buf)
buf.seek(0)
loaded = joblib.load(buf)
predictions_match = bool(np.array_equal(model.predict(X), loaded.predict(X)))` },

    { type: 'exercise', difficulty: 'easy', title: 'Es. 18.2 — Experiment tracking manuale',
      prompt: '<p>Simula il tracking di 3 esperimenti in una lista di dizionari, ognuno con name, params (dict), metrics (dict). Trova quello con la migliore metrica <code class="inline">roc_auc</code>. Salva il nome del migliore in <code class="inline">best_name</code>.</p>',
      starter: `experiments = [
    {'name': 'exp1', 'params': {'C': 0.1}, 'metrics': {'roc_auc': 0.85, 'acc': 0.82}},
    {'name': 'exp2', 'params': {'C': 1.0}, 'metrics': {'roc_auc': 0.91, 'acc': 0.88}},
    {'name': 'exp3', 'params': {'C': 10}, 'metrics': {'roc_auc': 0.89, 'acc': 0.87}},
]

best_name = None
print(best_name)`,
      check: `if best_name == 'exp2':
    print(f"OK — {best_name}")
else:
    print(f"FAIL — atteso 'exp2', tuo '{best_name}'")`,
      solution: `best_name = max(experiments, key=lambda e: e['metrics']['roc_auc'])['name']` },

    { type: 'exercise', difficulty: 'easy', title: 'Es. 18.3 — Validazione input',
      prompt: '<p>Scrivi <code class="inline">validate_features(features, n_expected)</code> che ritorna True se features è una lista di n_expected numeri validi (no NaN, no infiniti), altrimenti False.</p>',
      starter: `import math
def validate_features(features, n_expected):
    return None

print(validate_features([1.0, 2.0, 3.0], 3))       # True
print(validate_features([1.0, 2.0], 3))            # False (len)
print(validate_features([1.0, float('nan'), 3], 3)) # False (NaN)
print(validate_features([1.0, 2, 3.0], 3))         # True (int è ok)`,
      check: `import math
tests = [
    ([1.0, 2.0, 3.0], 3, True),
    ([1.0, 2.0], 3, False),
    ([1.0, float('nan'), 3.0], 3, False),
    ([1.0, float('inf'), 3.0], 3, False),
    ([1.0, 2, 3.0], 3, True),
]
ok = all(validate_features(f, n) == e for f, n, e in tests)
print("OK" if ok else "FAIL")`,
      solution: `def validate_features(features, n_expected):
    if not isinstance(features, list) or len(features) != n_expected:
        return False
    for x in features:
        if not isinstance(x, (int, float)):
            return False
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return False
    return True` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 18.4 — Semplice registry',
      prompt: '<p>Implementa <code class="inline">ModelRegistry</code> con metodi register(name, version, model_dict), get_latest(name), list_versions(name). Ogni model_dict è un dizionario con metadata. Salva l\'istanza in <code class="inline">registry</code>.</p>',
      starter: `class ModelRegistry:
    def __init__(self):
        # TODO
        pass
    def register(self, name, version, model_dict):
        pass
    def get_latest(self, name):
        pass
    def list_versions(self, name):
        pass

registry = ModelRegistry()
registry.register('churn', 'v1', {'acc': 0.85})
registry.register('churn', 'v2', {'acc': 0.87})
print(registry.get_latest('churn'))
print(registry.list_versions('churn'))`,
      check: `latest = registry.get_latest('churn')
versions = registry.list_versions('churn')
if latest == {'acc': 0.87} and sorted(versions) == ['v1', 'v2']:
    print("OK")
else:
    print(f"FAIL — latest {latest}, versions {versions}")`,
      solution: `class ModelRegistry:
    def __init__(self):
        self.store = {}
    def register(self, name, version, model_dict):
        self.store.setdefault(name, {})[version] = model_dict
    def get_latest(self, name):
        return self.store[name][sorted(self.store[name].keys())[-1]]
    def list_versions(self, name):
        return sorted(self.store[name].keys())` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 18.5 — KS test per drift',
      prompt: '<p>Usa <code class="inline">scipy.stats.ks_2samp</code> per confrontare due distribuzioni. Salva un boolean <code class="inline">drifted</code> = True se p_value < 0.05.</p>',
      starter: `import numpy as np
from scipy.stats import ks_2samp
np.random.seed(0)
ref = np.random.normal(0, 1, 500)
cur = np.random.normal(0.5, 1, 500)

drifted = None`,
      check: `if drifted is True:
    print(f"OK — drift rilevato correttamente")
else:
    print(f"FAIL — {drifted}")`,
      solution: `stat, pval = ks_2samp(ref, cur)
drifted = bool(pval < 0.05)` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 18.6 — PSI',
      prompt: '<p>Implementa Population Stability Index. Formula: $PSI = \\sum (p_i - q_i) \\log(p_i/q_i)$ su bin. Con n_bins=10 e i breakpoint dai quantili di reference.</p>',
      starter: `import numpy as np
def psi(reference, current, n_bins=10):
    return None

np.random.seed(0)
ref = np.random.normal(0, 1, 1000)
cur = np.random.normal(0.3, 1, 1000)
print("PSI:", psi(ref, cur))`,
      check: `import numpy as np
np.random.seed(0)
ref = np.random.normal(0, 1, 1000)
cur = np.random.normal(0.3, 1, 1000)
p = psi(ref, cur)
if p is not None and 0.05 < p < 0.5:
    print(f"OK — PSI = {p:.4f} (shift moderato)")
else:
    print(f"FAIL — PSI = {p}")`,
      solution: `def psi(reference, current, n_bins=10):
    bp = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bp[0] = -np.inf; bp[-1] = np.inf
    ref_freq, _ = np.histogram(reference, bins=bp)
    cur_freq, _ = np.histogram(current, bins=bp)
    ref_pct = np.clip(ref_freq / len(reference), 1e-6, None)
    cur_pct = np.clip(cur_freq / len(current), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 18.7 — Sistema di logging',
      prompt: '<p>Crea <code class="inline">PredictionLogger</code> che accumula un log di predizioni (dict con timestamp, features, prediction, probability). Fornisce <code class="inline">get_stats()</code> che ritorna dict con n_predictions, mean_probability. Testalo con almeno 3 predizioni.</p>',
      starter: `import time
class PredictionLogger:
    def __init__(self):
        # TODO
        pass
    def log(self, features, prediction, probability):
        pass
    def get_stats(self):
        pass

logger = PredictionLogger()
logger.log([1.0, 2.0], 1, 0.8)
logger.log([1.5, 2.5], 0, 0.3)
logger.log([2.0, 3.0], 1, 0.9)
print(logger.get_stats())`,
      check: `stats = logger.get_stats()
if stats and stats.get('n_predictions') == 3 and abs(stats.get('mean_probability', 0) - 0.6666) < 0.01:
    print(f"OK — {stats}")
else:
    print(f"FAIL — {stats}")`,
      solution: `class PredictionLogger:
    def __init__(self):
        self.logs = []
    def log(self, features, prediction, probability):
        self.logs.append({
            'timestamp': time.time(),
            'features': features,
            'prediction': prediction,
            'probability': probability,
        })
    def get_stats(self):
        n = len(self.logs)
        if n == 0: return {'n_predictions': 0}
        mean_prob = sum(l['probability'] for l in self.logs) / n
        return {'n_predictions': n, 'mean_probability': mean_prob}` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 18.8 — Canary evaluation',
      prompt: '<p>Simula un canary deployment: hai vecchio modello (v1, accuracy 0.85) e nuovo (v2, accuracy 0.87). Su 1000 richieste, ruta il 10% al v2. Simula predizioni e calcola l\'accuracy osservata di ciascuno. Salva in <code class="inline">acc_v1, acc_v2</code>.</p>',
      starter: `import numpy as np
np.random.seed(0)
n = 1000
canary_pct = 0.1

# Simula "verità": chi va al canary?
routes = np.random.rand(n) < canary_pct
y_true = np.random.randint(0, 2, n)

# Predizioni simulate: v1 giusta 85% delle volte, v2 87%
pred_v1 = np.where(np.random.rand(n) < 0.85, y_true, 1 - y_true)
pred_v2 = np.where(np.random.rand(n) < 0.87, y_true, 1 - y_true)

# Aggregate predictions come farebbe il router
final_pred = np.where(routes, pred_v2, pred_v1)

acc_v1 = None  # accuracy solo sui non-canary
acc_v2 = None  # accuracy solo sui canary`,
      check: `if acc_v1 is not None and acc_v2 is not None:
    if abs(acc_v1 - 0.85) < 0.05 and abs(acc_v2 - 0.87) < 0.06:
        print(f"OK — v1 acc {acc_v1:.4f}, v2 acc {acc_v2:.4f}")
    else:
        print(f"WARN — {acc_v1:.4f}, {acc_v2:.4f}")`,
      solution: `import numpy as np
acc_v1 = float((pred_v1[~routes] == y_true[~routes]).mean())
acc_v2 = float((pred_v2[routes] == y_true[routes]).mean())` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 18.9 — Rate limiter per API',
      prompt: '<p>Implementa <code class="inline">RateLimiter(max_per_minute=60)</code> con metodo <code class="inline">allow(client_id)</code> che ritorna True se il client può fare una nuova richiesta (non ha superato max_per_minute), False altrimenti. Usa un dict {client_id: list_of_timestamps}.</p>',
      starter: `import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_per_minute=60):
        # TODO
        pass
    def allow(self, client_id):
        pass

limiter = RateLimiter(max_per_minute=3)
results = [limiter.allow('user_A') for _ in range(5)]
print("Prime 5 richieste user_A:", results)`,
      check: `# 3 dovrebbero passare, le altre no (limite 3/min)
if results[:3] == [True, True, True] and results[3:] == [False, False]:
    print("OK")
else:
    print(f"FAIL — {results}")`,
      solution: `class RateLimiter:
    def __init__(self, max_per_minute=60):
        self.max = max_per_minute
        self.window = 60
        self.log = defaultdict(list)
    def allow(self, client_id):
        now = time.time()
        # Purge old
        self.log[client_id] = [t for t in self.log[client_id] if now - t < self.window]
        if len(self.log[client_id]) < self.max:
            self.log[client_id].append(now)
            return True
        return False` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 18.10 — Feature validation schema',
      prompt: '<p>Implementa <code class="inline">FeatureSchema(feature_names, min_vals, max_vals)</code> con metodo <code class="inline">validate(features_dict)</code> che ritorna una lista di errori (stringhe) o [] se tutto ok. Deve controllare: chiavi corrette, tipi numerici, range min-max.</p>',
      starter: `class FeatureSchema:
    def __init__(self, feature_names, min_vals, max_vals):
        # TODO
        pass
    def validate(self, features):
        pass

schema = FeatureSchema(['age', 'income'], [0, 0], [120, 1e6])
print(schema.validate({'age': 25, 'income': 50000}))       # []
print(schema.validate({'age': 200, 'income': 50000}))       # error range
print(schema.validate({'age': 25}))                          # missing key`,
      check: `errs1 = schema.validate({'age': 25, 'income': 50000})
errs2 = schema.validate({'age': 200, 'income': 50000})
errs3 = schema.validate({'age': 25})
if errs1 == [] and len(errs2) >= 1 and len(errs3) >= 1:
    print("OK")
else:
    print(f"FAIL — {errs1}, {errs2}, {errs3}")`,
      solution: `class FeatureSchema:
    def __init__(self, feature_names, min_vals, max_vals):
        self.names = feature_names
        self.mins = dict(zip(feature_names, min_vals))
        self.maxs = dict(zip(feature_names, max_vals))
    def validate(self, features):
        errors = []
        for name in self.names:
            if name not in features:
                errors.append(f'Missing feature: {name}')
                continue
            v = features[name]
            if not isinstance(v, (int, float)):
                errors.append(f'{name} not numeric')
                continue
            if v < self.mins[name] or v > self.maxs[name]:
                errors.append(f'{name} out of range [{self.mins[name]}, {self.maxs[name]}]: got {v}')
        for k in features:
            if k not in self.names:
                errors.append(f'Unexpected feature: {k}')
        return errors` },
  ],
  test: [
    { type: 'quiz', questions: [
      { q: 'MLflow serve principalmente a:', options: ['Fare deploy', 'Tracciare esperimenti (parametri, metriche, artifacts) e organizzarli', 'Fare feature engineering', 'Scrivere DAG'], answer: 1, explain: 'Sperimentazione tracciata, poi anche model registry e serving.' },
      { q: 'Canary deployment significa:', options: ['Deploy solo su GPU', 'Sostituire il modello vecchio istantaneamente', 'Servire una piccola frazione di traffico al nuovo modello prima di rollare al 100%', 'Deployare canari.png'], answer: 2, explain: 'Rilascio graduale con monitoring.' },
      { q: 'Data drift si rileva con:', options: ['Solo con accuracy', 'Test statistici come KS, PSI, chi-squared sulle distribuzioni delle feature', 'Log delle richieste', 'MSE sul training'], answer: 1, explain: 'Confrontare distribuzioni training vs produzione.' },
      { q: 'Training-serving skew è:', options: ['Un errore di codifica', 'Discrepanza tra come le feature sono calcolate in training vs inference, causa perdite di performance in produzione', 'Un warning di Python', 'Un problema di GPU'], answer: 1, explain: 'Es. missing imputation con la media del training non identica in produzione. Feature store lo previene.' },
      { q: 'FastAPI vs Flask:', options: ['Sono uguali', 'FastAPI è async, valida automaticamente i tipi con Pydantic, genera docs OpenAPI', 'Flask è più veloce', 'FastAPI non supporta async'], answer: 1, explain: 'FastAPI moderno, Flask legacy (comunque valido).' },
    ] },
    { type: 'exercise', difficulty: 'hard', title: 'Test finale — pipeline serializzata',
      prompt: '<p>Prepara un modello per produzione: 1) fit una pipeline (scaler + logistic) su breast_cancer, 2) serializza in bytes, 3) simula il caricamento in un "servizio", 4) fai una predizione sul primo esempio del test set. Salva <code class="inline">served_prob</code> = probabilità classe 1.</p>',
      starter: `import io
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

served_prob = None`,
      check: `if served_prob is not None and 0 <= served_prob <= 1:
    print(f"OK — served prob = {served_prob:.4f}")
else:
    print(f"FAIL — {served_prob}")`,
      solution: `pipe = Pipeline([('s', StandardScaler()), ('lr', LogisticRegression(max_iter=5000))]).fit(Xt, yt)
buf = io.BytesIO()
joblib.dump(pipe, buf)
buf.seek(0)
loaded = joblib.load(buf)
served_prob = float(loaded.predict_proba(Xe[:1])[0, 1])` },

    { type: 'exercise', difficulty: 'hard', title: 'Bonus expert — drift dashboard',
      prompt: '<p><strong>Sfida applicata</strong>. Simula un sistema di monitoring: dati training di riferimento, e 4 batch di produzione (mesi 1-4) con drift crescente. Per ogni batch e ogni feature, calcola PSI. Trova il primo mese in cui almeno una feature ha PSI > 0.2 (soglia "significativo"). Salvalo in <code class="inline">first_alert_month</code>.</p>',
      starter: `import numpy as np
np.random.seed(0)
ref = np.random.normal(0, 1, (2000, 3))
prod_months = [np.random.normal(shift, 1, (500, 3)) for shift in [0.05, 0.15, 0.35, 0.6]]

def psi(reference, current, n_bins=10):
    bp = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bp[0] = -np.inf; bp[-1] = np.inf
    ref_freq, _ = np.histogram(reference, bins=bp)
    cur_freq, _ = np.histogram(current, bins=bp)
    ref_pct = np.clip(ref_freq / len(reference), 1e-6, None)
    cur_pct = np.clip(cur_freq / len(current), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

first_alert_month = None`,
      check: `if first_alert_month is not None and 2 <= first_alert_month <= 3:
    print(f"OK — primo alert al mese {first_alert_month}")
else:
    print(f"FAIL — atteso 2 o 3, tuo {first_alert_month}")`,
      solution: `for month_idx, batch in enumerate(prod_months, start=1):
    psis = [psi(ref[:, j], batch[:, j]) for j in range(3)]
    if max(psis) > 0.2:
        first_alert_month = month_idx
        break` },
  ]
});
