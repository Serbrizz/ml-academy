/* Lezione: Cos'è il Machine Learning */
window.LESSONS.push({
  "id": "l01-intro",
  "title": "Cos'è il Machine Learning",
  "tag": "Fondamenti",
  "summary": "Definizioni, tipi di apprendimento, terminologia essenziale.",
  "teoria": [
    {
      "type": "md",
      "content": "\n<h3>1.1 Definizione operativa</h3>\n<p>Il machine learning è la disciplina che studia algoritmi capaci di <strong>migliorare le proprie prestazioni su un compito attraverso l'esperienza</strong>. La definizione classica di Tom Mitchell (1997) è quasi una formula: un programma apprende dall'esperienza $E$ rispetto a un compito $T$ misurato da $P$ se le prestazioni su $T$, misurate da $P$, migliorano con $E$.</p>\n<p>In termini fisici puoi pensarci così: non stai scrivendo l'equazione del moto a mano, stai facendo <em>fit di una funzione</em> a partire dai dati. La differenza rispetto al fit classico è che le funzioni che consideriamo hanno milioni di parametri e non hanno una struttura fisica imposta a priori.</p>\n\n<h3>1.2 Il paradigma: dai dati alla funzione</h3>\n<p>In programmazione tradizionale scrivi: <code class=\"inline\">regole + dati &rarr; risultati</code>. Nel machine learning inverti: <code class=\"inline\">dati + risultati &rarr; regole</code>. L'algoritmo di apprendimento è quindi un <em>meta-programma</em> che produce un programma (il modello) a partire da esempi.</p>\n\n<h3>1.3 I tre grandi paradigmi</h3>\n<p><strong>Apprendimento supervisionato.</strong> Hai coppie $(x_i, y_i)$ dove $x_i \\in \\mathbb{R}^d$ è un vettore di features e $y_i$ è l'etichetta. Vuoi imparare $f : \\mathbb{R}^d \\to \\mathcal{Y}$ tale che $f(x) \\approx y$ su nuovi dati. Il tipo di problema dipende da $\\mathcal{Y}$: se è continuo è <em>regressione</em>, se è discreto è <em>classificazione</em>.</p>\n<p><strong>Apprendimento non supervisionato.</strong> Hai solo $x_i$, senza etichette. Vuoi scoprire struttura latente: cluster, direzioni principali di variabilità, distribuzione dei dati, anomalie.</p>\n<p><strong>Reinforcement learning.</strong> Un agente interagisce con un ambiente, osserva stati $s_t$, sceglie azioni $a_t$, riceve ricompense $r_t$. Vuoi imparare una policy $\\pi(a|s)$ che massimizza la ricompensa cumulativa attesa $\\mathbb{E}\\left[\\sum_t \\gamma^t r_t\\right]$.</p>\n<p><strong>Semi-supervisionato e self-supervised.</strong> Nel semi-supervisionato hai molti dati non etichettati e pochi etichettati (etichettare è costoso). Nel self-supervised le etichette sono generate automaticamente dai dati stessi — es. predire la parola successiva in un testo: questa è la base dei Large Language Model.</p>\n\n<h3>1.4 Tassonomia del supervisionato: regressione e classificazione</h3>\n<p>La distinzione è fondamentale e ricorre in tutti gli esercizi. Dipende dallo spazio dei target $\\mathcal{Y}$.</p>\n\n<h4>Regressione &mdash; $\\mathcal{Y} \\subseteq \\mathbb{R}$ (o $\\mathbb{R}^k$)</h4>\n<ul>\n<li><strong>Regressione semplice</strong>: una sola feature $x \\in \\mathbb{R}$ &rarr; scalare. Es: superficie casa &rarr; prezzo.</li>\n<li><strong>Regressione multipla</strong>: vettore di feature $x \\in \\mathbb{R}^d$ &rarr; scalare. Es: età, reddito, anni di esperienza &rarr; stipendio.</li>\n<li><strong>Regressione multivariata</strong>: $x \\in \\mathbb{R}^d$ &rarr; vettore $y \\in \\mathbb{R}^k$. Es: input sensori &rarr; coordinate 3D del robot.</li>\n</ul>\n<p>La loss standard è l'errore quadratico medio (MSE): $\\ell(\\hat{y}, y) = (\\hat{y} - y)^2$. Varianti robuste: MAE $= |\\hat{y}-y|$, Huber (quadratica vicino a zero, lineare lontano).</p>\n\n<h4>Classificazione binaria &mdash; $\\mathcal{Y} = \\{0, 1\\}$</h4>\n<p>Esattamente due classi, una detta \"positiva\" (1) e una \"negativa\" (0). Esempi: spam/non-spam, tumore maligno/benigno, transazione fraudolenta/legittima.</p>\n<p>Il modello produce una probabilità $\\hat{p} = P(y=1|x) \\in [0,1]$, poi si applica una <strong>soglia</strong> $\\tau$ (default 0.5) per ottenere la classe: se $\\hat{p} \\geq \\tau$ &rarr; classe 1, altrimenti classe 0. La soglia <em>non</em> è appresa durante il training: è un iperparametro che si sceglie in base al costo relativo di falsi positivi e falsi negativi.</p>\n<p>La loss standard è la <em>binary cross-entropy</em>: $\\ell(\\hat{p}, y) = -[y \\log \\hat{p} + (1-y)\\log(1-\\hat{p})]$.</p>\n\n<h4>Classificazione multi-classe &mdash; $\\mathcal{Y} = \\{1, 2, \\ldots, K\\}$ con $K > 2$</h4>\n<p>$K$ classi <strong>mutualmente esclusive</strong>: ogni esempio appartiene a esattamente una classe. Esempi: riconoscere la cifra scritta a mano (0-9, $K=10$), classificare un articolo di giornale per tema, riconoscere una specie animale da una foto.</p>\n<p>Il modello produce un vettore di $K$ probabilità $[\\hat{p}_1, \\ldots, \\hat{p}_K]$ con $\\sum_k \\hat{p}_k = 1$ tramite la funzione <strong>softmax</strong>. La classe predetta è $\\hat{y} = \\arg\\max_k \\hat{p}_k$. La loss è la <em>categorical cross-entropy</em>.</p>\n<p>Due strategie per adattare classificatori binari al caso multi-classe:</p>\n<ul>\n<li><strong>One-vs-Rest (OvR)</strong>: per ogni classe $k$, allena un classificatore binario \"classe $k$ vs tutte le altre\". Produce $K$ modelli.</li>\n<li><strong>One-vs-One (OvO)</strong>: allena un classificatore binario per ogni coppia di classi. Produce $K(K-1)/2$ modelli.</li>\n</ul>\n\n<h4>Classificazione multi-label &mdash; più etichette per esempio</h4>\n<p>Un esempio può appartenere a <strong>più classi contemporaneamente</strong>. Esempi: un'immagine contiene sia un cane che una macchina; una canzone è rock, live e anni-'90 allo stesso tempo; un articolo riguarda sia economia che politica estera.</p>\n<p><strong>Attenzione: non è multi-classe.</strong> In multi-classe le etichette si escludono (una sola classe per esempio); in multi-label no. Si tratta tipicamente come $K$ classificatori binari indipendenti, uno per etichetta, ognuno con la propria soglia.</p>\n\n<h4>Classificazione ordinale</h4>\n<p>Le classi hanno un ordine naturale ma non una distanza numerica precisa. Esempi: rating 1-5 stelle, grado di malattia (lieve/moderato/grave), livello di accordo in una survey. Non è né classificazione pura né regressione: trattarla come regressione è approssimativo ma spesso funziona; esistono metodi dedicati (ordinal regression, threshold models).</p>\n\n<h3>1.5 Formalizzazione dell'apprendimento supervisionato</h3>\n<p>Assumiamo che i dati provengano da una distribuzione congiunta ignota $\\mathcal{D}$ su $\\mathcal{X} \\times \\mathcal{Y}$. Data una funzione di loss $\\ell(\\hat{y}, y)$, definiamo il <em>rischio atteso</em>:</p>\n<p>$$R(f) = \\mathbb{E}_{(x,y) \\sim \\mathcal{D}}\\left[\\ell(f(x), y)\\right]$$</p>\n<p>Non conosciamo $\\mathcal{D}$, quindi minimizziamo il <em>rischio empirico</em> su un training set $\\{(x_i, y_i)\\}_{i=1}^n$:</p>\n<p>$$\\hat{R}(f) = \\frac{1}{n}\\sum_{i=1}^n \\ell(f(x_i), y_i)$$</p>\n<p>Il grande tema del ML è: quando il minimo di $\\hat{R}$ è vicino al minimo di $R$? Questa è la questione della <em>generalizzazione</em>.</p>\n\n<h3>1.6 Bias-Variance Tradeoff</h3>\n<p>Ogni modello commette errori. L'errore atteso su un punto $x$, mediato su tutti i possibili training set di dimensione $n$ campionati da $\\mathcal{D}$, si decompone esattamente in tre termini:</p>\n<p>$$\\mathbb{E}\\left[(f(x) - y)^2\\right] = \\underbrace{\\left(\\mathbb{E}[f(x)] - y^*\\right)^2}_{\\text{Bias}^2} + \\underbrace{\\mathbb{E}\\left[(f(x) - \\mathbb{E}[f(x)])^2\\right]}_{\\text{Varianza}} + \\underbrace{\\sigma^2_\\epsilon}_{\\text{Rumore}}$$</p>\n<p>dove $y^* = \\mathbb{E}[y|x]$ è il valore vero (la funzione da imparare) e $\\sigma^2_\\epsilon$ è il rumore irriducibile dei dati.</p>\n<ul>\n<li><strong>Bias elevato</strong>: il modello è sistematicamente sbagliato, troppo semplice per catturare il pattern reale &rarr; <em>underfitting</em>. Es: fittare una retta su dati parabolici.</li>\n<li><strong>Varianza elevata</strong>: il modello varia molto al variare del training set, memorizza il rumore invece del pattern &rarr; <em>overfitting</em>. Es: polinomio grado 15 su 10 punti.</li>\n<li><strong>Rumore irriducibile</strong> $\\sigma^2_\\epsilon$: errore intrinseco dei dati (misurazione imprecisa, variabilità naturale). Non eliminabile con nessun modello.</li>\n</ul>\n<p>Il <em>tradeoff</em>: aumentare la complessità del modello riduce il bias ma aumenta la varianza. Il punto ottimale minimizza la somma. Strumenti pratici: la <strong>regolarizzazione</strong> riduce la varianza penalizzando i parametri grandi; <strong>più dati</strong> riducono la varianza; un <strong>modello più espressivo</strong> riduce il bias.</p>\n\n<h3>1.7 Lo spazio delle ipotesi e la capacità</h3>\n<p>Quando scegliamo una famiglia di modelli (rette, alberi, reti neurali...) stiamo scegliendo uno <strong>spazio delle ipotesi</strong> $\\mathcal{H}$. L'algoritmo di apprendimento cerca:</p>\n<p>$$f^* = \\arg\\min_{f \\in \\mathcal{H}} \\hat{R}(f)$$</p>\n<p>La <em>capacità</em> (o complessità) di $\\mathcal{H}$ determina il tradeoff bias-varianza:</p>\n<ul>\n<li>$\\mathcal{H}$ ristretto (es. solo rette): alto bias, bassa varianza.</li>\n<li>$\\mathcal{H}$ ampio (es. polinomi di grado alto, reti neurali profonde): basso bias potenziale, alta varianza se non regolarizzato.</li>\n</ul>\n<p>Misure formali della capacità: <em>dimensione VC</em>, <em>Rademacher complexity</em>. Non richieste in questo corso, ma utile sapere che la teoria del ML studia formalmente quando la minimizzazione del rischio empirico garantisce la generalizzazione.</p>\n\n<h3>1.8 Terminologia essenziale</h3>\n<ul>\n<li><strong>Feature</strong> (o predittore, variabile indipendente): una componente di $x$.</li>\n<li><strong>Label</strong> (o target, variabile dipendente): $y$.</li>\n<li><strong>Modello</strong>: la funzione $f_\\theta$ parametrizzata da $\\theta$.</li>\n<li><strong>Iperparametro</strong>: parametro scelto prima del training, non appreso dai dati (es. grado del polinomio, soglia di classificazione, numero di alberi).</li>\n<li><strong>Training</strong>: il processo di trovare $\\theta$ ottimale minimizzando $\\hat{R}$.</li>\n<li><strong>Inference</strong>: usare il modello addestrato per fare previsioni su nuovi dati.</li>\n<li><strong>Overfitting</strong>: il modello memorizza il training set ma non generalizza. $\\hat{R}$ piccolo, $R$ grande. Alta varianza.</li>\n<li><strong>Underfitting</strong>: il modello è troppo semplice per catturare il pattern. Sia $\\hat{R}$ che $R$ grandi. Alto bias.</li>\n<li><strong>Generalizzazione</strong>: capacità di performare bene su dati nuovi, non visti durante il training.</li>\n<li><strong>Dataset split</strong>: <em>training set</em> (addestrare i parametri), <em>validation set</em> (scegliere iperparametri e confrontare modelli), <em>test set</em> (stimare la performance reale &mdash; usato UNA volta sola alla fine).</li>\n</ul>\n"
    },
    {
      "type": "callout",
      "variant": "note",
      "title": "Analogia con la fisica",
      "content": "Overfitting è come fittare un polinomio di grado 20 su 10 punti sperimentali: passi per tutti i punti ma il modello non ha significato fisico. Il bias-variance tradeoff (sezione 1.6) è la formalizzazione matematica di questa intuizione: la complessità ottimale bilancia errore sistematico e sensibilità al campione."
    },
    {
      "type": "callout",
      "variant": "tip",
      "title": "Classificazione: binaria vs multi-classe vs multi-label",
      "content": "Regola pratica — conta le classi e chiedi se un esempio può averne più di una. Due classi esclusive → binaria. Più di due classi esclusive → multi-classe. Più etichette per esempio → multi-label. Ordine ma non metrica → ordinale. Queste distinzioni cambiano loss, metriche e architettura del modello."
    }
  ],
  "esempi": [
    {
      "type": "md",
      "content": "<h3>Esempio: da dati a modello in 10 righe</h3><p>Prendiamo un problema banale (predire il prezzo di una casa dalla sua superficie) e vediamo l'intero flusso.</p>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nfrom sklearn.linear_model import LinearRegression\n\n# Dati sintetici: superficie in mq -> prezzo in migliaia di euro\nX = np.array([[50], [65], [80], [100], [120], [150], [180], [220]])\ny = np.array([120, 155, 190, 235, 280, 340, 400, 480])\n\n# Modello: cerchiamo f(x) = w*x + b\nmodel = LinearRegression()\nmodel.fit(X, y)\n\nprint(f\"Coefficiente (w): {model.coef_[0]:.3f}\")\nprint(f\"Intercetta  (b): {model.intercept_:.3f}\")\nprint(f\"Predizione per 90 mq: {model.predict([[90]])[0]:.1f} k€\")\nprint(f\"R^2 sul training: {model.score(X, y):.4f}\")"
    },
    {
      "type": "md",
      "content": "<p>Nota: abbiamo scelto una famiglia di funzioni (retta), abbiamo trovato i parametri che minimizzano l'errore quadratico, abbiamo valutato. Questo è il pattern che ripeteremo ossessivamente, cambiando solo la famiglia di funzioni e la loss.</p><h3>Esempio: classificazione binaria</h3>"
    },
    {
      "type": "run",
      "content": "from sklearn.datasets import make_classification\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split\n\nX, y = make_classification(n_samples=200, n_features=2, n_informative=2,\n                            n_redundant=0, random_state=42)\nX_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)\n\nclf = LogisticRegression()\nclf.fit(X_tr, y_tr)\n\nprint(f\"Accuracy training: {clf.score(X_tr, y_tr):.3f}\")\nprint(f\"Accuracy test:     {clf.score(X_te, y_te):.3f}\")\nprint(\"Prima predizione:\", clf.predict(X_te[:1]), \"vera:\", y_te[:1])"
    },
    {
      "type": "md",
      "content": "<h3>Esempio: classificazione multi-classe (Iris, K=3)</h3><p>Il dataset Iris ha 3 specie (classi). Il modello produce 3 probabilità che sommano a 1 via softmax. Notiamo come l'accuracy da sola non è sempre sufficiente: stampiamo anche il report completo per classe.</p>"
    },
    {
      "type": "run",
      "content": "from sklearn.datasets import load_iris\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.metrics import classification_report\nimport numpy as np\n\nX, y = load_iris(return_X_y=True)\nnomi_classi = ['setosa', 'versicolor', 'virginica']\n\nX_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)\nclf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))\nclf.fit(X_tr, y_tr)\n\nprob = clf.predict_proba(X_te)\nprint('Esempio: probabilità per primo campione del test:')\nfor nome, p in zip(nomi_classi, prob[0]):\n    print(f'  {nome}: {p:.3f}')\nprint(f'Predizione: {nomi_classi[clf.predict(X_te[:1])[0]]}  |  Vera: {nomi_classi[y_te[0]]}')\nprint()\nprint(classification_report(y_te, clf.predict(X_te), target_names=nomi_classi))"
    },
    {
      "type": "md",
      "content": "<h3>Esempio: multi-label &mdash; un'immagine ha più etichette</h3><p>Simuliamo tag cinematografici: un film può essere contemporaneamente 'azione', 'commedia', 'dramma'. Non si escludono. Usiamo un classificatore binario per ogni etichetta.</p>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.multioutput import MultiOutputClassifier\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import classification_report\n\nnp.random.seed(42)\nn = 400\n# feature: ritmo, umorismo, intensita_emotiva\nX = np.random.randn(n, 3)\n# etichette multi-label: azione, commedia, dramma\nazione   = (X[:,0] > 0.2).astype(int)\ncommedia = (X[:,1] > 0.3).astype(int)\ndramma   = (X[:,2] > 0.1).astype(int)\nY = np.column_stack([azione, commedia, dramma])\n\nprint('Film con piu di una etichetta:', (Y.sum(axis=1) > 1).sum(), '/', n)\nprint('Distribuzione etichette: azione', azione.mean():.2f,\n      'commedia', commedia.mean():.2f, 'dramma', dramma.mean():.2f)\n\nX_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.3, random_state=0)\n# un LogisticRegression per ogni colonna di Y\nmodel = MultiOutputClassifier(LogisticRegression()).fit(X_tr, Y_tr)\nY_pred = model.predict(X_te)\nfor i, tag in enumerate(['azione', 'commedia', 'dramma']):\n    from sklearn.metrics import f1_score\n    f1 = f1_score(Y_te[:,i], Y_pred[:,i])\n    print(f'{tag:10s}: F1 = {f1:.3f}')"
    },
    {
      "type": "md",
      "content": "<h3>Esempio: bias-variance tradeoff visivo</h3><p>Stessa funzione, stessi punti, polinomi di grado crescente. Osserva come MSE train scende sempre, mentre MSE test ha un minimo: questo è il punto di bias-varianza ottimale.</p>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nimport matplotlib.pyplot as plt\n\nnp.random.seed(7)\nx_tr = np.linspace(0, 1, 12)\ny_tr = np.sin(2*np.pi*x_tr) + 0.15*np.random.randn(12)\nx_te = np.linspace(0, 1, 200)\ny_te = np.sin(2*np.pi*x_te)\n\ngradi = range(1, 13)\nmse_tr, mse_te = [], []\nfor g in gradi:\n    c = np.polyfit(x_tr, y_tr, g)\n    mse_tr.append(np.mean((np.polyval(c, x_tr) - y_tr)**2))\n    mse_te.append(np.mean((np.polyval(c, x_te) - y_te)**2))\n\nplt.figure(figsize=(8,4))\nplt.plot(list(gradi), mse_tr, 'o-', label='MSE train')\nplt.plot(list(gradi), mse_te, 's-', label='MSE test')\nplt.axvline(x=np.argmin(mse_te)+1, color='gray', linestyle='--', label=f'ottimo grado={np.argmin(mse_te)+1}')\nplt.xlabel('Grado polinomio'); plt.ylabel('MSE'); plt.title('Bias-Variance Tradeoff')\nplt.ylim(0, min(max(mse_te)*1.5, 2)); plt.legend(); plt.grid(alpha=0.3)\nplt.tight_layout(); plt.show()\nprint(f'Grado ottimo: {np.argmin(mse_te)+1}, MSE test min: {min(mse_te):.4f}')\nprint(f'Grado 1  (alto bias):    MSE test = {mse_te[0]:.4f}')\nprint(f'Grado 11 (alta varianza): MSE test = {mse_te[10]:.4f}')"
    }
  ],
  "esercizi": [
    {
      "type": "exercise",
      "difficulty": "easy",
      "title": "Es. 1.1 — Classifica il problema",
      "prompt": "<p>Per ognuno dei seguenti scenari, decidi se è <strong>regressione</strong>, <strong>classificazione binaria</strong>, <strong>classificazione multi-classe</strong>, <strong>clustering</strong>, o <strong>reinforcement learning</strong>. Modifica il dizionario nel codice e poi clicca \"Verifica soluzione\".</p><ol><li>Predire il consumo elettrico di una casa domani (kWh)</li><li>Riconoscere se un'email è spam</li><li>Raggruppare clienti simili senza avere una categoria predefinita</li><li>Riconoscere la cifra scritta a mano (0-9)</li><li>Allenare un bot a giocare a scacchi</li></ol>",
      "starter": "risposte = {\n    1: \"?\",  # scegli tra: regressione, binaria, multi, clustering, rl\n    2: \"?\",\n    3: \"?\",\n    4: \"?\",\n    5: \"?\",\n}\nfor k, v in risposte.items():\n    print(f\"{k}: {v}\")",
      "check": "_atteso = {1:\"regressione\", 2:\"binaria\", 3:\"clustering\", 4:\"multi\", 5:\"rl\"}\n_giusti = sum(1 for k in _atteso if risposte.get(k) == _atteso[k])\nprint(f\"\\n--- {_giusti}/5 corrette ---\")\nfor k,v in _atteso.items():\n    ok = \"OK\" if risposte.get(k)==v else \"NO\"\n    print(f\"[{ok}] {k}: atteso '{v}', tuo '{risposte.get(k)}'\")",
      "solution": "risposte = {1: \"regressione\", 2: \"binaria\", 3: \"clustering\", 4: \"multi\", 5: \"rl\"}"
    },
    {
      "type": "exercise",
      "difficulty": "easy",
      "title": "Es. 1.2 — Calcola rischio empirico",
      "prompt": "<p>Data una lista di predizioni <code class=\"inline\">y_pred</code> e valori veri <code class=\"inline\">y_true</code>, calcola il rischio empirico con loss quadratica $\\ell(\\hat{y}, y) = (\\hat{y}-y)^2$. Il risultato deve essere in <code class=\"inline\">rischio</code>.</p>",
      "starter": "y_true = [3.0, 5.0, 2.5, 7.0, 4.5]\ny_pred = [2.8, 5.2, 3.0, 6.5, 4.7]\n\n# calcola qui il rischio empirico (media dei quadrati degli scarti)\nrischio = None\n\nprint(\"Rischio empirico:\", rischio)",
      "check": "_atteso = sum((a-b)**2 for a,b in zip(y_true,y_pred)) / len(y_true)\nif rischio is None:\n    print(\"FAIL: rischio è None\")\nelif abs(rischio - _atteso) < 1e-6:\n    print(f\"OK! Valore corretto: {rischio:.4f}\")\nelse:\n    print(f\"FAIL: atteso {_atteso:.4f}, ottenuto {rischio}\")",
      "solution": "rischio = sum((yp-yt)**2 for yp,yt in zip(y_pred,y_true)) / len(y_true)"
    },
    {
      "type": "exercise",
      "difficulty": "medium",
      "title": "Es. 1.3 — Overfitting visibile",
      "prompt": "<p>Usa <code class=\"inline\">numpy.polyfit</code> per fittare un polinomio di grado 1 e uno di grado 15 sugli stessi 10 punti rumorosi. Stampa gli errori MSE sul training e su un test set separato. Osserva la differenza.</p>",
      "starter": "import numpy as np\nnp.random.seed(0)\nx_tr = np.linspace(0, 1, 10)\ny_tr = np.sin(2*np.pi*x_tr) + 0.1*np.random.randn(10)\nx_te = np.linspace(0, 1, 100)\ny_te = np.sin(2*np.pi*x_te)\n\n# fitta polinomio grado 1 e grado 15, calcola MSE su train e test\n# usa np.polyfit(x, y, deg) e np.polyval(coef, x)\n\n# TODO\n",
      "check": "import numpy as np\nc1 = np.polyfit(x_tr, y_tr, 1); c15 = np.polyfit(x_tr, y_tr, 15)\ntr1 = np.mean((np.polyval(c1,x_tr)-y_tr)**2); te1 = np.mean((np.polyval(c1,x_te)-y_te)**2)\ntr15 = np.mean((np.polyval(c15,x_tr)-y_tr)**2); te15 = np.mean((np.polyval(c15,x_te)-y_te)**2)\nprint(f\"grado 1  : MSE train {tr1:.4f}, test {te1:.4f}\")\nprint(f\"grado 15 : MSE train {tr15:.4f}, test {te15:.4f}\")\nprint(\"\\nOsserva: grado 15 ha MSE train quasi zero ma test enorme -> OVERFITTING\")",
      "solution": "import numpy as np\nc1  = np.polyfit(x_tr, y_tr, 1)\nc15 = np.polyfit(x_tr, y_tr, 15)\nfor c, deg in [(c1,1),(c15,15)]:\n    tr = np.mean((np.polyval(c,x_tr)-y_tr)**2)\n    te = np.mean((np.polyval(c,x_te)-y_te)**2)\n    print(f\"grado {deg}: MSE train {tr:.4f}, test {te:.4f}\")"
    }
  ],
  "test": [
    {
      "type": "md",
      "content": "<h3>Test di comprensione</h3><p>Rispondi alle domande. Alla fine, l'esercizio pratico verifica l'applicazione.</p>"
    },
    {
      "type": "quiz",
      "questions": [
        {
          "q": "Un algoritmo che raggruppa articoli di giornale in temi senza conoscere le categorie è un esempio di:",
          "options": [
            "Supervisionato",
            "Non supervisionato",
            "Reinforcement learning",
            "Semi-supervisionato"
          ],
          "answer": 1,
          "explain": "Non ci sono etichette, cerchiamo struttura latente: clustering, tipico del non supervisionato."
        },
        {
          "q": "Il rischio empirico $\\hat{R}$ è:",
          "options": [
            "La stessa cosa del rischio atteso $R$",
            "Il valore di $R$ sul test set",
            "La media della loss sul training set",
            "La varianza delle predizioni"
          ],
          "answer": 2,
          "explain": "Per definizione, $\\hat{R} = \\frac{1}{n}\\sum_i \\ell(f(x_i), y_i)$ sul training set. Il rischio atteso $R$ è la stessa quantità ma sotto la distribuzione vera ignota."
        },
        {
          "q": "Un modello con training accuracy 99% e test accuracy 60% è tipicamente:",
          "options": [
            "Underfit (alto bias)",
            "Overfit (alta varianza)",
            "Ben calibrato",
            "Ottimamente regolarizzato"
          ],
          "answer": 1,
          "explain": "Grande gap train-test è la firma dell'overfitting (alta varianza): il modello memorizza il training set invece di generalizzare."
        },
        {
          "q": "Un'immagine è etichettata come 'gatto' E 'esterno' E 'notturna'. Che tipo di problema è?",
          "options": [
            "Classificazione multi-classe",
            "Classificazione multi-label",
            "Classificazione binaria ripetuta",
            "Regressione ordinale"
          ],
          "answer": 1,
          "explain": "Più etichette non esclusive sullo stesso esempio → multi-label. In multi-classe ogni esempio ha ESATTAMENTE una classe."
        },
        {
          "q": "Il numero di neuroni in uno strato nascosto di una rete è:",
          "options": [
            "Un parametro appreso durante il training",
            "Un iperparametro scelto prima del training",
            "Sempre uguale al numero di feature",
            "Determinato automaticamente dai dati"
          ],
          "answer": 1,
          "explain": "I parametri (pesi, bias) sono appresi durante il training. Gli iperparametri (architettura, learning rate, regolarizzazione) si scelgono prima e tipicamente via validation set o cross-validation."
        },
        {
          "q": "Nella decomposizione bias-varianza, il 'rumore irriducibile' $\\sigma^2_\\epsilon$:",
          "options": [
            "Si elimina con un modello più complesso",
            "Si elimina con più dati",
            "Non si può ridurre con nessun modello",
            "È uguale al bias al quadrato"
          ],
          "answer": 2,
          "explain": "$\\sigma^2_\\epsilon$ è l'errore intrinseco dei dati (misurazioni imprecise, variabilità naturale del fenomeno). Nessun algoritmo può fare meglio del 'bayes error' fissato da questo rumore."
        },
        {
          "q": "Nella classificazione binaria, la soglia $\\tau = 0.5$ è:",
          "options": [
            "Un parametro appreso durante il training con la cross-entropy",
            "Un iperparametro che si può cambiare in base al costo dei falsi positivi/negativi",
            "Sempre la scelta ottimale",
            "Fissata dalla funzione softmax"
          ],
          "answer": 1,
          "explain": "La soglia non è appresa: si sceglie dopo il training sul validation set. Se i falsi negativi sono più costosi (es. diagnosi medica), conviene abbassarla per aumentare il recall."
        },
        {
          "q": "Nella regressione lineare $y = wx + b$, i parametri appresi sono:",
          "options": [
            "Solo $w$",
            "Solo $b$",
            "$w$ e $b$",
            "$x$ e $y$"
          ],
          "answer": 2,
          "explain": "$x$ e $y$ sono i dati (fissi); $w$ (peso) e $b$ (bias/intercetta) sono i parametri da apprendere minimizzando l'MSE."
        }
      ]
    },
    {
      "type": "md",
      "content": "<h3>Notebook finale</h3><p>Applica quanto imparato: costruisci un modello che predice se il numero passato è pari usando la sua rappresentazione binaria. Segui lo scheletro.</p>"
    },
    {
      "type": "exercise",
      "difficulty": "medium",
      "title": "Test pratico",
      "prompt": "<p>Crea un classificatore che, dati 100 interi tra 0 e 255 rappresentati come vettore binario di 8 bit, predica se il numero è pari. Usa <code class=\"inline\">LogisticRegression</code> di scikit-learn. Deve raggiungere <strong>almeno 95% accuracy</strong> sul test.</p>",
      "starter": "import numpy as np\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split\n\nnp.random.seed(0)\nnums = np.random.randint(0, 256, size=200)\n\ndef to_bits(n):\n    return [(n >> i) & 1 for i in range(8)]\n\nX = np.array([to_bits(n) for n in nums])\ny = (nums % 2 == 0).astype(int)\n\n# TODO: split, fit, valuta test accuracy in variable 'acc'\nacc = 0.0\nprint(\"Test accuracy:\", acc)",
      "check": "if acc >= 0.95:\n    print(f\"OK — accuracy {acc*100:.1f}%. Nota: il primo bit E' la parità, il modello lo scopre.\")\nelse:\n    print(f\"FAIL — accuracy {acc*100:.1f}%, servono >= 95%.\")",
      "solution": "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)\nclf = LogisticRegression().fit(X_tr, y_tr)\nacc = clf.score(X_te, y_te)\nprint(\"Test accuracy:\", acc)"
    }
  ]
});
