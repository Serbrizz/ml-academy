/* Lezione: Alberi di decisione e Random Forest */
window.LESSONS.push({
  "id": "l08-trees",
  "title": "Alberi di decisione e Random Forest",
  "tag": "Supervisionato",
  "summary": "Modelli non lineari, interpretabili, potenti. La base dei modelli ensemble che vincono le Kaggle.",
  "teoria": [
    {
      "type": "md",
      "content": "\n<h3>8.1 Cos'è un albero di decisione</h3>\n<p>Una struttura a nodi in cui ogni nodo interno è un test su una feature (es. \"$x_3 \\leq 2.5$?\"), e le foglie contengono predizioni. Predire significa scendere l'albero seguendo i test.</p>\n<p>Un albero binario partiziona $\\mathbb{R}^d$ in regioni rettangolari e assegna una predizione costante ad ogni regione.</p>\n\n<h3>8.2 Come si costruisce (greedy)</h3>\n<p>Ad ogni nodo, tra tutte le possibili scelte (feature $j$, soglia $t$), scegli quella che <em>massimizza la riduzione di impurità</em> dei figli:</p>\n<p>$$\\Delta = I(\\text{parent}) - \\left(\\frac{n_L}{n} I(L) + \\frac{n_R}{n} I(R)\\right)$$</p>\n<p>Poi ricorri su ogni figlio finché non si soddisfa un criterio di stop (profondità max, dimensione minima foglia, purezza raggiunta).</p>\n\n<h3>8.3 Misure di impurità</h3>\n<p>Per classificazione (con $p_c$ = frazione di classe $c$):</p>\n<ul>\n<li><strong>Gini</strong>: $1 - \\sum_c p_c^2$. Zero se puro, massimo con classi uniformi.</li>\n<li><strong>Entropia</strong>: $-\\sum_c p_c \\log p_c$.</li>\n</ul>\n<p>Praticamente equivalenti. Per regressione si usa la varianza (o MSE).</p>\n\n<h3>8.4 Overfitting</h3>\n<p>Un albero cresciuto senza limiti memorizza il training set: profondità $= n-1$ nel caso estremo. Contromisure:</p>\n<ul>\n<li><strong>Pre-pruning</strong>: <code class=\"inline\">max_depth</code>, <code class=\"inline\">min_samples_split</code>, <code class=\"inline\">min_samples_leaf</code></li>\n<li><strong>Post-pruning</strong>: cost-complexity pruning con parametro $\\alpha$</li>\n</ul>\n\n<h3>8.5 Pro e contro degli alberi singoli</h3>\n<p><strong>Pro</strong>: interpretabili, robusti a scala e outlier, gestiscono categoriche, senza bisogno di normalizzazione.</p>\n<p><strong>Contro</strong>: alta varianza (piccole modifiche ai dati &rarr; alberi molto diversi), tendono a overfittare, non catturano bene relazioni lineari o additive.</p>\n\n<h3>8.6 Ensemble: bagging e Random Forest</h3>\n<p><strong>Bagging</strong> (Bootstrap Aggregating): allena $B$ alberi su $B$ campioni bootstrap del training. Aggrega con media (regressione) o voto (classificazione). Riduce la varianza.</p>\n<p><strong>Random Forest</strong>: bagging + a ogni split considera solo un sottoinsieme casuale di feature (tipicamente $\\sqrt{d}$). Questo decorrelazione ulteriore riduce ancora la varianza.</p>\n<p>Il beneficio è dimostrabile: se hai $B$ predittori i.i.d. con varianza $\\sigma^2$ e correlazione $\\rho$, la varianza della media è:</p>\n<p>$$\\sigma^2_{\\text{avg}} = \\rho \\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2$$</p>\n<p>Random Forest cerca di ridurre $\\rho$.</p>\n\n<h3>8.7 Boosting</h3>\n<p>Diverso dal bagging: costruisce alberi <em>in sequenza</em>, ognuno concentrato sugli errori del precedente. AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost. Tipicamente il non-plus-ultra per dati tabulari.</p>\n\n<h3>8.8 Feature importance</h3>\n<p>Ogni feature ha un'importanza data dalla riduzione totale di impurità che ha causato attraverso tutti gli split. Utile per interpretare il modello, ma con limiti (biased verso feature con più valori possibili).</p>\n"
    },
    {
      "type": "callout",
      "variant": "tip",
      "title": "Sui dati tabulari",
      "content": "Nel 2026 come nel 2016, per la maggior parte dei problemi tabulari (righe = esempi, colonne = feature), un gradient boosting ben tarato (XGBoost, LightGBM) batte tutto: reti neurali comprese. Se vuoi vincere una Kaggle su tabulari, parti da lì."
    },
    {
      "type": "md",
      "content": "\n<h3>8.9 CART in dettaglio</h3>\n<p>CART (Classification And Regression Trees) costruisce alberi <em>binari</em>: ogni nodo fa esattamente un test binario su una sola feature. La costruzione è <strong>greedy top-down</strong>:</p>\n<ul>\n<li>Per ogni feature $j = 1, \\ldots, d$ e ogni soglia $t$ (valori unici di $x_j$ nel nodo corrente), calcola l'impurità pesata dei due figli candidati.</li>\n</ul>\n<p>$$I_{\\text{pesata}}(j,t) = \\frac{n_L}{n} I(L) + \\frac{n_R}{n} I(R)$$</p>\n<p>Scegli la coppia ottima che minimizza questa quantità, poi ricorri su ciascun figlio. <strong>Stopping criteria</strong>: <code class=\"inline\">max_depth</code> (profondità massima), <code class=\"inline\">min_samples_split</code> (non dividere nodi troppo piccoli), <code class=\"inline\">min_impurity_decrease</code> (non dividere se il guadagno è trascurabile).</p>\n<p><strong>Complessità training</strong>: $O(d \\cdot n \\log n)$ — per ogni livello dell'albero si ordinano i valori di ogni feature una sola volta.</p>\n\n<h3>8.10 Confronto misure di impurità</h3>\n<p>Per classificazione, con $p_k$ = frazione della classe $k$ nel nodo:</p>\n<ul>\n<li><strong>Gini</strong>: $\\text{Gini} = 1 - \\sum_k p_k^2$ — computazionalmente leggera, default sklearn. Zero se puro.</li>\n<li><strong>Entropia</strong>: $H = -\\sum_k p_k \\log_2 p_k$ — penalizza di più i nodi impuri, a volte preferita per ottenere probabilità meglio calibrate.</li>\n<li><strong>MSE</strong> (regressione): $\\frac{1}{n}\\sum_i (y_i - \\bar{y})^2$ — minimizza la varianza del target nei figli; scelta naturale per regressione.</li>\n</ul>\n<p><strong>Gini vs Entropia in pratica</strong>: quasi equivalenti — la differenza sull'accuracy finale è tipicamente &lt; 0.5%. Entrambe tendono a favorire split bilanciati. Se hai bisogno di probabilità calibrate, usa <code class=\"inline\">criterion='log_loss'</code> (sklearn &ge; 1.1).</p>\n"
    },
    {
      "type": "md",
      "content": "\n<h3>8.11 Pruning</h3>\n<p><strong>Pre-pruning</strong>: ferma la crescita dell'albero durante la costruzione tramite hyperparameter: <code class=\"inline\">max_depth</code>, <code class=\"inline\">min_samples_split</code>, <code class=\"inline\">min_samples_leaf</code>, <code class=\"inline\">min_impurity_decrease</code>. Semplice e veloce, ma richiede tuning manuale e può fermarsi troppo presto (miopia).</p>\n<p><strong>Post-pruning — cost-complexity pruning</strong>: lascia crescere l'albero completo, poi rimuove nodi calcolando il trade-off tra complessità e accuracy tramite il parametro $\\alpha$:</p>\n<p>$$R_\\alpha(T) = R(T) + \\alpha \\cdot |T|$$</p>\n<p>dove $R(T)$ è l'errore dell'albero e $|T|$ è il numero di foglie. Aumentare <code class=\"inline\">ccp_alpha</code> rimuove progressivamente più nodi interni. In sklearn: <code class=\"inline\">clf.cost_complexity_pruning_path(X, y)</code> restituisce la sequenza di $\\alpha$ ottimali; scegli il migliore con cross-validation. Produce modelli più interpretabili e con meno overfitting rispetto al solo pre-pruning.</p>\n\n<h3>8.12 Out-of-Bag (OOB) Error</h3>\n<p>Nel bootstrap di $n$ esempi con reinserimento, la probabilità che un dato esempio non venga mai estratto è:</p>\n<p>$$\\left(1 - \\frac{1}{n}\\right)^n \\xrightarrow{n \\to \\infty} e^{-1} \\approx 36.8\\%$$</p>\n<p>Quindi circa il 36.8% dei dati è <em>out-of-bag</em> (OOB) per ogni albero. Per fare una predizione su $x_i$, si usano <em>solo</em> gli alberi che non l'hanno visto nel training: la media (o il voto) di queste predizioni costituisce la stima OOB.</p>\n<p>In sklearn: <code class=\"inline\">RandomForestClassifier(oob_score=True)</code>, poi <code class=\"inline\">rf.oob_score_</code>. <strong>Vantaggio</strong>: stima gratuita dell'errore di generalizzazione — non serve una cross-validation separata per valutare la foresta.</p>\n"
    },
    {
      "type": "callout",
      "variant": "tip",
      "title": "OOB come CV gratuita",
      "content": "Abilita sempre oob_score=True nei tuoi RandomForest: ottieni una stima affidabile dell'errore di generalizzazione senza overhead computazionale aggiuntivo. È particolarmente utile per confrontare rapidamente diverse configurazioni (numero di alberi, profondità) senza dover lanciare una CV completa per ciascuna."
    },
    {
      "type": "md",
      "content": "\n<h3>8.13 ExtraTrees vs Random Forest</h3>\n<p><strong>Random Forest</strong>: a ogni split considera un sottoinsieme casuale di $\\sqrt{d}$ feature, poi cerca la <em>soglia ottimale</em> per ognuna (scansiona tutti i valori unici).</p>\n<p><strong>ExtraTrees</strong> (Extremely Randomized Trees): considera un sottoinsieme casuale di feature E sceglie la soglia in modo <em>completamente casuale</em>, senza ricercare il punto di split ottimale. Effetti:</p>\n<ul>\n<li><strong>Più veloce</strong> in training: nessuna ricerca della soglia ottimale ad ogni nodo.</li>\n<li><strong>Più bias, meno varianza</strong>: la casualità extra diversifica ulteriormente gli alberi, riducendo la correlazione tra loro.</li>\n<li><strong>Performance spesso simile</strong> a RF; talvolta migliore su dataset rumorosi dove lo split ottimale può essere ingannevole.</li>\n</ul>\n<p>In sklearn: <code class=\"inline\">ExtraTreesClassifier</code> / <code class=\"inline\">ExtraTreesRegressor</code>. Ottima scelta quando il training time è il bottleneck e si vogliono più alberi nello stesso budget computazionale.</p>\n\n<h3>8.14 Stacking e Blending</h3>\n<p>Tecniche per combinare modelli eterogenei in un meta-ensemble:</p>\n<p><strong>Stacking</strong>:</p>\n<ul>\n<li><em>Livello 1</em>: $K$ modelli diversi (es. RF, SVM, regressione logistica) — ciascuno produce le proprie predizioni.</li>\n<li><em>Livello 2 — meta-learner</em>: apprende come combinare le predizioni del livello 1 in una predizione finale.</li>\n<li><strong>OOF per evitare leakage</strong>: il meta-learner si addestra sulle predizioni <em>out-of-fold</em> — per ogni fold di CV, il modello di livello 1 predice sui dati che non ha visto. Questo è cruciale per evitare che il meta-learner sfrutti l'overfitting dei level-1.</li>\n</ul>\n<p><strong>Blending</strong>: variante semplificata — split manuale train/holdout invece di CV. Il meta-learner si addestra sulle predizioni sul holdout. Più veloce ma usa meno dati e può sovrastimare le performance se il holdout è piccolo.</p>\n<p>In sklearn: <code class=\"inline\">StackingClassifier</code> / <code class=\"inline\">StackingRegressor</code> implementano lo stacking con OOF correttamente.</p>\n\n<h3>8.15 Voting Classifier</h3>\n<p>L'ensemble più semplice: aggrega le predizioni di più modelli eterogenei senza addestramento aggiuntivo.</p>\n<ul>\n<li><strong>Hard voting</strong>: predici la classe con più voti (maggioranza semplice). Funziona anche se i modelli non espongono <code class=\"inline\">predict_proba</code>.</li>\n<li><strong>Soft voting</strong>: calcola la media delle probabilità $P(y=c|x)$ tra tutti i modelli e predice la classe con probabilità media più alta. Richiede <code class=\"inline\">predict_proba</code> su tutti i modelli. Quasi sempre <em>superiore</em> all'hard voting perché sfrutta la confidenza — un modello molto sicuro pesa di più.</li>\n</ul>\n<p>In sklearn: <code class=\"inline\">VotingClassifier(estimators=[('rf', rf), ('svc', svc), ...], voting='soft')</code>. Ottima baseline ensemble rapida prima di esplorare stacking.</p>\n"
    },
    {
      "type": "callout",
      "variant": "warn",
      "title": "Stacking: attenzione al data leakage",
      "content": "Nel blending, se il holdout usato per addestrare il meta-learner è lo stesso dataset che usi per valutare le performance finali, stai contaminando la valutazione. Usa sempre le predizioni OOF (stacking con CV) per una stima onesta. StackingClassifier di sklearn gestisce questo correttamente in automatico."
    }
  ],
  "esempi": [
    {
      "type": "md",
      "content": "<h3>Esempio 1: albero singolo su Iris</h3>"
    },
    {
      "type": "run",
      "content": "from sklearn.tree import DecisionTreeClassifier, export_text\nfrom sklearn.datasets import load_iris\nfrom sklearn.model_selection import train_test_split\n\niris = load_iris()\nXt, Xe, yt, ye = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target)\n\ntree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xt, yt)\nprint(f\"Accuracy: {tree.score(Xe, ye):.4f}\")\nprint(\"\\nAlbero:\")\nprint(export_text(tree, feature_names=list(iris.feature_names)))"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 2: bias/variance con la profondità</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.datasets import make_moons\nfrom sklearn.model_selection import cross_val_score\n\nX, y = make_moons(n_samples=500, noise=0.3, random_state=0)\nprint(\"depth | train | CV mean +/- std\")\nfor d in [1, 2, 5, 10, None]:\n    m = DecisionTreeClassifier(max_depth=d, random_state=0)\n    m.fit(X, y)\n    tr = m.score(X, y)\n    cv = cross_val_score(m, X, y, cv=5)\n    print(f\"{str(d):>5s} | {tr:.3f} | {cv.mean():.3f} +/- {cv.std():.3f}\")"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 3: Random Forest e feature importance</h3>"
    },
    {
      "type": "run",
      "content": "from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.datasets import load_wine\nfrom sklearn.model_selection import train_test_split\nimport numpy as np\n\ndata = load_wine()\nX, y = data.data, data.target\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)\n\nrf = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xt, yt)\nprint(f\"Accuracy: {rf.score(Xe, ye):.4f}\")\n\nimp = sorted(zip(rf.feature_importances_, data.feature_names), reverse=True)\nprint(\"\\nTop 5 feature:\")\nfor i, (v, n) in enumerate(imp[:5]):\n    print(f\"  {i+1}. {n:30s} {v:.4f}\")"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 4: bagging da zero</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.datasets import make_classification\nfrom sklearn.model_selection import train_test_split\nfrom scipy.stats import mode\n\nX, y = make_classification(n_samples=500, n_features=8, random_state=0)\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42)\n\nn = len(Xt); B = 50\nrng = np.random.default_rng(0)\ntrees = []\nfor _ in range(B):\n    idx = rng.choice(n, n, replace=True)  # bootstrap\n    t = DecisionTreeClassifier(random_state=None).fit(Xt[idx], yt[idx])\n    trees.append(t)\n\n# voto di maggioranza\npreds = np.array([t.predict(Xe) for t in trees])\nfinal = mode(preds, axis=0, keepdims=False).mode\nprint(f\"Bagging accuracy: {np.mean(final == ye):.4f}\")\nprint(f\"Singolo albero:   {DecisionTreeClassifier(random_state=0).fit(Xt, yt).score(Xe, ye):.4f}\")"
    }
  ],
  "esercizi": [
    {
      "type": "exercise",
      "difficulty": "easy",
      "title": "Es. 8.1 — Gini a mano",
      "prompt": "<p>Implementa <code class=\"inline\">gini(labels)</code>: dato un array di etichette, ritorna l'impurità di Gini $1 - \\sum p_c^2$.</p>",
      "starter": "import numpy as np\ndef gini(labels):\n    # TODO\n    return 0.0\nprint(gini(np.array([0,0,0,1,1])))  # atteso 1 - (0.6^2 + 0.4^2) = 0.48\nprint(gini(np.array([0,0,0,0])))    # atteso 0\nprint(gini(np.array([0,1,0,1])))    # atteso 0.5",
      "check": "import numpy as np\nif abs(gini(np.array([0,0,0,1,1])) - 0.48) < 1e-6 and    gini(np.array([0,0,0,0])) == 0 and    abs(gini(np.array([0,1,0,1])) - 0.5) < 1e-6:\n    print(\"OK\")\nelse:\n    print(\"FAIL\")",
      "solution": "def gini(labels):\n    _, c = np.unique(labels, return_counts=True)\n    p = c / len(labels)\n    return 1 - (p**2).sum()"
    },
    {
      "type": "exercise",
      "difficulty": "medium",
      "title": "Es. 8.2 — Miglior split",
      "prompt": "<p>Data una feature 1-D <code class=\"inline\">x</code> e etichette <code class=\"inline\">y</code>, trova la soglia $t$ (tra i valori unici di $x$) che minimizza il Gini pesato dei due figli. Salva in <code class=\"inline\">best_t</code>.</p>",
      "starter": "import numpy as np\ndef gini(l):\n    _,c = np.unique(l, return_counts=True); p=c/len(l); return 1-(p**2).sum()\nx = np.array([1,2,3,4,5,6,7,8])\ny = np.array([0,0,0,0,1,1,1,1])\n\n# TODO\nbest_t = None\nprint(\"soglia ottima:\", best_t)",
      "check": "import numpy as np\ndef gini(l):\n    _,c = np.unique(l, return_counts=True); p=c/len(l); return 1-(p**2).sum()\n_thresholds = np.unique(x)\n_scores = {}\nfor t in _thresholds:\n    L, R = y[x<=t], y[x>t]\n    if len(L)==0 or len(R)==0: continue\n    _scores[t] = (len(L)*gini(L) + len(R)*gini(R))/len(y)\n_atteso = min(_scores, key=_scores.get)\nif best_t == _atteso:\n    print(f\"OK — soglia {best_t}, gini {_scores[best_t]:.4f}\")\nelse:\n    print(f\"FAIL — atteso {_atteso}\")",
      "solution": "def score(t):\n    L,R = y[x<=t], y[x>t]\n    if len(L)==0 or len(R)==0: return 1\n    return (len(L)*gini(L) + len(R)*gini(R))/len(y)\nbest_t = min(np.unique(x), key=score)"
    },
    {
      "type": "exercise",
      "difficulty": "medium",
      "title": "Es. 8.3 — RF vs singolo albero",
      "prompt": "<p>Su make_moons, confronta accuracy di <code class=\"inline\">DecisionTreeClassifier(max_depth=None)</code> e <code class=\"inline\">RandomForestClassifier(n_estimators=100)</code> su un test set. Verifica che RF batta l'albero.</p>",
      "starter": "import numpy as np\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.datasets import make_moons\nfrom sklearn.model_selection import train_test_split\n\nX, y = make_moons(n_samples=1000, noise=0.35, random_state=0)\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42)\n\n# TODO\nacc_tree = None\nacc_rf = None\nprint(f\"Tree: {acc_tree}, RF: {acc_rf}\")",
      "check": "if acc_tree is None or acc_rf is None:\n    print(\"FAIL: None\")\nelif acc_rf > acc_tree:\n    print(f\"OK — RF {acc_rf:.4f} > Tree {acc_tree:.4f}\")\nelse:\n    print(f\"WARN — con seed diversi RF batte quasi sempre; qui tree={acc_tree:.4f}, rf={acc_rf:.4f}\")",
      "solution": "acc_tree = DecisionTreeClassifier(random_state=0).fit(Xt, yt).score(Xe, ye)\nacc_rf = RandomForestClassifier(n_estimators=100, random_state=0).fit(Xt, yt).score(Xe, ye)"
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Es. 8.4 — Feature importance",
      "prompt": "<p>Su un dataset in cui solo 3 di 10 feature contano davvero (le altre sono rumore), verifica che RF assegni importanza alta alle prime 3. Salva in <code class=\"inline\">top3</code> la lista degli indici delle 3 feature più importanti.</p>",
      "starter": "import numpy as np\nfrom sklearn.ensemble import RandomForestRegressor\n\nnp.random.seed(0)\nn = 1000\nX = np.random.randn(n, 10)\ny = 3*X[:,0] - 2*X[:,1] + X[:,2]**2 + 0.1*np.random.randn(n)\n\n# TODO: allena RF, estrai importanze, top 3\ntop3 = None\nprint(\"top 3:\", top3)",
      "check": "if top3 is not None and set(top3) == {0, 1, 2}:\n    print(f\"OK — top {top3}\")\nelse:\n    print(f\"FAIL — atteso {{0,1,2}}, tuo {top3}\")",
      "solution": "rf = RandomForestRegressor(n_estimators=200, random_state=0).fit(X, y)\ntop3 = list(np.argsort(rf.feature_importances_)[-3:])"
    }
  ],
  "test": [
    {
      "type": "quiz",
      "questions": [
        {
          "q": "Un albero senza limiti di profondità su training di $n$ esempi tipicamente ha accuracy training:",
          "options": [
            "~ 0.5",
            "Perfetta (100%)",
            "Uguale a Random Forest",
            "Zero"
          ],
          "answer": 1,
          "explain": "Memorizza ogni punto in una foglia dedicata."
        },
        {
          "q": "Random Forest riduce principalmente:",
          "options": [
            "Il bias",
            "La varianza",
            "Entrambi in ugual misura",
            "L'entropia"
          ],
          "answer": 1,
          "explain": "Il bagging riduce la varianza degli alberi ad alta varianza."
        },
        {
          "q": "Il \"random\" in Random Forest deriva da:",
          "options": [
            "Solo bootstrap dei dati",
            "Solo sottoinsiemi di feature",
            "Entrambi",
            "Random init dei pesi"
          ],
          "answer": 2,
          "explain": "Bootstrap sui dati + random subset di feature ad ogni split."
        },
        {
          "q": "Il boosting differisce dal bagging perché:",
          "options": [
            "Non usa alberi",
            "Costruisce modelli in sequenza focalizzati sugli errori",
            "Non ha regolarizzazione",
            "È più veloce"
          ],
          "answer": 1,
          "explain": "Sequenza vs parallelo. Ogni modello corregge il precedente."
        }
      ]
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Test pratico — RF su Wine",
      "prompt": "<p>Sul dataset wine, addestra un RandomForest e trova via GridSearch la miglior combinazione di <code class=\"inline\">n_estimators \\in \\{50, 200, 500\\}</code> e <code class=\"inline\">max_depth \\in \\{None, 5, 10\\}</code>. Salva l'accuracy test in <code class=\"inline\">acc_final</code> (target &gt; 0.96).</p>",
      "starter": "from sklearn.datasets import load_wine\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import GridSearchCV, train_test_split\n\nX, y = load_wine(return_X_y=True)\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)\n\n# TODO\nacc_final = None",
      "check": "if acc_final is None: print(\"FAIL: None\")\nelif acc_final > 0.96: print(f\"OK — {acc_final:.4f}\")\nelse: print(f\"FAIL — {acc_final:.4f}\")",
      "solution": "grid = {'n_estimators':[50,200,500], 'max_depth':[None,5,10]}\ngs = GridSearchCV(RandomForestClassifier(random_state=0), grid, cv=5).fit(Xt, yt)\nacc_final = gs.score(Xe, ye)"
    }
  ]
});
