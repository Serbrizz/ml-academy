/* Lezione: Cos'è il Machine Learning */
window.LESSONS.push({
  "id": "l01-intro",
  "title": "Cos'è il Machine Learning",
  "tag": "Fondamenti",
  "summary": "Definizioni, tipi di apprendimento, terminologia essenziale.",
  "teoria": [
    {
      "type": "md",
      "content": "\n<h3>1.1 Definizione operativa</h3>\n<p>Il machine learning è la disciplina che studia algoritmi capaci di <strong>migliorare le proprie prestazioni su un compito attraverso l'esperienza</strong>. La definizione classica di Tom Mitchell (1997) è quasi una formula: un programma apprende dall'esperienza $E$ rispetto a un compito $T$ misurato da $P$ se le prestazioni su $T$, misurate da $P$, migliorano con $E$.</p>\n<p>In termini fisici puoi pensarci così: non stai scrivendo l'equazione del moto a mano, stai facendo <em>fit di una funzione</em> a partire dai dati. La differenza rispetto al fit classico è che le funzioni che consideriamo hanno milioni di parametri e non hanno una struttura fisica imposta a priori.</p>\n\n<h3>1.2 Il paradigma: dai dati alla funzione</h3>\n<p>In programmazione tradizionale scrivi: <code class=\"inline\">regole + dati &rarr; risultati</code>. Nel machine learning inverti: <code class=\"inline\">dati + risultati &rarr; regole</code>. L'algoritmo di apprendimento è quindi un <em>meta-programma</em> che produce un programma (il modello) a partire da esempi.</p>\n\n<h3>1.3 I tre grandi paradigmi</h3>\n<p><strong>Apprendimento supervisionato.</strong> Hai coppie $(x_i, y_i)$ dove $x_i \\in \\mathbb{R}^d$ è un vettore di features e $y_i$ è l'etichetta. Vuoi imparare $f : \\mathbb{R}^d \\to \\mathcal{Y}$ tale che $f(x) \\approx y$ su nuovi dati. Se $\\mathcal{Y}$ è discreto è <em>classificazione</em>; se è continuo è <em>regressione</em>.</p>\n<p><strong>Apprendimento non supervisionato.</strong> Hai solo $x_i$, senza etichette. Vuoi scoprire struttura: cluster, direzioni principali di variabilità, anomalie.</p>\n<p><strong>Reinforcement learning.</strong> Un agente interagisce con un ambiente, osserva stati $s_t$, sceglie azioni $a_t$, riceve ricompense $r_t$. Vuoi imparare una policy $\\pi(a|s)$ che massimizza la ricompensa cumulativa attesa $\\mathbb{E}\\left[\\sum_t \\gamma^t r_t\\right]$.</p>\n\n<h3>1.4 Formalizzazione dell'apprendimento supervisionato</h3>\n<p>Assumiamo che i dati provengano da una distribuzione congiunta ignota $\\mathcal{D}$ su $\\mathcal{X} \\times \\mathcal{Y}$. Data una funzione di loss $\\ell(\\hat{y}, y)$, definiamo il <em>rischio atteso</em>:</p>\n<p>$$R(f) = \\mathbb{E}_{(x,y) \\sim \\mathcal{D}}\\left[\\ell(f(x), y)\\right]$$</p>\n<p>Non conosciamo $\\mathcal{D}$, quindi minimizziamo il <em>rischio empirico</em> su un training set $\\{(x_i, y_i)\\}_{i=1}^n$:</p>\n<p>$$\\hat{R}(f) = \\frac{1}{n}\\sum_{i=1}^n \\ell(f(x_i), y_i)$$</p>\n<p>Il grande tema del ML è: quando il minimo di $\\hat{R}$ è vicino al minimo di $R$? Questa è la questione della <em>generalizzazione</em>.</p>\n\n<h3>1.5 Terminologia essenziale</h3>\n<ul>\n<li><strong>Feature</strong> (o predittore, variabile indipendente): una componente di $x$.</li>\n<li><strong>Label</strong> (o target, variabile dipendente): $y$.</li>\n<li><strong>Modello</strong>: la funzione $f_\\theta$ parametrizzata da $\\theta$.</li>\n<li><strong>Training</strong>: il processo di trovare $\\theta$ ottimale.</li>\n<li><strong>Inference</strong>: usare il modello addestrato per fare previsioni.</li>\n<li><strong>Overfitting</strong>: il modello memorizza il training set ma non generalizza. $\\hat{R}$ piccolo, $R$ grande.</li>\n<li><strong>Underfitting</strong>: il modello è troppo semplice per catturare il pattern. Entrambi grandi.</li>\n</ul>\n"
    },
    {
      "type": "callout",
      "variant": "note",
      "title": "Analogia con la fisica",
      "content": "Overfitting è come fittare un polinomio di grado 20 su 10 punti sperimentali: passi per tutti i punti ma il modello non ha significato fisico. Il bias-variance tradeoff che vedremo dopo è essenzialmente la stessa questione della scelta del grado del polinomio, generalizzata."
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
          "explain": "Non ci sono etichette, cerchiamo struttura: clustering, non supervisionato."
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
          "explain": "Per definizione, $\\hat{R} = \\frac{1}{n}\\sum_i \\ell(f(x_i), y_i)$ sul training set."
        },
        {
          "q": "Un modello con training accuracy 99% e test accuracy 60% è tipicamente:",
          "options": [
            "Underfit",
            "Overfit",
            "Ben calibrato",
            "Non identificabile"
          ],
          "answer": 1,
          "explain": "Grande gap train-test è la firma dell'overfitting."
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
          "explain": "$x, y$ sono i dati (fissi); $w, b$ sono i parametri da apprendere."
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
