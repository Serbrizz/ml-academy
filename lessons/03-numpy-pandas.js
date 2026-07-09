/* Lezione: NumPy e Pandas */
window.LESSONS.push({
  "id": "l03-numpy-pandas",
  "title": "NumPy e Pandas",
  "tag": "Strumenti",
  "summary": "Vettorializzazione, broadcasting, DataFrame — l'ambiente in cui vive il ML.",
  "teoria": [
    {
      "type": "md",
      "content": "\n<h3>3.1 Perché NumPy</h3>\n<p>Python puro è lento per il calcolo numerico. NumPy sposta i loop dentro C compilato, con dati in memoria contigua. Un'operazione vettoriale in NumPy è tipicamente <strong>50-500 volte più veloce</strong> del loop Python equivalente.</p>\n<p>La regola d'oro: <em>se stai scrivendo un for-loop su un array, quasi certamente c'è un modo vettorializzato più elegante e più veloce.</em></p>\n\n<h3>3.2 Array: shape, dtype, axis</h3>\n<p>Un <code class=\"inline\">ndarray</code> ha una <em>shape</em> (tupla di dimensioni), un <em>dtype</em> (tipo dei singoli elementi), e uno <em>strides</em> (come si passa da un elemento all'altro in memoria). Non devi pensare agli strides quasi mai, ma capire che array multidimensionali sono <em>views</em> su un blocco piatto di memoria aiuta.</p>\n<p>Un dataset tipico ha shape <code class=\"inline\">(n_samples, n_features)</code>. Un batch di immagini RGB ha shape <code class=\"inline\">(batch, H, W, 3)</code>.</p>\n\n<h3>3.3 Broadcasting</h3>\n<p>Regola: quando fai un'operazione tra array di shape diverse, NumPy le \"estende\" (senza copiare memoria) se sono <em>compatibili</em>. Sono compatibili se, allineando le shape a destra, ogni dimensione è uguale o vale 1.</p>\n<pre class=\"code\">A shape (3, 4)\nb shape    (4,)     -> ok, b viene \"ripetuto\" sulle 3 righe\nc shape (3, 1)      -> ok, c viene \"ripetuto\" sulle 4 colonne\nd shape (2, 4)      -> ERRORE (2 vs 3)</pre>\n<p>Broadcasting è come sono implementati praticamente tutti i preprocessing di dati. Esempio classico: standardizzare le colonne.</p>\n\n<h3>3.4 Pandas: DataFrame</h3>\n<p>Pandas costruisce sopra NumPy la nozione di <em>tabella con nomi</em>: righe con indice, colonne con nome, tipi eterogenei. È il coltellino svizzero per la fase esplorativa.</p>\n<ul>\n<li><code class=\"inline\">df.loc[etichette]</code> selezione per etichetta</li>\n<li><code class=\"inline\">df.iloc[posizioni]</code> selezione per posizione intera</li>\n<li><code class=\"inline\">df.groupby('col').agg(...)</code> aggregazioni</li>\n<li><code class=\"inline\">df.merge(altro, on='...')</code> join tra tabelle</li>\n<li><code class=\"inline\">df.isna().sum()</code> conteggio valori mancanti</li>\n</ul>\n\n<h3>3.5 Split, indexing, boolean masks</h3>\n<p>Il modo idiomatico di filtrare dati:</p>\n<pre class=\"code\">mask = df['age'] &gt; 30\nadulti = df[mask]\ndf.loc[mask, 'salario'] *= 1.1</pre>\n"
    },
    {
      "type": "callout",
      "variant": "tip",
      "title": "Regola pratica",
      "content": "Ogni volta che scrivi un for-loop che itera su righe di un DataFrame, fermati 30 secondi e chiediti: \"posso fare la stessa cosa con un'operazione vettoriale, un groupby, o apply?\" Nel 90% dei casi la risposta è sì."
    }
  ],
  "esempi": [
    {
      "type": "md",
      "content": "<h3>Esempio 1: broadcasting per standardizzare</h3><p>Sottraiamo la media colonna per colonna e dividiamo per la deviazione standard. Un'operazione tipica nel preprocessing.</p>"
    },
    {
      "type": "run",
      "content": "import numpy as np\n\nnp.random.seed(0)\nX = np.random.randn(5, 3) * np.array([1, 10, 100]) + np.array([0, 50, 200])\nprint(\"Dati grezzi:\\n\", np.round(X, 2))\n\nmu = X.mean(axis=0)      # media per colonna, shape (3,)\nsigma = X.std(axis=0)    # std per colonna, shape (3,)\nX_std = (X - mu) / sigma # broadcasting: shape (5,3) - shape (3,)\n\nprint(\"\\nMedia dopo std:\", np.round(X_std.mean(axis=0), 6))\nprint(\"Std   dopo std:\", np.round(X_std.std(axis=0), 6))"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 2: fancy indexing e boolean mask</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\na = np.arange(10, 20)\nprint(\"array:\", a)\nprint(\"indici pari:\", a[[0, 2, 4, 6, 8]])\nprint(\"elementi > 15:\", a[a > 15])\nprint(\"modifica in place:\")\na[a > 15] = 0\nprint(a)"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 3: Pandas basics</h3>"
    },
    {
      "type": "run",
      "content": "import pandas as pd\nimport numpy as np\n\ndf = pd.DataFrame({\n    'nome': ['Anna','Bruno','Carlo','Diana','Elisa','Franco'],\n    'eta': [23, 45, 34, 28, 51, 39],\n    'citta': ['Roma','Milano','Roma','Torino','Milano','Roma'],\n    'salario': [28000, 55000, 42000, 33000, 68000, 47000]\n})\nprint(df)\nprint(\"\\nStatistiche:\\n\", df.describe())\nprint(\"\\nSalario medio per citta:\")\nprint(df.groupby('citta')['salario'].mean())\nprint(\"\\nPersone con salario > 40k:\")\nprint(df[df.salario > 40000][['nome','salario']])"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 4: pivot e valori mancanti</h3>"
    },
    {
      "type": "run",
      "content": "import pandas as pd\nimport numpy as np\n\ndf = pd.DataFrame({\n    'prodotto': ['A','A','B','B','C','C'],\n    'mese': ['gen','feb','gen','feb','gen','feb'],\n    'vendite': [100, 120, 80, np.nan, 150, 170]\n})\nprint(\"Dati lunghi:\\n\", df)\npivot = df.pivot(index='prodotto', columns='mese', values='vendite')\nprint(\"\\nPivot:\\n\", pivot)\nprint(\"\\nRiempo NaN con la media del prodotto:\")\npivot['feb'] = pivot['feb'].fillna(pivot.mean(axis=1))\nprint(pivot)"
    }
  ],
  "esercizi": [
    {
      "type": "exercise",
      "difficulty": "easy",
      "title": "Es. 3.1 — Vettorializza",
      "prompt": "<p>Riscrivi questo loop in versione vettorializzata NumPy. Il risultato deve essere identico ma senza <code class=\"inline\">for</code>.</p><pre class=\"code\">out = []\nfor v in x:\n    out.append(v**2 + 3*v - 1)\nresult = np.array(out)</pre>",
      "starter": "import numpy as np\nx = np.arange(-5, 6)\n\n# TODO: versione vettorializzata\nresult = None\nprint(result)",
      "check": "import numpy as np\n_atteso = x**2 + 3*x - 1\nif result is not None and np.allclose(result, _atteso):\n    print(\"OK\")\nelse:\n    print(f\"FAIL: atteso {_atteso}, tuo {result}\")",
      "solution": "result = x**2 + 3*x - 1"
    },
    {
      "type": "exercise",
      "difficulty": "medium",
      "title": "Es. 3.2 — Distanze pairwise",
      "prompt": "<p>Data una matrice <code class=\"inline\">X</code> di shape (n, d), calcola la matrice <code class=\"inline\">D</code> di shape (n, n) dove <code class=\"inline\">D[i,j]</code> è la distanza euclidea tra la riga $i$ e la riga $j$. Usa broadcasting, no loop.</p><p><em>Hint:</em> $\\|x_i - x_j\\|^2 = \\|x_i\\|^2 + \\|x_j\\|^2 - 2 x_i^T x_j$.</p>",
      "starter": "import numpy as np\nnp.random.seed(0)\nX = np.random.randn(5, 3)\n\n# TODO: matrice D di shape (5,5) senza loop\nD = None\nprint(np.round(D, 3))",
      "check": "import numpy as np\nfrom scipy.spatial.distance import cdist  # potrebbe non essere disponibile\n_atteso = np.sqrt(((X[:,None,:] - X[None,:,:])**2).sum(-1))\nif D is not None and np.allclose(D, _atteso, atol=1e-6):\n    print(\"OK\")\nelse:\n    print(\"FAIL\")",
      "solution": "D = np.sqrt(((X[:,None,:] - X[None,:,:])**2).sum(-1))\n# oppure: sq = (X**2).sum(1); D = np.sqrt(sq[:,None]+sq[None,:] - 2*X@X.T)"
    },
    {
      "type": "exercise",
      "difficulty": "medium",
      "title": "Es. 3.3 — Analisi su DataFrame",
      "prompt": "<p>Dato il DataFrame di partenza, trova per ogni <strong>città</strong> il salario massimo e il nome della persona che lo ha. Il risultato deve essere un DataFrame con colonne <code class=\"inline\">['citta', 'nome', 'salario']</code>, salvato in <code class=\"inline\">top_citta</code>.</p>",
      "starter": "import pandas as pd\ndf = pd.DataFrame({\n    'nome': ['Anna','Bruno','Carlo','Diana','Elisa','Franco'],\n    'eta': [23, 45, 34, 28, 51, 39],\n    'citta': ['Roma','Milano','Roma','Torino','Milano','Roma'],\n    'salario': [28000, 55000, 42000, 33000, 68000, 47000]\n})\n\n# TODO\ntop_citta = None\nprint(top_citta)",
      "check": "import pandas as pd\nif top_citta is None:\n    print(\"FAIL: None\")\nelse:\n    ok = (\n        set(top_citta['citta']) == {'Roma','Milano','Torino'} and\n        top_citta.loc[top_citta.citta=='Roma','nome'].iloc[0] == 'Franco' and\n        top_citta.loc[top_citta.citta=='Milano','nome'].iloc[0] == 'Elisa'\n    )\n    print(\"OK\" if ok else \"FAIL — controlla nomi/città\")",
      "solution": "idx = df.groupby('citta')['salario'].idxmax()\ntop_citta = df.loc[idx, ['citta','nome','salario']].reset_index(drop=True)"
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Es. 3.4 — Rolling mean",
      "prompt": "<p>Data una serie temporale <code class=\"inline\">ts</code>, calcola la media mobile su una finestra di 3 elementi senza usare <code class=\"inline\">pandas.rolling</code>. Il risultato dev'essere lungo <code class=\"inline\">len(ts) - 2</code>.</p>",
      "starter": "import numpy as np\nts = np.array([1., 2., 3., 4., 5., 6., 7., 8.])\n\n# TODO\nmm = None\nprint(mm)  # atteso [2, 3, 4, 5, 6, 7]",
      "check": "import numpy as np\natteso = np.array([2,3,4,5,6,7], dtype=float)\nif mm is not None and np.allclose(mm, atteso):\n    print(\"OK\")\nelse:\n    print(f\"FAIL: atteso {atteso}, tuo {mm}\")",
      "solution": "mm = (ts[:-2] + ts[1:-1] + ts[2:]) / 3\n# alternativa: np.convolve(ts, np.ones(3)/3, mode='valid')"
    }
  ],
  "test": [
    {
      "type": "quiz",
      "questions": [
        {
          "q": "Qual è la shape di <code class=\"inline\">np.zeros((3,4))@np.ones((4,))</code>?",
          "options": [
            "(3,4)",
            "(3,)",
            "(4,)",
            "Errore"
          ],
          "answer": 1,
          "explain": "Matrice (3,4) per vettore (4,) dà vettore (3,)."
        },
        {
          "q": "In Pandas, <code class=\"inline\">df.iloc[2]</code> restituisce:",
          "options": [
            "La riga con indice etichetta 2",
            "La terza riga (posizione 2)",
            "La seconda colonna",
            "Errore se non c'è etichetta 2"
          ],
          "answer": 1,
          "explain": "iloc è per posizione intera."
        },
        {
          "q": "Broadcasting tra shape (5,1,4) e (3,4) produce shape:",
          "options": [
            "(5,3,4)",
            "(5,4)",
            "(3,5,4)",
            "Errore"
          ],
          "answer": 0,
          "explain": "Allineando a destra: (5,1,4) e (_,3,4). 1 si estende a 3, dimensione mancante diventa 5. Risultato (5,3,4)."
        },
        {
          "q": "Per iterare su righe di un DataFrame di 1M righe conviene:",
          "options": [
            "for i, row in df.iterrows()",
            "Convertire a lista e iterare",
            "Vettorializzare o usare df.apply solo se necessario",
            "Usare un ciclo while"
          ],
          "answer": 2,
          "explain": "iterrows è lentissimo; sempre preferire operazioni vettoriali."
        }
      ]
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Test pratico — mini EDA",
      "prompt": "<p>Ti passo un DataFrame di transazioni. Devi produrre queste 3 statistiche:</p><ol><li><code class=\"inline\">stat_total</code>: totale speso da ogni cliente (Series)</li><li><code class=\"inline\">stat_top_prod</code>: prodotto più venduto (stringa)</li><li><code class=\"inline\">stat_gg_medio</code>: numero medio di transazioni al giorno (float)</li></ol>",
      "starter": "import pandas as pd\ndf = pd.DataFrame({\n    'giorno': ['2024-01-01','2024-01-01','2024-01-02','2024-01-02','2024-01-02','2024-01-03'],\n    'cliente': ['A','B','A','C','B','A'],\n    'prodotto': ['libro','penna','libro','laptop','libro','penna'],\n    'importo': [15, 3, 12, 900, 18, 4]\n})\n\nstat_total = None\nstat_top_prod = None\nstat_gg_medio = None\n\nprint(\"Totale per cliente:\\n\", stat_total)\nprint(\"Prodotto top:\", stat_top_prod)\nprint(\"Transazioni/giorno:\", stat_gg_medio)",
      "check": "ok = True\nif not (stat_total is not None and stat_total.get('A')==27 and stat_total.get('B')==21 and stat_total.get('C')==900): ok=False\nif stat_top_prod != 'libro': ok=False\nif not (abs(stat_gg_medio - 2.0) < 1e-6): ok=False\nprint(\"OK\" if ok else \"FAIL — controlla i tre valori\")",
      "solution": "stat_total = df.groupby('cliente')['importo'].sum()\nstat_top_prod = df['prodotto'].value_counts().idxmax()\nstat_gg_medio = df.groupby('giorno').size().mean()"
    }
  ]
});
