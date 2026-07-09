/* Lezione: Esplorazione dei dati e visualizzazione */
window.LESSONS.push({
  "id": "l04-eda",
  "title": "Esplorazione dei dati e visualizzazione",
  "tag": "Workflow",
  "summary": "L'80% del lavoro di un data scientist. Come guardare i dati prima di modellarli.",
  "teoria": [
    {
      "type": "md",
      "content": "\n<h3>4.1 Perché EDA</h3>\n<p>Prima di lanciare qualsiasi modello, devi <strong>guardare i dati</strong>. La Exploratory Data Analysis (Tukey, anni '60) è la fase in cui rispondi a domande come:</p>\n<ul>\n<li>Che tipo hanno le variabili? Quali sono numeriche, categoriche, ordinali?</li>\n<li>Ci sono valori mancanti? Come sono distribuiti?</li>\n<li>Le distribuzioni sono ragionevoli o ci sono outlier assurdi?</li>\n<li>Ci sono correlazioni evidenti tra variabili? Con il target?</li>\n<li>La classe target è bilanciata o sbilanciata?</li>\n</ul>\n<p>Se salti l'EDA, quasi certamente costruirai un modello che fallisce silenziosamente. Un modello mediocre su dati puliti batte un modello sofisticato su dati sporchi.</p>\n\n<h3>4.2 Statistica descrittiva</h3>\n<p>Le prime cose da guardare per una variabile numerica:</p>\n<ul>\n<li><strong>Tendenza centrale</strong>: media, mediana</li>\n<li><strong>Dispersione</strong>: deviazione standard, IQR (differenza tra 3° e 1° quartile)</li>\n<li><strong>Forma</strong>: skewness (asimmetria), kurtosi (code)</li>\n<li><strong>Estremi</strong>: min, max, percentili</li>\n</ul>\n<p>La differenza media-mediana è un ottimo detector di skewness: se media &gg; mediana, la distribuzione ha coda a destra (es. redditi).</p>\n\n<h3>4.3 Grafici essenziali</h3>\n<ul>\n<li><strong>Istogramma / KDE</strong>: distribuzione di una variabile numerica</li>\n<li><strong>Boxplot</strong>: mediana, quartili, outlier</li>\n<li><strong>Scatter plot</strong>: relazione tra due numeriche</li>\n<li><strong>Bar chart</strong>: conteggi per categoria</li>\n<li><strong>Heatmap di correlazioni</strong>: sguardo globale sulle relazioni lineari</li>\n<li><strong>Pair plot</strong>: tutti gli scatter a coppie in una griglia</li>\n</ul>\n\n<h3>4.4 Gestione dei missing values</h3>\n<p>Strategie principali:</p>\n<ol>\n<li><strong>Elimina</strong> le righe o colonne con troppi missing. Solo se sono pochi.</li>\n<li><strong>Imputa</strong> con media/mediana (numeriche) o moda (categoriche).</li>\n<li><strong>Imputa</strong> con un modello (KNN, regressione).</li>\n<li><strong>Missing come informazione</strong>: aggiungi una feature indicatrice \"era missing?\".</li>\n</ol>\n<p>Attenzione al <em>data leakage</em>: calcola le statistiche di imputazione SOLO sul training set, non su tutto il dataset.</p>\n\n<h3>4.5 Outlier: rimuovere o no?</h3>\n<p>Un outlier può essere:</p>\n<ul>\n<li>Un errore di misurazione &rarr; rimuovilo</li>\n<li>Un evento raro ma reale &rarr; tienilo (magari con un modello robusto)</li>\n<li>La cosa che vuoi predire (frode, guasti) &rarr; è l'oggetto stesso del task</li>\n</ul>\n<p>Regola pratica: guarda l'outlier, capisci perché è lì, poi decidi. Non applicare mai una regola \"rimuovi tutto oltre 3 sigma\" senza aver capito i dati.</p>\n\n<h3>4.6 Correlazione ≠ causalità</h3>\n<p>La correlazione di Pearson misura la relazione <em>lineare</em>. Due variabili possono avere correlazione zero ed essere fortemente dipendenti (es. $y = x^2$ con $x$ centrato). Usa anche Spearman (rank correlation) per relazioni monotone non lineari.</p>\n"
    },
    {
      "type": "callout",
      "variant": "warn",
      "title": "Attenzione",
      "content": "Ricorda il quartetto di Anscombe: quattro dataset con identiche statistiche riassuntive ma visualizzazioni totalmente diverse. La conclusione è che le statistiche riassuntive non bastano — devi vedere i grafici."
    }
  ],
  "esempi": [
    {
      "type": "md",
      "content": "<h3>Esempio 1: dataset Iris — il classico</h3><p>Iris è il \"dataset di riferimento\" del ML. 150 fiori, 4 misure, 3 specie.</p>"
    },
    {
      "type": "run",
      "content": "from sklearn.datasets import load_iris\nimport pandas as pd\n\niris = load_iris(as_frame=True)\ndf = iris.frame\ndf['specie'] = df['target'].map(dict(enumerate(iris.target_names)))\nprint(\"Shape:\", df.shape)\nprint(\"\\nPrime righe:\\n\", df.head())\nprint(\"\\nStatistiche:\\n\", df.describe())\nprint(\"\\nConteggio per specie:\\n\", df['specie'].value_counts())"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 2: istogrammi e boxplot</h3>"
    },
    {
      "type": "run",
      "content": "import matplotlib.pyplot as plt\nfrom sklearn.datasets import load_iris\nimport pandas as pd\n\niris = load_iris(as_frame=True)\ndf = iris.frame\n\nfig, axes = plt.subplots(2, 2, figsize=(10, 6))\nfor ax, col in zip(axes.flatten(), df.columns[:-1]):\n    for cls in [0,1,2]:\n        ax.hist(df.loc[df.target==cls, col], alpha=0.5,\n                bins=15, label=f'classe {cls}')\n    ax.set_title(col); ax.legend(fontsize=8)\nplt.tight_layout()\nplt.show()"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 3: matrice di correlazione</h3>"
    },
    {
      "type": "run",
      "content": "import matplotlib.pyplot as plt\nimport numpy as np\nfrom sklearn.datasets import load_iris\n\niris = load_iris(as_frame=True)\ndf = iris.frame\ncorr = df.drop('target', axis=1).corr()\nprint(\"Matrice di correlazione:\\n\", corr.round(3))\n\nfig, ax = plt.subplots(figsize=(6,5))\nim = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)\nax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))\nax.set_xticklabels(corr.columns, rotation=45, ha='right')\nax.set_yticklabels(corr.columns)\nfor i in range(len(corr)):\n    for j in range(len(corr)):\n        ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', color='k', fontsize=9)\nplt.colorbar(im, ax=ax)\nplt.tight_layout(); plt.show()"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 4: outlier detection con IQR</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nnp.random.seed(0)\ndata = np.concatenate([np.random.normal(50, 5, 100), [120, 130, -10]])  # 3 outlier\n\nq1, q3 = np.percentile(data, [25, 75])\niqr = q3 - q1\nlo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr\nprint(f\"Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}\")\nprint(f\"Range accettabile: [{lo:.2f}, {hi:.2f}]\")\noutliers = data[(data < lo) | (data > hi)]\nprint(f\"Outlier trovati: {outliers}\")"
    }
  ],
  "esercizi": [
    {
      "type": "exercise",
      "difficulty": "easy",
      "title": "Es. 4.1 — Statistiche di base",
      "prompt": "<p>Data una serie di valori con qualche NaN, calcola media, mediana, e conta i NaN. Salva rispettivamente in <code class=\"inline\">m, med, n_nan</code>.</p>",
      "starter": "import numpy as np\nimport pandas as pd\ns = pd.Series([1.2, 3.5, np.nan, 4.1, 2.8, np.nan, 5.0, 3.3, np.nan, 4.7])\n\nm = None\nmed = None\nn_nan = None\nprint(m, med, n_nan)",
      "check": "import numpy as np\nif abs(m - s.mean()) < 1e-9 and abs(med - s.median()) < 1e-9 and n_nan == 3:\n    print(\"OK\")\nelse:\n    print(f\"FAIL: attesi mean={s.mean():.3f}, median={s.median():.3f}, nan=3\")",
      "solution": "m = s.mean(); med = s.median(); n_nan = s.isna().sum()"
    },
    {
      "type": "exercise",
      "difficulty": "medium",
      "title": "Es. 4.2 — Skewness manuale",
      "prompt": "<p>Implementa la skewness campionaria: $\\text{skew} = \\frac{1}{n}\\sum \\left(\\frac{x_i - \\bar x}{s}\\right)^3$. Usa il numpy std di default (ddof=0). Testa su dati normali (skew ~ 0) e su dati esponenziali (skew > 0).</p>",
      "starter": "import numpy as np\ndef skew_manual(x):\n    # TODO\n    return 0.0\n\nnp.random.seed(0)\nprint(\"normale:      \", skew_manual(np.random.normal(0,1,10000)))\nprint(\"esponenziale: \", skew_manual(np.random.exponential(1,10000)))",
      "check": "import numpy as np\nnp.random.seed(0)\ns1 = skew_manual(np.random.normal(0,1,10000))\ns2 = skew_manual(np.random.exponential(1,10000))\nif abs(s1) < 0.1 and s2 > 1.5:\n    print(\"OK — skew normale ~ 0, esponenziale > 1.5\")\nelse:\n    print(f\"FAIL — normale {s1:.3f}, esp {s2:.3f}\")",
      "solution": "def skew_manual(x):\n    x = np.asarray(x, float)\n    z = (x - x.mean()) / x.std()\n    return np.mean(z**3)"
    },
    {
      "type": "exercise",
      "difficulty": "medium",
      "title": "Es. 4.3 — Imputazione mediana per gruppo",
      "prompt": "<p>Riempi i NaN nella colonna <code class=\"inline\">salario</code> con la mediana <strong>del gruppo</strong> a cui appartiene la riga (colonna <code class=\"inline\">reparto</code>). Salva il risultato in <code class=\"inline\">df['salario']</code>.</p>",
      "starter": "import pandas as pd\nimport numpy as np\ndf = pd.DataFrame({\n    'reparto': ['R&D','R&D','R&D','HR','HR','HR','R&D','HR'],\n    'salario':[50000, np.nan, 60000, 30000, 32000, np.nan, 55000, np.nan]\n})\n\n# TODO\nprint(df)",
      "check": "import pandas as pd\nok = df['salario'].isna().sum()==0\n# la mediana R&D è tra {50000,55000,60000} = 55000\n# la mediana HR è tra {30000,32000} = 31000\nok = ok and df.loc[df.reparto=='R&D','salario'].iloc[1] == 55000\nok = ok and df.loc[df.reparto=='HR','salario'].iloc[2] == 31000\nprint(\"OK\" if ok else \"FAIL — imputazione errata\")",
      "solution": "df['salario'] = df.groupby('reparto')['salario'].transform(lambda s: s.fillna(s.median()))"
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Es. 4.4 — Feature engineering",
      "prompt": "<p>Dato un DataFrame con timestamp di transazioni per utente, crea una nuova colonna <code class=\"inline\">giorni_da_ultimo</code>: quanti giorni sono passati tra ogni transazione e la precedente dello stesso utente. La prima transazione di ogni utente ha valore NaN.</p>",
      "starter": "import pandas as pd\ndf = pd.DataFrame({\n    'user': ['A','A','B','A','B','C','B'],\n    'data': pd.to_datetime(['2024-01-05','2024-01-10','2024-01-06','2024-01-20','2024-01-15','2024-01-08','2024-01-25'])\n}).sort_values(['user','data']).reset_index(drop=True)\n\n# TODO: crea colonna 'giorni_da_ultimo'\nprint(df)",
      "check": "import pandas as pd, numpy as np\nok = True\nif 'giorni_da_ultimo' not in df.columns: ok=False\nelse:\n    a = df[df.user=='A']['giorni_da_ultimo'].tolist()\n    b = df[df.user=='B']['giorni_da_ultimo'].tolist()\n    if not (np.isnan(a[0]) and a[1]==5 and a[2]==10): ok=False\n    if not (np.isnan(b[0]) and b[1]==9 and b[2]==10): ok=False\nprint(\"OK\" if ok else \"FAIL\")",
      "solution": "df['giorni_da_ultimo'] = df.groupby('user')['data'].diff().dt.days"
    }
  ],
  "test": [
    {
      "type": "quiz",
      "questions": [
        {
          "q": "Se media &gt; mediana in una distribuzione, la skewness è tipicamente:",
          "options": [
            "Negativa (coda a sinistra)",
            "Zero",
            "Positiva (coda a destra)",
            "Non definita"
          ],
          "answer": 2,
          "explain": "Coda lunga a destra tira la media in alto rispetto alla mediana."
        },
        {
          "q": "Il quartetto di Anscombe insegna che:",
          "options": [
            "La media è la migliore statistica",
            "Le statistiche riassuntive possono nascondere pattern importanti",
            "La correlazione è sempre affidabile",
            "I grafici sono superflui"
          ],
          "answer": 1,
          "explain": "4 dataset con stesse statistiche hanno grafici totalmente diversi."
        },
        {
          "q": "Calcolare la media per imputare i NaN usando tutto il dataset (train+test) è:",
          "options": [
            "La pratica corretta",
            "Data leakage: usa solo il training",
            "Ininfluente",
            "Consigliato in caso di pochi dati"
          ],
          "answer": 1,
          "explain": "Includere il test nel calcolo delle statistiche contamina la valutazione."
        },
        {
          "q": "Correlazione di Pearson tra $x$ e $y = x^2$ con $x \\sim \\mathcal{N}(0,1)$:",
          "options": [
            "~ 1",
            "~ -1",
            "~ 0",
            "Dipende dal campione"
          ],
          "answer": 2,
          "explain": "Simmetria: la relazione è perfettamente non lineare ma con correlazione lineare nulla."
        }
      ]
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Test pratico — EDA completa",
      "prompt": "<p>Sul dataset Iris, calcola:</p><ol><li><code class=\"inline\">v_max_corr</code>: la coppia di variabili numeriche con correlazione più alta (tupla di 2 nomi)</li><li><code class=\"inline\">v_min_var</code>: nome della variabile con varianza minore</li><li><code class=\"inline\">v_disc</code>: la variabile che meglio separa le classi (quella con maggior rapporto varianza tra-gruppi / varianza intra-gruppi)</li></ol>",
      "starter": "from sklearn.datasets import load_iris\nimport pandas as pd\nimport numpy as np\niris = load_iris(as_frame=True)\ndf = iris.frame.rename(columns={'target':'y'})\nfeatures = [c for c in df.columns if c != 'y']\n\nv_max_corr = None\nv_min_var = None\nv_disc = None\n\nprint(v_max_corr, v_min_var, v_disc)",
      "check": "ok = True\nif v_max_corr is None or set(v_max_corr) != {'petal length (cm)','petal width (cm)'}: ok=False\nif v_min_var != 'sepal width (cm)': ok=False\nif v_disc not in ('petal length (cm)','petal width (cm)'): ok=False\nprint(\"OK\" if ok else f\"FAIL: {v_max_corr}, {v_min_var}, {v_disc}\")",
      "solution": "import numpy as np\n# max correlation off-diagonal\nc = df[features].corr().abs()\nnp.fill_diagonal(c.values, 0)\ni, j = np.unravel_index(np.argmax(c.values), c.shape)\nv_max_corr = (c.index[i], c.columns[j])\n\nv_min_var = df[features].var().idxmin()\n\n# rapporto discriminante\ndef f_ratio(col):\n    mu = df[col].mean()\n    between = df.groupby('y')[col].mean().sub(mu).pow(2).mul(df.groupby('y').size()).sum()\n    within = df.groupby('y')[col].var(ddof=0).mul(df.groupby('y').size()).sum()\n    return between / within\nv_disc = max(features, key=f_ratio)"
    }
  ]
});
