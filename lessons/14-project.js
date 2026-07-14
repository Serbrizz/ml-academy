/* Lezione: Progetto finale: un modello end-to-end */
window.LESSONS.push({
  "id": "l14-project",
  "title": "Progetto finale: un modello end-to-end",
  "tag": "Progetto",
  "summary": "Mettere tutto insieme: EDA, feature engineering, modelli, tuning, valutazione.",
  "teoria": [
    {
      "type": "md",
      "content": "\n<h3>14.1 Il workflow di un progetto ML</h3>\n<p>Ora che conosci i pezzi, ecco l'ordine in cui li combini per un progetto reale.</p>\n<ol>\n<li><strong>Definisci il problema</strong>: cosa vuoi predire? Regressione o classificazione? Che metrica ti interessa davvero? (Accuracy non è sempre la risposta giusta)</li>\n<li><strong>Raccogli e ispeziona i dati</strong>: EDA, statistiche descrittive, grafici, valori mancanti, outlier.</li>\n<li><strong>Split train/val/test</strong>: <em>subito</em>, prima di guardare i dati troppo a lungo. Isola il test.</li>\n<li><strong>Baseline</strong>: modello stupido (majority class, media, regressione lineare). Ti dice il minimo accettabile.</li>\n<li><strong>Preprocessing e feature engineering</strong>: dentro Pipeline. Standardizza, encoding categoriche, feature derivate.</li>\n<li><strong>Prova più modelli</strong>: lineare, tree, ensemble, magari MLP. Con CV per stimare performance.</li>\n<li><strong>Tuning</strong>: grid search o random search sui migliori 1-2 modelli.</li>\n<li><strong>Valutazione finale</strong>: sul test set, una volta sola. Riporta con intervallo di confidenza.</li>\n<li><strong>Interpretazione</strong>: feature importance, esempi predetti male, casi limite. Un modello che non capisci è pericoloso.</li>\n</ol>\n\n<h3>14.2 Errori comuni</h3>\n<ul>\n<li><strong>Data leakage</strong>: statistiche calcolate su tutto il dataset, uso di feature che nascondono $y$</li>\n<li><strong>Guardare il test durante lo sviluppo</strong>: overfitting sul test</li>\n<li><strong>Ignorare l'imbalance</strong>: accuracy 99% con classe rara è fake</li>\n<li><strong>Trascurare i baseline</strong>: se un modello mediocre batte il tuo modello complesso, hai un bug</li>\n<li><strong>Cercare complessità prima di robustezza</strong>: prima fai funzionare la lineare, poi passi ai transformer</li>\n</ul>\n\n<h3>14.3 Prossimi passi</h3>\n<p>Fatta questa base, per continuare:</p>\n<ul>\n<li>Kaggle: 3-5 competizioni tabulari per consolidare. Poi immagini/testo.</li>\n<li>Corso di deep learning di Andrew Ng o fast.ai.</li>\n<li>Libro \"Hands-On Machine Learning\" di Géron dalla copertina alla copertina.</li>\n<li>Un progetto personale con dati che ti interessano davvero. La motivazione è il carburante.</li>\n</ul>\n"
    },
    {
      "type": "callout",
      "variant": "tip",
      "title": "Ultima raccomandazione",
      "content": "Il ML si impara costruendo modelli, non leggendo di modelli. Ogni ora di libro deve essere seguita da due ore di codice. E non aver paura di fallire: un modello che crolla ti insegna più di dieci che funzionano subito."
    },
    {
      "type": "md",
      "content": "\n<h3>14.1b Workflow completo ML — checklist dettagliata</h3>\n<p>Ogni progetto ML professionale segue questi 10 passi in sequenza. Saltarne uno di solito significa tornare indietro.</p>\n<ol>\n<li><strong>1. Problem framing</strong>: definisci la <em>metrica di business</em> (es. \"ridurre i falsi negativi del 20%\"), identifica il tipo di problema (classificazione binaria? regressione? ranking?), stima una <em>baseline umana o naive</em> come riferimento minimo. Senza questa chiarezza, ottimizzare la metrica sbagliata porta a modelli inutili in produzione.</li>\n<li><strong>2. Data acquisition e ispezione</strong>: controlla <code class=\"inline\">X.shape</code>, <code class=\"inline\">dtypes</code>, valori mancanti per colonna, distribuzione del target. Un target sbilanciato 99:1 richiede strategie dedicate. Stampa sempre <code class=\"inline\">df.describe()</code> e <code class=\"inline\">df.info()</code> come primo passo.</li>\n<li><strong>3. EDA</strong>: distribuzioni di ogni feature (istogrammi, boxplot), correlazioni (heatmap), outlier (regola IQR o z-score), relazione feature-target per le feature piu&#39; promettenti. L&#39;EDA non e&#39; opzionale — le sorprese nei dati emergono quasi sempre qui.</li>\n<li><strong>4. Feature engineering</strong>: encoding categoriche (OrdinalEncoder per ordinali, OneHotEncoder per nominali), scaling numerico (StandardScaler, MinMaxScaler), interazioni (prodotti tra feature), feature derivate (log per distribuzioni skewed, bin per valori continui, lag per serie temporali). Tutto dentro Pipeline.</li>\n<li><strong>5. Baseline model</strong>: <code class=\"inline\">DummyClassifier(strategy=&#39;most_frequent&#39;)</code> o <code class=\"inline\">DummyRegressor(strategy=&#39;mean&#39;)</code>. Poi modello lineare semplice (LogisticRegression, Ridge). Questo e&#39; il tuo <em>floor</em> — qualsiasi modello complesso deve battere questo valore.</li>\n<li><strong>6. Model selection</strong>: confronta 3-5 famiglie di modelli con CV stratificata e <em>metrica corretta</em> (AUC-ROC o F1 per classi sbilanciate, RMSE/MAE per regressione, log-loss per probabilita&#39; calibrate). Non scegliere il modello guardando il test set.</li>\n<li><strong>7. Hyperparameter tuning</strong>: GridSearchCV per spazi piccoli, RandomizedSearchCV per spazi grandi. Per una stima onesta del vero errore di generalizzazione usa <em>nested CV</em>: il loop esterno (StratifiedKFold) valuta, il loop interno (GridSearchCV) ottimizza. Riportare la best score della CV interna come stima finale e&#39; una forma sottile di overfitting.</li>\n<li><strong>8. Analisi degli errori</strong>: guarda i singoli esempi predetti male. Hanno qualcosa in comune? Quali sottogruppi soffrono di piu&#39; (slice analysis)? Confusion matrix normalizzata per vedere dove il modello si confonde. Un modello 95% overall con 40% su un sottogruppo critico e&#39; inaccettabile.</li>\n<li><strong>9. Interpretabilita&#39;</strong>: feature importance (permutation importance e&#39; piu&#39; affidabile di quella built-in di RandomForest), SHAP values per spiegazioni per-sample. Un modello che sai spiegare e&#39; piu&#39; affidabile, piu&#39; difendibile e piu&#39; facile da debuggare.</li>\n<li><strong>10. Deployment preparation</strong>: serializza l&#39;intera pipeline con <code class=\"inline\">joblib.dump(pipe, &#39;model.joblib&#39;)</code>, scrivi test di regressione (input noti con output attesi), definisci monitoring (drift dei dati, performance nel tempo). Un modello senza monitoring diventa stale silenziosamente.</li>\n</ol>\n\n<h3>14.4 Stratified split — perche&#39; e&#39; necessario</h3>\n<p>Con classi sbilanciate, uno split casuale puo&#39; concentrare quasi tutta la classe rara in train (o in test), rendendo la valutazione inaffidabile o completamente inutile.</p>\n<ul>\n<li><strong>Il problema concreto</strong>: supponi 100 esempi, 5 positivi. Uno split 80/20 casuale puo&#39; mettere tutti i 5 positivi in train e zero in test. Il modello ottiene accuracy 100% sul test (sempre predice negativo) ma non ha <em>mai</em> visto la classe rara in fase di valutazione.</li>\n<li><strong>La soluzione</strong>: <code class=\"inline\">train_test_split(X, y, test_size=0.2, stratify=y)</code> garantisce la stessa proporzione di classi in ogni split. Banale da aggiungere, elimina un&#39;intera categoria di bug silenziosi.</li>\n<li><strong>Regola pratica</strong>: per <em>qualsiasi</em> problema di classificazione usa sempre <code class=\"inline\">stratify=y</code>. Non costa nulla in termini di performance.</li>\n<li><strong>In cross-validation</strong>: <code class=\"inline\">StratifiedKFold</code> applica la stratificazione automaticamente per design — e&#39; questo il motivo per cui e&#39; preferito a <code class=\"inline\">KFold</code> per la classificazione.</li>\n</ul>\n\n<h3>14.5 Strategie per classi sbilanciate</h3>\n<p>Quando una classe e&#39; molto piu&#39; rara dell&#39;altra, il modello tende a ignorarla per massimizzare l&#39;accuracy. Questi sono gli strumenti principali, dall&#39;approccio piu&#39; semplice al piu&#39; complesso:</p>\n<ul>\n<li><strong>class_weight=&#39;balanced&#39;</strong>: disponibile in LogisticRegression, SVC, RandomForestClassifier. Pesa ogni campione inversamente alla frequenza della sua classe: un esempio della classe rara contribuisce alla loss in proporzione alla rarità della sua classe. Semplice, efficace, nessun campione aggiuntivo in memoria.</li>\n<li><strong>SMOTE (Synthetic Minority Oversampling Technique)</strong>: per ogni esempio della classe rara, sintetizza nuovi punti interpolando tra i suoi k nearest neighbors nello spazio delle feature. Aumenta la classe rara con varieta&#39; sintetica invece di semplice duplicazione. Libreria: <code class=\"inline\">imblearn.over_sampling.SMOTE</code>.</li>\n<li><strong>ADASYN (Adaptive Synthetic Sampling)</strong>: come SMOTE ma genera piu&#39; esempi sintetici nelle zone di confine difficili tra le classi. Piu&#39; aggressivo di SMOTE, utile quando il confine decisionale e&#39; complesso.</li>\n<li><strong>Undersampling</strong>: riduce la classe maggioritaria anziche&#39; aumentare la minoritaria. <em>Random undersampling</em>: rimuove casualmente esempi dalla classe maggioritaria. <em>Tomek links</em>: rimuove le coppie di campioni di classi diverse molto vicine tra loro. <em>ENN (Edited Nearest Neighbours)</em>: rimuove campioni misclassificati dai loro vicini, pulendo la zona di confine.</li>\n<li><strong>Regola critica — niente leakage</strong>: applica SMOTE e undersampling <em>dentro</em> la pipeline/CV, <strong>mai</strong> prima dello split train/test. Applicarlo prima causa data leakage: i campioni sintetici generati da esempi del test set vengono visti durante il training, gonfiando artificialmente le performance valutate.</li>\n</ul>\n\n<h3>14.6 Error analysis — analisi sistematica degli errori</h3>\n<p>La metrica aggregata (AUC, accuracy, F1) nasconde molto. Fermarsi a un numero unico significa non capire dove e perche&#39; il modello sbaglia.</p>\n<ul>\n<li><strong>Guarda i singoli errori</strong>: isola gli esempi predetti male con <code class=\"inline\">X_test[y_pred != y_test]</code>. Hanno feature particolari in comune? Range di valori inusuali? Appartengono a una sottocategoria specifica? Spesso il pattern e&#39; immediatamente visibile.</li>\n<li><strong>Slice analysis</strong>: calcola la metrica separatamente per sottogruppi rilevanti (fascia d&#39;eta&#39;, regione geografica, categoria prodotto, fascia di reddito). Un modello con 95% di AUC overall e 40% su un sottogruppo critico e&#39; inutile o pericoloso. Il bias spesso emerge solo con la slice analysis.</li>\n<li><strong>Confusion matrix normalizzata</strong>: normalizza per riga (<code class=\"inline\">confusion_matrix(y, yp, normalize=&#39;true&#39;)</code>) per vedere il recall per ogni classe. Rivela dove il modello si confonde sistematicamente tra classi specifiche.</li>\n<li><strong>Soglia ottimale</strong>: la soglia di default 0.5 non e&#39; quasi mai ottimale per classi sbilanciate. Usa la curva precision-recall per trovare la soglia che massimizza la metrica di business. Un threshold piu&#39; basso aumenta il recall (meno falsi negativi) a scapito della precision.</li>\n<li><strong>Cluster di errori</strong>: se i falsi positivi si concentrano in una regione dello spazio delle feature, potrebbe servire una feature aggiuntiva che discrimini quella zona, oppure un modello locale specializzato per quella sottoregione.</li>\n</ul>\n\n<h3>14.7 Model selection — criteri pratici oltre l&#39;accuracy</h3>\n<p>Scegliere il modello giusto non significa scegliere il piu&#39; preciso in assoluto. Significa scegliere quello che funziona meglio nel contesto specifico di deployment.</p>\n<ul>\n<li><strong>Principio di Occam</strong>: a parita&#39; di performance (differenza non statisticamente significativa tra modelli), scegli il modello piu&#39; semplice. Generalizza meglio su dati nuovi, e&#39; piu&#39; robusto al distributional shift, piu&#39; facile da debuggare e monitorare.</li>\n<li><strong>Performance vs complessita&#39;</strong>: se una regressione logistica ottiene AUC 0.92 e un gradient boosting ottiene 0.93 con 10x piu&#39; parametri e 50x piu&#39; tempo di training, la differenza e&#39; spesso trascurabile in produzione e non vale il costo.</li>\n<li><strong>Tempo di training</strong>: per sistemi con retraining frequente (ogni ora o ogni giorno), un modello lineare che si addestra in secondi batte un ensemble che richiede 20 minuti, anche se quest&#39;ultimo e&#39; marginalmente piu&#39; preciso.</li>\n<li><strong>Tempo di inference</strong>: in sistemi real-time con SLA di latenza (es. 10ms), un Random Forest con 1000 alberi o una SVM con molti support vectors puo&#39; essere troppo lento. Considera la latenza al 99mo percentile, non solo la media.</li>\n<li><strong>Dimensione del modello</strong>: deployment su dispositivi edge o mobile impone vincoli di memoria. Un SVM serializzato con molti support vectors puo&#39; pesare centinaia di MB.</li>\n<li><strong>Interpretabilita&#39;</strong>: in settori regolamentati (credito, medicina, assicurazioni), e&#39; spesso richiesta la capacita&#39; di spiegare ogni predizione. Una regressione logistica con feature ben costruite e&#39; difendibile davanti a un regolatore; un ensemble non lo e&#39;.</li>\n<li><strong>Robustezza al distributional shift</strong>: i modelli semplici degradano piu&#39; gradualmente quando i dati di produzione cambiano rispetto al training (concetto drift, feature drift). Monitora la distribuzione degli input e le performance nel tempo — un modello senza monitoring diventa rapidamente stale.</li>\n</ul>\n"
    },
    {
      "type": "callout",
      "variant": "warn",
      "title": "SMOTE: mai applicare prima dello split",
      "content": "L'errore piu' comune con SMOTE e' applicarlo sull'intero dataset prima del train/test split. I campioni sintetici vengono generati interpolando tra esempi del test set, che \"leakano\" cosi' nel training. Le performance valutate risultano artificialmente gonfiate. Usa sempre imblearn.pipeline.Pipeline per garantire che SMOTE venga applicato solo sul fold di training durante la CV."
    },
    {
      "type": "callout",
      "variant": "tip",
      "title": "Nested CV: la stima onesta dell'errore",
      "content": "Se fai hyperparameter tuning con GridSearchCV e poi riporti la best score della CV interna come stima delle performance finali, stai overfittando sul validation set. Usa nested CV: loop esterno (StratifiedKFold, es. 5 fold) per stimare il vero errore di generalizzazione, loop interno (GridSearchCV) per ottimizzare i parametri. Il test set rimane intoccato fino alla valutazione finale e viene usato una sola volta."
    }
  ],
  "esempi": [
    {
      "type": "md",
      "content": "<h3>Progetto completo su breast_cancer</h3><p>Un template end-to-end che puoi copiare e adattare a qualsiasi problema tabulare.</p>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nimport pandas as pd\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.svm import SVC\nfrom sklearn.metrics import classification_report, roc_auc_score, confusion_matrix\n\n# 1) Carica e ispeziona\ndata = load_breast_cancer()\nX, y = data.data, data.target\nprint(f\"Shape: {X.shape}, classi: {np.bincount(y)}\")\n\n# 2) Split\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n\n# 3) Baseline\nfrom sklearn.dummy import DummyClassifier\ndummy = DummyClassifier(strategy='most_frequent').fit(Xt, yt)\nprint(f\"Baseline (majority): {dummy.score(Xe, ye):.4f}\")\n\n# 4) Prova tre modelli con CV\nmodels = {\n    'Logistic':  Pipeline([('s', StandardScaler()), ('m', LogisticRegression(max_iter=5000))]),\n    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=0),\n    'SVM-RBF':   Pipeline([('s', StandardScaler()), ('m', SVC(probability=True))]),\n}\ncv = StratifiedKFold(5, shuffle=True, random_state=0)\nfor name, m in models.items():\n    s = cross_val_score(m, Xt, yt, cv=cv, scoring='roc_auc')\n    print(f\"{name:15s} CV AUC: {s.mean():.4f} +/- {s.std():.4f}\")\n\n# 5) Tuning del migliore\ngrid = {'m__C':[0.01, 0.1, 1, 10, 100]}\ngs = GridSearchCV(models['Logistic'], grid, cv=cv, scoring='roc_auc').fit(Xt, yt)\nprint(f\"\\nMigliori parametri: {gs.best_params_}\")\n\n# 6) Valutazione finale\nfinal = gs.best_estimator_\nyp = final.predict(Xe); yprob = final.predict_proba(Xe)[:,1]\nprint(f\"\\nTest AUC:      {roc_auc_score(ye, yprob):.4f}\")\nprint(f\"Test Accuracy: {(yp==ye).mean():.4f}\")\nprint(\"\\nMatrice di confusione:\\n\", confusion_matrix(ye, yp))\nprint(\"\\nReport:\\n\", classification_report(ye, yp, digits=3))"
    }
  ],
  "esercizi": [
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Progetto — California housing",
      "prompt": "<p>Task di regressione. Prevedi il target sulla base di 8 feature. Devi ottenere <strong>R² test > 0.75</strong>. Sei libero di scegliere modello, preprocessing, tuning.</p>",
      "starter": "import numpy as np\nfrom sklearn.datasets import fetch_california_housing\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\nfrom sklearn.linear_model import Ridge\n\ntry:\n    d = fetch_california_housing()\n    X, y = d.data, d.target\nexcept Exception as e:\n    # fallback se il download non è possibile in Pyodide\n    np.random.seed(0)\n    X = np.random.randn(2000, 8)\n    y = (2*X[:,0] - X[:,1] + 0.5*X[:,0]*X[:,2] + np.random.randn(2000)*0.5)\n\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=42)\n\nr2 = None\nprint(\"R²:\", r2)",
      "check": "if r2 is None: print(\"FAIL: None\")\nelif r2 > 0.75: print(f\"OK — R² = {r2:.4f}\")\nelse: print(f\"FAIL — R² = {r2:.4f}\")",
      "solution": "m = GradientBoostingRegressor(n_estimators=300, max_depth=4, random_state=0).fit(Xt, yt)\nr2 = m.score(Xe, ye)\n# Random Forest funziona quasi identico:\n# m = RandomForestRegressor(n_estimators=200, random_state=0).fit(Xt, yt)"
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Progetto — pipeline completa da zero",
      "prompt": "<p>Sul dataset wine, costruisci una pipeline che comprenda: standardizzazione, PCA (mantieni il 95% di varianza), e un classificatore a tua scelta. Fai tuning con GridSearchCV. Target: accuracy test > 0.95. Salva in <code class=\"inline\">acc</code>.</p>",
      "starter": "from sklearn.datasets import load_wine\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.svm import SVC\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import GridSearchCV, train_test_split\n\nX, y = load_wine(return_X_y=True)\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)\n\nacc = None",
      "check": "if acc is None: print(\"FAIL: None\")\nelif acc > 0.95: print(f\"OK — {acc:.4f}\")\nelse: print(f\"FAIL — {acc:.4f}\")",
      "solution": "pipe = Pipeline([('s', StandardScaler()),\n                 ('pca', PCA(n_components=0.95)),\n                 ('c', LogisticRegression(max_iter=5000))])\ngs = GridSearchCV(pipe, {'c__C':[0.1, 1, 10]}, cv=5).fit(Xt, yt)\nacc = gs.score(Xe, ye)"
    }
  ],
  "test": [
    {
      "type": "md",
      "content": "<h3>Test finale — riflessione</h3><p>Non c'è codice: rispondi alle domande. Sono le più importanti che ti farai iniziando un progetto reale.</p>"
    },
    {
      "type": "quiz",
      "questions": [
        {
          "q": "Un modello ha CV accuracy 0.95 ma sul test cala a 0.85. La spiegazione più probabile è:",
          "options": [
            "Test set piccolo (varianza alta)",
            "Data leakage durante training",
            "Hai fatto tuning contro il test set — hai overfittato sul test",
            "Tutte plausibili, andrebbero verificate una per una"
          ],
          "answer": 3,
          "explain": "Diagnosticare senza dati aggiuntivi richiede investigazione, non conclusioni affrettate."
        },
        {
          "q": "Il primo passo di un progetto è:",
          "options": [
            "Scegliere il modello",
            "Definire chiaramente cosa vuoi predire e con che metrica",
            "Cercare dati",
            "Installare le librerie"
          ],
          "answer": 1,
          "explain": "Senza definizione del problema, tutto il resto è casuale."
        },
        {
          "q": "Un baseline stupido è utile perché:",
          "options": [
            "Fa perdere tempo",
            "Stabilisce il floor che qualsiasi modello serio deve battere",
            "È divertente",
            "Non lo è"
          ],
          "answer": 1,
          "explain": "Se il tuo Deep Learning fa uguale alla media, hai un bug o un problema di segnale."
        },
        {
          "q": "Il test set va guardato:",
          "options": [
            "Ogni volta che iteri",
            "Solo alla fine, una sola volta",
            "Ogni epoch",
            "Solo se train val hanno grandi gap"
          ],
          "answer": 1,
          "explain": "Guardare il test durante lo sviluppo lo trasforma in un secondo val set."
        }
      ]
    },
    {
      "type": "md",
      "content": "<h3>Congratulazioni</h3><p>Se sei arrivato qui e hai completato gli esercizi, hai una base solida di machine learning classico. I prossimi passi naturali sono: (1) fare 3-5 progetti Kaggle su dati tabulari, (2) imparare gradient boosting avanzato (XGBoost, LightGBM), (3) iniziare deep learning con PyTorch, (4) leggere il libro di Géron per approfondire.</p><p><strong>Il ML si impara facendo. Adesso inizia a farlo.</strong></p>"
    }
  ]
});
