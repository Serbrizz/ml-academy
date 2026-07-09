# Roadmap Machine Learning — da zero a job-ready

Percorso pensato per chi ha una base matematica (fisica, ingegneria) e vuole arrivare in ~6 mesi a un livello sufficiente per candidarsi come **Junior ML Engineer / Data Scientist**. Assume 30 minuti al giorno, tutti i giorni. Se puoi fare 60 minuti, dimezzi i tempi.

## Il patto onesto

30 minuti al giorno per 6 mesi = **90 ore di studio**. Non è poco, ma non è nemmeno "molto" nel contesto ML. Per essere davvero pronto a un colloquio senior servirebbero 300-500 ore. La roadmap qui sotto ti porta a un livello **junior credibile**: capisci il campo, sai fare pipeline complete su dati tabulari, hai due-tre progetti solidi da mostrare, conosci le basi di deep learning e MLOps.

La differenza tra chi ci arriva e chi si arena è **una sola**: la costanza. Meglio 30 minuti tutti i giorni che 4 ore ogni domenica. Il ML è un dominio dove i concetti si sedimentano solo con la ripetizione. Da qui la streak giornaliera nell'app: non è gamification per divertimento, è la variabile predittiva più importante del tuo successo.

## Mappa dei 6 mesi

| Mese | Focus | Tempo cumulativo |
|------|-------|------------------|
| 1 | Fondamenta: math, Python, EDA, primo modello | ~15h |
| 2 | ML classico supervisionato | ~30h |
| 3 | ML avanzato + primo progetto reale | ~45h |
| 4 | Deep Learning con PyTorch | ~60h |
| 5 | Specializzazione (NLP o CV) + progetti | ~75h |
| 6 | MLOps, deployment, portfolio, preparazione colloqui | ~90h |

## Mese 1 — Fondamenta

**Obiettivo:** riprendere la matematica utile, essere fluente in NumPy e Pandas, fare EDA su un dataset reale, addestrare la prima regressione lineare seria.

### Settimana 1 — Ripasso math + Python

- **Lezione 1** dell'app (Cos'è il ML)
- **Lezione 2** dell'app (Ripasso matematico)
- 3Blue1Brown, "Essence of Linear Algebra" — playlist YouTube, guarda gli episodi 1-9
- Esercizi extra: implementa da zero prodotto scalare, norma L2, matrice trasposta, autovalori tramite algoritmo di potenza

**Checkpoint:** riesci a spiegare a te stesso cosa è un gradiente, una matrice di covarianza, una MLE, senza rileggere.

### Settimana 2 — NumPy, Pandas, primo dataset

- **Lezione 3** dell'app (NumPy e Pandas)
- Scarica un dataset da [Kaggle Learn Datasets](https://www.kaggle.com/datasets), inizia con qualcosa di semplice tipo "Palmer Penguins" o "Titanic"
- Fai 50 operazioni Pandas diverse su quel dataset (filtraggio, groupby, pivot, merge)

**Checkpoint:** puoi rispondere in 5 minuti a domande tipo "qual è il salario medio per città e reparto, esclusi gli outlier?"

### Settimana 3 — EDA seria

- **Lezione 4** dell'app (EDA e visualizzazione)
- Corso Kaggle "Data Visualization" (gratis, 4 ore)
- Fai una EDA completa del dataset scelto: 15 grafici minimo, un report scritto di 1 pagina con le tue osservazioni

**Checkpoint:** dal tuo report EDA sei capace di formulare 3-5 domande predittive concrete.

### Settimana 4 — Prima regressione lineare fatta bene

- **Lezione 5** dell'app (Regressione lineare)
- Andrew Ng, Machine Learning Specialization su Coursera — **Course 1** interamente (~10 ore)
- Implementa la regressione lineare da zero con NumPy, poi con scikit-learn, confronta i risultati

**Checkpoint mese 1:** hai un notebook su GitHub con EDA + regressione lineare su un dataset reale, con conclusioni scritte.

**Risorse consigliate mese 1:**
- Libro: *"Python Data Science Handbook"* di Jake VanderPlas — capitoli 2, 3, 4 (gratis online)
- Corso: [Kaggle Learn](https://www.kaggle.com/learn) — Python, Pandas, Data Visualization
- Video: 3Blue1Brown Linear Algebra, Calculus (playlist)

---

## Mese 2 — ML classico supervisionato

**Obiettivo:** conoscere tutti i modelli classici, sapere quando usare ciascuno, saper valutare correttamente.

### Settimana 5 — Regressione logistica + classificazione

- **Lezione 6** dell'app (Logistica)
- Coursera Andrew Ng, Course 2 (classification, moduli 1-2)
- Progetto: predici se un dataset di Kaggle "Heart Disease" ha malattia, con metriche complete (precision, recall, F1, ROC-AUC)

### Settimana 6 — KNN + Decision Trees

- **Lezioni 7 e 8** dell'app
- Leggi il capitolo 6 di *Hands-On ML* (Géron)
- Confronta 4 modelli sullo stesso dataset (Logistic, KNN, Tree, Random Forest) e produci una tabella di risultati

### Settimana 7 — SVM + valutazione avanzata

- **Lezioni 9 e 10** dell'app
- Impara cross-validation, grid search, learning curves fatti bene
- Rifai un progetto del mese 1 con valutazione seria (nested CV) — vedrai quanto era gonfio prima

### Settimana 8 — Non supervisionato

- **Lezioni 11 e 12** dell'app (K-Means, PCA)
- Progetto: sul dataset "Mall Customer Segmentation" di Kaggle, fai clustering e produci 3-4 personas con interpretazione

**Checkpoint mese 2:** hai 2-3 notebook su GitHub, ognuno con almeno 3 modelli confrontati, valutazione seria, interpretazione dei risultati.

**Risorsa chiave mese 2:**
- Libro: *"Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow"* di Aurélien Géron — capitoli 1-9. Questo libro è **il migliore che esista** per imparare ML pratico. Se ne compri solo uno, questo.

---

## Mese 3 — ML avanzato + primo progetto vero

**Obiettivo:** dominare i modelli che vincono davvero, saper gestire dati sporchi/sbilanciati, completare il primo progetto significativo.

### Settimana 9 — Gradient Boosting

- Studia XGBoost e LightGBM (nell'app ci sarà la lezione dedicata, in Fase 3)
- Documentazione ufficiale XGBoost: [xgboost.readthedocs.io](https://xgboost.readthedocs.io)
- Progetto: fai una submission Kaggle sulla competition "House Prices - Advanced Regression Techniques" (per principianti) e mira alla top 50%

### Settimana 10 — Feature Engineering avanzato

- Target encoding, mean encoding, feature interaction, embedding per categoriche ad alta cardinalità
- Libro Géron capitolo 2 (fine) + [Kaggle Learn Feature Engineering](https://www.kaggle.com/learn/feature-engineering)
- Rifai il progetto house prices con feature engineering serio, punta al top 30%

### Settimana 11 — Dati sbilanciati e problemi reali

- SMOTE, class weights, cost-sensitive learning
- Metriche giuste per classi rare (PR curve, focal loss)
- Progetto: dataset "Credit Card Fraud" (Kaggle) — sbilanciamento estremo, ottimo caso di studio

### Settimana 12 — Interpretabilità

- SHAP values, permutation importance, partial dependence plots
- Libreria `shap` per Python
- Aggiungi interpretazione ai tuoi 3 progetti principali

**Checkpoint mese 3:** primo progetto grande su GitHub con README completo, submission Kaggle nel top 30%, capitolo di interpretazione. Questo è già "showcase material" per un CV.

---

## Mese 4 — Deep Learning con PyTorch

**Obiettivo:** capire davvero come funzionano le reti neurali, saperle costruire in PyTorch, addestrare una CNN da zero.

### Settimana 13 — Basi reti neurali

- **Lezione 13** dell'app + capitoli 10-11 di Géron (versione TF nel libro, ma i concetti sono identici)
- Installa PyTorch, prova gli esempi ufficiali del tutorial "60 minute blitz"
- Implementa un MLP da zero in NumPy per MNIST (senza PyTorch prima)

### Settimana 14 — PyTorch fondamenta

- Corso [Deep Learning con PyTorch: Zero to GANs](https://jovian.ai) o il tutorial ufficiale PyTorch
- Costruisci un MLP in PyTorch per Fashion-MNIST, ottieni >85% accuracy
- Capisci Dataset, DataLoader, optimizer, scheduler

### Settimana 15 — CNN

- Karpathy CS231n, prime lezioni disponibili gratis su YouTube
- Costruisci una CNN piccola per CIFAR-10, arriva a >70% accuracy
- Fine-tuning di ResNet pre-addestrata su un dataset di immagini a tua scelta

### Settimana 16 — Transfer learning + best practices

- torchvision.models: prova ResNet, EfficientNet, ViT
- Data augmentation, mixup, learning rate scheduling
- Progetto piccolo di classificazione immagini con dataset personale (raccoglilo da Google/Unsplash o usa uno pubblico)

**Checkpoint mese 4:** hai addestrato una rete neurale che riconosce qualcosa di non banale, e sai spiegare cosa succede a ogni layer.

**Risorse mese 4:**
- Corso: [fast.ai Practical Deep Learning](https://course.fast.ai) — approccio top-down, molto pratico
- Video: Andrej Karpathy, "Neural Networks: Zero to Hero" su YouTube — **imperdibile**, ti mostra come costruire GPT da zero
- Libro: *"Deep Learning with PyTorch"* di Stevens/Antiga/Viehmann

---

## Mese 5 — Specializzazione

Scegli **una** delle due direzioni. Non provare a fare entrambe seriamente in un mese.

### Opzione A — Computer Vision

- **Sett. 17:** object detection (YOLO, Faster R-CNN) — capisci i concetti, prova YOLOv8 pre-addestrata
- **Sett. 18:** segmentazione (UNet) — implementa una UNet per un task medico o satellitare
- **Sett. 19:** GAN o diffusion (uno solo) — solo intuizione + esempio pre-addestrato
- **Sett. 20:** progetto CV completo: prendi un problema reale (contatore veicoli, classificatore di piante, quello che ti passiona) e costruiscilo end-to-end

### Opzione B — NLP

- **Sett. 17:** tokenizzazione, word embeddings (Word2Vec, GloVe), classificatore testi con embeddings
- **Sett. 18:** RNN, LSTM (concettuale) + BERT pre-addestrato via HuggingFace Transformers
- **Sett. 19:** fine-tuning BERT su un task di classificazione tuo (sentiment, topic classification)
- **Sett. 20:** progetto NLP completo: chatbot semplice, generatore di testo, o classificatore fine-tuned per problema reale

**Checkpoint mese 5:** un progetto verticalizzato su CV o NLP, con codice pulito, README, e post scritto su Medium/LinkedIn per spiegarlo.

**Risorse:**
- HuggingFace [NLP Course](https://huggingface.co/learn/nlp-course) — gratis, eccellente
- Papers With Code — per scoprire lo stato dell'arte del tuo problema

---

## Mese 6 — MLOps, deployment, portfolio, colloqui

**Obiettivo:** trasformare i tuoi modelli in cose "shippabili" e costruire il pacchetto per candidarti.

### Settimana 21 — MLOps essenziali

- **Version control per modelli e dati:** DVC, MLflow, Weights & Biases (scegline uno)
- **Sperimentazione tracciata:** logga metriche, parametri, artefatti per tutti gli esperimenti
- Rifai un progetto vecchio usando MLflow per tracciare gli esperimenti

### Settimana 22 — Deployment

- **API con FastAPI:** metti un modello dietro un endpoint HTTP
- **Docker:** containerizza l'applicazione
- **Cloud:** deployala gratis su Render, Fly.io, o Hugging Face Spaces
- Fine settimana: il tuo modello è pubblicamente utilizzabile via URL

### Settimana 23 — CI/CD, monitoraggio, testing

- GitHub Actions per test automatici del codice ML (pytest)
- Data validation con Great Expectations o Pandera
- Monitoraggio drift dei dati (concetto + tool tipo Evidently AI)

### Settimana 24 — Portfolio + colloqui

- **GitHub sistemato:** 3-5 progetti con README curati, immagini, risultati chiari
- **CV** con sezione ML: elenco progetti, tecnologie, risultati misurabili
- **LinkedIn:** post su 1-2 dei tuoi progetti, con visualizzazioni
- **Preparazione colloqui:** 
  - Domande teoriche classiche (bias-variance, overfitting, cross-validation, backprop, ecc.)
  - Case study: prendi 3 problemi ML e disegnati la soluzione end-to-end su carta
  - Coding: pratica su LeetCode livello medium (arrays, strings, algoritmi base) — 30 minuti al giorno

**Checkpoint finale:** puoi mostrare a un recruiter:
1. Un profilo GitHub con 3-5 progetti robusti
2. Uno di questi progetti deployato e vivo online
3. Post di blog o LinkedIn dove spieghi il tuo processo
4. Un CV chiaro con storia coerente

---

## Cosa fare oltre i 6 mesi

Se dopo 6 mesi vuoi passare da junior a mid-senior, i temi da attaccare sono:
- **LLM in profondità:** transformer architecture da zero, RAG, fine-tuning con LoRA, RLHF
- **Distributed training:** DDP in PyTorch, multi-GPU, TPU
- **Ricerca:** lettura sistematica di paper (1 paper a settimana da arxiv-sanity), riproduzione di risultati
- **Sistemi ML:** design di sistemi (feature store, model registry, A/B testing), scale-up
- **Specializzazione verticale:** reinforcement learning, causal ML, graph ML, time series in profondità

---

## Regole d'oro del percorso

1. **Codice ogni giorno.** Anche 15 minuti. Anche una singola riga. Non spezzare la catena.
2. **Progetti > tutorial.** Ogni volta che finisci un capitolo teorico, applicalo a qualcosa di tuo entro 3 giorni, o lo dimentichi.
3. **Documenta tutto.** README dettagliato per ogni progetto. Se non sai spiegarlo per iscritto, non lo capisci davvero.
4. **Dubita dei tuoi modelli.** Il 90% dei bug del ML sono dati sbagliati, non modelli sbagliati. Controlla sempre i dati due volte.
5. **Interagisci con la community.** Kaggle forum, Reddit r/MachineLearning, HuggingFace discord. Impari velocissimo vedendo problemi diversi.
6. **Non inseguire le novità.** Gli LLM sono figherrimi, ma se non hai le basi non capisci cosa fanno. Impara le basi bene, e il paper del mese lo leggi in 20 minuti invece che 2 giorni.

---

## Risorse chiave (ordine di priorità)

**Libri (in ordine di importanza)**
1. *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow* — Aurélien Géron (essenziale)
2. *Deep Learning with PyTorch* — Stevens, Antiga, Viehmann
3. *Designing Machine Learning Systems* — Chip Huyen (per MLOps)
4. *The Elements of Statistical Learning* — Hastie, Tibshirani, Friedman (riferimento teorico, gratis online)

**Corsi online**
1. Andrew Ng, Machine Learning Specialization su Coursera (ML classico)
2. fast.ai Practical Deep Learning (deep learning pratico, gratis)
3. Karpathy, Neural Networks Zero to Hero (YouTube, gratis)
4. HuggingFace NLP Course (gratis)

**Community**
- Kaggle (competizioni + Learn)
- Papers With Code
- r/MachineLearning
- Twitter/X: segui Karpathy, Yann LeCun, Chip Huyen, Jeremy Howard
- HuggingFace Discord

**Tool essenziali da imparare**
- Python, NumPy, Pandas, matplotlib, seaborn (mese 1)
- scikit-learn (mesi 2-3)
- XGBoost, LightGBM (mese 3)
- PyTorch (mese 4)
- HuggingFace Transformers (mese 5)
- MLflow, FastAPI, Docker (mese 6)

Buono studio. Alla fine di questi 6 mesi non sarai un genio del ML, ma sarai qualcuno che può contribuire seriamente in un team, e da lì la curva di apprendimento accelera moltissimo perché entri nel loop dove impari sui problemi veri di ogni giorno.
