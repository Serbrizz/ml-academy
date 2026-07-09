# ML Academy — Corso interattivo di Machine Learning (v5)

Web app didattica offline-capable per imparare il machine learning da zero fino a un livello job-ready. **19 lezioni** con teoria, esempi eseguibili, esercizi con verifica automatica e test finali. Python (NumPy, Pandas, scikit-learn, matplotlib) gira direttamente nel browser via Pyodide.

## Novità versione 5

- **Modalità mobile attivabile a mano**: interruttore virtuale nella sidebar per switchare tra layout PC (default) e mobile. La modalità mobile: sidebar collassabile con hamburger, editor più grandi con bottone Tab touch-friendly, tabs delle lezioni scrollabili, tutti i bottoni min 44px per touch.
- **Modalità sola lettura opzionale**: nasconde gli editor di codice, utile per leggere teoria e formule su schermo piccolo.
- **PWA installabile**: puoi aggiungere l'app alla home screen del telefono. Funziona offline dopo il primo caricamento (grazie al service worker).
- Preferenze salvate in localStorage — quando riapri l'app la trovi come l'hai lasciata.

## Come usare la modalità mobile

1. Apri l'app sul telefono all'URL `https://serbrizz.github.io/ml-academy/`.
2. Se non vedi la sidebar, questa è nascosta di default in mobile: tocca il pulsante ☰ in alto a sinistra.
3. In fondo alla sidebar, attiva l'interruttore "Modalità mobile".
4. Il layout si adatta: sidebar collassabile, bottoni più grandi, tabs scrollabili, editor comodi da toccare.
5. Se vuoi solo leggere senza scrivere codice (utile in metro), attiva anche "Sola lettura".

## Come installare come app sul telefono

**Android (Chrome)**: vai su `https://serbrizz.github.io/ml-academy/`, tocca i 3 puntini in alto a destra → "Aggiungi a schermata Home" o "Installa app".

**iOS (Safari)**: apri il sito, tocca il pulsante Condividi → "Aggiungi a Home".

Dopo l'installazione l'app ha un'icona sulla home screen come qualsiasi altra app. Al secondo lancio funziona anche offline (i file dell'app sono in cache, Pyodide anche una volta scaricato).

## Statistiche corso

- **19 lezioni** strutturate progressivamente
- **152 esercizi** con verifica automatica del codice
- **85 domande** di quiz interattivi
- **89 esempi** Python eseguibili nel browser
- **141 termini** nel glossario ML
- **20 cheat sheet** stampabili

## Struttura file

```
ml-academy/
├── index.html
├── styles.css
├── app.js
├── glossary.js
├── cheatsheets.js
├── pdf-library.js
├── manifest.json          # PWA v5
├── sw.js                  # service worker per offline v5
├── lessons/
│   ├── index.js
│   └── 01-intro.js ... 19-fraud-project.js
├── ROADMAP.md
├── README.md
└── .nojekyll
```

## Come aggiornare il repo esistente

```powershell
cd $HOME\Documents\ml-academy

Get-ChildItem -Force | Where-Object { $_.Name -ne '.git' } | Remove-Item -Recurse -Force

Copy-Item "C:\Users\BRIZZIP2\AppData\Local\Claude-3p\local-agent-mode-sessions\608d2558\00000000\local_612ad815-5777-4fe2-8125-528e46a0f732\outputs\*" -Destination . -Recurse -Force

git add -A
git commit -m "v5: modalita mobile + PWA installabile"
git push
```

## Come funziona la streak

Ogni azione (aprire una lezione, eseguire codice, rispondere a un quiz, salvare note) conta come attività del giorno e dà XP. La streak sale se studi ogni giorno consecutivo, si azzera se salti più di un giorno.

- Aprire una lezione: 3 XP
- Eseguire un esempio o esercizio: 2 XP
- Rispondere correttamente a un quiz: 3 XP
- Risolvere un esercizio (output OK): 10 XP
- Completare una lezione: 25 XP
- Scrivere note: 1 XP

## Note tecniche mobile / PWA

**Pyodide su mobile**: il primo caricamento è pesante (~50MB). Fallo su Wi-Fi. Successivamente è cached e va offline. Su smartphone di fascia media l'esecuzione è 3-5x più lenta del desktop; usa la modalità mobile per lettura, il PC per scrivere codice.

**Service worker**: la cache è chiamata `ml-academy-v4`. Se aggiorni l'app e il vecchio worker resta appeso, force refresh con Ctrl+Shift+R (o cache clear del browser). In produzione bisogna incrementare `CACHE_NAME` in sw.js ogni volta che si fa una release, cosi il vecchio worker viene sostituito.

**LocalStorage cross-device**: le note, lo streak, i progressi sono per browser. Se apri sul telefono e sul PC, sono due sessioni distinte. Esportare/importare i dati è una futura feature possibile.

## Licenza

Uso libero, personale e non. Attribuzione gradita ma non obbligatoria.
