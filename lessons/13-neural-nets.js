/* Lezione: Introduzione alle reti neurali */
window.LESSONS.push({
  "id": "l13-neural-nets",
  "title": "Introduzione alle reti neurali",
  "tag": "Deep Learning",
  "summary": "Percettrone, MLP, backpropagation. La transizione al deep learning.",
  "teoria": [
    {
      "type": "md",
      "content": "\n<h3>13.1 Dal lineare al non lineare</h3>\n<p>La regressione logistica è un neurone singolo. Se componiamo tanti neuroni in strati otteniamo un <em>perceptron multistrato</em> (MLP), capace di approssimare qualsiasi funzione continua (teorema di approssimazione universale).</p>\n\n<h3>13.2 Un neurone artificiale</h3>\n<p>Un neurone calcola:</p>\n<p>$$a = \\phi(w^T x + b)$$</p>\n<p>dove $\\phi$ è la <em>funzione di attivazione</em> non lineare. Senza $\\phi$, comporre neuroni sarebbe equivalente a una singola trasformazione lineare — inutile.</p>\n\n<h3>13.3 Attivazioni</h3>\n<ul>\n<li><strong>Sigmoide</strong>: $\\sigma(z) = 1/(1+e^{-z})$. Ora poco usata: gradienti che svaniscono.</li>\n<li><strong>Tanh</strong>: $\\tanh(z)$. Centrata su zero, meglio della sigmoide.</li>\n<li><strong>ReLU</strong>: $\\max(0, z)$. Standard moderno. Semplice, veloce, evita gradient vanishing.</li>\n<li><strong>Leaky ReLU, ELU, GELU</strong>: varianti che risolvono limiti di ReLU.</li>\n<li><strong>Softmax</strong>: per l'output di classificazione multi-classe.</li>\n</ul>\n\n<h3>13.4 Un MLP</h3>\n<p>Strato per strato, con notazione compatta:</p>\n<p>$$h^{(l+1)} = \\phi(W^{(l+1)} h^{(l)} + b^{(l+1)})$$</p>\n<p>con $h^{(0)} = x$ e output $\\hat y = h^{(L)}$. I parametri sono tutti i $W^{(l)}, b^{(l)}$.</p>\n<p>Il numero di neuroni per strato e il numero di strati sono <em>iperparametri architetturali</em>.</p>\n\n<h3>13.5 Loss e training</h3>\n<p>Per regressione: MSE. Per classificazione binaria: BCE. Per multi-classe con softmax: cross-entropy. Il training minimizza la loss con gradient descent (o Adam, RMSprop...).</p>\n\n<h3>13.6 Backpropagation</h3>\n<p>Il gradiente della loss rispetto a ogni parametro si calcola con la <em>regola della catena</em> applicata all'indietro. Concettualmente semplice, tecnicamente delicato solo per la contabilità delle dimensioni.</p>\n<p>Per uno strato $l$:</p>\n<p>$$\\delta^{(l)} = (W^{(l+1)T} \\delta^{(l+1)}) \\odot \\phi'(z^{(l)})$$</p>\n<p>$$\\frac{\\partial \\mathcal{L}}{\\partial W^{(l)}} = \\delta^{(l)} (h^{(l-1)})^T$$</p>\n<p>I framework moderni (PyTorch, TensorFlow) fanno tutto automaticamente via <em>autograd</em>: tu scrivi il forward, loro calcolano il backward.</p>\n\n<h3>13.7 Regolarizzazione</h3>\n<ul>\n<li><strong>L2 weight decay</strong>: penalità $\\lambda \\|W\\|^2$</li>\n<li><strong>Dropout</strong>: azzera casualmente frazione dei neuroni durante training</li>\n<li><strong>Batch normalization</strong>: normalizza le attivazioni per stabilizzare</li>\n<li><strong>Early stopping</strong>: fermati quando val loss smette di scendere</li>\n</ul>\n\n<h3>13.8 Ottimizzatori</h3>\n<ul>\n<li><strong>SGD</strong>: base, spesso con momentum</li>\n<li><strong>Adam</strong>: adatta il learning rate per parametro. Default ragionevole.</li>\n<li><strong>RMSprop, AdaGrad</strong>: varianti</li>\n</ul>\n\n<h3>13.9 Oltre l'MLP</h3>\n<ul>\n<li><strong>CNN</strong>: per immagini, sfruttano la struttura spaziale</li>\n<li><strong>RNN, LSTM</strong>: per sequenze</li>\n<li><strong>Transformer</strong>: attualmente dominanti per testo, immagini, quasi tutto</li>\n</ul>\n<p>Sono la lezione successiva di un corso, ma tutto si costruisce sulle stesse basi.</p>\n"
    }
  ],
  "esempi": [
    {
      "type": "md",
      "content": "<h3>Esempio 1: MLP con scikit-learn</h3>"
    },
    {
      "type": "run",
      "content": "from sklearn.neural_network import MLPClassifier\nfrom sklearn.datasets import load_digits\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\n\nX, y = load_digits(return_X_y=True)\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)\n\npipe = Pipeline([('s', StandardScaler()),\n                 ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32),\n                                       activation='relu',\n                                       max_iter=200,\n                                       random_state=0))])\npipe.fit(Xt, yt)\nprint(f\"Train acc: {pipe.score(Xt, yt):.4f}\")\nprint(f\"Test acc:  {pipe.score(Xe, ye):.4f}\")\nprint(f\"Num layer: {len(pipe.named_steps['mlp'].coefs_)}\")\nfor i, W in enumerate(pipe.named_steps['mlp'].coefs_):\n    print(f\"  layer {i}: shape {W.shape}\")"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 2: MLP from scratch (2 layer)</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nnp.random.seed(0)\n\n# dati XOR — non separabile linearmente\nX = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)\ny = np.array([[0],[1],[1],[0]], dtype=float)\n\n# architettura: 2 -> 4 -> 1\nW1 = np.random.randn(2, 4) * 0.5\nb1 = np.zeros(4)\nW2 = np.random.randn(4, 1) * 0.5\nb2 = np.zeros(1)\n\ndef sig(z): return 1/(1+np.exp(-z))\n\nlr = 0.5\nfor step in range(5000):\n    # forward\n    z1 = X @ W1 + b1; a1 = sig(z1)\n    z2 = a1 @ W2 + b2; y_hat = sig(z2)\n    # loss (BCE)\n    loss = -np.mean(y*np.log(y_hat+1e-9) + (1-y)*np.log(1-y_hat+1e-9))\n    # backward\n    dz2 = (y_hat - y) / len(X)\n    dW2 = a1.T @ dz2; db2 = dz2.sum(0)\n    da1 = dz2 @ W2.T\n    dz1 = da1 * a1 * (1-a1)\n    dW1 = X.T @ dz1; db1 = dz1.sum(0)\n    # update\n    W1 -= lr*dW1; b1 -= lr*db1; W2 -= lr*dW2; b2 -= lr*db2\n    if step % 1000 == 0:\n        print(f\"step {step:4d}: loss = {loss:.5f}\")\n\nprint(\"\\nPredizioni:\", y_hat.round(3).ravel(), \"vere:\", y.ravel())"
    },
    {
      "type": "md",
      "content": "<h3>Esempio 3: effetto della profondità</h3>"
    },
    {
      "type": "run",
      "content": "import numpy as np\nfrom sklearn.neural_network import MLPClassifier\nfrom sklearn.datasets import make_moons\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.model_selection import cross_val_score\n\nX, y = make_moons(n_samples=500, noise=0.3, random_state=0)\nprint(f\"{'arch':>20} {'CV':>10}\")\nfor arch in [(2,), (10,), (10,10), (50,), (50,50,50), (200,200)]:\n    p = Pipeline([('s', StandardScaler()),\n                  ('m', MLPClassifier(hidden_layer_sizes=arch, max_iter=500, random_state=0))])\n    s = cross_val_score(p, X, y, cv=5).mean()\n    print(f\"{str(arch):>20} {s:>10.4f}\")"
    }
  ],
  "esercizi": [
    {
      "type": "exercise",
      "difficulty": "medium",
      "title": "Es. 13.1 — Forward pass manuale",
      "prompt": "<p>Implementa il forward di un MLP a 2 layer: input dim 3, hidden 4 con ReLU, output 1 con sigmoide. Data <code class=\"inline\">x</code>, <code class=\"inline\">W1, b1, W2, b2</code>, salva l'output in <code class=\"inline\">y_hat</code>.</p>",
      "starter": "import numpy as np\nnp.random.seed(0)\nx = np.array([1.0, -0.5, 2.0])\nW1 = np.random.randn(3, 4); b1 = np.zeros(4)\nW2 = np.random.randn(4, 1); b2 = np.zeros(1)\n\n# TODO\ny_hat = None\nprint(y_hat)",
      "check": "import numpy as np\n_z1 = x @ W1 + b1; _a1 = np.maximum(0, _z1)\n_z2 = _a1 @ W2 + b2; _yh = 1/(1+np.exp(-_z2))\nif y_hat is not None and np.allclose(y_hat, _yh):\n    print(\"OK\")\nelse:\n    print(f\"FAIL — atteso {_yh}\")",
      "solution": "z1 = x @ W1 + b1\na1 = np.maximum(0, z1)  # ReLU\nz2 = a1 @ W2 + b2\ny_hat = 1/(1+np.exp(-z2))  # sigmoide"
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Es. 13.2 — MLP su moons",
      "prompt": "<p>Su make_moons, addestra un MLP con architettura a scelta. Target: test accuracy > 0.9.</p>",
      "starter": "import numpy as np\nfrom sklearn.datasets import make_moons\nfrom sklearn.neural_network import MLPClassifier\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.model_selection import train_test_split\n\nX, y = make_moons(n_samples=1000, noise=0.25, random_state=0)\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42)\n\nacc = None",
      "check": "if acc is None: print(\"FAIL: None\")\nelif acc > 0.9: print(f\"OK — {acc:.4f}\")\nelse: print(f\"FAIL — {acc:.4f}\")",
      "solution": "p = Pipeline([('s', StandardScaler()),\n              ('m', MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=500, random_state=0))])\np.fit(Xt, yt)\nacc = p.score(Xe, ye)"
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Es. 13.3 — ReLU e derivata",
      "prompt": "<p>Implementa ReLU e la sua derivata. La derivata è 1 per $z > 0$, 0 altrove.</p>",
      "starter": "import numpy as np\ndef relu(z):\n    return None\ndef relu_grad(z):\n    return None\n\nz = np.array([-2., 0., 1., 3.])\nprint(relu(z), relu_grad(z))",
      "check": "import numpy as np\nif np.allclose(relu(z), [0,0,1,3]) and np.allclose(relu_grad(z), [0,0,1,1]):\n    print(\"OK\")\nelse:\n    print(\"FAIL\")",
      "solution": "def relu(z): return np.maximum(0, z)\ndef relu_grad(z): return (z > 0).astype(float)"
    }
  ],
  "test": [
    {
      "type": "quiz",
      "questions": [
        {
          "q": "Senza attivazioni non lineari, un MLP profondo è equivalente a:",
          "options": [
            "Un modello lineare",
            "Una foresta",
            "Un SVM",
            "Niente"
          ],
          "answer": 0,
          "explain": "Composizione di lineari = lineare."
        },
        {
          "q": "ReLU è preferita alla sigmoide perché:",
          "options": [
            "È più facile da scrivere",
            "Ha gradiente non-nullo per $z>0$ senza saturare",
            "È non lineare",
            "È simmetrica"
          ],
          "answer": 1,
          "explain": "Sigmoide satura per $|z|$ grande, gradient vanishing."
        },
        {
          "q": "Backpropagation applica:",
          "options": [
            "Il gradient descent",
            "La regola della catena all'indietro nel grafo di calcolo",
            "La trasformata di Fourier",
            "L'ottimizzazione lineare"
          ],
          "answer": 1,
          "explain": "Calcola gradienti composti."
        },
        {
          "q": "Il dropout serve per:",
          "options": [
            "Velocizzare training",
            "Regolarizzare: azzera casualmente neuroni per prevenire co-adaptation",
            "Ridurre memoria",
            "Aumentare accuracy training"
          ],
          "answer": 1,
          "explain": "Riduce overfitting."
        }
      ]
    },
    {
      "type": "exercise",
      "difficulty": "hard",
      "title": "Test pratico — MLP su digits",
      "prompt": "<p>Costruisci un MLP che raggiunga accuracy > 0.97 su digits. Puoi scegliere qualsiasi architettura, regolarizzazione, ottimizzatore. Salva in <code class=\"inline\">acc</code>.</p>",
      "starter": "from sklearn.datasets import load_digits\nfrom sklearn.neural_network import MLPClassifier\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.model_selection import train_test_split\n\nX, y = load_digits(return_X_y=True)\nXt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)\n\nacc = None",
      "check": "if acc is None: print(\"FAIL: None\")\nelif acc > 0.97: print(f\"OK — {acc:.4f}\")\nelse: print(f\"FAIL — {acc:.4f}\")",
      "solution": "p = Pipeline([('s', StandardScaler()),\n              ('m', MLPClassifier(hidden_layer_sizes=(128, 64),\n                                  activation='relu',\n                                  alpha=1e-4,\n                                  max_iter=500,\n                                  random_state=0))])\np.fit(Xt, yt)\nacc = p.score(Xe, ye)"
    }
  ]
});
