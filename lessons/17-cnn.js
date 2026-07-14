/* Lezione: CNN e Deep Learning pratico */
window.LESSONS.push({
  id: 'l17-cnn',
  title: 'CNN e Deep Learning pratico',
  tag: 'Deep Learning',
  summary: 'Reti convoluzionali per immagini: convoluzione, pooling, architetture moderne, transfer learning. Con codice PyTorch da eseguire in locale.',
  teoria: [
    { type: 'md', content: `
<h3>17.1 Perché le CNN</h3>
<p>Se applichi un MLP a un\'immagine 224x224 RGB, il primo layer avrebbe 224*224*3 = 150.528 input, e con 1000 neuroni nel primo layer sarebbero 150 milioni di parametri solo per il primo strato. Improponibile.</p>
<p>Le <em>Convolutional Neural Networks</em> risolvono il problema con tre idee:</p>
<ol>
<li><strong>Local connectivity</strong>: ogni neurone vede solo una piccola porzione dell\'immagine (un "recettore" locale).</li>
<li><strong>Weight sharing</strong>: gli stessi filtri sono applicati a tutte le posizioni dell\'immagine. Se un filtro riconosce un bordo verticale, lo riconosce ovunque.</li>
<li><strong>Hierarchical features</strong>: layer successivi combinano feature semplici (bordi) in feature complesse (occhi, poi facce).</li>
</ol>

<h3>17.2 La convoluzione</h3>
<p>Data un\'immagine $I$ e un filtro (kernel) $K$ di dimensione $k \\times k$, la convoluzione produce una feature map:</p>
<p>$$(I * K)_{ij} = \\sum_{u=0}^{k-1}\\sum_{v=0}^{k-1} I_{i+u, j+v} \\cdot K_{uv}$$</p>
<p>Un kernel è un piccolo tensore di pesi (es. 3x3) che scivola sull\'immagine. Ogni posizione produce un valore nella feature map.</p>
<p>Parametri chiave:</p>
<ul>
<li><strong>Kernel size</strong>: 3x3 e 5x5 sono i più comuni. Kernel più grandi vedono un contesto più ampio ma più parametri.</li>
<li><strong>Stride</strong>: passo con cui il kernel scivola. Stride 2 dimezza la dimensione dell\'output.</li>
<li><strong>Padding</strong>: aggiunge zeri ai bordi per preservare la dimensione. "same" padding mantiene HxW, "valid" no.</li>
<li><strong>Channels</strong>: per input RGB (3 canali), il filtro è 3D (kxkx3). L\'output ha un canale per filtro; con 64 filtri, output ha 64 canali.</li>
</ul>

<h3>17.3 Pooling</h3>
<p>Riduzione dimensionale + invarianza traslazionale. Il pooling più comune è <em>max pooling 2x2 con stride 2</em>: prende il massimo di ogni finestra 2x2, dimezzando altezza e larghezza.</p>
<p>Effetto: rende la rete un po\' invariante a piccole traslazioni ("l\'occhio non deve essere esattamente nel pixel 45,32").</p>

<h3>17.4 Un\'architettura CNN classica</h3>
<pre class="code">Input (224x224x3)
  ↓ Conv 3x3, 64 filtri, ReLU
  ↓ Conv 3x3, 64 filtri, ReLU
  ↓ MaxPool 2x2      → (112x112x64)
  ↓ Conv 3x3, 128 filtri, ReLU
  ↓ Conv 3x3, 128 filtri, ReLU
  ↓ MaxPool 2x2      → (56x56x128)
  ↓ ... (blocchi ripetuti)
  ↓ Flatten
  ↓ Dense 512, ReLU
  ↓ Dropout
  ↓ Dense 1000, softmax   (per ImageNet 1000 classi)</pre>
<p>La progressione tipica: <strong>H,W diminuiscono, canali aumentano</strong>. Man mano che la risoluzione spaziale cala, la profondità semantica sale.</p>

<h3>17.5 Architetture storiche</h3>
<ul>
<li><strong>LeNet-5</strong> (1998): la prima CNN moderna. Riconosceva cifre MNIST.</li>
<li><strong>AlexNet</strong> (2012): vinse ImageNet con margine enorme, iniziò l\'era del deep learning.</li>
<li><strong>VGG</strong> (2014): solo conv 3x3 e pool 2x2, molto profonda.</li>
<li><strong>ResNet</strong> (2015): introdusse <em>residual connections</em>, permise reti da 50-1000 layer. Vincitore ImageNet 2015.</li>
<li><strong>EfficientNet</strong> (2019): scaling composto di depth/width/resolution.</li>
<li><strong>Vision Transformer</strong> (2020): trasferisce l\'architettura Transformer alle immagini. Compete con CNN a scala.</li>
</ul>

<h3>17.6 Residual connections</h3>
<p>Un blocco residuale calcola:</p>
<p>$$y = F(x) + x$$</p>
<p>invece di $y = F(x)$. Il gradiente può fluire attraverso lo skip senza attraversare F, risolvendo il vanishing gradient in reti molto profonde. Semplice ma rivoluzionario.</p>

<h3>17.7 Transfer learning</h3>
<p>La tecnica pratica più importante del deep learning applicato:</p>
<ol>
<li>Prendi una rete pre-addestrata (es. ResNet50 su ImageNet).</li>
<li>Rimuovi l\'ultimo layer (originale per 1000 classi ImageNet).</li>
<li>Sostituisci con un layer per il tuo problema (es. 5 classi custom).</li>
<li>Addestra solo l\'ultimo layer (fine-tuning veloce), o tutto il modello con lr basso (fine-tuning completo).</li>
</ol>
<p>Perché funziona: i primi layer imparano feature generali (bordi, texture) che sono utili ovunque. Serve solo adattare i layer finali al task specifico.</p>
<p>Con 1000-10000 esempi di training questa tecnica supera facilmente reti addestrate da zero con 100000 esempi.</p>

<h3>17.8 Data augmentation</h3>
<p>Per immagini è cruciale. Trasformazioni comuni:</p>
<ul>
<li>Random crop, resize</li>
<li>Horizontal flip</li>
<li>Color jitter (brightness, contrast, saturation)</li>
<li>Random rotation, translation, shear</li>
<li>Cutout, MixUp, CutMix (moderni)</li>
</ul>
<p>Effetto: aumenta artificialmente il dataset di 5-100x, regolarizza fortemente.</p>

<h3>17.9 PyTorch: il framework standard</h3>
<p>Nella pratica industriale si usa <strong>PyTorch</strong> (o TensorFlow, ma PyTorch domina la ricerca e sempre più anche produzione). Ha:</p>
<ul>
<li>Autograd automatico: tu scrivi il forward, i gradienti sono gratis.</li>
<li>Modulo <code class="inline">nn</code> con layer pronti (Conv2d, Linear, BatchNorm2d).</li>
<li>Ottimizzatori (SGD, Adam, AdamW) e loss (CrossEntropyLoss, MSELoss).</li>
<li>DataLoader con multi-processing, augmentation via torchvision.transforms.</li>
<li>Modelli pretrained via torchvision.models (ResNet, EfficientNet, ViT).</li>
</ul>
<p>⚠️ <strong>PyTorch non è disponibile in Pyodide</strong> per limiti di dimensione. Il codice sotto lo esegui in locale con <code class="inline">pip install torch torchvision</code>. In questa lezione useremo NumPy per capire i concetti e sklearn/scipy per esempi eseguibili.</p>
` },
    { type: 'callout', variant: 'note', title: 'Setup PyTorch in locale', content: 'Sul tuo PC: <code>pip install torch torchvision</code> (~1GB per la versione CPU-only, molto di più per GPU). Se hai una GPU NVIDIA, installa la versione CUDA seguendo <a href="https://pytorch.org/get-started/locally/">pytorch.org</a>. Con Colab hai una GPU gratis.' },
    { type: 'md', content: `
<h3>17.10 Formula dimensione output convoluzione</h3>
<p>Data una feature map $H \\times W$, dopo una conv con kernel $k$, stride $s$, padding $p$:</p>
<p>$$H_{out} = \\left\\lfloor \\frac{H + 2p - k}{s} \\right\\rfloor + 1$$</p>
<p>Stessa formula per la larghezza. Esempi con input $H = 32$:</p>
<ul>
<li><strong>Same padding</strong>: $k=3, p=1, s=1$ → $\\lfloor(32+2-3)/1\\rfloor+1 = 32$ (dimensione invariata)</li>
<li><strong>Stride 2</strong>: $k=3, p=1, s=2$ → $\\lfloor(32+2-3)/2\\rfloor+1 = 16$ (dimezza)</li>
<li><strong>Valid, stride 2</strong>: $k=5, p=0, s=2$ → $\\lfloor(32-5)/2\\rfloor+1 = 14$</li>
</ul>
<p><strong>Regola same padding</strong>: per kernel dispari $k$, il padding $p = (k-1)/2$ garantisce $H_{out} = H_{in}$ con $s=1$. Per kernel pari non esiste una soluzione intera — ecco perché si usano quasi sempre kernel dispari (1, 3, 5, 7).</p>

<h3>17.11 Receptive Field</h3>
<p>Il <em>receptive field</em> (RF) è la porzione dell'immagine originale che influenza una singola attivazione in un dato layer. Con $L$ strati conv di kernel $k$ e stride 1:</p>
<p>$$RF = 1 + L \\cdot (k - 1)$$</p>
<p>Esempi pratici:</p>
<ul>
<li>1 layer 3×3: RF = 3</li>
<li>2 layer 3×3: RF = 5 (equivalente a una conv 5×5, ma con meno parametri: $2 \\cdot 9 = 18$ vs $25$)</li>
<li>5 layer 3×3: RF = 11</li>
</ul>
<p>Con stride $s > 1$ al layer $l$, il RF cresce moltiplicativamente: un MaxPool 2×2 (stride 2) raddoppia il RF di tutti i layer successivi.</p>
<p><strong>Questo motiva le architetture deep</strong>: i primi layer hanno RF piccolo → catturano feature locali (bordi, texture); i layer profondi hanno RF grande → catturano strutture globali (forme, oggetti interi). Usare due conv 3×3 invece di una 5×5 dà stesso RF, meno parametri, più non-linearità (due ReLU invece di una).</p>

<h3>17.12 Batch Normalization nelle CNN</h3>
<p>Nelle reti conv, BatchNorm normalizza <strong>per canale</strong>: media e varianza calcolate su (batch, H, W) per ogni canale $c$:</p>
<p>$$\\hat{x}_c = \\frac{x_c - \\mu_c}{\\sqrt{\\sigma_c^2 + \\varepsilon}}, \\quad y_c = \\gamma_c \\hat{x}_c + \\beta_c$$</p>
<p>I parametri $\\gamma_c$ (scala) e $\\beta_c$ (shift) sono appresi durante il training — solo $2C$ parametri in totale, molto pochi rispetto ai pesi.</p>
<p><strong>Posizionamento moderno</strong>: <code class="inline">Conv → BN → ReLU</code>, non Conv → ReLU → BN. BN prima della non-linearità mantiene una distribuzione più stabile degli input a ReLU, evitando saturazione.</p>
<p><strong>Alternative per batch piccoli</strong> (es. object detection con batch 2-4):</p>
<ul>
<li><strong>Group Normalization</strong>: divide i $C$ canali in $G$ gruppi (es. $G=32$) e normalizza dentro ogni gruppo. Non dipende dalla dimensione del batch — la scelta standard per detection e segmentation.</li>
<li><strong>Layer Normalization</strong>: normalizza su tutti i canali per ogni singolo esempio. Standard nei Vision Transformers (ViT).</li>
</ul>

<h3>17.13 Global Average Pooling (GAP)</h3>
<p>Il GAP sostituisce <code class="inline">Flatten → Dense</code>: invece di appiattire tutta la feature map, calcola la <strong>media spaziale</strong> di ogni canale:</p>
<p>$$GAP_c = \\frac{1}{H \\cdot W} \\sum_{i=1}^{H} \\sum_{j=1}^{W} x_{c,i,j}$$</p>
<p>Risultato: $(C, H, W) \\to (C,)$. Poi un singolo <code class="inline">Linear(C, num_classes)</code> fa la classificazione finale.</p>
<p><strong>Vantaggi rispetto a Flatten + Dense:</strong></p>
<ul>
<li><strong>Invarianza alla dimensione dell'input</strong>: lo stesso modello addestrato su 224×224 accetta immagini di qualsiasi risoluzione in inference (reti fully-convolutional)</li>
<li><strong>Drastica riduzione parametri</strong>: $7 \\times 7 \\times 512 \\to$ Dense(10) = 25.6M pesi; GAP(512) $\\to$ Dense(10) = soli 5.1K pesi</li>
<li><strong>Meno overfitting</strong>: nessun layer denso enorme da regolarizzare con dropout</li>
</ul>
<p>Usato in ResNet, EfficientNet, MobileNet, e praticamente ogni architettura post-2015. Le fully-convolutional networks basate su GAP sono la norma nell'era moderna.</p>

<h3>17.14 Depthwise Separable Convolutions</h3>
<p>Una conv standard $K \\times K$, $C_{in} \\to C_{out}$ ha costo computazionale $K^2 \\cdot C_{in} \\cdot C_{out}$ per ogni posizione spaziale. La <strong>depthwise separable conv</strong> fattorizza in due step:</p>
<ol>
<li><strong>Depthwise conv</strong>: un kernel $K \\times K$ applicato <em>indipendentemente</em> a ciascuno dei $C_{in}$ canali (non combina canali tra loro). Costo: $K^2 \\cdot C_{in}$.</li>
<li><strong>Pointwise conv</strong>: conv $1 \\times 1$ che ricombina i canali. Costo: $C_{in} \\cdot C_{out}$.</li>
</ol>
<p>Risparmio computazionale rispetto alla conv standard:</p>
<p>$$\\frac{K^2 C_{in} + C_{in} C_{out}}{K^2 C_{in} C_{out}} = \\frac{1}{C_{out}} + \\frac{1}{K^2}$$</p>
<p>Per $K=3$: $\\approx 1/9$, ovvero <strong>8-9× meno operazioni</strong>. Base di <strong>MobileNet</strong> (modelli per mobile/edge), <strong>Xception</strong>, e componenti di <strong>EfficientNet</strong>. La qualità rimane quasi identica a parità di parametri totali.</p>

<h3>17.15 Strategie di fine-tuning</h3>
<p>Tre approcci in ordine crescente di aggressività:</p>
<ol>
<li><strong>Feature extraction</strong> (freeze_all → train_head): congela tutti i pesi pre-addestrati (<code class="inline">requires_grad=False</code>), addestra solo il nuovo classificatore. Veloce (minuti), basso rischio di <em>catastrophic forgetting</em>. Ottimo se il dataset è simile a ImageNet.</li>
<li><strong>Gradual unfreeze</strong>: sblocca i layer dall'ultimo al primo progressivamente (un blocco per volta), con LR molto piccolo per gli strati profondi. Tecnica popolarizzata da fastai — bilancia adattamento e conservazione delle feature.</li>
<li><strong>Discriminative learning rates</strong>: LR diversi per ogni layer o gruppo — piccolo vicino all'input (feature generali già buone, non vuoi cambiarle), grande vicino all'output (classificatore da adattare). Tipicamente LR/10 per ogni strato verso l'input.</li>
</ol>
<p><strong>Regola pratica</strong>: dataset simile a ImageNet (fotografie naturali) → congela di più; dataset molto diverso (immagini medicali, satellitari, microscopia) → addestra più strati con LR piccolo (1e-4 o meno). Con pochissimi dati (&lt;500 esempi) usa quasi sempre solo feature extraction.</p>

<h3>17.16 Mixed Precision Training (FP16)</h3>
<p>Le GPU con Tensor Core (NVIDIA V100, A100, RTX 30xx/40xx) eseguono operazioni FP16 2-8× più velocemente di FP32. La <em>mixed precision</em> combina i punti di forza di entrambi:</p>
<ul>
<li><strong>FP32</strong>: copia master dei pesi, accumulatori nell'optimizer (necessario per stabilità numerica)</li>
<li><strong>FP16</strong>: forward pass, backward pass, gradienti — metà memoria, molto più veloce</li>
</ul>
<p><strong>Gradient scaling</strong>: i gradienti in FP16 possono fare underflow a zero perché i valori assoluti sono troppo piccoli per la precisione FP16 (minimo $\approx 6 \\times 10^{-5}$). Soluzione: moltiplica la loss per un fattore di scala $s$ prima del backward, poi dividi prima dell'update dell'optimizer. PyTorch gestisce questo automaticamente con <code class="inline">GradScaler</code>:</p>
<pre class="code">from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

for x, y in loader:
    optimizer.zero_grad()
    with autocast():               # forward in FP16
        loss = loss_fn(model(x), y)
    scaler.scale(loss).backward()  # backward con gradient scaling
    scaler.step(optimizer)         # unscale + clip + optimizer step
    scaler.update()                # aggiusta scaling factor automaticamente</pre>
<p>Risultato pratico: quasi zero perdita di accuracy, ~2× speedup, ~2× riduzione memoria GPU. Una delle ottimizzazioni con il miglior rapporto sforzo/beneficio.</p>
` },
    { type: 'callout', variant: 'tip', title: 'Calcolo rapido dimensioni CNN', content: 'Con same padding (p=(k-1)/2, s=1) la feature map mantiene H&times;W invariato. Stride 2 dimezza, MaxPool 2&times;2 dimezza. Con input 224&times;224: dopo 5 dimezzamenti &rarr; 224/32 = 7. Scrivi queste dimensioni <em>prima</em> di costruire il modello per evitare errori di shape nel codice. PyTorch mostra un errore oscuro se le dimensioni non tornano nel classificatore finale.' },
    { type: 'callout', variant: 'warn', title: 'Fine-tuning: attenzione al catastrophic forgetting', content: 'Se usi un LR troppo alto durante il fine-tuning completo, i layer pre-addestrati perdono le feature imparate su ImageNet (catastrophic forgetting). Usa sempre LR basso per i layer profondi (1e-4 o meno). Se hai meno di 500 esempi, usa quasi sempre feature extraction pura (freeze tutto tranne l&apos;ultima head) per evitare overfitting drastico.' },
  ],
  esempi: [
    { type: 'md', content: '<h3>Esempio 1: convoluzione manuale su un\'immagine</h3>' },
    { type: 'run', content: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Carichiamo una cifra 8x8
digits = load_digits()
img = digits.images[0]   # cifra "0"
print(f"Immagine {img.shape}")

# Filtro Sobel per bordi verticali
Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

def conv2d(img, K):
    h, w = img.shape
    kh, kw = K.shape
    out = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = (img[i:i+kh, j:j+kw] * K).sum()
    return out

edges_x = conv2d(img, Kx)
edges_y = conv2d(img, Ky)
edges_mag = np.sqrt(edges_x**2 + edges_y**2)

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
axes[0].imshow(img, cmap='gray'); axes[0].set_title('Originale')
axes[1].imshow(edges_x, cmap='gray'); axes[1].set_title('Bordi verticali (Sobel x)')
axes[2].imshow(edges_y, cmap='gray'); axes[2].set_title('Bordi orizzontali (Sobel y)')
axes[3].imshow(edges_mag, cmap='gray'); axes[3].set_title('Magnitudine bordi')
for ax in axes: ax.axis('off')
plt.tight_layout(); plt.show()
print("Le CNN imparano automaticamente filtri come questi, ma anche molto più complessi.")` },
    { type: 'md', content: '<h3>Esempio 2: max pooling da zero</h3>' },
    { type: 'run', content: `import numpy as np
def max_pool_2x2(x, stride=2):
    h, w = x.shape
    out_h, out_w = h // stride, w // stride
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = x[i*stride:(i+1)*stride, j*stride:(j+1)*stride].max()
    return out

x = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]], dtype=float)
print("Input 4x4:")
print(x)
print("\\nMax pool 2x2:")
print(max_pool_2x2(x))
# Espansione: prende il max di ogni sotto-blocco 2x2` },
    { type: 'md', content: '<h3>Esempio 3: numero di parametri e cost of MLP vs CNN</h3>' },
    { type: 'run', content: `# Confronto stimato numero parametri
# Immagine 32x32 RGB (CIFAR-10)
input_size = 32 * 32 * 3   # 3072

# MLP con hidden layer da 128
mlp_params = input_size * 128 + 128 + 128 * 10 + 10
print(f"MLP (3072 -> 128 -> 10): {mlp_params:,} parametri")

# CNN piccola: 2 conv layer + 1 FC
# Conv1: 3x3x3 -> 32 canali = 3*3*3*32 + 32 = 896
# Conv2: 3x3x32 -> 64 canali = 3*3*32*64 + 64 = 18496
# Dopo 2 pool 2x2: 8x8x64 = 4096 flatten
# FC: 4096 -> 10 = 40970
cnn_params = 896 + 18496 + 4096 * 10 + 10
print(f"CNN piccola: {cnn_params:,} parametri")

# Immagine più grande (224x224 RGB) l'MLP diventa insostenibile
input_big = 224*224*3
mlp_big = input_big * 128 + 128
print(f"\\nMLP su 224x224 RGB con 128 neuroni: {mlp_big:,} parametri (~{mlp_big/1e6:.1f}M)")
# ResNet50 completo ha 25M parametri e input 224x224. Molto più efficiente.
print("Le CNN riusano gli stessi pesi per posizioni diverse: enorme risparmio.")` },
    { type: 'md', content: '<h3>Esempio 4: filtro appreso vs disegnato</h3>' },
    { type: 'run', content: `import numpy as np
import matplotlib.pyplot as plt

# Immagine sintetica con un bordo verticale
img = np.zeros((20, 20))
img[:, 10:] = 1
img += 0.05 * np.random.randn(20, 20)

# Filtro random iniziale
np.random.seed(0)
K = np.random.randn(3, 3) * 0.5
lr = 0.1

# Obiettivo: filtro che risponde massimamente al bordo
# Feature map target: alta risposta nella colonna del bordo
target = np.zeros((18, 18))
target[:, 8:11] = 1

def conv2d(img, K):
    h, w = img.shape; kh, kw = K.shape
    out = np.zeros((h-kh+1, w-kw+1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i,j] = (img[i:i+kh, j:j+kw] * K).sum()
    return out

# Training del filtro con gradient descent
for it in range(50):
    out = conv2d(img, K)
    loss = np.mean((out - target)**2)
    # Gradient del filtro (semplice, per un solo esempio)
    grad_K = np.zeros_like(K)
    err = 2 * (out - target) / out.size
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            grad_K += err[i, j] * img[i:i+3, j:j+3]
    K = K - lr * grad_K

print("Filtro finale (dovrebbe essere simile al Sobel x):")
print(K.round(2))
print("\\nSobel x per confronto:")
print(np.array([[-1,0,1],[-2,0,2],[-1,0,1]]))
print("\\nLa CNN impara automaticamente questi pattern dai dati.")` },
    { type: 'md', content: '<h3>Esempio 5: MLP per approssimare una CNN su MNIST</h3>' },
    { type: 'run', content: `from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# MLP piccolo su MNIST (8x8)
p = Pipeline([('s', StandardScaler()),
               ('m', MLPClassifier(hidden_layer_sizes=(256, 128),
                                   max_iter=300, random_state=0))])
p.fit(Xt, yt)
print(f"MLP su digits 8x8: test acc = {p.score(Xe, ye):.4f}")
print("\\nSu immagini più grandi (MNIST 28x28, CIFAR-10 32x32), CNN batterebbe MLP.")
print("PyTorch code equivalente (da eseguire in locale):")
print('''
import torch.nn as nn
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
''')` },
    { type: 'md', content: '<h3>Esempio 6: codice PyTorch di riferimento (esegui in locale)</h3>' },
    { type: 'code', content: `# Questo codice richiede: pip install torch torchvision
# NON gira in Pyodide, salvalo in locale ed esegui con Python normale.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Data augmentation + normalizzazione
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 2. Dataset
train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
test_ds  = datasets.CIFAR10('./data', train=False, transform=transform_test)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)

# 3. Modello CNN semplice
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),   # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),   # 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),   # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
loss_fn = nn.CrossEntropyLoss()

# 4. Training loop
for epoch in range(50):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    scheduler.step()

    # Eval
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"Epoch {epoch+1}: test acc = {100*correct/total:.2f}%")
` },
    { type: 'md', content: '<h3>Esempio 7: transfer learning con ResNet (codice PyTorch)</h3>' },
    { type: 'code', content: `# Transfer learning: fine-tuning ResNet18 su un dataset custom
# Esegui in locale con: pip install torch torchvision

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Preprocessing standard per modelli pre-addestrati ImageNet
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Assumendo che tu abbia una cartella ./data/train/classe_A, ./data/train/classe_B ...
train_ds = ImageFolder('./data/train', transform=transform)
num_classes = len(train_ds.classes)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# Carichiamo ResNet18 pre-addestrato
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Strategia 1: freeze tutto tranne l'ultimo layer (feature extraction)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, num_classes)   # nuovo classificatore

# Strategia 2 (fine-tuning completo): non fare il freeze, ma usa lr piccolo
# model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)   # solo il classificatore
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done")
` },
  ],
  esercizi: [
    { type: 'exercise', difficulty: 'easy', title: 'Es. 17.1 — Dimensione output di conv',
      prompt: '<p>Data una feature map di shape (H, W), applicando un conv con kernel_size=k, stride=s, padding=p, l\'output ha shape $(\\lfloor(H+2p-k)/s\\rfloor+1, \\lfloor(W+2p-k)/s\\rfloor+1)$. Calcola le dimensioni per H=W=32, k=3, s=1, p=1 e per k=5, s=2, p=0. Salvale in <code class="inline">shape1, shape2</code> come tuple.</p>',
      starter: `shape1 = None  # per k=3, s=1, p=1
shape2 = None  # per k=5, s=2, p=0
print(shape1, shape2)`,
      check: `if shape1 == (32, 32) and shape2 == (14, 14):
    print(f"OK — {shape1}, {shape2}")
else:
    print(f"FAIL — atteso (32,32) e (14,14), tuo {shape1}, {shape2}")`,
      solution: `# formula: floor((H + 2p - k) / s) + 1
# (32 + 2 - 3) / 1 + 1 = 32
# (32 + 0 - 5) / 2 + 1 = floor(27/2) + 1 = 13 + 1 = 14
shape1 = (32, 32)
shape2 = (14, 14)` },

    { type: 'exercise', difficulty: 'easy', title: 'Es. 17.2 — Convoluzione manuale',
      prompt: '<p>Implementa <code class="inline">conv2d_valid(img, K)</code>: convoluzione (in realtà cross-correlazione) con padding=0, stride=1. Ritorna la feature map.</p>',
      starter: `import numpy as np
def conv2d_valid(img, K):
    return None

img = np.arange(25).reshape(5, 5).astype(float)
K = np.array([[1, 0], [0, -1]], dtype=float)
print(conv2d_valid(img, K))`,
      check: `import numpy as np
h, w = img.shape; kh, kw = K.shape
expected = np.zeros((h-kh+1, w-kw+1))
for i in range(expected.shape[0]):
    for j in range(expected.shape[1]):
        expected[i,j] = (img[i:i+kh, j:j+kw] * K).sum()
result = conv2d_valid(img, K)
if result is not None and np.allclose(result, expected):
    print("OK")
else:
    print("FAIL")`,
      solution: `def conv2d_valid(img, K):
    h, w = img.shape
    kh, kw = K.shape
    out = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = (img[i:i+kh, j:j+kw] * K).sum()
    return out` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 17.3 — Padding "same"',
      prompt: '<p>Modifica la convoluzione per usare "same" padding (output stessa dimensione dell\'input). Il padding necessario per kernel k è $(k-1)/2$. Implementa <code class="inline">conv2d_same(img, K)</code> con K quadrato di dimensione dispari.</p>',
      starter: `import numpy as np
def conv2d_same(img, K):
    return None

img = np.random.randn(6, 6)
K = np.ones((3, 3)) / 9
out = conv2d_same(img, K)
print("Input shape:", img.shape, "-> Output shape:", out.shape if out is not None else None)`,
      check: `import numpy as np
if out is not None and out.shape == img.shape:
    print(f"OK — shape preservata {out.shape}")
else:
    print(f"FAIL — atteso {img.shape}")`,
      solution: `def conv2d_same(img, K):
    kh, kw = K.shape
    p = kh // 2
    padded = np.pad(img, ((p, p), (p, p)), mode='constant')
    h, w = img.shape
    out = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            out[i, j] = (padded[i:i+kh, j:j+kw] * K).sum()
    return out` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 17.4 — Max pooling con stride variabile',
      prompt: '<p>Implementa <code class="inline">max_pool(x, k, stride)</code> generale, con finestra k×k e stride variabile. Testa con k=3, stride=2 su una griglia 8×8.</p>',
      starter: `import numpy as np
def max_pool(x, k, stride):
    return None

x = np.arange(64).reshape(8, 8).astype(float)
print("Max pool k=3 stride=2:")
print(max_pool(x, 3, 2))`,
      check: `import numpy as np
h, w = x.shape
out_h = (h - 3) // 2 + 1
out_w = (w - 3) // 2 + 1
_expected = np.zeros((out_h, out_w))
for i in range(out_h):
    for j in range(out_w):
        _expected[i,j] = x[i*2:i*2+3, j*2:j*2+3].max()
result = max_pool(x, 3, 2)
if result is not None and np.allclose(result, _expected):
    print("OK")
else:
    print("FAIL")`,
      solution: `def max_pool(x, k, stride):
    h, w = x.shape
    out_h = (h - k) // stride + 1
    out_w = (w - k) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = x[i*stride:i*stride+k, j*stride:j*stride+k].max()
    return out` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 17.5 — Numero di parametri',
      prompt: '<p>Calcola il numero totale di parametri (pesi + bias) di una CNN: Conv2d(3->32, kernel 3), Conv2d(32->64, kernel 3), Linear(64*7*7 -> 128), Linear(128 -> 10). Salva in <code class="inline">n_params</code>.</p>',
      starter: `# Formula conv: (kh*kw*in_channels + 1) * out_channels
# Formula linear: (in_features + 1) * out_features   (il +1 è per il bias, ma bias è out_features)
# Più precisamente: pesi = in*out, bias = out. Totale = in*out + out.

n_params = None
print(n_params)`,
      check: `# Conv1: 3*3*3*32 + 32 = 864 + 32 = 896
# Conv2: 3*3*32*64 + 64 = 18432 + 64 = 18496
# Linear1: 64*7*7*128 + 128 = 401408 + 128 = 401536
# Linear2: 128*10 + 10 = 1290
_expected = 896 + 18496 + 401536 + 1290
if n_params == _expected:
    print(f"OK — {n_params:,} parametri")
else:
    print(f"FAIL — atteso {_expected:,}, tuo {n_params}")`,
      solution: `conv1 = 3*3*3*32 + 32
conv2 = 3*3*32*64 + 64
lin1 = 64*7*7*128 + 128
lin2 = 128*10 + 10
n_params = conv1 + conv2 + lin1 + lin2` },

    { type: 'exercise', difficulty: 'medium', title: 'Es. 17.6 — Filtro Gaussian blur',
      prompt: '<p>Crea un filtro Gaussiano 5×5 con sigma=1.0 e applicalo a un\'immagine. Il kernel deve sommarsi a 1. Salva il kernel in <code class="inline">K</code>.</p>',
      starter: `import numpy as np
def gaussian_kernel(size, sigma):
    return None

K = gaussian_kernel(5, 1.0)
print("Kernel Gaussian 5x5 sigma=1:")
print(K.round(3))
print("Somma:", K.sum())`,
      check: `import numpy as np
if K is not None and K.shape == (5, 5) and abs(K.sum() - 1) < 1e-6:
    # verifica il picco al centro
    if K[2, 2] == K.max():
        print(f"OK — kernel Gaussiano valido, picco al centro {K[2,2]:.4f}")
    else:
        print("FAIL — picco non al centro")
else:
    print(f"FAIL — shape {K.shape if K is not None else None}, somma {K.sum() if K is not None else None}")`,
      solution: `def gaussian_kernel(size, sigma):
    ax = np.arange(-(size//2), size//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    K = np.exp(-(xx**2 + yy**2) / (2*sigma**2))
    return K / K.sum()` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 17.7 — CNN concettuale con NumPy',
      prompt: '<p>Implementa un mini CNN forward pass in NumPy: Conv(1 canale -> 4 canali, 3×3, ReLU) + MaxPool 2×2. Input 8×8 (una cifra di digits). Salva output in <code class="inline">out</code>. Deve avere shape (4, 3, 3).</p>',
      starter: `import numpy as np
from sklearn.datasets import load_digits

img = load_digits().images[0]  # 8x8
np.random.seed(0)
filters = np.random.randn(4, 3, 3) * 0.3  # 4 filtri 3x3
biases = np.zeros(4)

def conv2d_multi(img, filters, biases):
    return None

def relu(z):
    return np.maximum(0, z)

def max_pool_2x2(x):
    h, w = x.shape
    out = np.zeros((h//2, w//2))
    for i in range(h//2):
        for j in range(w//2):
            out[i,j] = x[i*2:i*2+2, j*2:j*2+2].max()
    return out

# TODO: applica conv -> relu -> max pool
out = None
print("Shape:", out.shape if out is not None else None)`,
      check: `import numpy as np
if out is not None and out.shape == (4, 3, 3):
    if (out >= 0).all():
        print(f"OK — shape {out.shape}, valori non-negativi (ReLU applicata)")
    else:
        print("FAIL — valori negativi (ReLU non applicata)")
else:
    print(f"FAIL — atteso (4, 3, 3), tuo {out.shape if out is not None else None}")`,
      solution: `def conv2d_multi(img, filters, biases):
    n_filters = filters.shape[0]
    kh = filters.shape[1]
    h, w = img.shape
    out = np.zeros((n_filters, h-kh+1, w-kh+1))
    for f in range(n_filters):
        for i in range(out.shape[1]):
            for j in range(out.shape[2]):
                out[f, i, j] = (img[i:i+kh, j:j+kh] * filters[f]).sum() + biases[f]
    return out

conv_out = conv2d_multi(img, filters, biases)
relu_out = relu(conv_out)
out = np.stack([max_pool_2x2(relu_out[f]) for f in range(4)])` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 17.8 — Global Average Pooling',
      prompt: '<p>Il Global Average Pooling (GAP) è un modo moderno di ridurre da (C, H, W) a (C,) prendendo la media spaziale di ogni canale. Implementa <code class="inline">gap(x)</code> per x di shape (C, H, W).</p>',
      starter: `import numpy as np
def gap(x):
    return None

x = np.random.randn(64, 7, 7)
out = gap(x)
print("Input shape:", x.shape, "-> GAP shape:", out.shape if out is not None else None)`,
      check: `import numpy as np
_expected = x.mean(axis=(1, 2))
if out is not None and np.allclose(out, _expected):
    print(f"OK — shape {out.shape}")
else:
    print("FAIL")`,
      solution: `def gap(x):
    return x.mean(axis=(1, 2))` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 17.9 — Batch normalization forward',
      prompt: '<p>Implementa il forward pass di BatchNorm: normalizza ogni canale rispetto alla dimensione batch. Input di shape (N, C, H, W). Output stessa shape, ma media 0 e varianza 1 per ogni canale. Ignora scale/shift learnable per semplicità.</p>',
      starter: `import numpy as np
def batch_norm_forward(x, eps=1e-5):
    return None

np.random.seed(0)
x = np.random.randn(4, 3, 5, 5) * 10 + 5
out = batch_norm_forward(x)
print("Media per canale post-BN:", out.mean(axis=(0, 2, 3)) if out is not None else None)
print("Std per canale post-BN:", out.std(axis=(0, 2, 3)) if out is not None else None)`,
      check: `import numpy as np
if out is not None:
    means = out.mean(axis=(0, 2, 3))
    stds = out.std(axis=(0, 2, 3))
    if np.allclose(means, 0, atol=1e-5) and np.allclose(stds, 1, atol=0.05):
        print("OK — media ~0, std ~1")
    else:
        print(f"FAIL — media {means}, std {stds}")`,
      solution: `def batch_norm_forward(x, eps=1e-5):
    mean = x.mean(axis=(0, 2, 3), keepdims=True)
    var = x.var(axis=(0, 2, 3), keepdims=True)
    return (x - mean) / np.sqrt(var + eps)` },

    { type: 'exercise', difficulty: 'hard', title: 'Es. 17.10 — Feature map response',
      prompt: '<p>Dato un filtro pre-definito (bordo verticale Sobel x) e un batch di 3 immagini digits, applica la conv e trova quale immagine ha la <strong>massima somma di attivazioni</strong> (segno di più bordi verticali). Salva l\'indice in <code class="inline">idx_max</code>.</p>',
      starter: `import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()
imgs = digits.images[:3]   # 3 immagini 8x8

K = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)

def conv2d_valid(img, K):
    h, w = img.shape; kh, kw = K.shape
    out = np.zeros((h-kh+1, w-kw+1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i,j] = (img[i:i+kh, j:j+kw] * K).sum()
    return out

idx_max = None
print("Indice immagine con più bordi verticali:", idx_max)`,
      check: `import numpy as np
responses = np.array([np.abs(conv2d_valid(img, K)).sum() for img in imgs])
_atteso = int(responses.argmax())
if idx_max == _atteso:
    print(f"OK — immagine {idx_max} ha massima risposta {responses[idx_max]:.2f}")
else:
    print(f"FAIL — atteso {_atteso}")`,
      solution: `responses = [np.abs(conv2d_valid(img, K)).sum() for img in imgs]
idx_max = int(np.argmax(responses))` },
  ],
  test: [
    { type: 'quiz', questions: [
      { q: 'Weight sharing nelle CNN significa:', options: ['Ogni neurone ha pesi diversi', 'Lo stesso kernel viene applicato a tutte le posizioni dell\'immagine', 'I pesi non vengono aggiornati', 'Solo l\'ultimo layer condivide pesi'], answer: 1, explain: 'Un filtro impara a rilevare un pattern che poi cerca ovunque.' },
      { q: 'Il max pooling introduce:', options: ['Più parametri', 'Invarianza a piccole traslazioni + riduzione dimensionale', 'Non-linearità', 'Regolarizzazione L2'], answer: 1, explain: 'Prende il max di una finestra, quindi lievi shift non cambiano l\'output.' },
      { q: 'Perché una CNN è più efficiente di un MLP per immagini?', options: ['Per il numero di layer', 'Grazie a weight sharing e local connectivity riduce enormemente i parametri', 'È più veloce a stampare', 'Non è vero'], answer: 1, explain: 'Un MLP su 224x224 avrebbe centinaia di milioni di parametri; una CNN qualche milione.' },
      { q: 'Transfer learning consiste in:', options: ['Copiare dati da un dataset all\'altro', 'Riutilizzare un modello pre-addestrato adattandolo a un nuovo task', 'Trasferire su GPU', 'Salvare e caricare i pesi'], answer: 1, explain: 'ResNet pre-addestrata su ImageNet è ottima base per quasi ogni task di visione.' },
      { q: 'Residual connections (ResNet) risolvono:', options: ['Il costo di memoria', 'Il vanishing gradient in reti molto profonde', 'La normalizzazione', 'Il pooling'], answer: 1, explain: '$y = F(x) + x$ permette al gradiente di fluire senza attraversare F.' },
    ] },
    { type: 'exercise', difficulty: 'hard', title: 'Test finale — mini CNN concettuale',
      prompt: '<p>Applica una pipeline completa CNN-like su digits (8×8): 4 filtri random 3×3 con ReLU, max pool 2×2, flatten, poi Logistic Regression sopra la rappresentazione. Ottieni test accuracy in <code class="inline">acc</code>. Target > 0.92.</p>',
      starter: `import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

digits = load_digits()
imgs = digits.images   # (1797, 8, 8)
y = digits.target

np.random.seed(0)
filters = np.random.randn(4, 3, 3) * 0.3

def conv2d(img, K):
    h, w = img.shape; kh, kw = K.shape
    out = np.zeros((h-kh+1, w-kh+1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i,j] = (img[i:i+kh, j:j+kw] * K).sum()
    return out

def max_pool_2x2(x):
    h, w = x.shape
    out = np.zeros((h//2, w//2))
    for i in range(h//2):
        for j in range(w//2):
            out[i,j] = x[i*2:i*2+2, j*2:j*2+2].max()
    return out

# TODO: applica filters + relu + pool + flatten a ogni immagine, poi LogisticRegression
acc = None`,
      check: `if acc is None: print("FAIL: None")
elif acc > 0.92: print(f"OK — {acc:.4f}")
else: print(f"FAIL — {acc:.4f}")`,
      solution: `features = []
for img in imgs:
    fs = []
    for K in filters:
        c = np.maximum(0, conv2d(img, K))
        p = max_pool_2x2(c)
        fs.extend(p.ravel())
    features.append(fs)
X = np.array(features)
Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
sc = StandardScaler().fit(Xt)
lr = LogisticRegression(max_iter=5000, multi_class='multinomial').fit(sc.transform(Xt), yt)
acc = lr.score(sc.transform(Xe), ye)` },

    { type: 'exercise', difficulty: 'hard', title: 'Bonus expert — analisi filtri appresi',
      prompt: '<p>Se addestriamo un MLP su digits e guardiamo i pesi del PRIMO layer come immagini 8×8, spesso si vedono pattern simili a filtri di edge detection. Verificalo: addestra MLP con hidden_layer_sizes=(16,) e visualizza il primo peso come immagine 8×8. Salva in <code class="inline">first_filter</code> un array 8×8.</p>',
      starter: `import numpy as np
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
X = StandardScaler().fit_transform(X)
m = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=0).fit(X, y)

# Prendi il PRIMO filtro del primo layer (colonna 0 di coefs_[0])
first_filter = None
print(first_filter.shape if first_filter is not None else None)`,
      check: `import numpy as np
if first_filter is not None and first_filter.shape == (8, 8):
    print(f"OK — filtro shape {first_filter.shape}")
    print("Valori (spesso pattern di edge detection):")
    print(first_filter.round(2))
else:
    print("FAIL")`,
      solution: `first_filter = m.coefs_[0][:, 0].reshape(8, 8)` },
  ]
});
