# SAM-Augmented Ensemble (Refactoring)

⚠️ **PROGETTO IN FASE DI REFACTORING** ⚠️

Questo repository contiene un framework per l'augmentazione di dataset di immagini mediche utilizzando Segment Anything Model (SAM/SAM2) e il training di modelli di segmentazione (HSNet, PolypPVT).

**Il progetto è attualmente in fase di ristrutturazione** per migliorare modularità, leggibilità e manutenibilità del codice. La struttura finale e alcuni componenti potrebbero subire modifiche.

## 📋 Struttura del Progetto

```
├── configs/                    # File di configurazione YAML
│   ├── augmentation.yaml      # Configurazione per l'augmentazione
│   └── hsnet_vanilla.yaml     # Configurazione per il training HSNet
├── data/                      # Directory per dataset e output
│   ├── datasets/             # Dataset originali
│   └── augmented/            # Dataset augmentati (output)
├── scripts/                   # Script eseguibili
│   └── run_augmentation.py   # Script principale per l'augmentazione
├── src/                      # Codice sorgente modulare
│   ├── augmentation/         # Moduli per l'augmentazione
│   │   ├── methods.py       # Metodi di augmentazione
│   │   └── sam_loader.py    # Caricamento modelli SAM
│   └── models/              # Modelli di segmentazione
│       ├── HSNet/
│       ├── PolypPVT/
│       └── SAM/
├── segment-anything-2/       # Repository SAM2 (clonato automaticamente)
└── requirements.txt          # Dipendenze Python
```

## 🚀 Setup Iniziale

### 1. Installazione

Eseguire lo script di setup automatico che:
- Crea l'ambiente virtuale Python 3.11
- Clona il repository SAM2
- Installa tutte le dipendenze
- Scarica i checkpoint di SAM1 e SAM2

```bash
bash setup.sh
```

### 2. Attivazione Ambiente

```bash
source venv_newSAMAug/bin/activate
```

## 📊 Utilizzo

### Augmentazione Dataset

**Script da avviare:** [`scripts/run_augmentation.py`](scripts/run_augmentation.py)

**Configurazione:** Modifica il file [`configs/augmentation.yaml`](configs/augmentation.yaml) per specificare:
- Dataset da processare (`datasets.folders`)
- Metodi di augmentazione (`augmentation.methods`)
- Versioni di SAM da utilizzare (`sam.versions`)
- Percorsi dei checkpoint (`paths.checkpoints_root`)

**Esecuzione:**

```bash
python scripts/run_augmentation.py --config configs/augmentation.yaml
```

**Output:** I dataset augmentati vengono salvati in `data/augmented/sam{1,2}/{metodo}/`

### Training Modelli

**Script da avviare:** [`src/models/HSNet/Train.py`](src/models/HSNet/Train.py)

**Configurazione:** Modifica il file [`configs/hsnet_vanilla.yaml`](configs/hsnet_vanilla.yaml) per specificare parametri di training.

**Esecuzione:**

⚠️ **NOTA TEMPORANEA:** Per il momento, il training deve essere avviato **dall'interno della cartella del modello HSNet** a causa dei path relativi non ancora aggiornati. Successivamente verranno modificate tutte le directory relative per poterlo avviare direttamente dalla home del progetto.

```bash
cd src/models/HSNet
python Train.py --config ../../../configs/hsnet_vanilla.yaml
```

## 🔧 Metodi di Augmentazione Disponibili

I metodi implementati in [`src/augmentation/methods.py`](src/augmentation/methods.py):

- `ourSAMAug` - Augmentazione SAM personalizzata
- `RG_segPrior` - Random Gaussian con prior di segmentazione
- `SV_segPrior` - Salt & Vinegar con prior di segmentazione
- `RG_logits` - Random Gaussian basato su logits
- `PCA_segPrior` - PCA con prior di segmentazione

## 📦 Dipendenze Principali

- **Python 3.11**
- **PyTorch** (installato con SAM2)
- **segment-anything** (SAM v1)
- **segment-anything-2** (SAM v2)
- **timm** - Per backbone dei modelli
- **opencv-python, scikit-image** - Elaborazione immagini
- **PyYAML** - Gestione configurazioni

## 📝 Note

- I checkpoint SAM vengono scaricati automaticamente in `checkpoints_sam/`
- Assicurarsi di avere GPU CUDA disponibile per il training e l'inferenza SAM
- La struttura dei dataset deve seguire il formato: `{dataset}/images/` e `{dataset}/masks/`

## 📂 Progetto Originale

https://github.com/LorisNanni/Exploring-SAM-Augmented-Ensembles

---

*Ultimo aggiornamento: 15 Dicembre 2025*
