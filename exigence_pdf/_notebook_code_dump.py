

# ============ CELL 0 (markdown) ============
# 🛡️ Détection de visages Réels vs Générés par IA (Deepfake)
### Pipeline MLOps complet — *Adaptive Fusion Detector* (EfficientNetV2 + analyse fréquentielle FFT)

Ce notebook fusionne deux travaux existants :

| Source | Ce qui a été conservé |
|---|---|
| `arabe_effnet-1792-fft-erasing-no-noise-adaptive` | Le **modèle** (double flux spatial + FFT avec porte adaptative), la recette d'entraînement et les **poids déjà entraînés** (`AdaptiveFusion_best_model.pth`) |
| `NLP_MLOPS_Project_v1` | La **démarche professionnelle** : environnement Colab, téléchargement Kaggle, EDA, visualisation des données, `summary` / graphe du modèle, callbacks, courbes, matrice de confusion, ROC, **Grad-CAM**, exemples qualitatifs |

> Le second notebook est écrit en Keras/TensorFlow, le premier en PyTorch. Tout a été **porté en PyTorch** (`torchinfo` remplace `model.summary()`, `torchview` remplace `plot_model`, Grad-CAM réécrit pour PyTorch).

---
## 🧭 Deux modes d'utilisation

| Mode | Réglage dans la cellule *Configuration* | Que se passe-t-il ? |
|---|---|---|
| **A — Inférence** *(par défaut)* | `train_model=False` | Aucun entraînement. Le notebook charge les poids placés dans **`/content/model_output`** et fait évaluation, Grad-CAM et prédictions. |
| **B — Entraînement** | `train_model=True` | Le notebook entraîne le modèle sur le dataset 140k, sauvegarde les poids + historique + configuration dans **`/content/model_output`**, puis enchaîne avec les mêmes étapes d'évaluation. |

Les deux modes partagent exactement la même suite d'évaluation : seul l'origine des poids change.

## ⚡ Démarrage rapide (sans entraîner)
1. Colab → *Exécution* → *Modifier le type d'exécution* → GPU (T4) recommandé (le CPU fonctionne pour de petites prédictions).
2. Mettre les poids dans `/content/model_output/AdaptiveFusion_best_model.pth` — **ou** laisser le notebook les télécharger (`weights_source="kaggle_kernel"`, voir section 7).
3. *Exécution → Tout exécuter*. Pour ne tester que le modèle sur vos images : `prepare_dataset=False`.

## 📚 Sommaire
1. Configuration
2. Environnement et dépendances
3. Données : téléchargement, EDA, visualisation
4. Pipeline de données (transformations, DataLoaders)
5. Architecture du modèle (summary, graphe, complexité)
6. Entraînement *(optionnel)*
7. Chargement du modèle entraîné
8. Évaluation quantitative
9. Interprétabilité : poids de fusion et Grad-CAM
10. Inférence : prédire sur ses propres images
11. Performances (latence, mémoire, MACs)
12. Artefacts MLOps (manifest, model card, sauvegarde)

## 📌 Résultats de référence (rapportés par le notebook source)
- Validation (mélange de 5 sources, 4000 images) : **Acc 99,12 %**, F1 0,9913, arrêt anticipé à l'époque 28.
- Test externe (Face-VAE vs GANsformer, 1054 images) : Acc 83,87 %, AUC 0,952, EER 0,087 → **bonne détection des vrais visages (99 %) mais 31 % de faux visages non détectés** : le modèle généralise moins bien à des générateurs jamais vus.
- Complexité : 31,8 M de paramètres, ≈ 3,0 G MACs.

# ============ CELL 1 (markdown) ============
---
# 1. Configuration

**C'est la seule cellule que vous avez normalement besoin de modifier.** Tout le reste du notebook lit cet objet `cfg`.

Points d'attention :
- `train_model` : `False` = utiliser les poids existants, `True` = entraîner.
- `weights_source` : comment obtenir les poids s'ils ne sont pas déjà dans `model_dir` (`local`, `kaggle_kernel`, `upload`, `drive`).
- `quick_demo=True` lance un mini-entraînement (500 images/classe, 2 époques) pour **valider le pipeline en quelques minutes** avant un vrai entraînement.
- Le lien Kaggle fourni dans le cahier des charges (`regnet2-1296-20-noise-...`) est un **autre notebook** (variante RegNet) que le notebook `effnet-1792` fourni. Le chargement des poids détecte automatiquement l'architecture (section 7), mais vérifiez que vous téléchargez bien la sortie du notebook dont vous voulez les poids.

# ============ CELL 2 (markdown) ============
# preparation envirennement kaggel et datasets

# ============ CELL 3 (code) ============
!pip install -q kaggle

# ============ CELL 4 (code) ============
!mkdir -p ~/.kaggle && echo KGAT_ebb9db2468b5a8c806c1f92d5f0492f5 > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token

# ============ CELL 5 (code) ============
!kaggle competitions list

# ============ CELL 6 (code) ============
# dataset 1
!kaggle datasets download -d xhlulu/140k-real-and-fake-faces \
    -p /content/datasets/140k-real-and-fake-faces --unzip

# ============ CELL 7 (code) ============
# dataset 2
!kaggle datasets download -d jessicali9530/celeba-dataset \
    -p /content/datasets/celeba --unzip

# ============ CELL 8 (code) ============
# dataset 3
!kaggle datasets download -d reubensuju/celeb-df-v2 \
    -p /content/datasets/celeb-df-v2 --unzip

# ============ CELL 9 (code) ============
# dataset 4
!kaggle datasets download -d xdxd003/ff-c23 \
    -p /content/datasets/faceforensics-c23 --unzip

# ============ CELL 10 (code) ============
# dataset 5
!kaggle datasets download -d arnaud58/flickrfaceshq-dataset-ffhq \
    -p /content/datasets/ffhq --unzip

# ============ CELL 11 (code) ============
# dataset 6
!kaggle datasets download -d jessicali9530/lfw-dataset \
    -p /content/datasets/lfw --unzip

# ============ CELL 12 (code) ============
# verification de import de datasets
import os

print(os.listdir("/content/datasets"))

# ============ CELL 13 (code) ============
# ============================================================
# 1. PROJECT CONFIGURATION  (the only cell you normally edit)
# ============================================================
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

IN_COLAB = "google.colab" in sys.modules
BASE_DIR = Path("/content") if Path("/content").exists() else Path.cwd()


@dataclass
class Config:
    # ---------- Execution mode ----------
    train_model: bool = False        # True -> run section 6 (training); False -> inference only
    prepare_dataset: bool = True     # Download/extract the 140k dataset (EDA, training, test evaluation)
    evaluate_on_test: bool = True    # Evaluate the loaded model on the dataset test split
    resume_training: bool = False    # Resume from model_dir/last_checkpoint.pt
    upload_own_images: bool = False  # Colab: open an upload dialog in section 10
    weights_source: str = "local"    # "local" | "kaggle_kernel" | "upload" | "drive"

    # ---------- Paths ----------
    model_dir: Path = BASE_DIR / "model_output"
    data_dir: Path = BASE_DIR / "dataset"
    zip_path: Path = BASE_DIR / "140k-real-and-fake-faces.zip"
    drive_model_dir: Path = Path("/content/drive/MyDrive/deepfake_project/model_output")
    sync_to_drive: bool = False      # Copy model_dir to Google Drive at the end

    # ---------- Remote resources ----------
    kaggle_dataset: str = "xhlulu/140k-real-and-fake-faces"
    kaggle_kernel: str = "securitysupport/regnet2-1296-20-noise-fft-erasing-adaptive-oct-29"
    weights_filename: str = "AdaptiveFusion_best_model.pth"

    # ---------- Model ----------
    backbone: str = "efficientnetv2_rw_s"     # 1792-d features
    backbone_candidates: Tuple[str, ...] = (  # tried automatically if the checkpoint does not match `backbone`
        "efficientnetv2_rw_s", "regnety_064", "regnety_040", "regnety_032",
        "convnext_tiny", "efficientnet_b3", "resnet50",
    )
    image_size: int = 224
    pretrained_backbone: bool = True          # ImageNet weights (training only)

    # ---------- Data subsets per class (None = full split) ----------
    train_per_class: Optional[int] = 10_000
    valid_per_class: Optional[int] = 2_000
    test_per_class: Optional[int] = 5_000

    # ---------- Training hyper-parameters (same recipe as the source notebook) ----------
    epochs: int = 30
    batch_size: int = 32
    patience: int = 5                # early stopping on validation F1
    weight_decay: float = 0.05
    lr_backbone: float = 1e-5
    lr_fusion: float = 1e-4          # frequency CNN, projections, fusion gate
    lr_classifier: float = 2e-4
    label_smoothing: float = 0.05    # targets become 0.05 / 0.95
    grad_clip: float = 1.0
    learn_residual_scale: bool = True  # the source notebook left `residual_scale` out of the optimizer
    use_amp: bool = True             # mixed precision (GPU only)
    num_workers: int = 2
    seed: int = 42
    quick_demo: bool = False         # 500 img/class, 2 epochs: validates the whole pipeline fast

    def __post_init__(self):
        if self.quick_demo:
            self.train_per_class, self.valid_per_class, self.test_per_class = 500, 200, 200
            self.epochs = 2
        self.model_dir = Path(self.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)


# ----- Edit here -----
cfg = Config(
    # train_model=True,                  # <- switch to training mode
    # quick_demo=True,                   # <- smoke test of the training pipeline
    # prepare_dataset=False,             # <- inference only, no 3.7 GB download
    # weights_source="kaggle_kernel",    # <- download the weights from the Kaggle notebook output
)

import pandas as pd
print(pd.Series({k: str(v) for k, v in cfg.__dict__.items()}, name="value").to_string())





# ============ CELL 14 (markdown) ============
---
# 2. Environnement et dépendances

# ============ CELL 15 (code) ============
%pip install -q timm torchinfo torchview graphviz thop kaggle

# ============ CELL 16 (code) ============
# ============================================================
# 2. IMPORTS, DETERMINISM, DEVICE
# ============================================================
import copy
import getpass
import hashlib
import json
import platform
import random
import shutil
import subprocess
import time
import zipfile
from dataclasses import asdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import Image as IPyImage
from IPython.display import Markdown, display
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold"})


def seed_everything(seed: int) -> None:
    """Make runs reproducible (data order, augmentation, weight init)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(cfg.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = bool(cfg.use_amp and device.type == "cuda")

print(f"Device       : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "  (no GPU: training would be very slow)"))
print(f"Mixed prec.  : {USE_AMP}")
print(f"Python       : {platform.python_version()} | torch {torch.__version__} | timm {timm.__version__}")
print(f"Model folder : {cfg.model_dir.resolve()}")

# ============ CELL 17 (markdown) ============
---
# 3. Données : téléchargement, EDA, visualisation

**Dataset** : [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) — 70 000 visages réels (Flickr-Faces-HQ, NVIDIA) et 70 000 visages synthétiques (StyleGAN), découpés en `train` (100 000), `valid` (20 000) et `test` (20 000), parfaitement équilibrés.

⚠️ **Convention des étiquettes.** Le modèle déjà entraîné utilise `real = 0` et `fake = 1`. Keras (`flow_from_directory`) et `torchvision.ImageFolder` classent les dossiers par ordre alphabétique (`fake = 0`, `real = 1`) : l'inverse. Ici le mapping est donc **explicite** (`LABEL2ID`) pour ne jamais inverser les prédictions.

**Économie de temps et d'espace.** Le ZIP (3,7 Go) est téléchargé une fois, mais seul le **sous-ensemble nécessaire** (`*_per_class` dans la configuration) est extrait, de façon déterministe.

🔐 **Identifiants Kaggle** : jamais dans le code. Le notebook cherche, dans l'ordre : un fichier existant, la variable d'environnement `KAGGLE_API_TOKEN`, le *Secret* Colab nommé `KAGGLE_API_TOKEN` (icône 🔑), sinon il demande le jeton en saisie masquée.

# ============ CELL 18 (code) ============
# ============================================================
# 3.1 DATASET CONSTANTS + KAGGLE HELPERS
# ============================================================
LABEL2ID = {"real": 0, "fake": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
SPLITS = ("train", "valid", "test")
IMG_EXT = (".jpg", ".jpeg", ".png")
ZIP_ROOT = "real_vs_fake/real-vs-fake"
DATA_ROOT = cfg.data_dir / ZIP_ROOT


def run_cli(cmd) -> None:
    """Run a shell command and stream its output; raise on failure."""
    print("$", " ".join(map(str, cmd)))
    subprocess.run([str(c) for c in cmd], check=True)


def setup_kaggle_credentials() -> None:
    """Configure Kaggle credentials without ever hard-coding a secret."""
    kdir = Path.home() / ".kaggle"
    if ((kdir / "kaggle.json").exists() or (kdir / "access_token").exists()
            or os.environ.get("KAGGLE_API_TOKEN")
            or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))):
        print("Kaggle credentials found.")
        return
    token = None
    try:
        from google.colab import userdata
        token = userdata.get("KAGGLE_API_TOKEN")
    except Exception:
        pass
    if not token:
        token = getpass.getpass("Paste your Kaggle API token (input is hidden): ").strip()
    kdir.mkdir(exist_ok=True)
    (kdir / "access_token").write_text(token)
    (kdir / "access_token").chmod(0o600)
    os.environ["KAGGLE_API_TOKEN"] = token
    print("Kaggle credentials configured for this session.")


def download_dataset() -> None:
    if cfg.zip_path.exists():
        print(f"ZIP already present: {cfg.zip_path}")
        return
    setup_kaggle_credentials()
    cfg.zip_path.parent.mkdir(parents=True, exist_ok=True)
    run_cli(["kaggle", "datasets", "download", "-d", cfg.kaggle_dataset, "-p", cfg.zip_path.parent])
    downloaded = cfg.zip_path.parent / (cfg.kaggle_dataset.split("/")[-1] + ".zip")
    if downloaded != cfg.zip_path and downloaded.exists():
        downloaded.rename(cfg.zip_path)


def index_zip(zip_path: Path) -> dict:
    """Return {(split, class): [member names]} without extracting anything."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.startswith(ZIP_ROOT + "/") and n.lower().endswith(IMG_EXT)]
    index = {}
    for name in names:
        _, _, split, cls, _ = name.split("/", 4)
        index.setdefault((split, cls), []).append(name)
    return index


def extract_subset(zip_path: Path, dest: Path, index: dict, limits: dict, seed: int) -> None:
    """Extract at most `limits[split]` images per class (deterministic, idempotent)."""
    rng = random.Random(seed)
    selected = []
    for split in SPLITS:
        for cls in LABEL2ID:
            members = sorted(index.get((split, cls), []))
            limit = limits.get(split)
            if limit is not None and len(members) > limit:
                members = rng.sample(members, limit)
            selected += members
    todo = [n for n in selected if not (dest / n).exists()]
    print(f"Selected {len(selected):,} images | {len(todo):,} still to extract")
    with zipfile.ZipFile(zip_path) as zf:
        for name in tqdm(todo, desc="Extracting"):
            zf.extract(name, dest)

# ============ CELL 19 (code) ============
# ============================================================
# 3.2 PREPARE THE DATASET (download + subset extraction)
# ============================================================
HAS_DATA = False
full_counts = None

if cfg.prepare_dataset:
    if not cfg.zip_path.exists() and not DATA_ROOT.exists():
        download_dataset()

    limits = {
        "train": cfg.train_per_class if cfg.train_model else 500,  # inference: a small train sample for EDA only
        "valid": cfg.valid_per_class if cfg.train_model else 0,
        "test": cfg.test_per_class,
    }
    if cfg.zip_path.exists():
        zip_index = index_zip(cfg.zip_path)
        full_counts = (pd.Series({k: len(v) for k, v in zip_index.items()})
                       .rename_axis(["split", "class"]).unstack().loc[list(SPLITS)])
        print("Official dataset composition (images per split and class):")
        display(full_counts)
        extract_subset(cfg.zip_path, cfg.data_dir, zip_index, limits, cfg.seed)
    HAS_DATA = DATA_ROOT.exists()
    print(f"Working data root: {DATA_ROOT}")
else:
    print("prepare_dataset=False -> dataset steps (EDA, loaders, evaluation) are skipped.")

# ============ CELL 20 (code) ============
# ============================================================
# 3.3 IMAGE INDEX (DataFrame with split / class / label / path)
# ============================================================
def build_dataframe(root: Path) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for cls, label in LABEL2ID.items():
            folder = root / split / cls
            if folder.exists():
                rows += [{"split": split, "class": cls, "label": label, "path": str(p)}
                         for p in sorted(folder.iterdir()) if p.suffix.lower() in IMG_EXT]
    return pd.DataFrame(rows, columns=["split", "class", "label", "path"])


images_df = build_dataframe(DATA_ROOT) if HAS_DATA else pd.DataFrame(columns=["split", "class", "label", "path"])
HAS_DATA = len(images_df) > 0

if HAS_DATA:
    display(images_df.sample(5, random_state=cfg.seed))
    print("\nImages available in the working subset:")
    display(images_df.groupby(["split", "class"]).size().unstack())
else:
    print("No images indexed.")

# ============ CELL 21 (code) ============
# # ============================================================
# # 3.3b MULTI-SOURCE MIX (replaces train/valid of images_df; test stays 140k)
# # ============================================================
# def resolve_source_dir(src: dict) -> Path:
#     local = cfg.data_dir / "extra" / src["kaggle"].split("/")[-1]
#     for base in (Path("/kaggle/input/datasets") / src["kaggle"], local):   # Kaggle mount first
#         if (base / src["subdir"]).exists():
#             return base / src["subdir"]
#     setup_kaggle_credentials()
#     local.mkdir(parents=True, exist_ok=True)
#     run_cli(["kaggle", "datasets", "download", "-d", src["kaggle"], "-p", local, "--unzip"])
#     return local / src["subdir"]


# def infer_label(path: Path):
#     name = path.parent.name.lower()
#     if "real" in name or "youtube" in name:
#         return 0
#     if any(k in name for k in ("fake", "synth", "gan", "diffusion")):
#         return 1
#     return None


# if cfg.use_all_datasets and HAS_DATA:
#     rng = random.Random(cfg.seed)
#     rows = []

#     # source 0: 140k train images (test split is kept untouched)
#     base = images_df[images_df["split"] == "train"]
#     base = base.sample(min(len(base), cfg.max_per_source), random_state=cfg.seed)
#     rows += [{"source": "140k", "label": int(l), "path": p} for l, p in zip(base["label"], base["path"])]

#     for src in cfg.extra_sources:
#         root = resolve_source_dir(src)
#         files = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT]
#         files = rng.sample(files, min(len(files), cfg.max_per_source))
#         for p in files:
#             label = src["label"] if src["label"] is not None else infer_label(p)
#             if label is not None:
#                 rows.append({"source": src["name"], "label": label, "path": str(p)})
#         print(f"{src['name']:>14}: {len(files):,} images from {root}")

#     mix = pd.DataFrame(rows)
#     n = int(mix["label"].value_counts().min())                      # strict 50/50 balance
#     mix = pd.concat([g.sample(n, random_state=cfg.seed) for _, g in mix.groupby("label")])
#     n_val = int(n * cfg.valid_fraction)
#     mix["split"] = "train"
#     for _, g in mix.groupby("label"):
#         mix.loc[g.sample(n_val, random_state=cfg.seed).index, "split"] = "valid"
#     mix["class"] = mix["label"].map(ID2LABEL)

#     test_rows = images_df[images_df["split"] == "test"]
#     images_df = pd.concat([mix, test_rows], ignore_index=True)
#     train_loader = make_loader("train", train_transform, shuffle=True)   # rebuild loaders on the new mix
#     val_loader = make_loader("valid", eval_transform, shuffle=False)
#     display(mix.groupby(["source", "class"]).size().unstack(fill_value=0))

# ============ CELL 22 (markdown) ============
## 3.4 Analyse exploratoire (EDA)
Trois contrôles : équilibre des classes, aperçu visuel, et **spectre de Fourier moyen** réel vs faux. Ce dernier justifie le second flux du modèle : les générateurs GAN laissent des artefacts périodiques dans les hautes fréquences, invisibles à l'œil mais visibles dans le spectre.

# ============ CELL 23 (code) ============
# ============================================================
# 3.4a CLASS BALANCE + SAMPLE GRID
# ============================================================
def plot_class_balance(df: pd.DataFrame) -> None:
    counts = df.groupby(["split", "class"]).size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=counts, x="split", y="count", hue="class", palette={"real": "#2b5c8f", "fake": "#d95f02"}, ax=ax)
    for bar in ax.patches:
        if bar.get_height() > 0:
            ax.annotate(f"{int(bar.get_height()):,}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")
    ax.set_title("Class balance per split (working subset)")
    ax.set_xlabel("")
    ax.set_ylabel("Images")
    plt.tight_layout()
    plt.show()


def plot_samples(df: pd.DataFrame, n_per_class: int = 6, seed: int = 0) -> None:
    fig, axes = plt.subplots(2, n_per_class, figsize=(2.2 * n_per_class, 5))
    for r, cls in enumerate(LABEL2ID):
        subset = df[df["class"] == cls].sample(n_per_class, random_state=seed)
        for c, path in enumerate(subset["path"]):
            axes[r, c].imshow(Image.open(path).convert("RGB"))
            axes[r, c].axis("off")
            axes[r, c].set_title(cls.upper(), color="#2b5c8f" if cls == "real" else "#d95f02", fontsize=10)
    plt.suptitle("Random samples (top: real, bottom: fake)", fontweight="bold")
    plt.tight_layout()
    plt.show()


if HAS_DATA:
    plot_class_balance(images_df)
    plot_samples(images_df, n_per_class=6)

# ============ CELL 24 (code) ============
# ============================================================
# 3.4b IMAGE PROPERTIES + MEAN FOURIER SPECTRUM (real vs fake)
# ============================================================
def mean_log_spectrum(paths, size: int = 224) -> np.ndarray:
    acc = np.zeros((size, size), dtype=np.float64)
    for p in paths:
        gray = np.asarray(Image.open(p).convert("L").resize((size, size)), dtype=np.float32) / 255.0
        acc += np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    return acc / max(len(paths), 1)


if HAS_DATA:
    sample = images_df.sample(min(300, len(images_df)), random_state=cfg.seed)
    sizes = pd.Series([Image.open(p).size for p in sample["path"]]).value_counts()
    print("Image sizes (W, H) in a 300-image sample:")
    print(sizes.to_string(), "\n")

    n = 200
    real_paths = images_df[images_df["class"] == "real"]["path"].head(n).tolist()
    fake_paths = images_df[images_df["class"] == "fake"]["path"].head(n).tolist()
    spec_real, spec_fake = mean_log_spectrum(real_paths), mean_log_spectrum(fake_paths)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, data, title, cmap in [(axes[0], spec_real, "Mean spectrum - REAL", "magma"),
                                  (axes[1], spec_fake, "Mean spectrum - FAKE", "magma"),
                                  (axes[2], spec_fake - spec_real, "Difference (FAKE - REAL)", "coolwarm")]:
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle("log(1+|FFT|) averaged over images: the frequency stream of the model exploits these differences", fontsize=11)
    plt.tight_layout()
    plt.show()

# ============ CELL 25 (markdown) ============
---
# 4. Pipeline de données

- **Entraînement** : redimensionnement 256 → *RandomResizedCrop* 224, flip horizontal, variations de couleur / niveaux de gris, puis `RandomErasing` (effacement aléatoire de petites zones, d'où « erasing » dans le nom du notebook source).
- **Validation / test** : redimensionnement 224 et normalisation ImageNet uniquement, **aucune augmentation**, pour une évaluation stable.
- Une image illisible est remplacée par une image noire pour ne jamais interrompre un entraînement long.

# ============ CELL 26 (code) ============
# ============================================================
# 4.1 TRANSFORMS, DATASET, DATALOADERS
# ============================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
S = cfg.image_size

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(S, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.1), transforms.RandomGrayscale(p=0.1)], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
])

eval_transform = transforms.Compose([
    transforms.Resize((S, S)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class FaceDataset(Dataset):
    """Binary dataset (real=0 / fake=1) built from a DataFrame with `path` and `label` columns."""

    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:                       # corrupted file -> black image, training keeps going
            img = Image.new("RGB", (S, S))
        return self.transform(img), self.labels[idx]


def make_loader(split: str, transform, shuffle: bool) -> Optional[DataLoader]:
    subset = images_df[images_df["split"] == split]
    if subset.empty:
        return None
    return DataLoader(FaceDataset(subset, transform), batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))


train_loader = make_loader("train", train_transform, shuffle=True)
val_loader = make_loader("valid", eval_transform, shuffle=False)
test_loader = make_loader("test", eval_transform, shuffle=False)   # shuffle=False keeps predictions aligned with the DataFrame

for name, loader in [("train", train_loader), ("valid", val_loader), ("test", test_loader)]:
    print(f"{name:>5}: " + (f"{len(loader.dataset):>6,} images | {len(loader):>4} batches" if loader else "not available"))

# ============ CELL 27 (code) ============
# ============================================================
# 4.2 INSPECT A BATCH (shapes, value range, augmentation preview)
# ============================================================
def denormalize(t: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t.cpu() * std + mean).clamp(0, 1)


if train_loader is not None:
    images, labels = next(iter(train_loader))
    print(f"Batch images : {tuple(images.shape)}  dtype={images.dtype}")
    print(f"Batch labels : {tuple(labels.shape)}  dtype={labels.dtype}  (0=real, 1=fake)")
    print(f"Value range  : [{images.min():.2f}, {images.max():.2f}]  (ImageNet-normalized)")
    print(f"Class mapping: {LABEL2ID}")

    n_show = min(16, len(images))
    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n_show:
            ax.imshow(denormalize(images[i]).permute(1, 2, 0).numpy())
            ax.set_title(ID2LABEL[int(labels[i])].upper(), fontsize=10,
                         color="#2b5c8f" if labels[i] == 0 else "#d95f02")
    plt.suptitle("Augmented training batch (what the network actually sees)", fontweight="bold")
    plt.tight_layout()
    plt.show()
else:
    print("No train loader available (prepare_dataset=False or no data).")

# ============ CELL 28 (markdown) ============
---
# 5. Architecture du modèle

**Adaptive Fusion Detector** : deux flux complémentaires dont la contribution est pondérée **image par image** par une porte apprise.

```
                    ┌─► Flux spatial     : EfficientNetV2-RW-S (ImageNet) ──► Linear+LN+GELU ─► p_s ─┐
 Image 224×224×3 ───┤                                                                                  ├─► Porte adaptative ─► (w_s, w_f)
                    └─► Flux fréquentiel : FFT2D ─► log(1+|F|) ─► petit CNN ─► Linear+LN+GELU ─► p_f ─┘            │
                                                                                                                    ▼
                         z = p_s + α · (w_s·p_s + w_f·p_f)   (α = residual_scale, appris)  ─►  Classifieur  ─►  logit  ─►  σ  ─►  P(fake)
```

Règle de décision : `logit > 0` ⇔ `P(fake) ≥ 0,5` ⇒ **FAKE** (étiquette 1), sinon **REAL** (étiquette 0).

Notes d'implémentation :
- Les noms de modules (`spatial_net`, `freq_net`, `spatial_projection`, `frequency_projection`, `fusion_gate`, `classifier`, `residual_scale`) sont **identiques** au notebook source : les poids existants se chargent directement.
- La classe s'appelait `ConvNeXtAdaptiveFusionDetector` alors qu'elle utilise EfficientNetV2 : renommée `AdaptiveFusionDetector` (sans effet sur les poids).
- La FFT est calculée en **float32** même en précision mixte (la FFT en float16 n'accepte que des tailles puissances de 2, or 224 n'en est pas une).

# ============ CELL 29 (code) ============
# ============================================================
# 5.1 MODEL DEFINITION
# ============================================================
class AdaptiveFusionDetector(nn.Module):
    """Dual-stream (spatial + frequency) deepfake detector with an adaptive fusion gate."""

    def __init__(self, backbone: str = "efficientnetv2_rw_s", pretrained: bool = True,
                 projection_dim: Optional[int] = None):
        super().__init__()
        self.backbone_name = backbone

        # 1. Spatial stream
        self.spatial_net = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.spatial_features_dim = self.spatial_net.num_features

        # 2. Frequency stream (FFT log-magnitude -> small CNN)
        self.freq_net = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
        )
        self.freq_features_dim = 64 * 7 * 7
        self.projection_dim = projection_dim or self.spatial_features_dim

        # 3. Projections into a shared space
        self.spatial_projection = nn.Sequential(
            nn.Linear(self.spatial_features_dim, self.projection_dim), nn.LayerNorm(self.projection_dim), nn.GELU())
        self.frequency_projection = nn.Sequential(
            nn.Linear(self.freq_features_dim, self.projection_dim), nn.LayerNorm(self.projection_dim), nn.GELU())

        # 4. Adaptive gate (starts with equal weights 0.5 / 0.5)
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(self.projection_dim * 2),
            nn.Linear(self.projection_dim * 2, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 2),
        )
        nn.init.zeros_(self.fusion_gate[-1].weight)
        nn.init.zeros_(self.fusion_gate[-1].bias)
        self.residual_scale = nn.Parameter(torch.tensor(0.5))

        # 5. Classifier (one logit: > 0 means FAKE)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.projection_dim),
            nn.Linear(self.projection_dim, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    @staticmethod
    def frequency_input(x: torch.Tensor) -> torch.Tensor:
        """log(1 + |FFT2(x)|), centered. Always float32 (cuFFT fp16 needs power-of-2 sizes)."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.fftshift(torch.fft.fft2(x.float(), dim=(-2, -1)), dim=(-2, -1))
            return torch.log1p(torch.abs(spectrum))

    def forward_detailed(self, x: torch.Tensor):
        """Return (logits [B,1], (w_spatial [B,1], w_frequency [B,1]), spatial feature map [B,C,h,w])."""
        fmap = self.spatial_net.forward_features(x)
        spatial_feat = self.spatial_net.forward_head(fmap)
        p_s = self.spatial_projection(spatial_feat)

        p_f = self.frequency_projection(self.freq_net(self.frequency_input(x)))

        gate = torch.softmax(self.fusion_gate(torch.cat([p_s, p_f], dim=1)), dim=1)
        w_s, w_f = gate[:, 0:1], gate[:, 1:2]

        fused = p_s + self.residual_scale * (w_s * p_s + w_f * p_f)
        return self.classifier(fused), (w_s, w_f), fmap

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        logits, weights, _ = self.forward_detailed(x)
        return (logits, weights) if return_weights else logits

# ============ CELL 30 (code) ============
# ============================================================
# 5.2 ARCHITECTURE SANITY TEST + PARAMETER BREAKDOWN
# (untrained instance used only for inspection: no ImageNet download needed)
# ============================================================
model_viz = AdaptiveFusionDetector(cfg.backbone, pretrained=False).eval()

with torch.no_grad():
    dummy = torch.randn(2, 3, S, S)
    logits, (ws, wf) = model_viz(dummy, return_weights=True)

assert logits.shape == (2, 1), "unexpected output shape"
assert torch.allclose(ws + wf, torch.ones_like(ws), atol=1e-6), "gate weights must sum to 1"
print("Output shape        :", tuple(logits.shape))
print("Spatial features    :", model_viz.spatial_features_dim)
print("Frequency features  :", model_viz.freq_features_dim)
print("Projection dim      :", model_viz.projection_dim)
print("Initial gate weights:", ws.flatten().tolist(), wf.flatten().tolist(), "(equal at init)")
print("Architecture test passed.\n")

breakdown = pd.DataFrame(
    [(n, sum(p.numel() for p in m.parameters())) for n, m in model_viz.named_children()]
    + [("residual_scale", model_viz.residual_scale.numel())],
    columns=["component", "parameters"])
breakdown["share_%"] = (100 * breakdown["parameters"] / breakdown["parameters"].sum()).round(2)
breakdown["parameters"] = breakdown["parameters"].map("{:,}".format)
display(breakdown)

# ============ CELL 31 (code) ============
# ============================================================
# 5.3 MODEL SUMMARY (equivalent of Keras model.summary())
# ============================================================
from torchinfo import summary

_ = summary(
    model_viz, input_size=(1, 3, S, S), depth=2, device="cpu",
    col_names=("input_size", "output_size", "num_params", "trainable"),
    row_settings=("var_names",),
)

# ============ CELL 32 (code) ============
# ============================================================
# 5.4 ARCHITECTURE DIAGRAMS
#  (a) hand-drawn block diagram (always works)
#  (b) torchview computational graph (equivalent of tf.keras.utils.plot_model)
# ============================================================
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def plot_architecture_diagram(save_path: Optional[Path] = None) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 6.2))
    ax.set_xlim(0, 15.6)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                    fc=color, ec="#333333", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.3, color="#333333"))

    blue, orange, green, purple, grey = "#cfe2f3", "#fce5cd", "#d9ead3", "#d9d2e9", "#eeeeee"
    box(0.2, 2.35, 1.7, 1.0, "Image\n3 × 224 × 224", grey)

    box(2.3, 4.2, 3.4, 1.0, f"Spatial stream\n{cfg.backbone}\n(features {model_viz.spatial_features_dim}-d)", blue)
    box(6.1, 4.2, 2.0, 1.0, "Linear + LN + GELU\np_s", blue)
    box(2.3, 0.9, 1.6, 1.0, "FFT 2D + shift\nlog(1+|F|)", orange)
    box(4.1, 0.9, 1.8, 1.0, "CNN freq.\n2×(Conv-BN-GELU)\npool 7×7", orange)
    box(6.1, 0.9, 2.0, 1.0, "Linear + LN + GELU\np_f", orange)

    box(8.7, 2.1, 2.3, 1.5, "Adaptive gate\nsoftmax → (w_s, w_f)\nz = p_s + α·(w_s·p_s + w_f·p_f)", green)
    box(11.4, 2.35, 1.8, 1.0, "Classifier\nLN-Linear-GELU\nDropout-Linear", purple)
    box(13.6, 2.35, 1.8, 1.0, "logit → σ\nP(fake)\nlogit>0 ⇒ FAKE", grey)

    arrow(1.9, 3.1, 2.3, 4.5); arrow(1.9, 2.6, 2.3, 1.5)
    arrow(5.7, 4.7, 6.1, 4.7); arrow(3.9, 1.4, 4.1, 1.4); arrow(5.9, 1.4, 6.1, 1.4)
    arrow(8.1, 4.5, 8.7, 3.4); arrow(8.1, 1.5, 8.7, 2.3)
    arrow(11.0, 2.85, 11.4, 2.85); arrow(13.2, 2.85, 13.6, 2.85)
    ax.set_title("Adaptive Fusion Detector - architecture", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


plot_architecture_diagram(cfg.model_dir / "architecture_diagram.png")

try:
    from torchview import draw_graph
    graph = draw_graph(model_viz, input_size=(1, 3, S, S), depth=2, device="cpu",
                       graph_name="AdaptiveFusionDetector", expand_nested=False)
    png = graph.visual_graph.render(str(cfg.model_dir / "model_graph"), format="png", cleanup=True)
    display(IPyImage(png))
except Exception as exc:  # graphviz binary missing, torchview incompatibility, ...
    print(f"torchview graph skipped ({type(exc).__name__}: {exc}). The block diagram above remains valid.")

# ============ CELL 33 (markdown) ============
---
# 6. Entraînement *(optionnel — `cfg.train_model=True`)*

Recette reprise du notebook source :
- **AdamW** avec taux d'apprentissage **différenciés** (`1e-5` pour le backbone pré-entraîné, `1e-4` pour le flux fréquentiel / projections / porte, `2e-4` pour le classifieur), `weight_decay=0.05`.
- **Cosine annealing** sur `epochs`, **label smoothing** (cibles 0,05 / 0,95), **gradient clipping** à 1,0.
- **Arrêt anticipé** sur le **F1 de validation** (`patience=5`), meilleurs poids sauvegardés.

Ajouts MLOps :
- **Précision mixte (AMP)** sur GPU : entraînement ≈ 1,5–2× plus rapide.
- **`last_checkpoint.pt`** à chaque époque (modèle + optimiseur + scheduler + historique) → reprise possible après une déconnexion Colab (`resume_training=True`).
- **`training_history.csv`**, **`model_config.json`** et courbes sauvegardés dans `model_dir`.
- `residual_scale` est maintenant **réellement appris** (`learn_residual_scale=True`). Dans le notebook source ce paramètre n'était dans aucun groupe de l'optimiseur et restait figé à 0,5 ; mettez `False` pour reproduire exactement l'ancienne recette.

⏱️ Ordre de grandeur sur GPU T4 avec 10 000 images/classe : quelques minutes par époque. Faites d'abord un essai avec `quick_demo=True`.

# ============ CELL 34 (code) ============
# ============================================================
# 6.1 METRICS, EVALUATION LOOP, HISTORY PLOTS (shared by sections 6 and 8)
# ============================================================
def calculate_eer(y_true, y_score) -> float:
    """Equal Error Rate: operating point where false-positive rate == false-negative rate."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["eer"] = calculate_eer(y_true, y_prob)
    return metrics


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, criterion=None, desc: str = "Inference", amp: bool = False) -> dict:
    """Run the model on a loader and return labels, logits, probabilities, predictions and fusion weights."""
    model.eval()
    logits_all, labels_all, ws_all, wf_all = [], [], [], []
    loss_sum = 0.0
    for images, y in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits, (w_s, w_f) = model(images, return_weights=True)
        logits = logits.float()
        if criterion is not None:
            loss_sum += criterion(logits, y.float().unsqueeze(1).to(device)).item() * images.size(0)
        logits_all.append(logits.squeeze(1).cpu())
        labels_all.append(y)
        ws_all.append(w_s.squeeze(1).float().cpu())
        wf_all.append(w_f.squeeze(1).float().cpu())
    logits_t = torch.cat(logits_all)
    y_true = torch.cat(labels_all).numpy().astype(int)
    return {
        "y_true": y_true,
        "logits": logits_t.numpy(),
        "y_prob": torch.sigmoid(logits_t).numpy(),
        "y_pred": (logits_t > 0).numpy().astype(int),
        "w_spatial": torch.cat(ws_all).numpy(),
        "w_frequency": torch.cat(wf_all).numpy(),
        "loss": loss_sum / len(y_true) if criterion is not None else float("nan"),
    }


def plot_training_history(history: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """2x2 grid: loss, accuracy, F1, balanced accuracy (train vs validation)."""
    panels = [("loss", "val_loss", "Loss"), ("acc", "val_acc", "Accuracy"),
              ("f1", "val_f1", "F1 score"), ("bal_acc", "val_bal_acc", "Balanced accuracy")]
    best_epoch = int(history.loc[history["val_f1"].idxmax(), "epoch"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (tr, va, title) in zip(axes.flat, panels):
        ax.plot(history["epoch"], history[tr], "o-", label="Train")
        ax.plot(history["epoch"], history[va], "s-", label="Validation")
        ax.axvline(best_epoch, color="gray", ls=":", label=f"Best epoch ({best_epoch})")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.suptitle("Training history", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

# ============ CELL 35 (code) ============
# ============================================================
# 6.2 TRAINING FUNCTION
# ============================================================
BEST_WEIGHTS_PATH = cfg.model_dir / cfg.weights_filename
LAST_CKPT_PATH = cfg.model_dir / "last_checkpoint.pt"
HISTORY_PATH = cfg.model_dir / "training_history.csv"


def build_optimizer(model: AdaptiveFusionDetector) -> optim.Optimizer:
    groups = [
        {"params": model.spatial_net.parameters(), "lr": cfg.lr_backbone},
        {"params": model.freq_net.parameters(), "lr": cfg.lr_fusion},
        {"params": model.spatial_projection.parameters(), "lr": cfg.lr_fusion},
        {"params": model.frequency_projection.parameters(), "lr": cfg.lr_fusion},
        {"params": model.fusion_gate.parameters(), "lr": cfg.lr_fusion},
        {"params": model.classifier.parameters(), "lr": cfg.lr_classifier},
    ]
    if cfg.learn_residual_scale:
        groups.append({"params": [model.residual_scale], "lr": cfg.lr_fusion})
    return optim.AdamW(groups, weight_decay=cfg.weight_decay)


def save_model_config(model: AdaptiveFusionDetector, metrics: Optional[dict] = None) -> Path:
    """Write the metadata needed to rebuild the model without any other file."""
    meta = {
        "architecture": "AdaptiveFusionDetector",
        "backbone": model.backbone_name,
        "spatial_features_dim": model.spatial_features_dim,
        "projection_dim": model.projection_dim,
        "image_size": S,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "label2id": LABEL2ID,
        "decision_rule": "logit > 0 -> fake (label 1)",
        "weights_file": cfg.weights_filename,
        "best_metrics": metrics or {},
        "torch_version": torch.__version__,
        "timm_version": timm.__version__,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = cfg.model_dir / "model_config.json"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return path


def fit(model: AdaptiveFusionDetector, train_loader: DataLoader, val_loader: DataLoader) -> pd.DataFrame:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=USE_AMP)

    history, best_f1, bad_epochs, start_epoch = [], -1.0, 0, 0
    if cfg.resume_training and LAST_CKPT_PATH.exists():
        ckpt = torch.load(LAST_CKPT_PATH, map_location=device, weights_only=False)  # our own file
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        history, best_f1, bad_epochs, start_epoch = ckpt["history"], ckpt["best_f1"], ckpt["bad_epochs"], ckpt["epoch"]
        print(f"Resumed from epoch {start_epoch} (best val F1 so far: {best_f1:.4f})")

    print(f"Training for up to {cfg.epochs} epochs | batch {cfg.batch_size} | AMP={USE_AMP} | early-stop patience={cfg.patience}")
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        model.train()
        loss_sum, y_true_ep, y_pred_ep = 0.0, [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}", leave=False)
        for images, y in pbar:
            images = images.to(device, non_blocking=True)
            y = y.float().unsqueeze(1).to(device)
            target = y * (1 - 2 * cfg.label_smoothing) + cfg.label_smoothing      # 0 -> 0.05, 1 -> 0.95

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_AMP):
                logits = model(images)
            loss = criterion(logits.float(), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item() * images.size(0)
            y_true_ep.append(y.detach().cpu().numpy().ravel())
            y_pred_ep.append((logits.detach() > 0).float().cpu().numpy().ravel())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        y_true_ep, y_pred_ep = np.concatenate(y_true_ep).astype(int), np.concatenate(y_pred_ep).astype(int)
        train_m = compute_metrics(y_true_ep, y_pred_ep)
        train_loss = loss_sum / len(y_true_ep)

        val = predict_loader(model, val_loader, criterion=criterion, desc="Validation", amp=USE_AMP)
        val_m = compute_metrics(val["y_true"], val["y_pred"], val["y_prob"])

        history.append({
            "epoch": epoch + 1, "loss": train_loss, "val_loss": val["loss"],
            "acc": train_m["accuracy"], "val_acc": val_m["accuracy"],
            "f1": train_m["f1"], "val_f1": val_m["f1"],
            "bal_acc": train_m["balanced_accuracy"], "val_bal_acc": val_m["balanced_accuracy"],
            "val_precision": val_m["precision"], "val_recall": val_m["recall"],
            "val_auc": val_m.get("auc", float("nan")), "val_eer": val_m.get("eer", float("nan")),
            "lr_backbone": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0,
        })

        improved = val_m["f1"] > best_f1
        if improved:
            best_f1, bad_epochs = val_m["f1"], 0
            torch.save(model.state_dict(), BEST_WEIGHTS_PATH)       # raw state_dict: same format as the source notebook
            save_model_config(model, {"val_" + k: v for k, v in val_m.items()})
        else:
            bad_epochs += 1

        scheduler.step()
        pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "history": history, "best_f1": best_f1,
                    "bad_epochs": bad_epochs, "epoch": epoch + 1}, LAST_CKPT_PATH)

        h = history[-1]
        print(f"Epoch {epoch + 1:02d}/{cfg.epochs} | loss {h['loss']:.4f}/{h['val_loss']:.4f} | "
              f"acc {h['acc']:.2%}/{h['val_acc']:.2%} | F1 {h['f1']:.4f}/{h['val_f1']:.4f} | "
              f"{h['seconds'] / 60:.1f} min {'  * best, weights saved' if improved else f'  (no gain {bad_epochs}/{cfg.patience})'}")
        if bad_epochs >= cfg.patience:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    model.load_state_dict(torch.load(BEST_WEIGHTS_PATH, map_location=device, weights_only=True))
    print(f"Best validation F1 = {best_f1:.4f} -> weights restored from {BEST_WEIGHTS_PATH.name}")
    return pd.DataFrame(history)

# ============ CELL 36 (code) ============
# ============================================================
# 6.3 RUN TRAINING (only if cfg.train_model=True)
# ============================================================
history_df = None
if cfg.train_model:
    assert train_loader is not None and val_loader is not None, (
        "Training needs the dataset: set prepare_dataset=True (and valid_per_class > 0).")
    seed_everything(cfg.seed)
    trained_model = AdaptiveFusionDetector(cfg.backbone, pretrained=cfg.pretrained_backbone).to(device)
    history_df = fit(trained_model, train_loader, val_loader)
    plot_training_history(history_df, cfg.model_dir / "training_curves.png")
    display(history_df.tail(3).round(4))
    del trained_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
else:
    print("Training skipped (cfg.train_model=False). Section 7 will load existing weights from:", cfg.model_dir)

# ============ CELL 37 (markdown) ============
---
# 7. Chargement du modèle entraîné

Cette section est **commune aux deux modes** : après un entraînement elle recharge les meilleurs poids sauvegardés, sinon elle les récupère selon `cfg.weights_source` :

| `weights_source` | Action |
|---|---|
| `"local"` | Utilise les fichiers déjà présents dans `/content/model_output` (glisser-déposer dans le panneau de fichiers Colab). |
| `"kaggle_kernel"` | Télécharge la **sortie** du notebook Kaggle `cfg.kaggle_kernel` (`kaggle kernels output`). |
| `"upload"` | Ouvre une boîte de dialogue d'upload Colab. |
| `"drive"` | Copie `cfg.drive_model_dir` (Google Drive) vers `model_dir`. |

Le fichier attendu est `AdaptiveFusion_best_model.pth` (un `state_dict` PyTorch). À défaut, le `.pth`/`.pt` le plus récent du dossier est utilisé.

**Détection automatique de l'architecture.** La dimension des projections est lue dans les poids ; `model_config.json` (créé à l'entraînement) est utilisé s'il existe ; sinon le notebook essaie `cfg.backbone` puis `cfg.backbone_candidates` jusqu'à ce que le chargement **strict** réussisse. Si aucun ne convient, une erreur explicite indique l'écart.

🔒 Les poids sont chargés avec `weights_only=True` : un fichier téléchargé ne peut pas exécuter de code arbitraire.

# ============ CELL 38 (code) ============
# extraire le vrai .pth du ZIP.
import zipfile, shutil, os
outer = "/content/model_output/AdaptiveFusion_best_model.pth"
zip_copy = "/content/kaggle_output.zip"
shutil.move(outer, zip_copy)                       # the "fake" .pth is really a ZIP

with zipfile.ZipFile(zip_copy) as z:
    z.extract("dest/AdaptiveFusion_best_model.pth", "/content/tmp_extract")
shutil.move("/content/tmp_extract/dest/AdaptiveFusion_best_model.pth", outer)

p = outer
print(os.path.getsize(p) / 1e6, "MB |", open(p, "rb").read(2))

# ============ CELL 39 (code) ============
# ============================================================
# 7.1 LOCATE / ACQUIRE THE WEIGHTS
# ============================================================
def find_checkpoint(model_dir: Path) -> Optional[Path]:
    exact = sorted(model_dir.rglob(cfg.weights_filename))
    if exact:
        return exact[0]
    candidates = [p for p in model_dir.rglob("*")
                  if p.suffix in {".pth", ".pt"} and p.name != "last_checkpoint.pt"]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def acquire_weights() -> None:
    src = cfg.weights_source
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    if src == "local":
        return
    if src == "kaggle_kernel":
        setup_kaggle_credentials()
        run_cli(["kaggle", "kernels", "output", cfg.kaggle_kernel, "-p", cfg.model_dir])
    elif src == "upload":
        from google.colab import files
        for name in files.upload():
            shutil.move(name, cfg.model_dir / name)
    elif src == "drive":
        from google.colab import drive
        drive.mount("/content/drive")
        shutil.copytree(cfg.drive_model_dir, cfg.model_dir, dirs_exist_ok=True)
    else:
        raise ValueError(f"Unknown weights_source: {src!r}")


checkpoint_path = find_checkpoint(cfg.model_dir)
if checkpoint_path is None:
    acquire_weights()
    checkpoint_path = find_checkpoint(cfg.model_dir)

if checkpoint_path is None:
    raise FileNotFoundError(
        f"No weights found in {cfg.model_dir}.\n"
        f"  Option 1: copy {cfg.weights_filename} into that folder, then re-run this cell.\n"
        f"  Option 2: set weights_source='kaggle_kernel' (or 'upload' / 'drive') in the configuration cell.\n"
        f"  Option 3: set train_model=True to train the model.")
print(f"Weights found: {checkpoint_path}  ({checkpoint_path.stat().st_size / 1e6:.1f} MB)")

# ============ CELL 40 (code) ============
import zipfile, os
p = "/content/model_output/AdaptiveFusion_best_model.pth"
print("exists:", os.path.exists(p), "| is a file:", os.path.isfile(p))
print("size  :", os.path.getsize(p) / 1e6, "MB")
print("valid zip:", zipfile.is_zipfile(p))
if zipfile.is_zipfile(p):
    print(zipfile.ZipFile(p).namelist()[:6])

# ============ CELL 41 (code) ============
# ============================================================
# 7.2 BUILD THE MODEL THAT MATCHES THE CHECKPOINT AND LOAD IT (strict)
# ============================================================
def load_state_dict_safely(path: Path) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and isinstance(obj.get("model"), dict):      # full training checkpoint
        obj = obj["model"]
    return {k.removeprefix("module."): v for k, v in obj.items()
            if not k.endswith(("total_ops", "total_params"))}           # profiling buffers added by thop


def build_model_from_state(state: dict) -> Tuple[AdaptiveFusionDetector, str]:
    if "spatial_projection.0.weight" not in state:
        raise ValueError("This file is not an AdaptiveFusionDetector checkpoint (missing 'spatial_projection.0.weight').")
    proj_dim, spatial_dim = state["spatial_projection.0.weight"].shape

    meta_path = cfg.model_dir / "model_config.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    names = list(dict.fromkeys([meta.get("backbone"), cfg.backbone, *cfg.backbone_candidates]))
    errors = []
    for name in filter(None, names):
        try:
            candidate = AdaptiveFusionDetector(name, pretrained=False, projection_dim=proj_dim)
            if candidate.spatial_features_dim != spatial_dim:
                errors.append(f"{name}: features {candidate.spatial_features_dim} != checkpoint {spatial_dim}")
                continue
            candidate.load_state_dict(state, strict=True)
            return candidate, name
        except Exception as exc:
            errors.append(f"{name}: {str(exc).splitlines()[0][:110]}")
    raise RuntimeError(
        f"No backbone matches the checkpoint (spatial dim {spatial_dim}, projection dim {proj_dim}).\n"
        "It was probably produced by a different notebook. Tried:\n  - " + "\n  - ".join(errors) +
        "\nSet cfg.backbone to the backbone used for training, or download the weights of the right notebook.")


state_dict = load_state_dict_safely(checkpoint_path)
model, detected_backbone = build_model_from_state(state_dict)
model = model.to(device).eval()
cfg.backbone = detected_backbone

n_params = sum(p.numel() for p in model.parameters())
sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
print(f"Model loaded  : AdaptiveFusionDetector(backbone={detected_backbone}, projection_dim={model.projection_dim})")
print(f"Parameters    : {n_params / 1e6:.2f} M | residual_scale = {model.residual_scale.item():.4f}")
print(f"SHA-256       : {sha[:16]}...")
print(f"Device        : {device}")

with torch.no_grad():
    check_logits = model(torch.randn(2, 3, S, S, device=device))
assert check_logits.shape == (2, 1)
print("Forward pass OK.")

if not (cfg.model_dir / "model_config.json").exists():
    save_model_config(model)       # cache the detected architecture next to the weights
    print("model_config.json written next to the weights.")

if HISTORY_PATH.exists():
    print("\nTraining history found -> plotting curves")
    plot_training_history(pd.read_csv(HISTORY_PATH))

# ============ CELL 42 (markdown) ============
---
# 8. Évaluation quantitative

Équivalent PyTorch des cellules *Quantitative results* du notebook Keras (matrice de confusion, ROC/AUC), enrichi de : EER, précision/rappel par classe, distribution des scores et analyse des erreurs.

> **Lecture des résultats.** Le test du dataset 140k est *in-distribution* (mêmes générateurs StyleGAN et mêmes sources réelles que l'entraînement) : attendez-vous à des scores très élevés. Le notebook source a montré qu'avec un **générateur inconnu** les performances chutent (voir *Résultats de référence*). Ne présentez donc pas ce chiffre comme une garantie de généralisation.

# ============ CELL 43 (code) ============
# ============================================================
# 8.1 EVALUATION HELPERS (confusion matrix, ROC, score distribution, report)
# ============================================================
def report_evaluation(results: dict, name: str, save_prefix: Optional[Path] = None) -> dict:
    y_true, y_pred, y_prob = results["y_true"], results["y_pred"], results["y_prob"]
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = results["loss"]
    metrics["real_accuracy"] = float((y_pred[y_true == 0] == 0).mean())
    metrics["fake_accuracy"] = float((y_pred[y_true == 1] == 1).mean())
    metrics["n_samples"] = int(len(y_true))

    print("=" * 62)
    print(f"{name}  ({len(y_true):,} images)")
    print("=" * 62)
    print(f"Accuracy {metrics['accuracy']:.2%} | Balanced acc {metrics['balanced_accuracy']:.2%}")
    print(f"Precision {metrics['precision']:.4f} | Recall {metrics['recall']:.4f} | F1 {metrics['f1']:.4f}")
    print(f"AUC {metrics['auc']:.4f} | EER {metrics['eer']:.4f} | Loss {metrics['loss']:.4f}")
    print(f"Real correctly identified: {metrics['real_accuracy']:.2%} | Fake correctly identified: {metrics['fake_accuracy']:.2%}\n")
    print(classification_report(y_true, y_pred, target_names=["REAL", "FAKE"], digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=axes[0], xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"])
    axes[0].set(title="Confusion matrix (counts)", xlabel="Predicted", ylabel="Actual")
    sns.heatmap(cm / cm.sum(axis=1, keepdims=True), annot=True, fmt=".2%", cmap="Blues", ax=axes[1],
                xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"])
    axes[1].set(title="Confusion matrix (per actual class)", xlabel="Predicted", ylabel="Actual")

    axes[2].plot(fpr, tpr, lw=2.5, color="#ff7f0e", label=f"AUC = {metrics['auc']:.4f}\nEER = {metrics['eer']:.4f}")
    axes[2].plot([0, 1], [0, 1], "--", color="#1f77b4", lw=1.2)
    axes[2].plot(metrics["eer"], 1 - metrics["eer"], "ro", label="EER point")
    axes[2].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
    axes[2].legend(loc="lower right")

    for label, color in [(0, "#2b5c8f"), (1, "#d95f02")]:
        axes[3].hist(y_prob[y_true == label], bins=40, range=(0, 1), alpha=0.65, color=color, label=ID2LABEL[label].upper())
    axes[3].axvline(0.5, color="black", ls="--", lw=1)
    axes[3].set(title="Score distribution P(fake)", xlabel="P(fake)", ylabel="Images", yscale="log")
    axes[3].legend()

    plt.suptitle(name, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_evaluation.png", dpi=150, bbox_inches="tight")
    plt.show()
    return metrics

# ============ CELL 44 (code) ============
# ============================================================
# 8.2 RUN THE EVALUATION
# ============================================================
eval_results, eval_metrics = {}, {}
if cfg.evaluate_on_test and (test_loader is not None or val_loader is not None):
    criterion = nn.BCEWithLogitsLoss()
    for split, loader in [("valid", val_loader), ("test", test_loader)]:
        if loader is None:
            continue
        eval_results[split] = predict_loader(model, loader, criterion=criterion, desc=f"Evaluating {split}")
        eval_metrics[split] = report_evaluation(eval_results[split], f"{split.upper()} split - 140k Real and Fake Faces",
                                                save_prefix=cfg.model_dir / split)
    (cfg.model_dir / "evaluation_metrics.json").write_text(json.dumps(eval_metrics, indent=2))
    display(pd.DataFrame(eval_metrics).T.round(4))
else:
    print("Evaluation skipped (evaluate_on_test=False or dataset not prepared).")

# ============ CELL 45 (code) ============
# ============================================================
# 8.3 ERROR ANALYSIS: most confident mistakes on the test split
# ============================================================
def plot_errors(results: dict, df: pd.DataFrame, n: int = 8) -> None:
    wrong = np.where(results["y_pred"] != results["y_true"])[0]
    print(f"{len(wrong)} misclassified images out of {len(results['y_true']):,}")
    if len(wrong) == 0:
        return
    confidence = np.abs(results["logits"][wrong])
    worst = wrong[np.argsort(-confidence)][:n]
    ncols = min(4, len(worst))
    nrows = int(np.ceil(len(worst) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.8 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, idx in zip(axes.flat, worst):
        ax.imshow(Image.open(df.iloc[idx]["path"]).convert("RGB"))
        ax.set_title(f"true {ID2LABEL[int(results['y_true'][idx])].upper()}\nP(fake) = {results['y_prob'][idx]:.3f}",
                     fontsize=10, color="#c0392b")
    plt.suptitle("Most confident errors", fontweight="bold")
    plt.tight_layout()
    plt.show()


if "test" in eval_results:
    test_df = images_df[images_df["split"] == "test"].reset_index(drop=True)   # same order as test_loader (shuffle=False)
    plot_errors(eval_results["test"], test_df)

# ============ CELL 46 (markdown) ============
---
# 9. Interprétabilité : poids de fusion et Grad-CAM

### 9.1 Poids de fusion adaptatifs
Pour chaque image, la porte produit `(w_spatial, w_frequency)` (somme = 1). Comparer leur distribution entre vrais et faux visages montre **quel flux le modèle exploite pour quel type d'image**. Dans le notebook source : spatial 0,86 / fréquentiel 0,14 pour les vrais, 0,74 / 0,26 pour les faux — le flux fréquentiel pèse davantage sur les faux.

### 9.2 Grad-CAM (porté de Keras vers PyTorch)
Grad-CAM utilise le gradient du score « FAKE » par rapport aux cartes de caractéristiques de la dernière couche convolutive du flux spatial pour montrer **quelles zones ont le plus influencé la décision** (rouge = fort impact, bleu = faible, colormap *jet*). Il explique le flux spatial ; l'effet du flux fréquentiel est résumé par les poids de la porte.

# ============ CELL 47 (code) ============
# ============================================================
# 9.1 ADAPTIVE FUSION WEIGHTS BY CLASS
# ============================================================
def analyse_fusion_weights(results: dict, save_path: Optional[Path] = None) -> pd.DataFrame:
    df = pd.DataFrame({"class": [ID2LABEL[int(i)].upper() for i in results["y_true"]],
                       "spatial_weight": results["w_spatial"], "frequency_weight": results["w_frequency"]})
    stats = df.groupby("class")[["spatial_weight", "frequency_weight"]].agg(["mean", "std", "median"]).round(4)
    display(stats)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    palette = {"REAL": "#2b5c8f", "FAKE": "#d95f02"}
    for ax, col, title in [(axes[0], "spatial_weight", "Spatial weight"), (axes[1], "frequency_weight", "Frequency weight")]:
        sns.histplot(data=df, x=col, hue="class", bins=30, palette=palette, alpha=0.6, ax=ax)
        ax.set_title(f"{title} distribution")
    sns.scatterplot(data=df.sample(min(2000, len(df)), random_state=0), x="spatial_weight", y="frequency_weight",
                    hue="class", palette=palette, alpha=0.5, s=18, ax=axes[2])
    axes[2].set_title("Dynamic fusion weights per image")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return df


if "test" in eval_results:
    fusion_df = analyse_fusion_weights(eval_results["test"], cfg.model_dir / "fusion_weights.png")
    n_unique = fusion_df["spatial_weight"].nunique()
    print(f"Distinct gate values: {n_unique:,} -> the gate is {'input-dependent (adaptive)' if n_unique > 10 else 'almost constant'}.")
else:
    print("Needs the test evaluation (section 8).")

# ============ CELL 48 (code) ============
# ============================================================
# 9.2 INFERENCE API + GRAD-CAM
# ============================================================
class DeepfakePredictor:
    """Small inference service: path / PIL / ndarray in, structured prediction out."""

    def __init__(self, model: AdaptiveFusionDetector, transform, device: torch.device, threshold: float = 0.5):
        self.model, self.transform, self.device, self.threshold = model.eval(), transform, device, threshold

    @staticmethod
    def to_pil(source) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, np.ndarray):
            return Image.fromarray(source).convert("RGB")
        return Image.open(source).convert("RGB")

    def to_tensor(self, source) -> torch.Tensor:
        return self.transform(self.to_pil(source)).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, source) -> dict:
        logits, (w_s, w_f), _ = self.model.forward_detailed(self.to_tensor(source))
        p_fake = torch.sigmoid(logits).item()
        return {"label": "FAKE" if p_fake >= self.threshold else "REAL",
                "prob_fake": p_fake, "prob_real": 1 - p_fake, "logit": logits.item(),
                "spatial_weight": w_s.item(), "frequency_weight": w_f.item()}

    def predict_many(self, sources) -> pd.DataFrame:
        rows = [{"source": str(s) if not isinstance(s, (Image.Image, np.ndarray)) else "<image>", **self.predict(s)} for s in sources]
        return pd.DataFrame(rows)

    def gradcam(self, source, target: str = "fake") -> Tuple[np.ndarray, np.ndarray]:
        """Return (resized RGB image in [0,1], heatmap in [0,1]) for the chosen class ('fake' or 'real')."""
        x = self.to_tensor(source)
        with torch.enable_grad():
            logits, _, fmap = self.model.forward_detailed(x)
            score = logits[:, 0] if target == "fake" else -logits[:, 0]
            grads = torch.autograd.grad(score.sum(), fmap)[0]
        channel_weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((channel_weights * fmap).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8)).detach().cpu().numpy()
        return denormalize(x[0]).permute(1, 2, 0).numpy(), cam


predictor = DeepfakePredictor(model, eval_transform, device)


def show_predictions(sources, true_labels=None, ncols: int = 4) -> pd.DataFrame:
    """Grid of images with predicted class + confidence (green = correct, red = wrong when truth is known)."""
    preds = predictor.predict_many(sources)
    nrows = int(np.ceil(len(sources) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.9 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (ax, src) in enumerate(zip(axes.flat, sources)):
        row = preds.iloc[i]
        conf = row["prob_fake"] if row["label"] == "FAKE" else row["prob_real"]
        color, truth = "black", ""
        if true_labels is not None:
            ok = row["label"] == true_labels[i]
            color, truth = ("#1e8449" if ok else "#c0392b"), f"true: {true_labels[i]}\n"
        ax.imshow(predictor.to_pil(src))
        ax.set_title(f"{truth}pred: {row['label']} ({conf:.1%})", color=color, fontsize=10)
    plt.tight_layout()
    plt.show()
    return preds


def show_gradcam(sources, true_labels=None, alpha: float = 0.45) -> None:
    jet = plt.get_cmap("jet")
    fig, axes = plt.subplots(len(sources), 3, figsize=(10, 3.4 * len(sources)), squeeze=False)
    for i, src in enumerate(sources):
        img, cam = predictor.gradcam(src, target="fake")
        pred = predictor.predict(src)
        overlay = np.clip((1 - alpha) * img + alpha * jet(cam)[..., :3], 0, 1)
        title = f"{pred['label']} - P(fake) {pred['prob_fake']:.1%}" + (f"\ntrue: {true_labels[i]}" if true_labels else "")
        for ax, data, sub in zip(axes[i], [img, cam, overlay], [title, "Grad-CAM heatmap", "Overlay"]):
            ax.imshow(data, cmap="jet" if data.ndim == 2 else None)
            ax.set_title(sub, fontsize=10)
            ax.axis("off")
    plt.suptitle("Grad-CAM (target: FAKE score) - red = most influential regions", fontweight="bold")
    plt.tight_layout()
    plt.show()

# ============ CELL 49 (code) ============
# ============================================================
# 9.3 GRAD-CAM ON TEST IMAGES (2 real + 2 fake)
# ============================================================
if HAS_DATA and (images_df["split"] == "test").any():
    test_pool = images_df[images_df["split"] == "test"]
    picks = pd.concat([test_pool[test_pool["class"] == "real"].sample(2, random_state=1),
                       test_pool[test_pool["class"] == "fake"].sample(2, random_state=1)])
    show_gradcam(picks["path"].tolist(), true_labels=[c.upper() for c in picks["class"]])
else:
    print("No test images available - use section 10 with your own images to see Grad-CAM.")

# ============ CELL 50 (markdown) ============
---
# 10. Inférence : prédire sur ses propres images

Équivalent des cellules *Image example - Qualitative results* du notebook Keras. Aucun entraînement n'est nécessaire : seul le modèle chargé en section 7 est utilisé.

- **Sorties** : classe, probabilité, et poids de fusion de l'image.
- Le modèle a été entraîné avec du *label smoothing* : ses probabilités plafonnent autour de 0,95 / 0,05 et sont donc moins « extrêmes » que celles d'un réseau classique. C'est normal.
- Pour de bons résultats, fournissez un **visage cadré** (comme dans FFHQ/StyleGAN). Une photo de scène large ou très compressée est hors distribution.

Utilisation : renseignez `my_images` avec des chemins, ou mettez `upload_own_images=True` (Colab) pour un dialogue d'upload.

# ============ CELL 51 (code) ============
# ============================================================
# 10.1 QUALITATIVE EXAMPLES FROM THE TEST SPLIT
# ============================================================
if HAS_DATA and (images_df["split"] == "test").any():
    pool = images_df[images_df["split"] == "test"]
    sel = pd.concat([pool[pool["class"] == "real"].sample(4, random_state=7),
                     pool[pool["class"] == "fake"].sample(4, random_state=7)]).sample(frac=1, random_state=7)
    table = show_predictions(sel["path"].tolist(), true_labels=[c.upper() for c in sel["class"]])
    display(table.assign(source=table["source"].str.split("/").str[-1]).round(4))
else:
    print("No dataset images loaded; use your own images below.")

# ============ CELL 52 (code) ============
!mkdir photo_test

# ============ CELL 53 (code) ============
# ============================================================
# 10.2 YOUR OWN IMAGES
# ============================================================
my_images = [
    "/content/photo_test/WIN_20260312_13_48_10_Pro.jpg",
    "/content/photo_test/WIN_20230502_01_14_45_Pro.jpg",
    "/content/photo_test/WIN_20260312_13_48_04_Pro.jpg",
    "/content/photo_test/WIN_20260312_13_48_10_Pro.jpg",
    "/content/photo_test/WIN_20261102_23_10_29_Pro - Copie.jpg",
    "/content/photo_test/WIN_20261113_01_05_13_Pro.jpg",
    "/content/photo_test/tinywow_profile_photo_50803976.png",
    "/content/photo_test/rimel2.jpg",
    "/content/photo_test/rimel1.jpg",
    "/content/photo_test/git.jpg",
    "/content/photo_test/WhatsApp Image 2022-02-01 at 11.46.46.jpeg",
    "/content/fake_img/6.png",
    "/content/fake_img/7.png",
    "/content/fake_img/8.png",
    "/content/fake_img/9.png",
    "/content/fake_img/101.png",
    "/content/fake_img/102.jpg",
    "/content/fake_img/100.jpg"




]

if cfg.upload_own_images and IN_COLAB:
    from google.colab import files
    my_images += [str(Path(name).resolve()) for name in files.upload()]

if my_images:
    own = show_predictions(my_images)
    display(own.round(4))
    show_gradcam(my_images)
else:
    print("my_images is empty. Add file paths above, or set upload_own_images=True in the configuration.")

# ============ CELL 54 (code) ============
!mkdir fake_img

# ============ CELL 55 (markdown) ============
---
# 11. Performances : paramètres, MACs, latence, mémoire

Mesures sur la machine courante (les chiffres changent selon le GPU/CPU). Corrections par rapport au notebook source :
- la latence est mesurée **avec synchronisation GPU** (sans elle, le FPS est surestimé car CUDA est asynchrone) ;
- `thop` compte des **MACs** (multiplications-accumulations), pas des FLOPs (FLOPs ≈ 2 × MACs) ;
- `thop` est exécuté sur une **copie** du modèle : il ajoute des buffers (`total_ops`, `total_params`) qui polluent sinon le `state_dict`.

# ============ CELL 56 (code) ============
# ============================================================
# 11.1 EFFICIENCY BENCHMARK
# ============================================================
def sync() -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark(model: nn.Module, batch_sizes=(1, 8, 32), warmup: int = 3) -> pd.DataFrame:
    model.eval()
    n_iter = 20 if device.type == "cuda" else 3
    rows = []
    for bs in batch_sizes:
        x = torch.randn(bs, 3, S, S, device=device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup):
            model(x)
        sync()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(x)
        sync()
        dt = (time.perf_counter() - t0) / n_iter
        rows.append({"batch_size": bs, "latency_ms": dt * 1e3, "throughput_img_s": bs / dt,
                     "peak_gpu_mem_MB": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")})
    return pd.DataFrame(rows).round(2)


efficiency = {"parameters_M": round(n_params / 1e6, 2)}
try:
    from thop import profile
    macs, _ = profile(copy.deepcopy(model).cpu().eval(), inputs=(torch.randn(1, 3, S, S),), verbose=False)
    efficiency["MACs_G"] = round(macs / 1e9, 2)
except Exception as exc:
    print(f"thop skipped: {exc}")

bench = benchmark(model)
efficiency["latency_ms_batch1"] = float(bench.loc[bench["batch_size"] == 1, "latency_ms"].iloc[0])
efficiency["throughput_img_s_best"] = float(bench["throughput_img_s"].max())
print(f"Parameters: {efficiency['parameters_M']} M | MACs: {efficiency.get('MACs_G', 'n/a')} G | device: {device}")
display(bench)

# ============ CELL 57 (markdown) ============
---
# 12. Artefacts MLOps

Pour que le modèle soit **traçable et réutilisable** sans ce notebook, on consigne dans `model_dir` :

| Fichier | Rôle |
|---|---|
| `AdaptiveFusion_best_model.pth` | Poids (`state_dict`) |
| `model_config.json` | Architecture + prétraitement + règle de décision (reconstruction sans ambiguïté) |
| `model_manifest.json` | Empreinte SHA-256 des poids, métriques, performances, versions logicielles, configuration d'exécution |
| `MODEL_CARD.md` | Fiche modèle lisible (usage prévu, données, métriques, limites) |
| `training_history.csv`, `training_curves.png`, `*_evaluation.png`, … | Historique et figures |

➡️ **Pour réutiliser le modèle sans entraîner** : copier ce dossier dans `/content/model_output` et garder `train_model=False`.

# ============ CELL 58 (code) ============
# ============================================================
# 12.1 MANIFEST + MODEL CARD
# ============================================================
def jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)


manifest = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "weights_file": checkpoint_path.name,
    "weights_sha256": sha,
    "weights_size_mb": round(checkpoint_path.stat().st_size / 1e6, 2),
    "architecture": {"name": "AdaptiveFusionDetector", "backbone": detected_backbone,
                     "projection_dim": model.projection_dim, "parameters": n_params},
    "preprocessing": {"image_size": S, "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    "classes": {"0": "real", "1": "fake"},
    "decision_rule": "sigmoid(logit) >= 0.5 -> FAKE",
    "evaluation_140k": eval_metrics,
    "efficiency": efficiency,
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__,
                    "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
    "run_config": asdict(cfg),
}
(cfg.model_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2, default=jsonable, ensure_ascii=False))

test_m = eval_metrics.get("test", {})
metric_line = (f"Accuracy {test_m['accuracy']:.2%}, F1 {test_m['f1']:.4f}, AUC {test_m['auc']:.4f}, EER {test_m['eer']:.4f} "
               f"({test_m['n_samples']:,} test images of the 140k dataset)") if test_m else "not evaluated in this run"
card = f"""# Model card - Adaptive Fusion Detector

**Task**: binary classification of face images, REAL (0) vs AI-GENERATED / FAKE (1).
**Architecture**: {detected_backbone} spatial stream + FFT log-magnitude CNN stream, adaptive softmax gate, residual fusion, MLP classifier ({n_params / 1e6:.2f} M parameters).
**Input**: RGB, {S}x{S}, ImageNet normalization. **Output**: one logit (> 0 means FAKE).

## Data
Evaluated on the test split of *140k Real and Fake Faces* (real: Flickr-Faces-HQ, fake: StyleGAN). Weights come from `{checkpoint_path.name}` (SHA-256 `{sha[:16]}...`).

## Metrics (this run)
{metric_line}

## Intended use
Research / educational demonstration of deepfake-face detection on cropped, aligned faces.

## Limitations
- In-distribution scores on 140k are optimistic. The source notebook measured 83.87 % accuracy and 68.69 % recall on fakes for generators unseen during training (Face-VAE / GANsformer).
- Not validated on diffusion-based images, video frames, heavy compression, non-face images or demographic fairness.
- Probabilities are compressed by label smoothing and should not be read as calibrated confidence.
- Not suitable as the sole basis for decisions affecting individuals.

Generated on {datetime.now():%Y-%m-%d %H:%M}.
"""
(cfg.model_dir / "MODEL_CARD.md").write_text(card)

files_table = pd.DataFrame([{"file": p.name, "size_MB": round(p.stat().st_size / 1e6, 3)}
                            for p in sorted(cfg.model_dir.iterdir()) if p.is_file()])
print(f"Contents of {cfg.model_dir}:")
display(files_table)

# ============ CELL 59 (code) ============
# ============================================================
# 12.2 OPTIONAL: PERSIST THE MODEL FOLDER TO GOOGLE DRIVE
# (Colab storage is wiped when the runtime is recycled)
# ============================================================
if cfg.sync_to_drive:
    from google.colab import drive
    drive.mount("/content/drive")
    cfg.drive_model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cfg.model_dir, cfg.drive_model_dir, dirs_exist_ok=True)
    print(f"Copied to Google Drive: {cfg.drive_model_dir}")
else:
    print("sync_to_drive=False -> remember to download or copy model_output before the Colab runtime is recycled.")

# ============ CELL 60 (markdown) ============
---
# ✅ Conclusion et dépannage

**Ce notebook permet de** : entraîner le modèle (`train_model=True`) ou **le réutiliser sans réentraîner** (`train_model=False` + poids dans `/content/model_output`), l'évaluer de façon complète, l'expliquer (poids de fusion, Grad-CAM) et prédire sur ses propres images, en laissant des artefacts traçables.

| Problème | Solution |
|---|---|
| `FileNotFoundError: No weights found` | Copier le `.pth` dans `/content/model_output` ou régler `weights_source`. |
| `No backbone matches the checkpoint` | Les poids viennent d'un autre notebook (ex. RegNet). Régler `cfg.backbone` ou télécharger la sortie du bon notebook Kaggle. |
| `kaggle kernels output` échoue | Le notebook Kaggle doit être public ou lancé avec un jeton valide ; sinon télécharger le `.pth` depuis l'onglet *Output* de Kaggle et utiliser `weights_source="upload"`. |
| Mémoire GPU saturée | Réduire `batch_size` (ex. 16). |
| Colab déconnecté en cours d'entraînement | Relancer avec `resume_training=True` (reprend `last_checkpoint.pt`). |
| Entraînement trop long | `quick_demo=True` pour valider, puis réduire `train_per_class` / `epochs`. |

> 🔐 **Sécurité** : ne jamais coller un jeton d'API dans une cellule. Le notebook `NLP_MLOPS_Project_v1` en contient un en clair : **révoquez-le** dans Kaggle (*Settings → API*) et utilisez les *Secrets* Colab.