# Deepfake Adaptive Fusion — MLOps

[![CI tests](https://github.com/OWNER/REPO/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci-tests.yml)
[![Docker build](https://github.com/OWNER/REPO/actions/workflows/docker-build.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/docker-build.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Détection de deepfakes par **fusion adaptative spatiale + fréquentielle** :
un backbone convolutif analyse l'image RGB, un second flux analyse son spectre de
Fourier, et une porte apprise (`fusion_gate`) pondère dynamiquement les deux
représentations. Le projet industrialise le notebook d'origine : code
refactorisé dans `src/`, pipeline DVC, suivi MLflow, export ONNX quantifié,
API FastAPI, interface React, Docker Compose, CI/CD et rapport LaTeX.

> **Avertissement** — ce détecteur est un outil d'aide, **jamais une preuve
> d'authenticité**. Il généralise mal aux générateurs inconnus et aux images
> fortement recompressées. Voir [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

---

## Sommaire

- [Architecture](#architecture)
- [Démarrage en 3 commandes](#démarrage-en-3-commandes)
- [URLs](#urls)
- [Installation détaillée](#installation-détaillée)
- [Variables d'environnement](#variables-denvironnement)
- [Données et DVC](#données-et-dvc)
- [Entraînement et évaluation](#entraînement-et-évaluation)
- [Expériences et MLflow](#expériences-et-mlflow)
- [Tests et qualité](#tests-et-qualité)
- [Structure du dépôt](#structure-du-dépôt)
- [Limites connues](#limites-connues)
- [Crédits et licences](#crédits-et-licences)

---

## Architecture

```
                 ┌──────────────────────────────┐
 image RGB ─────►│  spatial_net (ConvNeXt-T)    │──► spatial_projection ─┐
                 └──────────────────────────────┘                        │
                                                                          ├─► fusion_gate ─► classifier ─► logit (fake si > 0)
                 ┌──────────────────────────────┐                        │        │
 |FFT| log1p ───►│  freq_net (CNN fréquentiel)  │──► frequency_projection┘        └─► w_spatial / w_frequency
                 └──────────────────────────────┘
```

- `freq_input = log1p(|fftshift(fft2(image))|)` est calculé **hors du graphe ONNX**
  (Option A) : le modèle exporté prend deux entrées (`image`, `freq_input`) et
  renvoie trois sorties (`logit`, `w_spatial`, `w_frequency`).
- Labels : `real = 0`, `fake = 1` ; décision `logit > 0` ⇔ `P(fake) ≥ seuil`.
- Grad-CAM nécessite les gradients : `/predict` utilise ONNX Runtime (rapide),
  `/predict/visualize` charge PyTorch **paresseusement** pour la carte Grad-CAM.
- Détails et justifications : [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Démarrage en 3 commandes

```bash
git clone https://github.com/OWNER/REPO.git && cd REPO
dvc pull                       # données + checkpoint + ONNX (remote DVC configuré)
docker compose up --build      # frontend + backend + MLflow
```

Sans remote DVC (génère automatiquement les artefacts factices de démonstration) :

```bash
./scripts/setup_demo.sh
# ou : python scripts/make_dummy_model.py --out models && docker compose up --build
```

## URLs

| Service | URL | Description |
|---|---|---|
| Frontend | http://localhost:3000 | drag & drop, galerie, webcam, Grad-CAM |
| API Swagger | http://localhost:8000/docs | documentation interactive complète |
| API ReDoc | http://localhost:8000/redoc | documentation alternative |
| Health | http://localhost:8000/health | état, providers ONNX, GPU |
| MLflow | http://localhost:5000 | runs, métriques, artefacts, registry |

## Installation détaillée

Prérequis : Python 3.11, Node.js ≥ 20, Docker ≥ 24 + Compose v2, `git`, `dvc`
(optionnel mais recommandé). GPU CUDA optionnel pour l'entraînement.

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt          # entraînement + évaluation + export

cp .env.example .env                     # renseigner les secrets (jamais commités)
pre-commit install                       # black, flake8, nbstripout, detect-secrets
```

Frontend en développement :

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173, proxy /api -> localhost:8000
```

Backend en développement :

```bash
uvicorn backend.app.main:app --reload --port 8000
```

## Variables d'environnement

Toutes les variables sont documentées dans [`.env.example`](.env.example).

| Variable | Défaut | Rôle |
|---|---|---|
| `KAGGLE_API_TOKEN` | — | téléchargement du dataset (ou `KAGGLE_USERNAME`/`KAGGLE_KEY`) |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlruns/mlflow.db` | backend MLflow |
| `MODEL_PATH` | `models/deepfake_fusion.onnx` | modèle ONNX servi |
| `PYTORCH_CHECKPOINT_PATH` | `models/AdaptiveFusion_best_model.pth` | checkpoint pour Grad-CAM |
| `MODEL_CONFIG_PATH` | `models/model_config.json` | métadonnées du modèle |
| `METRICS_PATH` | `reports/metrics.json` | métriques exposées par l'API |
| `MAX_UPLOAD_MB` | `10` | taille maximale d'upload (413 au-delà) |
| `DECISION_THRESHOLD` | `0.5` | seuil `P(fake)` |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | origines autorisées |
| `VITE_API_URL` | vide | vide = même origine ; `/api` derrière nginx |

## Données et DVC

Dataset principal : [`xhlulu/140k-real-and-fake-faces`](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
(splits officiels `train`/`valid`/`test`, 70k/20k/50k). Le notebook
sous-échantillonne de façon **déterministe** (seed 42) pour tenir sur Colab.

```bash
# 1. télécharger + extraire le sous-ensemble (secrets via variables d'env)
export KAGGLE_API_TOKEN=...
python scripts/download_data.py --config params.yaml
# ou : python scripts/download_data.py --quick-demo          (500 img/classe)
# ou : python scripts/download_data.py --zip /content/140k.zip

# 2. initialiser DVC (une seule fois) puis pousser les données
./scripts/init_dvc.sh
dvc push

# 3. rejouer tout le pipeline
dvc repro
dvc metrics show
```

- **Anti-fuite** : splits officiels, sous-échantillonnage par classe, test MD5
  (`tests/test_no_leakage.py`). Aucune augmentation avant le split.
- **Cross-dataset** : les jeux externes (Celeb-DF v2, FF-C23, CelebA, LFW) ne
  servent **jamais** à l'entraînement ; voir `data/external/README.md`.
- Remote par défaut : dossier local `../dvc-remote`. Pour S3/GCS/Azure :
  `dvc remote add -d storage s3://bucket/path` puis `dvc push`.

## Entraînement et évaluation

```bash
# smoke test CPU : 500 images/classe, 2 époques
python -m src.models.train --config params.yaml --quick-demo

# run complet (GPU recommandé)
python -m src.models.train --config params.yaml

# baseline / ablations
python -m src.models.train --mode spatial_only --run-name R1_spatial
python -m src.models.train --mode freq_only    --run-name R2_freq
python -m src.models.train --mode concat_no_gate --run-name R3_concat
python -m src.models.train --mode fusion       --run-name R4_fusion

# évaluation : métriques, calibration, robustesse, benchmark, export ONNX
python -m src.models.evaluate --config params.yaml --export-onnx --quantize

# étude comparative complète (expériences + ablations)
python scripts/run_experiments.py --config params.yaml
```

Sorties : `models/AdaptiveFusion_best_model.pth`, `models/model_config.json`,
`models/deepfake_fusion{,_fp16,_int8}.onnx`, `reports/metrics.json`,
`reports/figures/`, `reports/comparison_table.csv`, `reports/onnx_parity.json`.

Le fichier `models/model_config.json` enregistre le backbone, le mode et le
seuil ; le backend l'expose via `/model/metadata`.

## Expériences et MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

- Expérience : `deepfake-adaptive-fusion`.
- Runs obligatoires : `R1_spatial_only`, `R2_freq_only`, `R3_concat_no_gate`,
  `R4_fusion`, puis les ablations de `experiments.yaml` (RandAugment/MixUp,
  Focal Loss, gel/dégel, robustesse).
- Le meilleur run est enregistré sous `deepfake-adaptive-fusion` ; passer
  `tracking.register: true` et `tracking.alias: production` dans `params.yaml`
  pour le promouvoir. Le backend privilégie l'artefact MLflow « Production »
  puis retombe sur `models/deepfake_fusion.onnx`.
- Aucune métrique n'est codée en dur : le rapport et l'API lisent
  `reports/metrics.json` / MLflow.

## Tests et qualité

```bash
# backend : API, prétraitement, Grad-CAM, mapping des labels, couverture >= 80 %
pytest backend -q --cov=backend/app --cov-report=term-missing --cov-fail-under=80

# tests racine : anti-fuite + contrat ONNX (parité réelle marquée `slow`)
pytest tests -q -m "not slow"

# style
flake8 src backend tests
black --check src backend tests

# frontend
cd frontend && npm install && npm run lint && npm test && npm run build

# smoke test HTTP (stack démarrée)
./scripts/smoke_test.sh path/to/face.jpg
```

Les tests backend utilisent un **modèle ONNX factice** de même signature : aucun
poids réel ni dataset n'est nécessaire. La parité PyTorch ↔ ONNX sur le vrai
modèle est un test `slow`, exécuté quand les artefacts sont présents.

## Structure du dépôt

```
.
├── .github/workflows/        # ci-tests.yml, docker-build.yml
├── backend/                  # FastAPI : app/, tests/, Dockerfile, requirements.txt
├── data/                     # raw/ + processed/ (DVC, jamais dans Git)
├── docs/                     # DECISIONS.md, MODEL_CARD.md, SOUTENANCE.md
├── frontend/                 # React 18 + Vite + Tailwind, Dockerfile + nginx
├── models/                   # .pth / .onnx (DVC, jamais dans Git)
├── notebooks/                # notebook nettoyé (démonstration, sans secrets)
├── rapport/                  # sources LaTeX -> rapport_technique.pdf
├── scripts/                  # download, experiments, DVC, sanitize, smoke...
├── src/                      # data/, models/, utils/, config.py, tracking.py
├── tests/                    # anti-fuite, contrat/parité ONNX
├── docker-compose.yml
├── dvc.yaml  params.yaml  experiments.yaml
├── requirements.txt  requirements-ci.txt
└── README.md
```

## Limites connues

- **Généralisation** : les performances chutent sur des générateurs absents du
  train (StyleGAN3, diffusion récente) et sur les vidéos fortement compressées.
- **Données** : 140k contient surtout StyleGAN ; risque de biais démographique
  et de raccourcis de compression. Voir la model card.
- **Seuil** : `0.5` par défaut ; à recalibrer selon le coût métier (faux positifs
  vs faux négatifs). La calibration est tracée dans `reports/threshold_calibration.json`.
- **Grad-CAM** : approximation sur la dernière feature map spatiale ; ne prouve
  pas la causalité.
- **INT8** : à ne pas déployer si la dégradation d'AUC dépasse 1 point (comparer
  les rapports générés).
- **Webcam** : l'inférence continue dépend de la lumière et de la qualité du
  flux ; ce n'est pas une mesure forensique.

## Crédits et licences

- Dataset : `xhlulu/140k-real-and-fake-faces` (Kaggle), dérivé de *Real and Fake
  Face Dataset* (140k) — licence à vérifier sur la page Kaggle.
- Backbone : [timm](https://github.com/huggingface/pytorch-image-models)
  (`convnext_tiny`, poids ImageNet).
- Code : licence MIT (voir `LICENSE`). Les poids et données ne sont pas couverts
  par cette licence ; ils restent soumis à leurs licences d'origine.
- Inspirations : littérature sur la détection fréquentielle de deepfakes
  (voir `rapport/refs.bib`).
