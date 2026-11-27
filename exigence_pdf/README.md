# Deepfake Adaptive Fusion — MLOps

[![CI tests](https://github.com/OWNER/REPO/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci-tests.yml)
[![Docker build](https://github.com/OWNER/REPO/actions/workflows/docker-build.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/docker-build.yml)
[![CD deploy](https://github.com/OWNER/REPO/actions/workflows/cd-deploy.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/cd-deploy.yml)
[![Continuous Training](https://github.com/OWNER/REPO/actions/workflows/continuous-training.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/continuous-training.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Détection de deepfakes par **fusion adaptative spatiale + fréquentielle** :
un backbone convolutif analyse l'image RGB, un second flux analyse son spectre de
Fourier, et une porte apprise (`fusion_gate`) pondère dynamiquement les deux
représentations. Le projet industrialise le notebook d'origine avec un pipeline
MLOps complet : versioning Git + DVC, suivi d'expériences MLflow, orchestration
Prefect, monitoring Prometheus + Grafana + Evidently AI, CI/CD/CT, et déploiement
Docker Compose.

> **Projet MLOps encadré par Sonia Gharsalli** — Tek-Up Engineering, ING-5 SDIA

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

### Architecture du modèle

```
                 ┌──────────────────────────────┐
 image RGB ─────►│  spatial_net (ConvNeXt-T)    │──► spatial_projection ─┐
                 └──────────────────────────────┘                        │
                                                                          ├─► fusion_gate ─► classifier ─► logit (fake si > 0)
                 ┌──────────────────────────────┐                        │        │
 |FFT| log1p ───►│  freq_net (CNN fréquentiel)  │──► frequency_projection┘        └─► w_spatial / w_frequency
                 └──────────────────────────────┘
```

### Architecture MLOps globale

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          GitHub Actions (CI/CD/CT)                              │
│  ci-tests.yml ─── docker-build.yml ─── cd-deploy.yml ─── continuous-training   │
└────────────────────────────┬────────────────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────────────────┐
│                              Docker Compose Stack                               │
│                                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐ │
│  │  Frontend   │──►│  Backend    │──►│  MLflow  │   │Prometheus│──►│Grafana │ │
│  │  React:3000 │   │  FastAPI    │   │  :5000   │   │  :9090   │   │ :3001  │ │
│  └─────────────┘   │  :8000      │   └──────────┘   └──────────┘   └────────┘ │
│                     └──────┬──────┘                                             │
│                            │/metrics                                           │
└────────────────────────────┼────────────────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────────────────┐
│                        Monitoring & Observability                               │
│                                                                                 │
│  • Prometheus : métriques système (CPU, RAM) + modèle (latence, prédictions)    │
│  • Grafana    : dashboards temps réel (overview, drift, performance)            │
│  • Evidently AI : détection de data drift (PSI, KS test)                        │
│  • Prediction Logger : JSONL pour analyse post-hoc                              │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Pipeline Orchestration (Prefect)                       │
│                                                                                 │
│  prepare ──► train ──► evaluate ──► quality_gate ──► export_onnx ──► register  │
│     ↕           ↕         ↕              ↕               ↕              ↕       │
│  retries     retries   retries      AUC≥0.85, F1≥0.80  retries       retries   │
└─────────────────────────────────────────────────────────────────────────────────┘
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
| **Prometheus** | http://localhost:9090 | métriques système et modèle (scraping) |
| **Grafana** | http://localhost:3001 | dashboards temps réel (admin/admin) |
| **/metrics** | http://localhost:8000/metrics | endpoint Prometheus exposition |
| **/monitoring/stats** | http://localhost:8000/monitoring/stats | statistiques des prédictions |
| **/monitoring/drift** | POST http://localhost:8000/monitoring/drift | déclencher l'analyse de drift |

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

## Orchestration des pipelines (Prefect)

L'orchestration est assurée par **Prefect** (`src/orchestration/`), qui enveloppe
les étapes DVC dans des flows avec retry, gestion d'erreurs et observabilité.

```bash
# Lancer le pipeline complet (prepare -> train -> evaluate -> export -> register)
python -m src.orchestration.flows --config params.yaml --register

# Lancer un réentraînement (triggered by drift ou dégradation)
python -m src.orchestration.flows --retrain --trigger data_drift

# Démarrer le serveur Prefect (optionnel, pour l'UI)
prefect server start
```

### Gestion des erreurs et reprise

Chaque tâche a sa propre politique de retry :

| Tâche | Retries | Délai | Raison |
|---|---|---|---|
| `prepare_data` | 2 | 30s | Téléchargement Kaggle peut échouer |
| `train_model` | 1 | 60s | Entraînement coûteux, retry limité |
| `evaluate_model` | 1 | 30s | Évaluation rapide |
| `export_onnx` | 2 | 30s | Export rapide, retry facile |
| `register_model` | 2 | 15s | Best-effort, ne bloque pas le pipeline |
| `check_model_quality` | 0 | — | Gate : échoue le flow si AUC < 0.85 |

Le flow `retrain_flow` est déclenché automatiquement quand :
- **Data drift** détecté par Evidently AI (score > 0.3)
- **Dégradation des métriques** (AUC ou F1 sous le seuil)
- **Trigger manuel** depuis l'UI Prefect ou GitHub Actions

## Continuous Integration (CI)

Deux workflows GitHub Actions assurent la qualité :

### `ci-tests.yml` — Lint + Tests
- **Backend** : black, flake8, pytest (couverture ≥ 80%)
- **Frontend** : npm lint, unit tests, production build
- **Core** : tests anti-fuite + contrat ONNX

### `docker-build.yml` — Smoke test Docker
- Build des images backend + frontend
- Démarre la stack (`docker compose up`)
- Vérifie `/health`, prédiction, frontend

## Continuous Delivery/Deployment (CD)

Le workflow `cd-deploy.yml` orchestre le déploiement en 4 étapes :

1. **Build** : images Docker poussées sur GitHub Container Registry
2. **Staging** : déploiement automatique sur `main`, smoke tests
3. **Production** : déploiement après validation (environnement protégé)
4. **Rollback** : restauration automatique si la production échoue

```
main push → Build → Deploy Staging → Smoke Tests → Deploy Production → Rollback (si fail)
```

## Continuous Training (CT)

Le workflow `continuous-training.yml` automatise le réentraînement :

- **Schedule** : vérification hebdomadaire (lundi 6h UTC)
- **Triggers** : data drift, dégradation métriques, dispatch externe
- **Pipeline** : retrain → validate → promote → deploy

```bash
# Déclencher manuellement un réentraînement
gh workflow run continuous-training.yml -f trigger=manual

# Via repository_dispatch (depuis un webhook monitoring)
curl -X POST https://api.github.com/repos/OWNER/REPO/dispatches \
  -H "Accept: application/vnd.github.v3+json" \
  -d '{"event_type": "retrain-requested"}'
```

## Monitoring

### Métriques système (Prometheus)
Le backend expose `/metrics` au format Prometheus :

| Métrique | Type | Description |
|---|---|---|
| `deepfake_http_requests_total` | Counter | Requêtes HTTP par méthode/endpoint/status |
| `deepfake_http_request_duration_seconds` | Histogram | Latence des requêtes |
| `deepfake_cpu_usage_percent` | Gauge | Utilisation CPU |
| `deepfake_memory_usage_bytes` | Gauge | Mémoire utilisée |
| `deepfake_predictions_total` | Counter | Prédictions par label |
| `deepfake_prediction_confidence` | Histogram | Distribution des confiances |
| `deepfake_prediction_latency_seconds` | Histogram | Latence par stage |
| `deepfake_fusion_weight_spatial` | Histogram | Poids spatial de la fusion |
| `deepfake_data_drift_score` | Gauge | Score de dérive des données |

### Métriques modèle (Evidently AI)
La détection de drift compare les prédictions récentes aux statistiques d'entraînement :

```bash
# Déclencher l'analyse de drift
curl -X POST http://localhost:8000/monitoring/drift

# Résultat : rapport JSON avec score de drift par feature
# Si drift_score > 0.3 → déclenche le CT (réentraînement automatique)
```

### Dashboards Grafana
Un dashboard pré-configuré est disponible sur http://localhost:3001 (admin/admin) :
- **HTTP Request Rate** et latence p95
- **Prediction Rate** par label (FAKE/REAL)
- **CPU/Memory** usage en temps réel
- **Fusion Weights** (spatial vs frequency)
- **Data Drift Score** avec alertes

### Prediction Logger
Chaque prédiction est logguée dans `monitoring_data/predictions.jsonl` pour :
- Analyse post-hoc de la distribution des prédictions
- Détection de drift avec Evidently AI
- Audit et traçabilité

## Structure du dépôt

```
.
├── .github/workflows/        # ci-tests, docker-build, cd-deploy, continuous-training
├── backend/                  # FastAPI : app/, tests/, Dockerfile, requirements.txt
├── data/                     # raw/ + processed/ (DVC, jamais dans Git)
├── docs/                     # DECISIONS.md, MODEL_CARD.md, SOUTENANCE.md
├── frontend/                 # React 18 + Vite + Tailwind, Dockerfile + nginx
├── models/                   # .pth / .onnx (DVC, jamais dans Git)
├── monitoring/               # Prometheus config + Grafana provisioning
├── monitoring_data/          # Prediction logs (JSONL, drift analysis)
├── notebooks/                # notebook nettoyé (démonstration, sans secrets)
├── rapport/                  # sources LaTeX -> rapport_technique.pdf
├── scripts/                  # download, experiments, DVC, sanitize, smoke...
├── src/
│   ├── data/                 # préparation, dataset, transforms, anti-leakage
│   ├── models/               # architecture, train, evaluate, losses, metrics, ONNX
│   ├── monitoring/           # metrics.py, drift.py, prediction_logger.py, api.py
│   ├── orchestration/        # Prefect flows (pipeline + retrain)
│   ├── utils/                # device, gradcam, io, logging, paths, plots, seed
│   ├── config.py             # configuration centralisée (params.yaml)
│   └── tracking.py           # MLflow tracking + Model Registry
├── tests/                    # anti-fuite, contrat/parité ONNX
├── docker-compose.yml        # Stack complète (backend, frontend, MLflow, Prometheus, Grafana)
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
