# DeepGuard — frontend React complet + backend Django

Cette brique intègre l'interface enrichie (landing, espace analyste, console admin,
CRUD/localStorage) et le backend Django du dépôt
[`houssem4512/deep-guard-deep-fake-`](https://github.com/houssem4512/deep-guard-deep-fake-)
**dans** le projet `exigence_pdf`, en conservant :

- le package partagé `src/` (inférence, export ONNX, monitoring…) ;
- les vrais artefacts du projet (`models/deepfake_fusion.onnx`,
  `models/AdaptiveFusion_best_model.pth`, `models/model_config.json`) ;
- les vraies métriques récupérées (`reports/metrics.json`) ;
- le déploiement Docker + le monitoring Prometheus/Grafana (ajoutés ici, absents
  du dépôt d'origine du collègue).

L'ancienne stack FastAPI (`backend/app/`) et son frontend minimal (`frontend/`)
restent intactes pour la CI et les tests.

## Arborescence

```
exigence_pdf/
├── deepguard/
│   ├── frontend/            React 18 + Vite 5 + Tailwind + React Router
│   ├── backend/             Django (API identique à la FastAPI) + média
│   ├── Dockerfile           build React + image Django/Gunicorn
│   ├── docker-compose.yml   DeepGuard + Prometheus + Grafana + MLflow
│   └── README.md            ce fichier
├── src/                     package partagé (réutilisé tel quel)
├── models/                  poids réels (montés en lecture seule)
├── reports/metrics.json     vraies métriques du checkpoint entraîné
└── monitoring/              configs Prometheus / Grafana
```

## 1. Démarrage local (2 terminaux)

> **Python 3.12** requis par `Django 6.1` (l'API FastAPI d'origine reste en
> 3.11). `deepguard/backend/manage.py` ajoute automatiquement la racine du projet
> au `PYTHONPATH` pour importer `src.*`.

**Terminal 1 — backend Django (port 8000)**

```bash
cd exigence_pdf
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r deepguard/backend/requirements.txt
python deepguard/backend/manage.py runserver
```

**Terminal 2 — frontend Vite (port 5173)**

```bash
cd exigence_pdf/deepguard/frontend
npm ci
npm run dev
```

Ouvrir <http://localhost:5173>. Le serveur Vite proxifie `/predict`, `/health`,
`/model` et `/media` vers `http://127.0.0.1:8000`.

Comptes de démonstration (stockés dans le `localStorage` du navigateur) :

| Rôle  | Email                  | Mot de passe |
|-------|------------------------|--------------|
| Admin | `admin@deepguard.io`   | `Admin@2026` |
| User  | `demo@deepguard.io`    | `Demo@2026`  |

**Mode mono-serveur** (Django sert le build React, aucune Node au runtime) :

```bash
cd exigence_pdf/deepguard/frontend && npm run build
cd ../.. && python deepguard/backend/manage.py runserver
# -> http://127.0.0.1:8000
```

## 2. Déploiement Docker + monitoring

```bash
cd exigence_pdf
docker compose -f deepguard/docker-compose.yml up --build
```

| Service    | URL                     | Notes |
|------------|-------------------------|-------|
| DeepGuard  | http://localhost:8010   | UI + API (Gunicorn) |
| Prometheus | http://localhost:9091   | scrape `/metrics` |
| Grafana    | http://localhost:3002   | `admin` / `admin` |
| MLflow     | http://localhost:5001   | tracking |

Le conteneur monte `../models` et `../reports` en lecture seule.

## 3. Monitoring ajouté

Le backend Django expose désormais les mêmes routes que la stack FastAPI :

| Méthode | Route                        | Description |
|---------|------------------------------|-------------|
| GET     | `/metrics`                   | exposition Prometheus |
| GET     | `/monitoring/stats`          | statistiques des prédictions (JSONL) |
| POST    | `/monitoring/drift`          | rapport de dérive (`?min_samples=50`) |
| POST    | `/monitoring/log-prediction` | journaliser une prédiction |

Chaque `POST /predict` enregistre le label, la confiance, les poids de fusion et
les latences (prétraitement/inférence/total), dans Prometheus **et** dans
`monitoring_data/predictions.jsonl` (analysable par `src/monitoring/drift.py`).
Une panne du monitoring ne casse jamais l'inférence.

## 4. Poids et métriques

- Les poids réels du projet (`models/*.onnx` / `*.pth`) sont utilisés : aucune
  modification de code n'est nécessaire.
- `reports/metrics.json` provient de l'évaluation du checkpoint entraîné
  (Acc test 99.79 %, F1 0.9979, AUC 1.000). Il alimente `/model/metadata` et la
  page admin « Model monitoring ».
- Les figures d'évaluation sont servies depuis
  `deepguard/backend/media/figures/`.

## 5. Différences avec l'ancienne stack

- Le frontend minimal (`exigence_pdf/frontend/`) et l'API FastAPI
  (`exigence_pdf/backend/`) restent présents et testés par la CI.
- DeepGuard est une seconde stack complète, isolée sous `deepguard/`, pour
  profiter de l'interface/CRUD sans casser l'existant.
