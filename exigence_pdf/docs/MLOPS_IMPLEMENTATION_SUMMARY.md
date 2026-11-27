# Résumé de l'implémentation MLOps

Ce document résume les composants ajoutés pour respecter les exigences du projet MLOps.

---

## ✅ Exigences couvertes

### 1. Présentation générale du projet ✅
- **README.md** : Architecture complète, idée, objectifs, choix techniques justifiés
- **docs/SOUTENANCE.md** : Guide de présentation structuré pour la soutenance
- **docs/DECISIONS.md** : Justifications techniques détaillées

### 2. Versioning du code et des données ✅

#### Git (code source)
- Branches : `main` (protégée), `dev` (intégration)
- Structure claire : `src/`, `backend/`, `frontend/`, `tests/`, `docs/`
- README avec badges CI
- .gitignore complet (data, models, monitoring_data, etc.)

#### DVC (données et artefacts)
- `dvc.yaml` : Pipeline reproductible (prepare → train → evaluate → export_onnx)
- `params.yaml` : Configuration centralisée
- `experiments.yaml` : Configuration des expériences
- Remote DVC configurable (local/S3/GCS)
- **Reproductibilité** : `dvc repro` relance tout le pipeline

### 3. Expérimentations avec MLflow ✅
- `src/tracking.py` : Intégration MLflow complète
- Paramètres enregistrés : backbone, mode, epochs, lr, augmentation, etc.
- Métriques enregistrées : accuracy, F1, AUC, EER, latence, calibration
- Expériences : R1 (spatial), R2 (freq), R3 (concat), R4 (fusion)
- Ablations : RandAugment, MixUp, Focal Loss, gel/dégel
- Comparaison des runs dans l'UI MLflow

### 4. Model Registry ✅
- `src/tracking.py` : `register_onnx_model()`, `promote()`
- Gestion des aliases : `staging`, `production`
- Backend charge le modèle "Production" depuis MLflow
- Fallback sur fichier local si MLflow indisponible
- Versioning des modèles avec métadonnées

### 5. Orchestration des pipelines ✅ **NOUVEAU**

#### Fichiers créés
- `src/orchestration/__init__.py`
- `src/orchestration/flows.py` : Flows Prefect avec gestion d'erreurs
- `src/orchestration/deployment.py` : Configuration des déploiements Prefect

#### Fonctionnalités
- **main_pipeline_flow** : prepare → train → evaluate → quality_gate → export → register
- **retrain_flow** : Déclenché par drift ou dégradation
- **Gestion des erreurs** : Retries par tâche (2x pour prepare, 1x pour train, 0x pour quality gate)
- **Mécanismes de reprise** : Prefect permet de relancer depuis l'échec
- **Observabilité** : UI Prefect avec logs en temps réel, artifacts JSON

#### Utilisation
```bash
python -m src.orchestration.flows --config params.yaml --register
python -m src.orchestration.flows --retrain --trigger data_drift
```

### 6. Continuous Integration (CI) ✅

#### Workflows GitHub Actions
- `ci-tests.yml` : Lint (black, flake8) + Tests (pytest ≥80%, vitest) + Build
- `docker-build.yml` : Build Docker + Smoke test (/health, /predict)

#### Tests mis en place
- **Unitaires** : validation API, prétraitement, mapping labels
- **Intégration** : anti-fuite (MD5), parité ONNX
- **Couverture** : ≥80% obligatoire
- **Frontend** : lint, unit tests, production build

### 7. Continuous Delivery/Deployment (CD) ✅ **NOUVEAU**

#### Fichier créé
- `.github/workflows/cd-deploy.yml`

#### Pipeline CD
1. **Build** : Images Docker → GitHub Container Registry
2. **Staging** : Déploiement automatique sur `main`, smoke tests
3. **Production** : Déploiement après validation (environnement protégé)
4. **Rollback** : Restauration automatique si échec

#### Utilisation
```bash
gh workflow run cd-deploy.yml -f environment=production
```

### 8. Continuous Training (CT) ✅ **NOUVEAU**

#### Fichier créé
- `.github/workflows/continuous-training.yml`

#### Fonctionnalités
- **Schedule** : Vérification hebdomadaire (lundi 6h UTC)
- **Triggers** : Data drift, dégradation métriques, dispatch externe
- **Pipeline** : check_drift → retrain → validate → promote → deploy

#### Utilisation
```bash
gh workflow run continuous-training.yml -f trigger=data_drift
```

### 9. Monitoring ✅ **NOUVEAU**

#### Fichiers créés
- `src/monitoring/__init__.py`
- `src/monitoring/metrics.py` : Collecteur Prometheus (système + modèle)
- `src/monitoring/drift.py` : Détection de drift (Evidently AI + fallback KS/PSI)
- `src/monitoring/prediction_logger.py` : Logger JSONL des prédictions
- `src/monitoring/api.py` : Endpoints FastAPI (/metrics, /monitoring/*)
- `monitoring/prometheus.yml` : Configuration Prometheus
- `monitoring/grafana/provisioning/` : Dashboards et datasources Grafana

#### Métriques système (Prometheus)
| Métrique | Type | Description |
|---|---|---|
| `deepfake_http_requests_total` | Counter | Requêtes HTTP |
| `deepfake_http_request_duration_seconds` | Histogram | Latence |
| `deepfake_cpu_usage_percent` | Gauge | CPU |
| `deepfake_memory_usage_bytes` | Gauge | Mémoire |
| `deepfake_predictions_total` | Counter | Prédictions par label |
| `deepfake_prediction_confidence` | Histogram | Distribution confiances |
| `deepfake_prediction_latency_seconds` | Histogram | Latence par stage |
| `deepfake_fusion_weight_spatial` | Histogram | Poids fusion spatial |
| `deepfake_data_drift_score` | Gauge | Score de drift |

#### Métriques modèle (Evidently AI)
- Détection de drift : PSI, KS test, Chi-squared
- Rapport JSON avec score par feature
- Seuil d'alerte : drift_score > 0.3

#### Endpoints
- `GET /metrics` : Exposition Prometheus
- `GET /monitoring/stats` : Statistiques des prédictions
- `POST /monitoring/drift` : Déclencher l'analyse de drift
- `POST /monitoring/log-prediction` : Logger manuellement une prédiction

#### Dashboards Grafana
- HTTP Request Rate et latence p95
- Prediction Rate par label (FAKE/REAL)
- CPU/Memory usage avec alertes
- Fusion Weights (spatial vs frequency)
- Data Drift Score avec seuils

#### Docker Compose
Ajout de deux services :
- **Prometheus** : http://localhost:9090
- **Grafana** : http://localhost:3001 (admin/admin)

---

## 📁 Structure des fichiers ajoutés

```
exigence_pdf/
├── .github/workflows/
│   ├── cd-deploy.yml              # CD : build → staging → prod → rollback
│   └── continuous-training.yml    # CT : drift check → retrain → validate → promote
│
├── src/
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── flows.py               # Prefect flows (pipeline + retrain)
│   │   └── deployment.py          # Configuration déploiements Prefect
│   │
│   └── monitoring/
│       ├── __init__.py
│       ├── metrics.py             # Collecteur Prometheus
│       ├── drift.py               # Détection de drift (Evidently + fallback)
│       ├── prediction_logger.py   # Logger JSONL des prédictions
│       └── api.py                 # Endpoints FastAPI monitoring
│
├── monitoring/
│   ├── prometheus.yml             # Configuration Prometheus
│   └── grafana/
│       └── provisioning/
│           ├── datasources/
│           │   └── prometheus.yml
│           └── dashboards/
│               ├── dashboards.yml
│               └── deepfake-overview.json
│
├── monitoring_data/
│   └── .gitkeep                   # Dossier pour logs de prédictions (ignoré par Git)
│
├── docker-compose.yml             # Ajout : Prometheus + Grafana
├── requirements.txt               # Ajout : prometheus-client, evidently, prefect, psutil
├── requirements-ci.txt            # Ajout : prometheus-client, psutil
├── backend/requirements.txt       # Ajout : prometheus-client, psutil
├── backend/app/main.py            # Ajout : monitoring middleware + prediction logging
├── .gitignore                     # Ajout : monitoring_data/
├── README.md                      # Mise à jour : orchestration, monitoring, CD/CT
└── docs/SOUTENANCE.md             # Mise à jour : guide complet pour la soutenance
```

---

## 🎯 Mapping exigences ↔ implémentation

| Exigence | Implémentation | Fichiers |
|---|---|---|
| **1. Présentation générale** | README complet, architecture, choix justifiés | README.md, docs/SOUTENANCE.md |
| **2. Versioning code** | Git branches, structure, README, .gitignore | .gitignore, README.md |
| **2. Versioning données** | DVC pipeline, params.yaml, reproductibilité | dvc.yaml, params.yaml |
| **3. Expérimentations MLflow** | Tracking params/métriques, comparaison runs | src/tracking.py |
| **4. Model Registry** | register_onnx_model, promote, aliases | src/tracking.py |
| **5. Orchestration** | Prefect flows, retries, gestion erreurs | src/orchestration/flows.py |
| **6. CI** | GitHub Actions, tests unitaires/intégration | .github/workflows/ci-tests.yml |
| **7. CD** | Build Docker, staging → prod, rollback | .github/workflows/cd-deploy.yml |
| **7. CT** | Réentraînement auto sur drift/dégradation | .github/workflows/continuous-training.yml |
| **8. Monitoring système** | Prometheus (CPU, RAM, requêtes) | src/monitoring/metrics.py |
| **8. Monitoring modèle** | Evidently AI (drift), prediction logger | src/monitoring/drift.py, prediction_logger.py |
| **8. Outils monitoring** | Prometheus, Grafana, Evidently AI | monitoring/, docker-compose.yml |

---

## 🚀 Commandes clés pour la démonstration

### Démarrer la stack complète
```bash
cd exigence_pdf
docker compose up --build
```

### Services disponibles
- Frontend : http://localhost:3000
- Backend API : http://localhost:8000/docs
- MLflow : http://localhost:5000
- **Prometheus** : http://localhost:9090
- **Grafana** : http://localhost:3001 (admin/admin)

### Pipeline MLOps
```bash
# Lancer le pipeline complet
python -m src.orchestration.flows --config params.yaml --register

# Lancer un réentraînement
python -m src.orchestration.flows --retrain --trigger data_drift
```

### Monitoring
```bash
# Voir les métriques Prometheus
curl http://localhost:8000/metrics

# Statistiques des prédictions
curl http://localhost:8000/monitoring/stats

# Déclencher l'analyse de drift
curl -X POST http://localhost:8000/monitoring/drift
```

### CI/CD/CT
```bash
# Voir les workflows
gh workflow list

# Déclencher un déploiement
gh workflow run cd-deploy.yml -f environment=production

# Déclencher un réentraînement
gh workflow run continuous-training.yml -f trigger=manual
```

---

## 📊 Points forts du projet

1. **Pipeline reproductible** : `dvc repro` ou `python -m src.orchestration.flows`
2. **Observabilité complète** : MLflow (expériences) + Prometheus (système) + Grafana (dashboards)
3. **Monitoring modèle** : Détection de drift automatique avec Evidently AI
4. **Orchestration robuste** : Prefect avec retries, gestion d'erreurs, reprise sur échec
5. **CI/CD/CT automatisé** : GitHub Actions pour tests, déploiement, réentraînement
6. **Traçabilité** : Chaque prédiction logguée, chaque modèle versionné
7. **Documentation** : README complet, guide soutenance, model card, décisions techniques

---

## 📝 Checklist pour la soutenance

- [ ] Démarrer la stack (`docker compose up --build`)
- [ ] Montrer l'interface frontend (drag & drop, galerie, webcam, Grad-CAM)
- [ ] Montrer MLflow UI (expériences, comparaison runs, registry)
- [ ] Montrer Prefect UI (flows, tasks, retries, logs)
- [ ] Montrer Grafana (dashboards temps réel)
- [ ] Montrer Prometheus (métriques brutes)
- [ ] Déclencher une analyse de drift (`POST /monitoring/drift`)
- [ ] Expliquer le pipeline CI/CD/CT (GitHub Actions)
- [ ] Expliquer la gestion des erreurs (retries, rollback)
- [ ] Justifier les choix techniques (Pourquoi FFT ? Pourquoi ONNX ? Pourquoi Prefect ?)

---

## 🔗 Ressources utiles

- **Prefect** : https://docs.prefect.io/
- **Prometheus** : https://prometheus.io/docs/
- **Grafana** : https://grafana.com/docs/
- **Evidently AI** : https://docs.evidentlyai.com/
- **MLflow** : https://mlflow.org/docs/latest/
- **DVC** : https://dvc.org/doc

---

**Projet MLOps complet et prêt pour la soutenance !** 🎉

Toutes les exigences sont couvertes :
- ✅ Présentation générale
- ✅ Versioning code + données
- ✅ Expérimentations MLflow
- ✅ Model Registry
- ✅ Orchestration Prefect
- ✅ CI (GitHub Actions)
- ✅ CD (Docker + déploiement)
- ✅ CT (réentraînement automatique)
- ✅ Monitoring (Prometheus + Grafana + Evidently)
