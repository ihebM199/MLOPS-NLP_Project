# Aide-mémoire soutenance (interne, pas de slides)

> **Projet MLOps — Détection de Deepfakes par Fusion Adaptative**
> Encadrante : **Sonia Gharsalli** — Tek-Up ING-5 SDIA

---

## Démo 5 minutes

1. **Démarrage** (30 s) : `docker compose up --build -d` puis
   `curl localhost:8000/health`. Montrer les 5 services (`docker compose ps`) et
   l'URL frontend `http://localhost:3000`.
2. **Drag & drop** (45 s) : glisser une image réelle, montrer `ProbabilityBars`
   animées, le verdict et `LatencyBadge` (client vs serveur).
3. **Galerie** (45 s) : cliquer une image réelle puis une générée (1 clic),
   comparer les poids de fusion (`FusionWeights`).
4. **Grad-CAM** (60 s) : onglet visualisation, basculer `fake`/`real`, jouer sur
   l'opacité, montrer original/overlay. Expliquer pourquoi PyTorch est chargé
   paresseusement.
5. **Webcam** (45 s) : Start, capture, inférence continue (~1–2 img/s, une seule
   requête en vol), Stop et arrêt propre du flux ; montrer le refus de permission.
6. **Panneau modèle** (30 s) : `/model/metadata` (architecture, seuil, métriques,
   version), avertissement sur les limites.

---

## 1. Présentation générale du projet

### Idée & objectifs
Détecter les deepfakes (visages générés par IA) en combinant deux flux :
- **Spatial** (CNN sur l'image RGB) — détecte les artefacts visuels
- **Fréquentiel** (CNN sur le spectre de Fourier) — détecte les traces spectrales

La **fusion adaptative** (porte apprise) pondère dynamiquement les deux flux.

### Architecture globale
- **Backend** : FastAPI + ONNX Runtime (inférence rapide)
- **Frontend** : React 18 + Vite + Tailwind (interface utilisateur)
- **MLflow** : suivi d'expériences + registre de modèles
- **Prefect** : orchestration du pipeline ML
- **Prometheus + Grafana** : monitoring système et modèle
- **Evidently AI** : détection de data drift
- **Docker Compose** : déploiement unifié

### Choix techniques justifiés
| Choix | Justification |
|---|---|
| ConvNeXt-Tiny | Backbone pré-entraîné ImageNet, bon compromis précision/vitesse |
| FFT | Les générateurs laissent des traces spectrales invisibles en RGB |
| ONNX Runtime | Portabilité CPU/GPU, latence maîtrisée, pas de dépendance PyTorch |
| FastAPI | Async natif, validation Pydantic, documentation OpenAPI auto |
| DVC | Versioning des données volumineuses, reproductibilité du pipeline |
| MLflow | Tracking d'expériences standard, registre de modèles |
| Prefect | Orchestration Python-native, retries, observabilité |
| Prometheus + Grafana | Standard industriel pour le monitoring temps réel |
| Evidently AI | Détection de drift spécialisée ML, intégration Python |

---

## 2. Versioning du code et des données

### Git (code source)
- **Branches** : `main` (protégée, PR + checks requis), `dev` (intégration)
- **Structure** : `src/` (code métier), `backend/`, `frontend/`, `tests/`, `docs/`
- **README** : documentation complète avec badges CI
- **.gitignore** : exclut `data/`, `models/`, `__pycache__`, `.env`, `node_modules/`

### DVC (données et artefacts)
- **Remote** : local `../dvc-remote` (configurable : S3/GCS/Azure)
- **dvc.yaml** : pipeline reproductible (prepare → train → evaluate → export_onnx)
- **params.yaml** : configuration centralisée (data, train, eval, export, tracking)
- **Reproductibilité** : `dvc repro` rejoue tout le pipeline depuis zéro

```bash
dvc dag          # Visualiser le graphe de dépendances
dvc status       # Vérifier si les étapes sont à jour
dvc repro        # Relancer le pipeline (ou rien si tout est à jour)
dvc metrics show # Afficher les métriques du dernier run
```

### Pipeline reproductible
Une seule commande : `dvc repro` (ou `python -m src.orchestration.flows`)

---

## 3. Expérimentations avec MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

### Expériences réalisées
- **R1** : `spatial_only` — uniquement le flux spatial
- **R2** : `freq_only` — uniquement le flux fréquentiel
- **R3** : `concat_no_gate` — concaténation simple sans porte
- **R4** : `fusion` — fusion adaptative (modèle complet)

### Paramètres enregistrés
- Backbone, mode, epochs, batch_size, learning rates
- Augmentation (RandAugment, MixUp, CutMix)
- Focal loss, label smoothing, weight decay

### Métriques enregistrées
- Accuracy, Precision, Recall, F1, AUC, EER
- Latence (preprocess, inference, total)
- Calibration (ECE), robustesse (JPEG, blur, resize)

### Comparaison des runs
Dans l'UI MLflow : table comparatif avec tri/filtrage, courbes d'entraînement,
artefacts (matrices de confusion, ROC, Grad-CAM, modèles ONNX).

---

## 4. Model Registry

### Modèles enregistrés
- **Nom** : `deepfake-adaptive-fusion`
- **Versions** : une par entraînement réussi
- **Aliases** : `staging` (test), `production` (déployé)

### Gestion (staging, promotion)
```python
# Enregistrer un modèle en staging
from src.tracking import register_onnx_model
register_onnx_model(cfg, "models/deepfake_fusion.onnx", alias="staging")

# Promouvoir en production
from src.tracking import promote
promote(cfg, version="3", alias="production")
```

### Évolution
- Le backend charge le modèle « Production » depuis MLflow
- Fallback sur `models/deepfake_fusion.onnx` si MLflow indisponible
- Chaque version a ses métadonnées (description, paramètres, métriques)

---

## 5. Orchestration des pipelines (Prefect)

```bash
python -m src.orchestration.flows --config params.yaml --register
```

### Interface Prefect
- **UI** : `prefect server start` → http://localhost:4200
- **Flows** : visualiser les exécutions, logs, durée
- **Tasks** : voir chaque étape (prepare, train, evaluate, export, register)

### Gestion des erreurs (dans le code)
```python
# src/orchestration/flows.py
@task(retries=2, retry_delay_seconds=30)
def prepare_data(config_path):
    """Retry 2x si le téléchargement Kaggle échoue."""
    ...

@task(retries=1, retry_delay_seconds=60)
def train_model(config_path):
    """Retry 1x (entraînement coûteux)."""
    ...

@task(retries=0)  # Pas de retry : gate de qualité
def check_model_quality(min_auc=0.85, min_f1=0.80):
    """Échoue le flow si les métriques sont insuffisantes."""
    ...
```

### Mécanismes de reprise
- **Retries automatiques** : chaque tâche a sa politique de retry
- **Flow re-run** : Prefect permet de relancer depuis l'échec
- **Artifacts** : chaque exécution crée un artifact (résumé JSON)
- **Logs** : toutes les erreurs sont loguées avec traceback

---

## 6. Continuous Integration (CI)

### Workflows GitHub Actions
| Workflow | Trigger | Tests |
|---|---|---|
| `ci-tests.yml` | push/PR sur main/dev | black, flake8, pytest (≥80%), vitest, build |
| `docker-build.yml` | push/PR | build Docker + smoke test (/health, /predict) |
| `cd-deploy.yml` | push sur main | build, staging, production, rollback |
| `continuous-training.yml` | schedule/dispatch | drift check, retrain, validate, promote |

### Tests mis en place
- **Unitaires** : validation API, prétraitement, mapping labels, couverture ≥ 80%
- **Intégration** : anti-fuite (MD5), parité ONNX (PyTorch ↔ ONNX)
- **Smoke** : script `smoke_test.sh` vérifie la stack complète
- **Frontend** : lint, unit tests (Vitest), production build

```bash
# Exécuter les tests localement
pytest backend -q --cov=backend/app --cov-fail-under=80
pytest tests -q -m "not slow"
cd frontend && npm test
```

---

## 7. Continuous Delivery/Deployment (CD) et Continuous Training (CT)

### CD (cd-deploy.yml)
1. **Build** : images Docker → GitHub Container Registry
2. **Staging** : déploiement auto sur `main`, smoke tests
3. **Production** : déploiement après validation (environnement protégé)
4. **Rollback** : restauration automatique si échec

```bash
# Déclencher manuellement
gh workflow run cd-deploy.yml -f environment=production
```

### CT (continuous-training.yml)
- **Schedule** : vérification hebdomadaire (lundi 6h UTC)
- **Triggers** : data drift, dégradation métriques, dispatch externe
- **Pipeline** : check_drift → retrain → validate → promote → deploy

```bash
# Déclencher un réentraînement
gh workflow run continuous-training.yml -f trigger=data_drift
```

---

## 8. Monitoring

### Métriques système (Prometheus)
Le backend expose `/metrics` au format Prometheus :

```bash
curl http://localhost:8000/metrics
```

| Métrique | Description |
|---|---|
| `deepfake_http_requests_total` | Requêtes HTTP (method, endpoint, status) |
| `deepfake_http_request_duration_seconds` | Latence des requêtes |
| `deepfake_cpu_usage_percent` | Utilisation CPU |
| `deepfake_memory_usage_bytes` | Mémoire utilisée |
| `deepfake_predictions_total` | Prédictions par label (FAKE/REAL) |
| `deepfake_prediction_confidence` | Distribution des confiances |
| `deepfake_prediction_latency_seconds` | Latence par stage (preprocess, inference, total) |
| `deepfake_fusion_weight_spatial` | Poids spatial de la fusion |
| `deepfake_data_drift_score` | Score de dérive des données |

### Métriques modèle (Evidently AI)
Détection de drift sur les prédictions :

```bash
# Déclencher l'analyse de drift
curl -X POST http://localhost:8000/monitoring/drift

# Résultat JSON : drift_score, n_drifted_features, per-feature analysis
# Si drift_score > 0.3 → alerte + déclenche le CT
```

### Outils utilisés
- **Prometheus** : collecte et stockage des métriques (scrape toutes les 10s)
- **Grafana** : dashboards temps réel (http://localhost:3001, admin/admin)
- **Evidently AI** : détection de drift (PSI, KS test)
- **Prediction Logger** : JSONL pour analyse post-hoc

### Dashboards Grafana
- HTTP Request Rate et latence p95
- Prediction Rate par label (FAKE/REAL par minute)
- CPU/Memory usage avec alertes (vert/jaune/rouge)
- Fusion Weights (spatial vs frequency)
- Data Drift Score avec seuils d'alerte
- Prediction Latency par stage

---

## Parcours MLflow (5 min)

- Ouvrir `http://localhost:5000`, expérience `deepfake-adaptive-fusion`.
- Comparer R1 `spatial_only`, R2 `freq_only`, R3 `concat_no_gate`, R4 `fusion` :
  accuracy, F1, AUC, EER, latence, paramètres.
- Montrer les ablations (RandAugment/MixUp, Focal Loss, gel/dégel, robustesse).
- Registry : modèle `deepfake-adaptive-fusion`, alias `staging` puis
  `production` ; expliquer le fallback local ONNX du backend.
- Ouvrir les artefacts : `training_history.csv`, courbes, matrice de confusion,
  ROC, Grad-CAM, `model_config.json`, ONNX.

## Parcours DVC / CI

- `dvc dag`, `dvc status`, `dvc metrics show`, `dvc repro --force` (quick demo).
- `git ls-files data models` : uniquement `.dvc`/`.gitkeep`.
- GitHub Actions : `ci-tests.yml` (black, flake8, pytest backend ≥ 80 %, vitest,
  build) et `docker-build.yml` (build + smoke `/health` + `/predict`).
- Montrer les badges README et la protection de `main` (PR + checks requis).

## FAQ technique par rôle

### Pourquoi la FFT ?
Les générateurs laissent des traces spectrales (hautes fréquences, grilles de
sur-échantillonnage) que le flux spatial peut manquer. Le flux fréquentiel
apporte un signal complémentaire, et la porte apprend quand lui faire confiance.

### Pourquoi ONNX ?
Portabilité CPU/GPU, latence maîtrisée, déploiement sans dépendance PyTorch pour
`/predict`, quantification FP16/INT8.

### Pourquoi la FFT sort du graphe ?
`torch.fft.fft2` s'exporte mal (opérateurs complexes). Option A : deux entrées,
calcul NumPy côté inférence, graphe simple et parité testée (< 1e-3).

### Quelle fuite de données évitée ?
Même image dans deux splits (MD5), frames d'une même vidéo des deux côtés
(découpage par vidéo/identité), FFHQ dupliqué dans les réels externes. Les
augmentations ne sont appliquées qu'au train, après le split.

### Pourquoi Grad-CAM en PyTorch ?
ONNX Runtime ne fournit pas de gradients. Le backend garde ONNX pour la vitesse
et charge PyTorch uniquement pour `/predict/visualize`.

### Pourquoi un seuil 0.5 ?
Valeur par défaut, mais calibrée sur validation (`threshold_calibration.json`).
Le seuil est exposé dans `/model/metadata` et modifiable par
`DECISION_THRESHOLD` ; le coût métier détermine le compromis.

### Pourquoi INT8 n'est pas toujours déployé ?
La quantification dynamique peut dégrader l'AUC. La règle est : si la perte
dépasse 1 point, on garde FP16/FP32 et on documente le compromis.

### Comment promouvoir un modèle ?
`tracking.register: true` + `tracking.alias: production` dans `params.yaml`,
puis run d'entraînement. Le backend charge « Production » si MLflow est
configuré, sinon le fichier local.

### Pourquoi Prefect pour l'orchestration ?
Python-native (pas de DAG XML comme Airflow), retries intégrées, observabilité
(UI avec logs en temps réel), et intégration naturelle avec le code existant.
Permet de transformer les étapes DVC en flow avec gestion d'erreurs fine.

### Pourquoi Prometheus + Grafana ?
- **Prometheus** : standard industriel, pull-based, stockage efficace des séries temporelles
- **Grafana** : dashboards configurables, alertes, intégration native Prometheus
- Ensemble, ils donnent une vue temps réel du système et du modèle en production.

### Pourquoi Evidently AI pour le drift ?
Spécialisé ML (pas générique), supporte PSI, KS test, Chi-squared, et génère
des rapports HTML. Détection de drift au niveau des features ET des prédictions.

### Comment fonctionne le Continuous Training ?
1. Monitoring détecte un drift (Evidently, score > 0.3) ou une dégradation
2. Webhook/GitHub Dispatch déclenche `continuous-training.yml`
3. Le workflow relance le pipeline (Prefect), valide la qualité, promeut le modèle
4. Le CD déploie automatiquement le nouveau modèle

### Que se passe-t-il si le déploiement échoue ?
Le workflow `cd-deploy.yml` a un job `rollback` qui se déclenche automatiquement
en cas d'échec du déploiement production. Il restaure la version précédente.

### Comment sont loguées les prédictions ?
Chaque prédiction est logguée dans `monitoring_data/predictions.jsonl` avec :
- Timestamp, features (statistiques image), prédiction (label, confiance)
- Poids de fusion, latence, version du modèle
- Ces logs servent à l'analyse de drift et à l'audit.

## Parcours Orchestration (3 min)

- Montrer l'UI Prefect : `prefect server start` puis http://localhost:4200
- Lancer le pipeline : `python -m src.orchestration.flows --config params.yaml`
- Montrer les retries dans le code (`src/orchestration/flows.py`)
- Expliquer le flow `retrain_flow` et son déclenchement

## Parcours Monitoring (3 min)

- Ouvrir Grafana : http://localhost:3001 (admin/admin)
- Montrer le dashboard "Deepfake Adaptive Fusion - Overview"
- Faire quelques prédictions → voir les métriques se mettre à jour
- Montrer `/metrics` (format Prometheus) et `/monitoring/stats`
- Déclencher un drift check : `curl -X POST localhost:8000/monitoring/drift`

## Parcours CI/CD/CT (3 min)

- GitHub Actions : montrer les 4 workflows et leurs badges
- Montrer un run CI (tests + couverture)
- Expliquer le pipeline CD (build → staging → prod → rollback)
- Expliquer le CT (schedule + triggers → retrain → validate → promote)

---

## Points à préparer (manuel)

- Révoquer le jeton Kaggle exposé, puis purger l'historique
  (`scripts/purge_secret_history.sh`) et force-push.
- Configurer le remote DVC (local/S3/GCS) et `dvc push`.
- Lancer l'entraînement GPU complet + `scripts/run_experiments.py`.
- Activer la protection de branche `main` (PR + checks requis).
- Compiler `rapport/main.tex` (TeX Live) et archiver `rapport_technique.pdf`.
- Démarrer Prefect : `prefect server start` et montrer l'UI.
- Vérifier Prometheus et Grafana : `docker compose up` puis http://localhost:3001.
- Tester le drift : faire plusieurs prédictions puis `POST /monitoring/drift`.
