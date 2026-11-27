# Explication détaillée du projet — Deepfake Adaptive Fusion

> Ce document explique le projet à un enseignant, avec un langage simple.
> Il insiste particulièrement sur les trois interfaces (frontend, API, MLflow)
> et sur le rôle de **MLflow sur le port 5000**.

---

## 1. Le problème traité

Le projet cherche à répondre à une question :

> « Cette image de visage est-elle une vraie photo, ou une image générée par une IA ? »

C'est donc un problème de **classification binaire** :

| Classe | Signification | Valeur utilisée dans le code |
|---|---|---|
| `real` | vrai visage / photo réelle | `0` |
| `fake` | visage généré ou manipulé (deepfake) | `1` |

Le modèle produit un score, puis on applique une règle de décision :

```
P(fake) = sigmoid(logit)
si P(fake) >= seuil  ->  FAKE
sinon                ->  REAL
```

Le seuil par défaut est `0.5`, mais il peut être calibré sur les données de validation.

**Point important pour la présentation :** ce modèle est un outil d'aide et de recherche.
Il ne doit jamais être présenté comme une preuve absolue qu'une image est authentique.

---

## 2. L'idée principale du modèle

Un deepfake laisse souvent deux types de traces :

1. des traces **spatiales** : forme du visage, texture, contours, incohérences visibles ;
2. des traces **fréquentielles** : motifs répétitifs, grilles, artefacts invisibles à l'œil nu
   mais présents dans le spectre de Fourier.

Le projet utilise donc **deux experts** :

- un réseau convolutif classique qui regarde l'image RGB : `spatial_net` ;
- un petit réseau qui regarde la transformée de Fourier de l'image : `freq_net` ;
- une **porte adaptative** (`fusion_gate`) qui décide, pour chaque image, quel expert croire
  davantage.

Schéma simplifié :

```
Image RGB 224x224
      |
      +------------------------------+
      |                              |
      v                              v
spatial_net (ConvNeXt-T)      FFT + freq_net
      |                              |
      v                              v
projection spatiale           projection fréquentielle
      |                              |
      +--------------+---------------+
                     |
                     v
              fusion_gate (poids)
                     |
                     v
                classifier
                     |
                     v
          logit  +  w_spatial  +  w_frequency
```

Formule utilisée en mode `fusion` :

```
z = p_s + alpha * (w_s * p_s + w_f * p_f)
```

- `p_s` = représentation spatiale ;
- `p_f` = représentation fréquentielle ;
- `w_s` et `w_f` = poids donnés par la porte, avec `w_s + w_f = 1` ;
- `alpha` = échelle résiduelle apprise (`residual_scale`).

En clair : le modèle part de l'analyse spatiale, puis ajoute une correction pondérée
par les deux points de vue. La porte peut donc préférer l'image ou le spectre selon le cas.

### Pourquoi la FFT est calculée en dehors du réseau ONNX

La transformée de Fourier est calculée par NumPy avec :

```
freq_input = log1p(|fftshift(fft2(image))|)
```

Elle n'est pas incluse dans le graphe ONNX, car `torch.fft.fft2` s'exporte mal en ONNX.
Le modèle exporté possède donc :

- **2 entrées** : `image` et `freq_input` ;
- **3 sorties** : `logit`, `w_spatial`, `w_frequency`.

C'est un contrat important : si on modifie l'export ou l'inférence, il faut conserver
cette signature.

---

## 3. Les données

Le dataset utilisé est :

```
xhlulu/140k-real-and-fake-faces  (Kaggle)
```

Il contient des splits officiels `train`, `valid`, `test`.

Pour rester dans un budget de calcul raisonnable (notamment sur Colab), le projet
sous-échantillonne de façon **déterministe** :

- `10 000` images par classe pour l'entraînement ;
- `2 000` par classe pour la validation ;
- `5 000` par classe pour le test ;
- graine fixe `seed = 42`.

Ces valeurs sont dans `params.yaml`.

### Protection contre les fuites de données

Une fuite de données, c'est par exemple la même image qui se retrouve à la fois dans
l'entraînement et dans le test. Le modèle aurait alors l'impression d'être bon sans
vraiment généraliser.

Le projet met en place plusieurs protections :

- utilisation des splits officiels ;
- sous-échantillonnage stratifié par classe ;
- comparaison MD5 des contenus entre les splits ;
- pour les vidéos, regroupement par vidéo/identité afin de ne pas mettre des images
  d'une même vidéo des deux côtés ;
- les jeux externes (Celeb-DF v2, FF-C23, CelebA, LFW) ne servent **jamais** à
  l'entraînement ; ils servent seulement à tester la généralisation.

Ces protections sont testées dans `tests/test_no_leakage.py`.

---

## 4. L'entraînement

Le fichier de configuration principal est `params.yaml`. Il pilote :

- les données ;
- l'entraînement ;
- l'évaluation ;
- l'export ONNX ;
- le suivi MLflow.

### Recette d'entraînement

- Optimiseur : **AdamW**.
- Plusieurs taux d'apprentissage différenciés (backbone, fusion, classifieur).
- Planificateur : **CosineAnnealingLR**.
- Mixed precision (AMP) quand un GPU est disponible.
- Perte : BCE avec label smoothing, ou Focal Loss en option.
- Early stopping sur le F1 de validation.
- Augmentations avancées : RandAugment, MixUp, CutMix (en option).

### Les variantes comparées

Le projet ne se contente pas du modèle final : il compare plusieurs architectures.

| Run | Mode | Description |
|---|---|---|
| R1 | `spatial_only` | uniquement le flux spatial (baseline) |
| R2 | `freq_only` | uniquement le flux fréquentiel |
| R3 | `concat_no_gate` | concaténation/moyenne fixe des deux flux, sans porte |
| R4 | `fusion` | fusion adaptative, modèle final |

Puis des ablations :

| Run | Ablation |
|---|---|
| A1 | sans RandAugment / MixUp / CutMix |
| A2 | Focal Loss au lieu de BCE |
| A3 | backbone gelé pendant 3 époques puis dégel progressif |
| A4 | sans AMP (FP32) |

Ces runs sont définis dans `experiments.yaml` et lancés par :

```bash
python scripts/run_experiments.py --config params.yaml
```

### Sorties de l'entraînement

- `models/AdaptiveFusion_best_model.pth` : meilleur checkpoint ;
- `models/model_config.json` : architecture, backbone, seuil, métadonnées ;
- `models/training_history.csv` : métriques par époque ;
- `reports/train_summary_*.json` : résumé du run.

---

## 5. L'évaluation

Le script principal est :

```bash
python -m src.models.evaluate --config params.yaml
```

Il calcule notamment :

| Métrique | Signification simple |
|---|---|
| Accuracy | pourcentage global de bonnes réponses |
| Balanced accuracy | moyenne des performances sur `real` et `fake` |
| Precision | parmi les images déclarées fake, combien le sont vraiment |
| Recall | parmi les vrais fakes, combien ont été détectés |
| F1 | compromis entre précision et rappel |
| AUC | capacité à bien classer, indépendamment du seuil |
| EER | taux d'erreur quand faux positifs et faux négatifs sont égaux |
| Matrice de confusion | détail des erreurs real/fake |

Le projet ajoute aussi :

- **calibration du seuil** sur la validation (`threshold_calibration.json`) ;
- **robustesse** : compression JPEG (75/50/30), flou (σ=1/2), redimensionnement
  (×0.5/×0.25) ;
- **cross-dataset** : test sur des jeux externes, quand ils sont présents ;
- **benchmark** : paramètres, MACs, latence ;
- **Grad-CAM** : cartes d'attention ;
- **comparaison ONNX** : FP32, FP16, INT8.

Les résultats sont écrits dans `reports/` et envoyés à MLflow.

---

## 6. Les composants MLOps du projet

Le projet ne contient pas seulement un modèle : il industrialise tout le cycle de vie.

| Composant | Rôle dans le projet |
|---|---|
| Git | versionne le code, les tests, la configuration |
| DVC | versionne les données et les gros artefacts (poids, ONNX) |
| MLflow | suit les expériences, les métriques, les paramètres et les artefacts |
| Prefect | orchestre le pipeline (lance les étapes dans le bon ordre) |
| Prometheus | collecte les métriques du système et du modèle |
| Grafana | affiche les métriques dans des tableaux de bord visuels |
| Evidently AI | détecte quand les données changent trop (data drift) |
| ONNX | format portable du modèle pour l'inférence |
| FastAPI | expose le modèle sous forme d'API HTTP |
| React + Vite | interface utilisateur de démonstration |
| Docker Compose | lance tous les services ensemble |
| GitHub Actions | CI (tests), CD (déploiement), CT (réentraînement) |
| Pytest / Vitest | tests backend et frontend |

### Le pipeline DVC

Défini dans `dvc.yaml` :

```
prepare -> train -> evaluate -> export_onnx
```

- `prepare` : télécharge et prépare un sous-ensemble du dataset ;
- `train` : entraîne le modèle et produit le checkpoint ;
- `evaluate` : calcule les métriques, la calibration, la robustesse ;
- `export_onnx` : exporte le modèle en ONNX et vérifie la parité PyTorch/ONNX.

Commande principale :

```bash
dvc repro
```

DVC ne remplace pas MLflow :

- **DVC** répond à : « quelles données et quels poids ont produit ce résultat ? »
- **MLflow** répond à : « quelles expériences ont été faites, avec quels paramètres,
  quelles métriques et quels artefacts ? »

Les deux sont complémentaires.

### L'orchestration avec Prefect

DVC lance les étapes dans le bon ordre, mais il ne gère pas bien les erreurs complexes
ni les reprises. C'est là qu'intervient **Prefect** (`src/orchestration/`).

Prefect enveloppe chaque étape du pipeline dans une **tâche** avec des règles :

```
Tâche              | Essais max | Délai entre essais
-------------------|------------|-------------------
prepare_data       | 2          | 30 secondes
train_model        | 1          | 60 secondes
evaluate_model     | 1          | 30 secondes
export_onnx        | 2          | 30 secondes
register_model     | 2          | 15 secondes
check_model_quality| 0 (gate)   | —
```

**Exemple simple** : si le téléchargement du dataset échoue à cause d'une panne réseau,
Prefect attend 30 secondes puis réessaye. Si ça échoue encore, il réessaye une dernière
fois. Après 3 échecs, la tâche est marquée comme échouée et on peut voir pourquoi dans
l'interface Prefect.

**La tâche `check_model_quality`** est différente : elle n'a **aucun retry**. C'est un
**contrôle qualité** (une « gate »). Si le modèle obtenu a un AUC < 0.85 ou un F1 < 0.80,
le pipeline s'arrête. On ne veut pas promouvoir un mauvais modèle.

Deux flows existent :
- **main_pipeline_flow** : le pipeline complet (prepare → train → evaluate → export → register)
- **retrain_flow** : déclenché quand on détecte que le modèle devient mauvais (voir Monitoring)

Commande pour lancer le pipeline via Prefect :

```bash
python -m src.orchestration.flows --config params.yaml --register
```

### Le monitoring (Prometheus + Grafana + Evidently AI)

En production, il faut surveiller deux choses :
1. **Le système** : est-ce que le serveur va bien ? Combien de requêtes par seconde ? Quelle latence ?
2. **Le modèle** : est-ce que les prédictions sont toujours fiables ? Les données n'ont-elles pas changé ?

#### Prometheus (métriques système et modèle)

Prometheus est un outil qui **collecte des chiffres** à intervalles réguliers.
Le backend expose un endpoint `/metrics` qui donne tous ces chiffres au format texte.

Voici ce qu'on mesure :

| Ce qu'on mesure | Pourquoi c'est important |
|---|---|
| Nombre de requêtes HTTP | Savoir si l'API est utilisée |
| Temps de réponse | Détecter les ralentissements |
| Utilisation CPU et mémoire | Savoir si le serveur sature |
| Nombre de prédictions FAKE/REAL | Comprendre l'usage |
| Confiance des prédictions | Détecter si le modèle est trop sûr ou trop incertain |
| Latence d'inférence | Vérifier que le modèle reste rapide |
| Poids de fusion (spatial/fréquentiel) | Comprendre quel flux est privilégié |
| Score de data drift | Savoir si les données ont changé |

#### Grafana (tableaux de bord visuels)

Grafana **lit les données de Prometheus** et les affiche dans des graphiques.
Accessible sur `http://localhost:3001` (identifiants : admin / admin).

On y voit :
- Combien de requêtes par minute
- Le temps de réponse (p95 = les 95% les plus rapides)
- Le nombre de prédictions FAKE vs REAL
- L'utilisation CPU et mémoire avec des jauges colorées (vert → jaune → rouge)
- La distribution des confiances
- Les poids de fusion spatial vs fréquentiel
- Le score de dérive des données

#### Evidently AI (détection de data drift)

Le **data drift**, c'est quand les données en production deviennent différentes de celles
utilisées à l'entraînement. Par exemple, si le modèle a été entraîné sur des photos de
bonne qualité et qu'en production il reçoit des images très compressées, ses prédictions
deviennent moins fiables.

Evidently AI **compare** les données récentes aux données d'entraînement avec des tests
statistiques (KS test, PSI). Si le score de dérive dépasse 0.3 (30% des features ont
dérivé), on sait qu'il faut probablement **réentraîner le modèle**.

Commande pour vérifier le drift :

```bash
curl -X POST http://localhost:8000/monitoring/drift
```

#### Prediction Logger

Chaque prédiction faite par l'API est **enregistrée** dans un fichier `monitoring_data/predictions.jsonl`.
Ce fichier contient :
- La date et l'heure
- Le résultat (label, confiance)
- Les poids de fusion
- La latence
- La version du modèle

Ces logs servent à :
- Analyser a posteriori la distribution des prédictions
- Détecter le drift avec Evidently AI
- Avoir une trace d'audit

### La Continuous Integration (CI)

La CI, c'est **tester automatiquement** le code à chaque modification.

Deux workflows GitHub Actions s'exécutent à chaque push ou pull request :

**`ci-tests.yml`** — Tests de qualité du code :
- `black --check` : vérifie que le code est bien formaté
- `flake8` : vérifie qu'il n'y a pas d'erreurs de style
- `pytest backend` : lance les tests du backend avec couverture ≥ 80%
- `pytest tests` : lance les tests d'anti-fuite et de contrat ONNX
- `npm run lint && npm test && npm run build` : vérifie le frontend

**`docker-build.yml`** — Test de la stack Docker :
- Construit les images Docker
- Démarre la stack complète
- Vérifie que `/health` répond 200
- Lance une prédiction de test
- Vérifie que le frontend est accessible

**En résumé** : si un développeur casse quelque chose, la CI le détecte immédiatement.

### La Continuous Delivery/Deployment (CD)

Le CD, c'est **déployer automatiquement** quand le code est validé.

Le workflow `cd-deploy.yml` fait 4 étapes :

```
1. BUILD         → Construit les images Docker et les pousse sur GitHub Container Registry
2. STAGING       → Déploie sur l'environnement de test, lance des smoke tests
3. PRODUCTION    → Déploie sur l'environnement réel (après validation humaine)
4. ROLLBACK      → Si la production échoue, revient automatiquement à la version précédente
```

**Exemple de scénario** :
1. Un développeur pousse du code sur `main`
2. La CI teste le code → tout passe ✅
3. Le CD construit les images Docker
4. Le CD déploie en staging → smoke tests ✅
5. Le CD déploie en production (avec validation humaine)
6. Si quelque chose plante → rollback automatique

**En résumé** : on peut déployer en un clic, et si ça se passe mal, on revient en arrière automatiquement.

### La Continuous Training (CT)

Le CT, c'est **réentraîner automatiquement** le modèle quand il devient moins bon.

Le workflow `continuous-training.yml` fait ceci :

```
1. CHECK DRIFT    → Vérifie si les données ont changé ou si les métriques ont baissé
2. RETRAIN        → Si nécessaire, relance le pipeline complet (Prefect)
3. VALIDATE       → Vérifie que le nouveau modèle est au moins aussi bon
4. PROMOTE        → Enregistre le nouveau modèle dans MLflow Registry
5. DEPLOY         → Lance le CD pour déployer le nouveau modèle
```

**Quand le CT se déclenche-t-il ?**
- **Automatiquement** : tous les lundis à 6h du matin (vérification de routine)
- **Manuellement** : un développeur peut lancer `gh workflow run continuous-training.yml`
- **Par webhook** : si le monitoring détecte un drift important, il peut déclencher le CT

**En résumé** : le modèle se met à jour tout seul quand c'est nécessaire, sans intervention manuelle.

---

## 7. Les interfaces et les ports

Le projet expose plusieurs services. Il est très important de ne pas les confondre.

| Port | Service | Rôle | Adresse |
|---|---|---|---|
| `3000` | Frontend React (nginx) | interface utilisateur pour importer une image | http://localhost:3000 |
| `8000` | Backend FastAPI | API de prédiction, santé, métadonnées, Grad-CAM, **monitoring** | http://localhost:8000/docs |
| `5000` | MLflow | interface de suivi des expériences | http://localhost:5000 |
| `9090` | Prometheus | collecte et stockage des métriques | http://localhost:9090 |
| `3001` | Grafana | tableaux de bord visuels (admin/admin) | http://localhost:3001 |
| `5173` | Vite (dev uniquement) | serveur de développement frontend | http://localhost:5173 |

### Qui parle à qui ?

```text
Navigateur
   |
   |  http://localhost:3000
   v
Frontend React (nginx)
   |
   |  /api/...  ->  proxy nginx
   v
Backend FastAPI :8000
   |
   +--> fichier ONNX local : models/deepfake_fusion.onnx
   +--> métadonnées du modèle : models/model_config.json
   +--> métriques : reports/metrics.json
   +--> PyTorch pour Grad-CAM (chargé seulement si demandé)
   +--> Prediction Logger (enregistre chaque prédiction)
   +--> /metrics (expose les métriques à Prometheus)

Prometheus :9090  <-- scrape les métriques de :8000 toutes les 10 secondes
Grafana :3001     <-- lit les données de Prometheus, affiche les dashboards
MLflow :5000      <-- outil séparé, utilisé pendant l'entraînement/évaluation
```

### Différence essentielle

- Le **frontend 3000** est l'interface pour l'utilisateur final : il dépose une image
  et voit REAL/FAKE.
- Le **backend 8000** fait le calcul réel et expose aussi Swagger + les métriques.
- **MLflow 5000** n'est pas l'application de détection. C'est le « carnet de laboratoire »
  du projet : il montre les expériences, les courbes, les métriques et les fichiers produits.
- **Prometheus 9090** collecte les chiffres de performance du système et du modèle.
- **Grafana 3001** affiche ces chiffres dans des graphiques jolis et faciles à lire.

---

## 8. MLflow sur le port 5000 : rôle détaillé

C'est la partie la plus importante pour la présentation.

### 8.1 Qu'est-ce que MLflow ?

MLflow est une plateforme de **suivi d'expériences de machine learning**.

Quand on entraîne un modèle, on veut savoir :

- quels paramètres ont été utilisés ?
- quel code / quel commit Git ?
- quelles métriques ont été obtenues ?
- quels fichiers ont été produits ?
- quel modèle est le meilleur ?
- peut-on comparer plusieurs essais ?

MLflow stocke et affiche toutes ces informations dans une interface web.

### 8.2 Pourquoi le port 5000 ?

Dans `docker-compose.yml`, le service MLflow est défini ainsi :

```yaml
mlflow:
  image: ghcr.io/mlflow/mlflow:v2.17.2
  command: >
    mlflow server
    --host 0.0.0.0
    --port 5000
    --backend-store-uri sqlite:////mlflow/mlflow.db
    --default-artifact-root /mlflow/artifacts
  ports:
    - "5000:5000"
  volumes:
    - mlflow-data:/mlflow
```

Cela signifie :

- un conteneur MLflow démarre ;
- il écoute sur le port **5000** à l'intérieur du réseau Docker ;
- le port est publié sur la machine hôte : `http://localhost:5000` ;
- la base de données MLflow est un fichier SQLite dans le volume Docker `mlflow-data` ;
- les artefacts (fichiers, images, modèles) sont stockés dans `/mlflow/artifacts`.

Le volume `mlflow-data` permet de conserver l'historique même si le conteneur est recréé.
Attention :

- `docker compose down` conserve le volume ;
- `docker compose down -v` supprime le volume et donc l'historique MLflow.

### 8.3 Ce que MLflow stocke

Pendant l'entraînement, `src/tracking.py` envoie à MLflow :

#### Paramètres

- configuration d'entraînement : epochs, batch size, learning rates, backbone, mode ;
- configuration des données : taille d'image, nombre d'images par classe, seed ;
- empreinte du dataset (`data.fingerprint`) ;
- environnement matériel ;
- description du modèle.

#### Tags

- `git_sha` : le commit Git utilisé ;
- `git_branch` : la branche ;
- `author` : auteur du run ;
- `axis` : axe de comparaison ou ablation ;
- `mode` : `fusion`, `spatial_only`, etc. ;
- `backbone` : nom du backbone.

#### Métriques

- par époque : loss, accuracy, F1, balanced accuracy, métriques de validation ;
- à la fin : meilleur F1, nombre d'époques ;
- à l'évaluation : accuracy, F1, AUC, EER, latence, paramètres, MACs, seuil.

#### Artefacts

- le checkpoint PyTorch ;
- `training_history.csv` ;
- les courbes d'entraînement ;
- `model_config.json` ;
- `metrics.json` ;
- les figures d'évaluation ;
- les exemples Grad-CAM ;
- le modèle ONNX ;
- le tableau de quantification FP32/FP16/INT8.

### 8.4 Ce qu'on voit dans l'interface MLflow

Quand on ouvre `http://localhost:5000`, on trouve généralement :

1. **La liste des expériences** à gauche.
   L'expérience principale s'appelle `deepfake-adaptive-fusion`.

2. **La table des runs**.
   Chaque ligne correspond à un entraînement : `R1_spatial_only`, `R2_freq_only`,
   `R3_concat_no_gate`, `R4_fusion`, `A1_baseline_augmentation`, etc.

3. **La comparaison des runs**.
   On peut sélectionner plusieurs runs et comparer accuracy, F1, AUC, EER, latence.

4. **Le détail d'un run**.
   - *Overview* : paramètres, tags, statut ;
   - *Metrics* : courbes par époque ;
   - *Artifacts* : fichiers produits ;
   - *Model* : modèle enregistré, si disponible.

5. **Le Model Registry**.
   Il permet de versionner un modèle et de lui donner un alias comme `staging` ou
   `production`.

### 8.5 Comment les runs arrivent dans MLflow

Le code d'entraînement utilise `src/tracking.py`.

Exemple conceptuel :

```python
with tracking.run(cfg, run_name="R4_fusion") as mlflow:
    tracking.log_params(cfg, {...})
    tracking.log_metrics(cfg, {...})
    tracking.log_artifact(cfg, "models/AdaptiveFusion_best_model.pth")
```

MLflow enregistre alors le run dans le store configuré.

Deux cas sont possibles :

#### Cas A — MLflow local (sans Docker)

Si `MLFLOW_TRACKING_URI` n'est pas définie, le projet utilise :

```
<projet>/mlruns/mlflow.db
```

C'est une base SQLite locale. On peut la consulter avec :

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Par défaut, cette commande utilise aussi le port 5000. Il faut donc éviter de la lancer
en même temps que le service MLflow Docker, ou choisir un autre port avec `--port 5001`.

#### Cas B — MLflow Docker (port 5000)

Pour que les runs apparaissent dans l'interface Docker sur `http://localhost:5000`,
il faut que l'entraînement envoie ses données à ce serveur :

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m src.models.train --config params.yaml
```

Dans Docker, le backend reçoit aussi `MLFLOW_TRACKING_URI=http://mlflow:5000`.
Mais attention : **l'entraînement n'est pas un service Docker** dans ce projet.
Il est lancé depuis la machine hôte. Le nom `mlflow` fonctionne donc à l'intérieur
du réseau Docker, tandis que depuis l'hôte on utilise `localhost:5000`.

### 8.6 Point de transparence important

Dans l'état actuel du code :

- le backend FastAPI charge directement le fichier local
  `models/deepfake_fusion.onnx` ;
- le backend ne va pas chercher automatiquement un modèle « Production » dans MLflow ;
- les fonctions de registre (`register_onnx_model`, `promote`) existent dans
  `src/tracking.py`, mais elles ne sont pas appelées automatiquement par le pipeline
  d'entraînement/export ;
- le Model Registry de MLflow peut donc être vide ou alimenté manuellement.

Autrement dit :

> **MLflow sert à suivre, comparer et archiver les expériences.**
> **Le serving des prédictions se fait avec le fichier ONNX local sur le port 8000.**

C'est une distinction honnête et importante à expliquer à l'enseignant. Les documents
du projet décrivent une promotion de modèle via MLflow comme évolution prévue, mais
le chemin d'inférence actuel reste local.

### 8.7 Si MLflow est vide le jour de la démonstration

C'est normal si aucun entraînement n'a encore été lancé avec
`MLFLOW_TRACKING_URI=http://localhost:5000`.

Le dépôt ne contient pas les poids ni les données : `models/` et `reports/` sont
volontairement ignorés par Git. Sans entraînement réel, MLflow n'a aucun run à afficher.

Deux options pour la démonstration :

1. expliquer que MLflow est prêt à recevoir les runs, et montrer l'interface vide ;
2. si le dataset est disponible, lancer un petit entraînement de démonstration :

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m src.models.train --config params.yaml --quick-demo
```

Le mode `--quick-demo` réduit à 500 images par classe en entraînement (200 en
validation/test) et 2 époques.

### 8.8 Ce qu'il faut retenir pour l'oral

Phrase simple à retenir :

> « MLflow sur le port 5000 est le tableau de bord de nos expériences.
> Il ne fait pas la prédiction. Il montre quel entraînement a été lancé, avec quels
> paramètres, quelles métriques et quels fichiers. Cela permet de comparer les modèles
> et de rendre le travail reproductible. L'application de démonstration, elle, est sur
> le port 3000, et l'API de prédiction sur le port 8000. »

---

## 9. Le backend FastAPI (port 8000)

Le backend est dans `backend/app/`.

### Endpoints principaux

| Endpoint | Méthode | Rôle |
|---|---|---|
| `/health` | GET | état du service et du modèle |
| `/model/metadata` | GET | fiche modèle, seuil, métriques, limites |
| `/predict` | POST | prédiction REAL/FAKE d'une image |
| `/predict/visualize` | POST | carte Grad-CAM en PNG ou JSON/base64 |
| `/metrics` | GET | **métriques Prometheus** (CPU, mémoire, prédictions, latence…) |
| `/monitoring/stats` | GET | **statistiques des prédictions** loguées |
| `/monitoring/drift` | POST | **lancer l'analyse de dérive des données** |
| `/monitoring/log-prediction` | POST | **logger manuellement une prédiction** |
| `/docs` | GET | documentation Swagger interactive |
| `/redoc` | GET | documentation alternative |

**Note** : les endpoints en gras sont les nouveaux endpoints de monitoring. Ils sont
essentiels pour le suivi en production.

### Fonctionnement de `/predict`

1. reçoit une image (JPEG, PNG, WebP ou BMP) ;
2. vérifie la taille et le format ;
3. redimensionne en 224×224 et normalise comme ImageNet ;
4. calcule la FFT ;
5. exécute le modèle ONNX ;
6. applique le seuil ;
7. **enregistre la prédiction** dans le monitoring (Prometheus + fichier JSONL) ;
8. renvoie :

```json
{
  "label": "real ou fake",
  "prob_fake": 0.87,
  "prob_real": 0.13,
  "fusion_weights": {
    "spatial": 0.62,
    "frequency": 0.38
  },
  "latency": {
    "preprocess_ms": 2.1,
    "inference_ms": 15.4,
    "total_server_ms": 17.5
  }
}
```

**Important pour le monitoring** : à chaque appel de `/predict`, le backend :
- incrémente le compteur `deepfake_predictions_total` (par label FAKE/REAL) ;
- enregistre la confiance dans l'histogramme `deepfake_prediction_confidence` ;
- enregistre la latence dans `deepfake_prediction_latency_seconds` ;
- enregistre les poids de fusion dans `deepfake_fusion_weight_spatial` et `_frequency` ;
- écrit la prédiction dans `monitoring_data/predictions.jsonl`.

Tout cela se fait automatiquement, sans ralentir la réponse. Le monitoring ne peut
jamais casser l'inférence.

### Grad-CAM

`/predict/visualize` est différent : il a besoin des gradients, donc il utilise PyTorch.
Pour ne pas ralentir le démarrage, PyTorch n'est chargé qu'au premier appel de
visualisation.

C'est pour cela que :

- `/predict` est rapide et utilise ONNX Runtime ;
- `/predict/visualize` est plus lent au premier appel et utilise PyTorch.

### Validation des images

Le backend ne se fie pas au nom du fichier. Il décode réellement l'image avec PIL.
Cela permet de refuser un fichier texte renommé en `.jpg`.

Codes d'erreur prévus :

- `400` : image vide, invalide ou corrompue ;
- `413` : fichier trop volumineux (10 Mo par défaut) ;
- `415` : format non supporté ;
- `422` : aucun fichier reçu ;
- `503` : modèle non chargé ou Grad-CAM indisponible.

---

## 10. Le frontend React (port 3000)

Le frontend est dans `frontend/`. Il propose :

- import par glisser-déposer ;
- galerie d'exemples ;
- webcam ;
- affichage du verdict REAL/FAKE ;
- barres de probabilité ;
- poids de fusion spatial/fréquentiel ;
- latence client et serveur ;
- Grad-CAM avec choix de la classe expliquée et de l'opacité ;
- panneau « fiche modèle » avec seuil, métriques et avertissements ;
- thème clair/sombre.

Il interroge l'API via des appels relatifs (`/health`, `/predict`, etc.).

En développement, Vite redirige `/api` vers `http://localhost:8000`.
En Docker, nginx reçoit `/api/...` et retire le préfixe avant de transmettre au backend.

---

## 11. Docker et les cinq services

`docker compose up --build` lance :

| Service | Image / construction | Port | Rôle |
|---|---|---|---|
| `frontend` | build React + nginx | 3000 → 80 | interface utilisateur |
| `backend` | build Python + FastAPI | 8000 → 8000 | API de prédiction + monitoring |
| `mlflow` | image officielle MLflow | 5000 → 5000 | suivi des expériences |
| `prometheus` | image officielle Prometheus | 9090 → 9090 | collecte des métriques |
| `grafana` | image officielle Grafana | 3001 → 3000 | tableaux de bord visuels |

Points importants :

- le backend monte `models/` et `reports/` en lecture seule ;
- le frontend attend que le backend soit en bonne santé (`depends_on`) ;
- MLflow possède un volume persistant `mlflow-data` ;
- Prometheus possède un volume persistant `prometheus-data` ;
- Grafana possède un volume persistant `grafana-data` ;
- le backend et MLflow sont sur le même réseau Docker : le backend peut joindre
  `http://mlflow:5000` ;
- Prometheus scrape automatiquement `http://backend:8000/metrics` toutes les 10 secondes ;
- Grafana est pré-configuré pour lire les données de Prometheus ;
- depuis le navigateur de la machine hôte, on utilise `http://localhost:5000` (MLflow),
  `http://localhost:9090` (Prometheus) et `http://localhost:3001` (Grafana).

Commandes utiles :

```bash
cd /workspaces/MLOPS-NLP_Project/exigence_pdf

docker compose up --build -d
docker compose ps
curl http://localhost:8000/health
curl http://localhost:8000/metrics          # métriques Prometheus
curl http://localhost:8000/monitoring/stats # statistiques des prédictions
docker compose logs -f
docker compose down        # conserve les données
docker compose down -v     # supprime aussi les volumes (MLflow, Prometheus, Grafana)
```

---

## 12. ONNX et le déploiement

Le modèle est exporté dans plusieurs variantes :

| Fichier | Description |
|---|---|
| `models/deepfake_fusion.onnx` | modèle FP32 standard |
| `models/deepfake_fusion_fp16.onnx` | poids en demi-précision |
| `models/deepfake_fusion_int8.onnx` | quantification dynamique INT8 |

L'export vérifie aussi la **parité PyTorch ↔ ONNX** : les deux doivent produire
des scores proches.

Règle de prudence du projet :

> Ne pas déployer l'INT8 si la perte d'AUC dépasse 1 point.

Commandes :

```bash
python -m src.models.evaluate --config params.yaml --export-onnx --quantize
# ou
python -m src.models.onnx_export --config params.yaml
```

Attention : le message affiché par `scripts/setup_demo.sh` mentionne `--export`,
mais l'option réelle est `--export-onnx`. C'est une petite incohérence de texte,
pas un problème de fonctionnement.

---

## 13. Tests et intégration continue

### Backend

```bash
pytest backend -q --cov=backend/app --cov-report=term-missing --cov-fail-under=80
```

Le backend est testé avec un **faux modèle ONNX** de même signature que le vrai.
Cela permet de tester toute l'API sans télécharger 132 Mo de poids.

### Tests racine

```bash
pytest tests -q -m "not slow"
```

Ils couvrent notamment :

- l'anti-fuite de données ;
- le contrat ONNX 2 entrées / 3 sorties ;
- la parité réelle, marquée `slow`, quand les vrais artefacts sont présents.

### Frontend

```bash
cd frontend
npm run lint
npm test
npm run build
```

### CI GitHub Actions

- `ci-tests.yml` : black, flake8, tests backend avec couverture ≥ 80 %,
  tests racine, lint/tests/build frontend.
- `docker-build.yml` : construit les images, démarre la stack, vérifie `/health`,
  lance une prédiction et vérifie que le frontend répond.

---

## 14. Où se trouve quoi dans le dépôt

Le vrai projet est dans `exigence_pdf/`, pas à la racine du dépôt.

```
exigence_pdf/
├── .github/workflows/
│   ├── ci-tests.yml            # CI : lint + tests + couverture
│   ├── docker-build.yml        # CI : build Docker + smoke test
│   ├── cd-deploy.yml           # CD : build → staging → production → rollback
│   └── continuous-training.yml # CT : drift check → retrain → validate → deploy
│
├── src/
│   ├── config.py               # configuration typée depuis params.yaml
│   ├── tracking.py             # helpers MLflow (log params/métriques, register, promote)
│   ├── data/                   # dataset, préparation, anti-fuite, transforms
│   ├── models/                 # architecture, train, evaluate, ONNX, metrics
│   ├── orchestration/          # Prefect : pipeline complet + retrain, retries, erreurs
│   │   ├── flows.py            # les flows (main_pipeline_flow, retrain_flow)
│   │   └── deployment.py       # configuration des déploiements Prefect
│   ├── monitoring/             # monitoring système et modèle
│   │   ├── metrics.py          # collecteur Prometheus (CPU, RAM, prédictions, etc.)
│   │   ├── drift.py            # détection de drift avec Evidently AI + fallback KS/PSI
│   │   ├── prediction_logger.py# logger JSONL des prédictions
│   │   └── api.py              # endpoints /metrics, /monitoring/*
│   └── utils/                  # chemins, device, Grad-CAM, logs, plots
│
├── backend/
│   ├── app/main.py             # application FastAPI + middleware monitoring
│   ├── app/inference.py        # ONNX Runtime + Grad-CAM
│   ├── app/settings.py         # configuration par variables d'environnement
│   └── tests/                  # tests API avec faux modèle ONNX
│
├── frontend/                   # React + Vite + Tailwind
│
├── monitoring/                 # configuration du monitoring
│   ├── prometheus.yml          # config Prometheus (quoi scraper, à quelle fréquence)
│   └── grafana/                # dashboards Grafana pré-configurés
│       └── provisioning/
│           ├── datasources/
│           └── dashboards/
│
├── monitoring_data/            # logs de prédictions (JSONL, ignoré par Git)
│
├── scripts/                    # download, expériences, démo, smoke tests
├── tests/                      # anti-fuite, contrat ONNX
├── docs/                       # DECISIONS.md, MODEL_CARD.md, SOUTENANCE.md
├── models/                     # poids (ignorés par Git, gérés par DVC)
├── reports/                    # métriques et figures (ignorés par Git)
├── params.yaml                 # source de vérité de la configuration
├── experiments.yaml            # plan des runs R1-R4 et ablations A1-A4
├── dvc.yaml                    # pipeline DVC
└── docker-compose.yml          # frontend + backend + MLflow + Prometheus + Grafana
```

---

## 15. Scénario de démonstration conseillé

### Avant la démonstration

```bash
cd /workspaces/MLOPS-NLP_Project/exigence_pdf
docker compose up --build -d
docker compose ps
curl http://localhost:8000/health
```

Vérifier que :

- `http://localhost:3000` s'ouvre (frontend) ;
- `http://localhost:8000/docs` s'ouvre (API) ;
- `http://localhost:5000` s'ouvre (MLflow) ;
- `http://localhost:9090` s'ouvre (Prometheus) ;
- `http://localhost:3001` s'ouvre (Grafana, admin/admin) ;
- le modèle est chargé (`/health` répond `ok`).

**Attention :** le dépôt ne contient pas les poids. Si
`models/deepfake_fusion.onnx` est absent, le backend répond `503 degraded` et le
frontend ne démarre pas, car `docker-compose.yml` attend que le backend soit
« healthy ». Pour une démonstration d'interface sans vrai modèle :

```bash
python scripts/make_dummy_model.py --out models
docker compose up --build -d
```

Ce modèle factice sert uniquement à faire fonctionner l'interface et les tests.
Ce n'est pas un vrai détecteur de deepfakes.

### Pendant la démonstration

1. **Présenter le problème** : real vs fake, pourquoi c'est difficile, pourquoi ce n'est
   pas une preuve.
2. **Montrer l'architecture** : deux flux + porte adaptative.
3. **Montrer le frontend** : glisser une image, commenter le verdict, les probabilités,
   les poids de fusion et la latence.
4. **Montrer Grad-CAM** : expliquer que c'est une indication d'attention, pas une preuve.
5. **Montrer la fiche modèle** : seuil, limites, version.
6. **Montrer l'API** : `/docs`, `/health`, `/predict`.
7. **Montrer MLflow** : expliquer que c'est le suivi des expériences sur le port 5000,
   comparer R1/R2/R3/R4 si les runs existent, montrer les artefacts.
8. **Montrer Prometheus** : aller sur `http://localhost:9090`, taper `deepfake_predictions_total`,
   montrer que les métriques se remplissent au fur et à mesure des prédictions.
9. **Montrer Grafana** : aller sur `http://localhost:3001` (admin/admin), ouvrir le dashboard
   "Deepfake Adaptive Fusion - Overview", montrer les graphiques en temps réel.
10. **Montrer l'orchestration** : expliquer que Prefect lance les étapes du pipeline dans
    le bon ordre, avec des retries en cas d'erreur. Montrer le code dans
    `src/orchestration/flows.py`.
11. **Montrer la CI/CD/CT** : aller sur GitHub Actions, montrer les workflows, expliquer
    que les tests se lancent automatiquement à chaque push, que le déploiement est
    automatisé, et que le réentraînement peut se déclencher tout seul.
12. **Montrer le monitoring de drift** : lancer `curl -X POST http://localhost:8000/monitoring/drift`
    et expliquer que si le score dépasse 0.3, le modèle est automatiquement réentraîné.
13. **Expliquer DVC** : reproductibilité, versionnement des données.
14. **Terminer par les limites** : généralisation, biais, seuil, pas de déploiement
    critique sans revue humaine.

Phrases de transition utiles :

> « Le port 3000 montre le résultat à l'utilisateur. Le port 8000 fait le calcul.
> Le port 5000 garde la mémoire scientifique du projet : les expériences, les métriques
> et les fichiers produits. Les ports 9090 et 3001 surveillent la santé du système
> et du modèle en temps réel. »

> « Quand on fait une prédiction, non seulement on obtient un résultat, mais on
> l'enregistre aussi. Cela nous permet de détecter si les données changent et de
> savoir quand réentraîner le modèle. C'est ça le monitoring. »

---

## 16. Limites et précautions

Le projet documente lui-même ses limites :

- le modèle généralise mal aux générateurs absents de l'entraînement
  (StyleGAN3, diffusion récente, etc.) ;
- il peut apprendre des artefacts de compression plutôt que la vraie sémantique du visage ;
- le dataset 140k est surtout composé de StyleGAN et présente un risque de biais
  démographique ;
- le seuil `0.5` n'est pas forcément le meilleur selon le coût des faux positifs
  et des faux négatifs ;
- Grad-CAM est un indice d'attention, pas une explication causale ;
- l'INT8 peut dégrader l'AUC ;
- la webcam dépend de la lumière et de la qualité du flux ;
- **le modèle ne doit jamais être la seule preuve d'authenticité d'un contenu**.

Ces limites sont aussi dans `docs/MODEL_CARD.md`.

---

## 17. Questions probables de l'enseignant

### « Pourquoi deux flux ? »

Parce que les deepfakes laissent des traces spatiales (texture, contours) mais aussi
des traces fréquentielles (artefacts de génération, grilles). Les deux signaux sont
complémentaires.

### « Pourquoi une porte adaptative ? »

Pour que le modèle décide image par image quel flux est le plus fiable, au lieu d'utiliser
un poids fixe.

### « Pourquoi ONNX ? »

Pour obtenir un modèle portable, rapide, utilisable sans PyTorch au moment de la
prédiction, et quantifiable en FP16/INT8.

### « Pourquoi la FFT est-elle en dehors du modèle ONNX ? »

Parce que l'export ONNX de `torch.fft.fft2` n'est pas fiable. On calcule donc la FFT
avec NumPy avant d'appeler ONNX Runtime.

### « À quoi sert MLflow exactement ? »

À suivre et comparer les expériences : paramètres, métriques, artefacts, versions.
Ce n'est pas le serveur de prédiction. Les prédictions passent par le backend sur le
port 8000.

### « Pourquoi le port 5000 est-il séparé ? »

Parce que MLflow est un outil de développement/MLOps, pas une partie de l'application
utilisateur. On peut le démarrer, l'arrêter ou le consulter indépendamment du frontend.

### « Où sont les vrais poids et les vraies métriques ? »

Ils ne sont pas dans Git. Les poids sont gérés par DVC et les métriques sont générées
par l'évaluation dans `reports/`. Le dépôt ne contient pas de chiffres inventés.

### « Pourquoi le backend peut-il être dégradé ? »

Si `models/deepfake_fusion.onnx` est absent, `/health` répond `503 degraded`.
C'est normal : le dépôt ne versionne pas les poids. Il faut soit les récupérer avec DVC,
soit entraîner, soit utiliser le modèle factice pour une démo d'interface.

### « MLflow montre-t-il le modèle utilisé pour la prédiction ? »

Pas automatiquement dans cette version. MLflow montre les runs et peut stocker le modèle,
mais le backend charge le fichier ONNX local. La promotion automatique via le registre
est une évolution prévue, pas le comportement actuel.

### « À quoi sert Prefet ? Pourquoi ne pas utiliser juste DVC ? »

DVC est très bien pour le versionnement des données et lancer les étapes dans le bon ordre.
Mais il ne gère pas bien les **erreurs complexes** : si une étape échoue, on veut pouvoir
**réessayer automatiquement** un certain nombre de fois, avec un délai entre les essais.
Prefect fait exactement ça. Il ajoute aussi une **interface visuelle** pour voir
l'exécution du pipeline en temps réel, avec les logs de chaque tâche.

**En résumé** : DVC versionne les données et définit le pipeline. Prefect l'exécute avec
de la robustesse (retries, gestion d'erreurs, observabilité).

### « C'est quoi le data drift ? Pourquoi c'est important ? »

Imaginez que vous entraînez un modèle sur des photos de bonne qualité. En production,
les utilisateurs envoient des photos floues, mal éclairées, très compressées.
Le modèle n'a jamais vu ce genre d'images : ses prédictions deviennent moins fiables.

C'est ça le **data drift** : les données en production deviennent différentes de celles
de l'entraînement. Le modèle ne « sait pas » qu'il devient moins bon. C'est pour ça
qu'on le surveille avec Evidently AI. Si on détecte un drift, on réentraîne le modèle
avec les nouvelles données.

### « Comment fonctionne le réentraînement automatique (CT) ? »

C'est un workflow GitHub Actions qui se déclenche quand :
- le monitoring détecte un data drift important (score > 0.3) ;
- les métriques du modèle baissent en dessous d'un seuil ;
- un développeur le lance manuellement.

Le workflow relance le pipeline d'entraînement (via Prefect), vérifie que le nouveau
modèle est au moins aussi bon que l'ancien, et le déploie automatiquement si tout va bien.

### « À quoi servent Prometheus et Grafana ? »

**Prometheus** est un collecteur de métriques. Il interroge le backend toutes les
10 secondes via `/metrics` et stocke les chiffres dans une base de données temporelle.

**Grafana** lit ces chiffres et les affiche dans des tableaux de bord. C'est plus facile
à lire que des lignes de texte. On voit en un coup d'œil si tout va bien :
- le nombre de requêtes par minute ;
- le temps de réponse ;
- l'utilisation CPU et mémoire ;
- la répartition FAKE/REAL des prédictions.

### « Que se passe-t-il si le déploiement échoue ? »

Le workflow CD a un job de **rollback** automatique. Si le déploiement en production
échoue, il revient à la version précédente. L'utilisateur ne voit rien.

### « Comment vous assurez-vous que le code est de bonne qualité ? »

Plusieurs niveaux :
1. **Tests unitaires** : chaque fonction est testée séparément (couverture ≥ 80%)
2. **Tests d'intégration** : on teste que les composants fonctionnent ensemble
3. **Lint** : le code doit respecter un style uniforme (black, flake8)
4. **CI automatique** : à chaque push, les tests se lancent tout seuls sur GitHub
5. **Smoke test Docker** : on vérifie que la stack complète démarre et répond

---

## 18. Résumé final en quelques phrases

Le projet détecte des visages réels ou générés en combinant une analyse spatiale
et une analyse fréquentielle. Il industrialise ce modèle avec un pipeline MLOps complet :

- **Git + DVC** pour versionner le code, les données et les poids ;
- **MLflow** pour suivre les expériences et gérer le registre de modèles ;
- **Prefect** pour orchestrer le pipeline avec gestion d'erreurs et retries ;
- **Prometheus + Grafana** pour surveiller le système et le modèle en temps réel ;
- **Evidently AI** pour détecter quand les données changent (data drift) ;
- **ONNX** pour l'inférence rapide et portable ;
- **FastAPI** pour l'API de prédiction ;
- **React** pour l'interface utilisateur ;
- **Docker Compose** pour lancer tous les services ensemble ;
- **GitHub Actions** pour la CI (tests), le CD (déploiement) et le CT (réentraînement).

L'interface **MLflow sur le port 5000** est le tableau de bord des expériences.
**Prometheus sur le port 9090** collecte les métriques. **Grafana sur le port 3001**
les affiche dans des graphiques. **Prefect** orchestre le pipeline. **GitHub Actions**
automatise les tests, le déploiement et le réentraînement.

Tout cela forme un **cycle de vie complet** : du développement du modèle à son
déploiement en production, en passant par le monitoring et le réentraînement
automatique quand nécessaire.
