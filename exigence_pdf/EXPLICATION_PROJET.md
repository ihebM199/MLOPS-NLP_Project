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
| ONNX | format portable du modèle pour l'inférence |
| FastAPI | expose le modèle sous forme d'API HTTP |
| React + Vite | interface utilisateur de démonstration |
| Docker Compose | lance frontend + backend + MLflow ensemble |
| GitHub Actions | lint, tests, couverture, build frontend, smoke test Docker |
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

---

## 7. Les interfaces et les ports

Le projet expose plusieurs services. Il est très important de ne pas les confondre.

| Port | Service | Rôle | Adresse |
|---|---|---|---|
| `3000` | Frontend React servi par nginx | interface utilisateur pour importer une image | http://localhost:3000 |
| `8000` | Backend FastAPI | API de prédiction, santé, métadonnées, Grad-CAM | http://localhost:8000/docs |
| `5000` | MLflow | interface de suivi des expériences | http://localhost:5000 |
| `5173` | Vite | serveur de développement frontend uniquement | http://localhost:5173 |

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

MLflow :5000  <-- outil séparé, utilisé pendant l'entraînement/évaluation
```

### Différence essentielle

- Le **frontend 3000** est l'interface pour l'utilisateur final : il dépose une image
  et voit REAL/FAKE.
- Le **backend 8000** fait le calcul réel et expose aussi Swagger.
- **MLflow 5000** n'est pas l'application de détection. C'est le « carnet de laboratoire »
  du projet : il montre les expériences, les courbes, les métriques et les fichiers produits.

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
| `/docs` | GET | documentation Swagger interactive |
| `/redoc` | GET | documentation alternative |

### Fonctionnement de `/predict`

1. reçoit une image (JPEG, PNG, WebP ou BMP) ;
2. vérifie la taille et le format ;
3. redimensionne en 224×224 et normalise comme ImageNet ;
4. calcule la FFT ;
5. exécute le modèle ONNX ;
6. applique le seuil ;
7. renvoie :

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

## 11. Docker et les trois services

`docker compose up --build` lance :

| Service | Image / construction | Port | Rôle |
|---|---|---|---|
| `frontend` | build React + nginx | 3000 → 80 | interface utilisateur |
| `backend` | build Python + FastAPI | 8000 → 8000 | API de prédiction |
| `mlflow` | image officielle MLflow | 5000 → 5000 | suivi des expériences |

Points importants :

- le backend monte `models/` et `reports/` en lecture seule ;
- le frontend attend que le backend soit en bonne santé (`depends_on`) ;
- MLflow possède un volume persistant `mlflow-data` ;
- le backend et MLflow sont sur le même réseau Docker : le backend peut joindre
  `http://mlflow:5000` ;
- depuis le navigateur de la machine hôte, on utilise `http://localhost:5000`.

Commandes utiles :

```bash
cd /workspaces/MLOPS-NLP_Project/exigence_pdf

docker compose up --build -d
docker compose ps
curl http://localhost:8000/health
docker compose logs -f
docker compose down        # conserve les données MLflow
docker compose down -v     # supprime aussi le volume MLflow
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
├── src/
│   ├── config.py               # configuration typée depuis params.yaml
│   ├── tracking.py             # helpers MLflow
│   ├── data/                   # dataset, préparation, anti-fuite, transforms
│   ├── models/                 # architecture, train, evaluate, ONNX, metrics
│   └── utils/                  # chemins, device, Grad-CAM, logs, plots
├── backend/
│   ├── app/main.py             # application FastAPI
│   ├── app/inference.py        # ONNX Runtime + Grad-CAM
│   ├── app/settings.py         # configuration par variables d'environnement
│   └── tests/                  # tests API avec faux modèle ONNX
├── frontend/                   # React + Vite + Tailwind
├── scripts/                    # download, expériences, démo, smoke tests
├── tests/                      # anti-fuite, contrat ONNX
├── docs/                       # DECISIONS.md, MODEL_CARD.md, SOUTENANCE.md
├── models/                     # poids (ignorés par Git, gérés par DVC)
├── reports/                    # métriques et figures (ignorés par Git)
├── params.yaml                 # source de vérité de la configuration
├── experiments.yaml            # plan des runs R1-R4 et ablations A1-A4
├── dvc.yaml                    # pipeline DVC
└── docker-compose.yml          # frontend + backend + MLflow
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

- `http://localhost:3000` s'ouvre ;
- `http://localhost:8000/docs` s'ouvre ;
- `http://localhost:5000` s'ouvre ;
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
8. **Expliquer DVC et la CI** : reproductibilité, tests automatiques, versionnement.
9. **Terminer par les limites** : généralisation, biais, seuil, pas de déploiement
   critique sans revue humaine.

Phrase de transition utile :

> « Le port 3000 montre le résultat à l'utilisateur. Le port 8000 fait le calcul.
> Le port 5000 garde la mémoire scientifique du projet : les expériences, les métriques
> et les fichiers produits. »

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

---

## 18. Résumé final en quelques phrases

Le projet détecte des visages réels ou générés en combinant une analyse spatiale
et une analyse fréquentielle. Il industrialise ce modèle avec DVC pour les données et
les poids, MLflow pour le suivi des expériences, ONNX pour l'inférence, FastAPI pour
l'API, React pour l'interface, Docker pour lancer le tout et GitHub Actions pour la
qualité.

L'interface **MLflow sur le port 5000** est le tableau de bord des expériences :
elle montre les runs, les paramètres, les métriques, les courbes et les artefacts.
Elle ne remplace ni l'application de démonstration sur le port 3000, ni l'API de
prédiction sur le port 8000. C'est un outil de traçabilité et de comparaison, essentiel
pour un projet MLOps sérieux.
