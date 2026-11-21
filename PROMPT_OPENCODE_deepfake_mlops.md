# PROMPT OPENCODE — Mise en conformité du projet « Deepfake Adaptive Fusion » avec le cahier des charges MLOps

> Colle ce document en entier dans OpenCode (ou enregistre-le comme `AGENTS.md` / `PROMPT.md` à la racine du dépôt et demande : « Exécute ce plan phase par phase, en t'arrêtant à chaque checkpoint »).

---

## 0. RÔLE ET RÈGLES DE TRAVAIL

Tu es un ingénieur MLOps / Full-Stack senior. Tu dois transformer un notebook Colab (`Deepfake_AdaptiveFusion_MLOps_officielle.ipynb`) en un **dépôt Git industriel** conforme au cahier des charges du module « Deep Learning Avancé » (Axe 1 : CNN avancé & Transfer Learning + Grad-CAM). Tu es évalué sur : modélisation (6 pts), MLOps (5 pts), full-stack (5 pts), rapport/doc (2 pts), soutenance technique (2 pts).

Règles impératives :
1. Travaille **phase par phase** (section 3). À la fin de chaque phase, lance les commandes de vérification indiquées et affiche leur résultat. Ne passe à la phase suivante que si la vérification passe.
2. **Ne jamais inventer de résultats.** Toute métrique affichée dans le README, le rapport ou le frontend doit venir d'une exécution réelle (MLflow, `metrics.json`, `model_config.json`). Si tu ne peux pas entraîner (pas de GPU), génère le code et un script de reproduction, et marque clairement « à exécuter sur Colab/Kaggle ».
3. **Aucun secret dans le code ni dans Git.** Voir phase 0.
4. **Aucun fichier lourd dans Git** (données, `.pth`, `.onnx`, checkpoints, `mlruns/`) : tout passe par DVC et `.gitignore`.
5. Respecte **exactement** l'arborescence imposée (section 2) et les noms de fichiers.
6. Conserve la compatibilité avec les poids existants : noms de modules `spatial_net`, `freq_net`, `spatial_projection`, `frequency_projection`, `fusion_gate`, `classifier`, `residual_scale` ; convention d'étiquettes **real = 0, fake = 1** ; règle de décision `logit > 0 ⇒ FAKE`.
7. Code propre : PEP8, `black`, `flake8`, type hints, docstrings courtes ; ESLint côté frontend. Commits atomiques (Conventional Commits : `feat:`, `fix:`, `ci:`, `docs:`, `chore:`), branches `main` / `dev` / `feature/*`.
8. Si une décision technique est ambiguë, choisis l'option la plus simple qui respecte le cahier des charges et note-la dans `docs/DECISIONS.md`.

---

## 1. DIAGNOSTIC DE L'EXISTANT (résultat de l'audit du notebook)

### 1.1 Ce qui est déjà bon (à réutiliser)
- Architecture `AdaptiveFusionDetector` (EfficientNetV2-RW-S + flux FFT + porte adaptative), ~31,8 M de paramètres.
- Pipeline PyTorch : transforms train/eval séparés (pas d'augmentation sur valid/test), `FaceDataset`, `LABEL2ID` explicite, seed global.
- Entraînement : AdamW à LR différenciés, cosine, label smoothing, grad-clip, AMP, early stopping sur F1, `last_checkpoint.pt`, reprise.
- Évaluation : accuracy, balanced accuracy, précision, rappel, F1, AUC, EER, matrice de confusion, ROC, analyse d'erreurs.
- Interprétabilité : Grad-CAM PyTorch, analyse des poids de fusion.
- Benchmark latence / mémoire / MACs, `model_config.json`, manifest, model card.

### 1.2 Écarts bloquants par rapport au PDF

| # | Exigence du PDF | État actuel | Gravité |
|---|---|---|---|
| 1 | Dépôt Git avec arborescence imposée (`src/`, `backend/`, `frontend/`, `.github/`…) | Un seul notebook | Bloquant |
| 2 | **DVC** (données + checkpoints) | Absent | Bloquant |
| 3 | **MLflow** tracking + Model Registry (Production/Staging) | Absent (seulement CSV/JSON locaux) | Bloquant |
| 4 | Export **ONNX** + inférence ONNX Runtime | Absent | Bloquant |
| 5 | **API FastAPI** (`/health`, `/model/metadata`, `/predict`, `/predict/visualize`) + Pydantic + tests | Absent (une classe `DeepfakePredictor` dans le notebook) | Bloquant |
| 6 | **Frontend React/Angular** (drag & drop, galerie, webcam, barres animées, latence) | Absent | Bloquant |
| 7 | **Docker** multi-stage + `docker-compose up --build` | Absent | Bloquant |
| 8 | **CI/CD GitHub Actions** (lint, pytest, build) | Absent | Bloquant |
| 9 | Tests **pytest** | Absents | Bloquant |
| 10 | Rapport **LaTeX** + README d'installation 1-clic | Absents | Majeur |
| 11 | **Analyse comparative baseline vs modèle avancé** (critère 1 du barème) | Absente (un seul modèle) | Majeur |
| 12 | Techniques attendues pour l'Axe 1 : MixUp/CutMix/RandAugment, quantification FP16/INT8 | Absentes (augmentation basique + RandomErasing ; FP16 seulement pour AMP) | Majeur |
| 13 | Rendu Grad-CAM **dans l'API et dans le frontend** | Grad-CAM seulement dans le notebook | Majeur |

### 1.3 Problèmes de sécurité / qualité détectés dans le notebook
1. **SECRET EXPOSÉ** : une cellule écrit un jeton Kaggle en clair (`KGAT_ebb9…`) dans `~/.kaggle/access_token`. Le notebook affirme pourtant « jamais d'identifiants dans le code ». **Action manuelle obligatoire pour l'utilisateur : révoquer ce jeton sur kaggle.com → Settings → API, puis en créer un nouveau.** Dans le code, ne jamais recopier ce jeton (ni dans un commit, ni dans le README, ni dans tes réponses).
2. Le fichier `.ipynb` pèse **~21 Mo** à cause des sorties embarquées (images, graphiques) → à nettoyer (`nbstripout`) avant tout commit.
3. **Bloc 3.3b (mélange multi-sources) commenté et cassé** : il référence `cfg.use_all_datasets`, `cfg.max_per_source`, `cfg.extra_sources`, `cfg.valid_fraction` qui n'existent pas dans `Config`. Les datasets 2 à 6 (CelebA, Celeb-DF v2, FF-C23, FFHQ, LFW) sont téléchargés mais **jamais utilisés**.
4. **Risque de fuite de données (Data Leakage)** si on active ce mélange : le jeu « 140k real-and-fake-faces » utilise des visages réels issus de **FFHQ**, et le dataset 5 est aussi **FFHQ** → des images du test 140k peuvent se retrouver dans le train. De plus, les frames de vidéos (Celeb-DF, FF-C23) d'une même vidéo/identité ne doivent jamais être réparties entre train et test.
5. **Provenance des poids** : `AdaptiveFusion_best_model.pth` vient d'un autre notebook (Kaggle) et le `kaggle_kernel` configuré pointe vers une variante **RegNet**, pas EfficientNet. Les résultats de référence (Acc 99,12 %, test externe 83,87 %) sont « rapportés par le notebook source » et **ne sont pas reproduits par l'équipe**. Le barème exige des runs MLflow réels → l'équipe doit **réentraîner** (au moins une fois) et tracer ses propres runs.
6. Faiblesse connue du modèle (à documenter honnêtement) : généralisation limitée à des générateurs non vus (test externe : ~31 % de faux non détectés).
7. Pas de gestion de version des dépendances (`%pip install` sans versions) → créer des `requirements.txt` épinglés.

---

## 2. ARBORESCENCE CIBLE (à respecter scrupuleusement)

```
deepfake-adaptive-fusion/
|-- .github/workflows/
|   |-- ci-tests.yml
|   `-- docker-build.yml
|-- data/                      # versionné via DVC (jamais dans Git)
|   |-- raw/
|   `-- processed/
|-- models/                    # checkpoints .pth / .onnx → DVC
|-- src/
|   |-- __init__.py
|   |-- data/dataset.py        # FaceDataset, transforms, DataLoaders, split anti-fuite
|   |-- models/
|   |   |-- architecture.py    # AdaptiveFusionDetector + baselines + ablations
|   |   |-- train.py           # entraînement + tracking MLflow
|   |   `-- evaluate.py        # métriques + export ONNX + quantification
|   `-- utils/gradcam.py       # Grad-CAM (+ Score-CAM optionnel)
|-- backend/
|   |-- Dockerfile
|   |-- requirements.txt
|   |-- app/{main.py, schemas.py, inference.py}
|   `-- tests/{conftest.py, test_api.py, test_inference.py}
|-- frontend/
|   |-- Dockerfile
|   |-- package.json
|   `-- src/
|       |-- components/        # Dropzone, Webcam, GradCamCanvas, ProbabilityBars, FusionWeights, LatencyBadge, ExampleGallery
|       |-- services/api.js
|       `-- App.jsx
|-- notebooks/                 # notebook nettoyé (nbstripout), usage exploratoire uniquement
|-- docs/                      # DECISIONS.md, MODEL_CARD.md, captures
|-- rapport/                   # sources LaTeX → rapport_technique.pdf
|-- docker-compose.yml
|-- dvc.yaml  params.yaml  data.dvc  .dvcignore
|-- .gitignore  .env.example  .pre-commit-config.yaml
|-- requirements.txt           # dépendances entraînement (src/)
|-- README.md
`-- rapport_technique.pdf
```

---

## 3. PLAN D'EXÉCUTION PAR PHASES

### PHASE 0 — Sécurité et hygiène (à faire en premier)
1. Créer le dépôt, `.gitignore` (données, `*.pth`, `*.onnx`, `*.pt`, `mlruns/`, `mlartifacts/`, `.env`, `node_modules/`, `__pycache__/`, `.ipynb_checkpoints/`, `dvc` cache si nécessaire).
2. Copier le notebook dans `notebooks/`, **supprimer la cellule contenant le jeton Kaggle** (et les cellules `!kaggle competitions list` inutiles), puis nettoyer les sorties : `pip install nbstripout && nbstripout notebooks/*.ipynb` (le fichier doit passer de ~21 Mo à quelques centaines de Ko).
3. Créer `.env.example` (`KAGGLE_USERNAME=`, `KAGGLE_KEY=`, `MLFLOW_TRACKING_URI=`, `MODEL_PATH=`…) ; lire les secrets uniquement via variables d'environnement.
4. Ajouter `pre-commit` (black, flake8, nbstripout, `detect-secrets` ou `gitleaks`) pour empêcher tout futur secret/gros fichier.
5. Vérifier : `git log -p | grep -i "KGAT_"` ne doit rien retourner ; `git ls-files | xargs du -ch | tail -1` doit rester petit.

### PHASE 1 — Refactorisation du notebook vers `src/` (Jalon 1-2)
1. `src/data/dataset.py` : constantes (`LABEL2ID`, `ID2LABEL`), `build_dataframe`, `FaceDataset`, transforms (train/eval), `make_loader`. Conserver le fallback « image illisible → image noire » mais **loguer un warning avec compteur** (ne pas masquer silencieusement).
2. `src/models/architecture.py` : `AdaptiveFusionDetector` (inchangé côté noms de modules) + variantes pour l'étude comparative (voir phase 3) via un paramètre `mode` : `fusion` (modèle final), `spatial_only` (baseline), `freq_only`, `concat_no_gate`. Fonction `build_model(cfg)`.
3. `src/models/train.py` : script CLI (`python -m src.models.train --config params.yaml`), paramètres lus dans `params.yaml` (hyperparamètres, tailles de sous-ensembles, seed). Reprendre la recette du notebook (AdamW LR différenciés, cosine, label smoothing, grad-clip, AMP, early stopping sur F1, reprise `last_checkpoint.pt`).
4. `src/models/evaluate.py` : métriques (accuracy, balanced acc, P/R/F1, AUC, EER), matrice de confusion, ROC, distribution des scores, analyse d'erreurs, poids de fusion par classe ; écrit `metrics.json` + figures dans `reports/`.
5. `src/utils/gradcam.py` : Grad-CAM sur la dernière feature map du flux spatial (cible `fake` ou `real`), retourne `(image_rgb, heatmap)` et une fonction `overlay(image, heatmap, alpha)` qui produit un PNG.
6. Le notebook allégé de `notebooks/` doit **importer** `src/` (pas de duplication de code).
7. Vérification : `python -m src.models.train --quick-demo` (500 img/classe, 2 époques) s'exécute de bout en bout ; `flake8 src` et `black --check src` OK.

### PHASE 2 — Données, anti-fuite et DVC (Jalon 1)
1. Dataset principal : `xhlulu/140k-real-and-fake-faces` dans `data/raw/` (téléchargement scripté via `scripts/download_data.py` qui lit les secrets dans l'environnement).
2. **Anti-fuite** : utiliser les splits officiels train/valid/test ; transformations de normalisation/augmentation **uniquement** via les transforms (jamais appliquées avant le split) ; sous-échantillonnage déterministe (seed). Écrire un test qui vérifie qu'aucun chemin/hash d'image n'apparaît dans deux splits (`tests/test_no_leakage.py`, hash MD5 des fichiers).
3. Datasets additionnels (CelebA, LFW, FFHQ, Celeb-DF v2, FF-C23) : **ne pas les mélanger dans l'entraînement par défaut**. Les utiliser uniquement comme **jeu de test externe (cross-dataset)** pour mesurer la généralisation. Règles : exclure FFHQ des reals externes (déjà présent dans le 140k) ; pour les vidéos, découper par vidéo/identité, jamais par frame ; dédupliquer par hash. Si tu supprimes ce volet, supprime aussi le code mort du bloc 3.3b.
4. DVC : `dvc init` ; `dvc add data/raw` (et `data/processed`) ; `dvc add models/` pour les checkpoints ; configurer un remote (dossier local `../dvc-remote` par défaut, avec instructions pour Google Drive/S3 dans le README) ; commiter les `.dvc`, `.dvcignore`, `dvc.yaml`.
5. `dvc.yaml` avec les stages `prepare` → `train` → `evaluate` → `export_onnx`, `params.yaml` pour les paramètres, `metrics.json` déclaré comme métrique DVC (`dvc metrics show`).
6. Vérification : `dvc status` propre ; `dvc repro` rejoue le pipeline (en `quick_demo`) ; `git ls-files data models` ne liste que des `.dvc`/`.gitkeep`.

### PHASE 3 — Expérimentation, rigueur scientifique et MLflow (Jalon 2) — 6 pts + 5 pts
1. **MLflow** : ajouter un service (voir phase 6) et, dans `train.py`, `mlflow.set_experiment("deepfake-adaptive-fusion")`. Pour chaque run, logger :
   - **params** : backbone, mode, lr par groupe, batch size, epochs, label smoothing, weight decay, taille d'image, tailles de sous-ensembles, seed, version des données (hash DVC) ;
   - **métriques par époque** : train/val loss, accuracy, F1, AUC ;
   - **métriques finales** : test accuracy, balanced acc, précision, rappel, F1, AUC, EER, latence, MACs, nb de paramètres ;
   - **artefacts** : `training_history.csv`, courbes, matrice de confusion, ROC, `model_config.json`, checkpoint, ONNX, 4 exemples Grad-CAM ;
   - **tags** : git commit SHA, auteur, axe.
2. **Étude comparative obligatoire (baseline vs avancé)** — au minimum 4 runs nommés et comparables (mêmes données, même seed) :
   - R1 : baseline `spatial_only` (EfficientNetV2 seul, sans FFT) ;
   - R2 : `freq_only` ;
   - R3 : `concat_no_gate` (fusion sans porte adaptative) ;
   - R4 : `fusion` (modèle final).
   Produire un tableau comparatif (accuracy, F1, AUC, EER, latence, paramètres) en **in-distribution** (test 140k) **et en cross-dataset** (test externe). Conclure honnêtement : si la fusion n'apporte rien en cross-dataset, le dire.
3. **Techniques avancées attendues pour l'Axe 1** (chaque technique = un run d'ablation tracé) :
   - augmentation avancée : **RandAugment** + **MixUp/CutMix** (appliqués au **train uniquement**) ; comparer avec la recette actuelle ;
   - perte : comparer BCE + label smoothing vs **Focal Loss** (même si les classes sont équilibrées, documenter pourquoi/pas d'effet) ;
   - fine-tuning sélectif : comparer backbone gelé → dégel progressif vs LR différenciés actuels ;
   - robustesse : évaluer sous compression JPEG / flou / redimensionnement (les deepfakes réels sont souvent recompressés) et rapporter la dégradation.
4. **Model Registry** : enregistrer le meilleur modèle sous le nom `deepfake-adaptive-fusion` ; lui donner l'alias/stage **Staging** puis **Production** après validation (MLflow ≥ 2.9 : utiliser les *aliases* `staging` / `production` ; sinon `transition_model_version_stage`). Le backend doit pouvoir charger le modèle « Production » (voir phase 4), avec fallback sur un fichier local ONNX.
5. **Seuil de décision** : calibrer et documenter le seuil (par défaut `P(fake) ≥ 0,5`) ; tracer la courbe précision/rappel ; exposer le seuil dans `/model/metadata`.
6. Vérification : l'UI MLflow montre ≥ 4 runs comparables, un modèle enregistré en `Production`, et des artefacts consultables.

### PHASE 4 — Export ONNX, quantification, optimisation (Jalon 3)
Point technique critique : **`torch.fft.fft2` ne s'exporte pas proprement en ONNX.** Choisir l'une de ces deux solutions, la documenter dans `docs/DECISIONS.md` et la tester :
- **Option A (recommandée, simple)** : sortir le calcul `log1p(|fftshift(fft2(x))|)` du graphe. Le modèle ONNX prend alors **deux entrées** (`image` normalisée, `freq_input` précalculé), le calcul FFT est fait côté inférence avec NumPy (`np.fft.fft2`), strictement identique à l'entraînement.
- **Option B** : réimplémenter la DFT 2D par multiplications matricielles (matrices DFT 224×224 réelles/imaginaires, `|F| = sqrt(re² + im²)`) dans un module exportable, et vérifier la parité numérique avec la version `torch.fft`.

Étapes :
1. `evaluate.py --export-onnx` : exporter vers `models/deepfake_fusion.onnx` (opset ≥ 17, axe batch dynamique), avec **3 sorties** : `logit`, `w_spatial`, `w_frequency`.
2. **Test de parité** : sur ≥ 64 images, `max|logit_torch − logit_onnx| < 1e-3` et 100 % de concordance des classes prédites. Ce test est obligatoire et fait partie de pytest.
3. **Quantification** : produire (a) ONNX **FP16** (conversion de poids), (b) ONNX **INT8** (quantification dynamique `onnxruntime.quantization`, ou statique avec calibration sur ~200 images de valid). Pour chaque variante, mesurer : taille du fichier, latence CPU (batch 1, médiane et p95 sur 100 itérations après warm-up), accuracy/F1/AUC sur le test. Présenter le compromis dans un tableau (PyTorch FP32 vs ONNX FP32 vs FP16 vs INT8). **Si INT8 dégrade trop l'AUC (> 1 point), ne pas le mettre en production** et le documenter.
4. **Grad-CAM et ONNX** : le Grad-CAM nécessite des gradients, impossible avec ONNX Runtime. Le backend utilise donc : ONNX Runtime pour `/predict` (rapide) et le modèle PyTorch (CPU, chargé paresseusement au premier appel) pour `/predict/visualize`. Documenter ce choix et sa latence.
5. Benchmark : reprendre le benchmark du notebook (params, MACs, latence, mémoire) dans un script `src/models/benchmark.py` qui logge vers MLflow.

### PHASE 5 — Backend FastAPI (Jalon 3) — spécifications du PDF
Fichiers : `backend/app/main.py`, `schemas.py`, `inference.py`, `backend/tests/`.

1. `inference.py` : classe `ONNXPredictor` (chargement unique au démarrage via *lifespan*), prétraitement identique à `eval_transform` (resize 224×224, ImageNet mean/std, RGB, float32), calcul du flux FFT (option A), sortie `{label, prob_fake, prob_real, logit, w_spatial, w_frequency, inference_ms}`. Classe `GradCamService` (PyTorch, chargement paresseux) pour la visualisation.
2. `schemas.py` — modèles **Pydantic stricts** : `PredictionResponse` (`label: Literal["real","fake"]`, `prob_real`, `prob_fake` ∈ [0,1], `confidence`, `fusion_weights{spatial, frequency}`, `latency{inference_ms, preprocess_ms, total_server_ms}`, `model_version`), `HealthResponse` (`status`, `model_loaded`, `gpu_available`, `model_version`, `onnxruntime_providers`), `ModelMetadata` (nom, architecture, backbone, taille d'entrée, classes `{0:real, 1:fake}`, règle de décision, seuil, métriques de validation/test et cross-dataset issues de `metrics.json`/MLflow, date d'entraînement, commit SHA), `ErrorResponse`.
3. Endpoints **obligatoires** :
   - `GET /health` → état, présence GPU, version du modèle ;
   - `GET /model/metadata` → caractéristiques et métriques ;
   - `POST /predict` (`UploadFile`) → `PredictionResponse` ;
   - `POST /predict/visualize` → image annotée (PNG ou JSON base64) avec **carte Grad-CAM superposée** + prédiction. Paramètres : `target` (`fake`|`real`), `alpha`.
4. **Asynchrone** : endpoints `async def` ; le calcul bloquant (ONNX/PyTorch) est exécuté via `run_in_threadpool` / `anyio.to_thread` pour ne pas bloquer la boucle d'événements.
5. **Gestion d'erreurs** : `400` fichier absent / non-image / image corrompue (valider par décodage PIL, pas seulement le content-type) ; `413` fichier trop gros (limite configurable, ex. 10 Mo) ; `415` format non supporté ; `422` validation Pydantic ; `503` modèle non chargé (avant que le *lifespan* ait fini ou si le fichier ONNX manque). Corps d'erreur homogène (`ErrorResponse`).
6. **CORS** configuré (origines via variable d'environnement), middleware de log (méthode, route, durée, code) en JSON, header `X-Process-Time-ms`.
7. Swagger : `http://localhost:8000/docs` doit être complet — `summary`, `description`, `response_model`, `responses={400:…, 503:…}`, et **exemples** (`json_schema_extra`) pour chaque schéma.
8. Configuration par variables d'environnement (`pydantic-settings`) : `MODEL_PATH`, `PYTORCH_CHECKPOINT_PATH`, `MAX_UPLOAD_MB`, `CORS_ORIGINS`, `MLFLOW_TRACKING_URI` (optionnel).
9. **Tests pytest** (`TestClient`, aucune dépendance à des poids réels : `conftest.py` génère un **mini modèle ONNX factice** de même signature) :
   - `/health` : 200 et champs attendus ; 503 quand le modèle n'est pas chargé ;
   - `/predict` : image valide → 200 + schéma valide + probabilités sommant à 1 ; fichier texte → 400 ; image tronquée → 400 ; fichier trop gros → 413 ; absence de fichier → 422 ;
   - `/predict/visualize` : retourne une image PNG décodable de la bonne taille ;
   - prétraitement : shape/dtype/plage de valeurs, ordre RGB ;
   - mapping des étiquettes (`logit > 0` ⇒ `fake`) ;
   - parité PyTorch ↔ ONNX (marqué `@pytest.mark.slow`, exécuté si un vrai modèle est présent) ;
   - couverture ≥ 80 % sur `backend/app`.
10. Vérification : `pytest backend -q --cov=backend/app` vert ; `uvicorn backend.app.main:app` puis `curl` sur les 4 endpoints.

### PHASE 6 — Frontend React (Vite + TailwindCSS) (Jalon 4) — 5 pts
Le PDF pénalise explicitement une « interface coquille vide » : il faut un **rendu graphique interactif**.

1. Stack : React 18 + Vite + TailwindCSS (ou shadcn/ui), Axios ou `fetch`, ESLint + Prettier.
2. `services/api.js` : `predict(file)`, `predictVisualize(file, opts)`, `getHealth()`, `getMetadata()` ; mesure de la **latence côté client** (`performance.now()`), gestion des erreurs HTTP avec messages clairs, URL de l'API via `import.meta.env.VITE_API_URL`.
3. **Modes d'ingestion (les trois sont obligatoires)** :
   - `Dropzone` : glisser-déposer + sélecteur de fichier + aperçu, validation côté client (type, taille) ;
   - `ExampleGallery` : galerie de ≥ 8 images d'exemple préchargées (4 réelles, 4 générées, petites, stockées dans `frontend/public/examples/`, sous licence compatible — extraites du test split du dataset 140k), test en **1 clic** ;
   - `Webcam` : capture d'une photo **et** inférence continue (échantillonnage ~1–2 images/s, **une seule requête en vol à la fois**, bouton Start/Stop, arrêt propre du flux caméra au démontage, gestion du refus de permission).
4. **Visualisation des résultats** :
   - `ProbabilityBars` : barres de progression **animées** P(real)/P(fake), couleur selon la classe, verdict `REAL`/`FAKE` bien visible, badge de niveau de confiance ;
   - `FusionWeights` : barre empilée spatial vs fréquentiel (interprétabilité du modèle) ;
   - `GradCamCanvas` : image + heatmap superposée sur `<canvas>`, **curseur d'opacité**, bascule `fake`/`real`, bouton original/overlay ;
   - `LatencyBadge` : latence **client** (aller-retour), **serveur** (`inference_ms`, `total_server_ms`) et écart réseau en ms ;
   - Panneau « Modèle » alimenté par `/model/metadata` (architecture, métriques, version) + indicateur de santé de l'API (`/health`) ;
   - Avertissement visible sur les limites du modèle (généralisation à des générateurs inconnus).
5. UX : états de chargement (skeleton/spinner), états d'erreur lisibles (API éteinte, 400, 503), design responsive, thème clair/sombre, accessibilité de base (labels, contrastes, navigation clavier).
6. Qualité : `npm run lint` sans erreur, `npm run build` OK ; tests Vitest + React Testing Library sur au moins `api.js` et `ProbabilityBars` (optionnel mais valorisé).

### PHASE 7 — Docker et Docker Compose (Jalon 3-4)
1. `backend/Dockerfile` **multi-stage** : stage `builder` (installation des dépendances dans un venv/wheels) → stage `runtime` `python:3.11-slim`, utilisateur non-root, `onnxruntime` CPU (PyTorch CPU uniquement si Grad-CAM embarqué ; sinon image séparée), `HEALTHCHECK` sur `/health`, `.dockerignore`. Les poids sont **montés par volume** (`./models:/models:ro`) ou récupérés via `dvc pull` au build — jamais commités.
2. `frontend/Dockerfile` **multi-stage** : `node:20-alpine` (build Vite) → `nginx:alpine` (servir `dist/`, fallback SPA, proxy optionnel `/api`), `.dockerignore`.
3. `docker-compose.yml` : services `backend` (port 8000), `frontend` (port 3000 ou 80), `mlflow` (port 5000, backend store SQLite ou Postgres + volume d'artefacts) ; `depends_on` avec `condition: service_healthy`, réseau dédié, volumes nommés, variables via `.env`.
4. Commande unique obligatoire : `docker-compose up --build` doit démarrer toute la pile. Prévoir un script `scripts/setup_demo.sh` qui récupère les modèles (`dvc pull`) avant le premier lancement.
5. Vérification : `docker-compose up --build -d` puis `curl localhost:8000/health` (200), ouverture du frontend, `localhost:5000` (MLflow) ; `docker images` pour contrôler la taille des images.

### PHASE 8 — CI/CD GitHub Actions
1. `.github/workflows/ci-tests.yml` (sur `push` et `pull_request`) : checkout → Python 3.11 + cache pip → `black --check` + `flake8` (src, backend) → `pytest backend --cov` (échec si < 80 %) → Node 20 + cache npm → `eslint` + `npm run build` (+ `vitest` si présent). Les tests ne doivent **pas** nécessiter de poids réels (modèle ONNX factice de `conftest.py`).
2. `.github/workflows/docker-build.yml` : `docker buildx build` du backend et du frontend (cache GHA), puis **smoke test** : `docker-compose up -d`, attendre `/health`, appeler `/predict` avec une image de test, `docker-compose down`. Optionnel : push vers GHCR sur `main`.
3. Badges de statut dans le README. Protéger `main` (PR + checks requis) et le documenter.
4. Vérification : les deux workflows passent au vert (capture d'écran dans `docs/`).

### PHASE 9 — Documentation, rapport LaTeX et soutenance (2 + 2 pts)
1. `README.md` : présentation, schéma d'architecture, prérequis, **installation en 3 commandes** (`git clone`, `dvc pull`, `docker-compose up --build`), URLs (frontend, `/docs`, MLflow), variables d'environnement, comment réentraîner (`dvc repro` / `python -m src.models.train`), comment lancer les tests, structure du dépôt, limites connues, crédits (dataset 140k, source des poids/architecture d'origine, licences).
2. `docs/MODEL_CARD.md` : usage prévu, données, métriques (in-distribution **et** cross-dataset), limites, biais, risques d'usage (ne pas utiliser comme preuve unique d'authenticité).
3. `rapport/` (LaTeX, `main.tex` + `refs.bib`) → `rapport_technique.pdf`. Plan : 1) contexte et problématique ; 2) données, EDA (équilibre, spectre de Fourier moyen réel vs faux), anti-fuite ; 3) architecture et justification du flux fréquentiel ; 4) protocole expérimental ; 5) **résultats : baseline vs fusion, ablations, cross-dataset, robustesse JPEG/flou** ; 6) interprétabilité (Grad-CAM, poids de fusion) ; 7) optimisation (ONNX FP32/FP16/INT8, latence) ; 8) architecture logicielle et MLOps (DVC, MLflow, Docker, CI/CD) ; 9) limites et perspectives ; 10) annexes (captures MLflow, Swagger, frontend, CI). Toutes les figures/tables proviennent des artefacts MLflow réels.
4. `docs/SOUTENANCE.md` (aide-mémoire interne, **pas de slides**) : script de démo de 5 min (`docker-compose up`, upload, galerie, webcam, Grad-CAM, latences), parcours MLflow (5 min : runs comparés, registry), parcours DVC/CI, et **FAQ technique par rôle** (pourquoi FFT ? pourquoi ONNX ? pourquoi la FFT sort du graphe ? quelle fuite de données évitée ? pourquoi Grad-CAM en PyTorch ?).

---

## 4. CHECKLIST FINALE (à exécuter et à afficher en fin de travail)

```
[ ] git log -p | grep -i "KGAT_"                      → aucun résultat (et jeton révoqué par l'utilisateur)
[ ] du -sh .git                                        → quelques Mo seulement
[ ] flake8 src backend && black --check src backend    → OK
[ ] pytest backend -q --cov=backend/app                → vert, ≥ 80 %
[ ] dvc status && dvc metrics show                     → à jour
[ ] dvc repro (quick_demo)                             → OK
[ ] mlflow ui : ≥ 4 runs comparables + modèle en Production
[ ] python -m src.models.evaluate --export-onnx        → parité PyTorch/ONNX validée (< 1e-3)
[ ] docker-compose up --build                          → 3 services healthy
[ ] curl localhost:8000/health | /model/metadata | /predict | /predict/visualize
[ ] http://localhost:8000/docs                         → Swagger complet avec exemples
[ ] Frontend : drag&drop, galerie, webcam, barres animées, Grad-CAM, latences client+serveur
[ ] GitHub Actions : ci-tests.yml et docker-build.yml au vert
[ ] README (installation 3 commandes) + rapport_technique.pdf + docs/MODEL_CARD.md
```

## 5. LIVRABLE ATTENDU DE TA PART
À la fin de chaque phase : (1) liste des fichiers créés/modifiés, (2) sortie des commandes de vérification, (3) problèmes rencontrés et décisions prises, (4) ce qui reste à faire **manuellement par l'utilisateur** (révocation du jeton Kaggle, entraînement GPU, configuration du remote DVC, activation de la protection de branche, etc.). Termine par un **tableau de conformité** reprenant les 5 dimensions du barème (6 / 5 / 5 / 2 / 2) avec, pour chaque critère, `OK / Partiel / Manquant` et la preuve (fichier, run MLflow, capture).
