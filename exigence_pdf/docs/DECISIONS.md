# Décisions techniques (ADR)

Ce document trace les choix structurants et leurs alternatives. Il complète le
rapport LaTeX (`rapport/main.tex`).

## 1. FFT hors du graphe ONNX — Option A

**Décision.** Le modèle exporté prend deux entrées : `image` (RGB normalisée,
`1×3×224×224`) et `freq_input` (`log1p(|fftshift(fft2(image))|)`). Le calcul FFT
est fait côté inférence avec NumPy, avant ONNX Runtime.

**Pourquoi.** `torch.fft.fft2` n'a pas d'export ONNX fiable (opérateurs
complexes, support runtime hétérogène). L'Option A garde le graphe simple,
portable et strictement identique à l'entraînement (même formule, même
normalisation). L'Option B (DFT par multiplications matricielles) a été écartée :
deux matrices 224×224 réelles/imaginaires alourdissent le graphe et le rendent
plus fragile.

**Conséquences.** L'API calcule la FFT une fois par image ; le coût est négligeable
devant le backbone. Le contrat est testé (`tests/test_onnx_parity.py`) :
signature 2 entrées / 3 sorties, `w_spatial + w_frequency = 1`.

## 2. Backbone `convnext_tiny`

**Décision.** `convnext_tiny` est le backbone par défaut de `params.yaml`, car
c'est l'architecture du checkpoint fourni (`models/AdaptiveFusion_best_model.pth`,
projection 768-d). `load_checkpoint` auto-détecte toutefois d'autres backbones
(`efficientnetv2_rw_s`, `regnety_*`, …) à partir de la forme des poids.

**Pourquoi.** Recharger le checkpoint sans réentraînement complet, tout en
laissant la porte ouverte aux ablations de backbone.

## 3. Grad-CAM en PyTorch, prédiction en ONNX

**Décision.** `/predict` utilise ONNX Runtime (CPU, rapide, sans gradient).
`/predict/visualize` charge paresseusement le checkpoint PyTorch pour calculer
les gradients et la carte Grad-CAM.

**Pourquoi.** ONNX Runtime n'expose pas de gradients. Charger PyTorch uniquement
au premier appel de visualisation garde l'image backend légère au démarrage et
n'impacte pas la latence de prédiction.

**Conséquences.** Le premier appel `/predict/visualize` est plus lent (chargement
du modèle). C'est documenté dans `docs/MODEL_CARD.md` et visible côté client
(`LatencyBadge`).

## 4. Pipeline DVC `prepare → train → evaluate → export_onnx`

**Décision.** Les données et les poids sont versionnés par DVC, jamais par Git.
Le checkpoint est un **output** du stage `train`, pas un `dvc add models/` :
DVC le cache automatiquement et `dvc pull` le restaure.

**Pourquoi.** Éviter les gros fichiers dans Git, rendre `dvc repro` reproductible
et conserver `git ls-files data models` propre (`.dvc`/`.gitkeep` uniquement).

## 5. Seuil de décision et calibration

**Décision.** Seuil par défaut `P(fake) ≥ 0.5`, exposé dans `/model/metadata` et
surchargeable par `DECISION_THRESHOLD`. La calibration (courbe précision/rappel,
seuil optimal) est calculée sur le split de validation et écrite dans
`reports/threshold_calibration.json`.

**Pourquoi.** Le seuil 0.5 n'est pas nécessairement optimal selon le coût des
faux positifs/négatifs. Le seuil retenu est explicite et traçable, jamais caché.

## 6. Registry MLflow avec fallback local

**Décision.** Le backend privilégie le modèle MLflow « Production » si
`MLFLOW_TRACKING_URI` est configuré, puis retombe sur
`models/deepfake_fusion.onnx`.

**Pourquoi.** Permettre une promotion de modèle sans redéploiement, tout en
gardant la démo autonome hors ligne. `tracking.register: true` et
`tracking.alias: production` activent l'enregistrement.

## 7. Anti-fuite

**Décision.** Splits officiels, sous-échantillonnage déterministe par classe
(seed 42), test MD5 des contenus entre splits. Les jeux externes sont réservés
au test cross-dataset ; FFHQ est exclu des réels externes (déjà dans 140k) et les
vidéos sont découpées par vidéo/identité, jamais par frame.

**Pourquoi.** Éviter les fuites classiques (même image dans train et test, frames
d'une même vidéo des deux côtés) qui gonflent artificiellement les métriques.

## 8. Frontend : même origine par défaut

**Décision.** `VITE_API_URL` vide ⇒ appels relatifs `/predict`. En Docker, le
build utilise `VITE_API_URL=/api` et nginx retire le préfixe `/api` avant de
proxyfier vers le backend.

**Pourquoi.** Un seul tunnel (Colab/ngrok) ou un seul domaine en production,
sans CORS supplémentaire ; le proxy nginx évite les problèmes de mixed-content.

## 9. Tests sans poids réels

**Décision.** `backend/tests/conftest.py` génère un mini graphe ONNX de même
signature (2 entrées / 3 sorties). Les tests API, prétraitement et Grad-CAM
tournent sur CPU en quelques secondes.

**Pourquoi.** CI rapide, déterministe, sans dataset ni checkpoint de 38 Mo. La
parité sur le vrai modèle reste un test `slow` exécuté localement/GPU.

## 10. Sécurité des secrets

**Décision.** Le notebook nettoyé ne contient plus de jeton ; les secrets sont lus
via variables d'environnement / `~/.kaggle`. `detect-secrets` et
`check-added-large-files` sont dans pre-commit.

**Rappel.** Le jeton Kaggle exposé dans l'historique Git doit être **révoqué**
puis l'historique purgé avec `scripts/purge_secret_history.sh` (force-push
explicite, après sauvegarde). Voir `docs/SOUTENANCE.md`.
