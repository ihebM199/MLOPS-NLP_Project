# Model Card — Deepfake Adaptive Fusion

## Identité

| Champ | Valeur |
|---|---|
| Nom | `deepfake-adaptive-fusion` |
| Tâche | classification binaire d'images de visages : `real` (0) / `fake` (1) |
| Architecture | `AdaptiveFusionDetector` : flux spatial + flux fréquentiel + porte adaptative |
| Backbone | `convnext_tiny` (poids ImageNet via `timm`) |
| Entrée | RGB 224×224, normalisation ImageNet ; `freq_input = log1p(|fftshift(fft2(x))|)` |
| Sortie | `logit`, `w_spatial`, `w_frequency` (ONNX) |
| Décision | `P(fake) = sigmoid(logit) ≥ seuil` (défaut 0.5) |
| Version | 1.0.0 (voir `/model/metadata`) |

## Usage prévu

- Démonstration pédagogique et aide à l'analyse d'images de visages.
- Détection **in-distribution** sur des visages similaires au dataset 140k
  (StyleGAN, visages frontaux, qualité web).
- Visualisation de l'attention spatiale (Grad-CAM) et des poids de fusion.

## Usage hors périmètre

- **Ne pas** utiliser comme preuve unique d'authenticité, ni dans un cadre
  judiciaire, journalistique ou de modération automatisée sans revue humaine.
- Pas de garantie sur les vidéos, les profils non frontaux, les visages très
  occultés, les images médicales/satellites, ou les générateurs récents absents
  du train (diffusion, StyleGAN3, etc.).
- Pas de détection de contenu synthétique non-visage (texte, audio).

## Données

- **Entraînement/validation/test** : `xhlulu/140k-real-and-fake-faces`
  (splits officiels). Sous-échantillonnage déterministe par classe (seed 42)
  défini dans `params.yaml` ; passer les tailles à `null` pour utiliser tout le
  split.
- **Cross-dataset** (optionnel, jamais en entraînement) : Celeb-DF v2, FF-C23,
  CelebA, LFW. FFHQ est exclu des réels externes (déjà présent dans 140k).
- **Anti-fuite** : splits officiels, test MD5 inter-splits, découpage vidéo par
  identité. Voir `tests/test_no_leakage.py`.

## Métriques

Les métriques ne sont **pas** recopiées ici pour éviter toute valeur inventée.
Elles sont générées par `python -m src.models.evaluate` et disponibles dans :

- `reports/metrics.json` : accuracy, balanced accuracy, précision, rappel, F1,
  AUC, EER, matrice de confusion, seuil ;
- `reports/threshold_calibration.json` : courbe précision/rappel ;
- `reports/comparison_table.csv` : baseline vs fusion (R1–R4) et ablations ;
- MLflow (`http://localhost:5000`) : métriques par époque, artefacts, latence,
  paramètres, MACs ;
- `reports/onnx_parity.json` : parité PyTorch ↔ ONNX.

**Lecture honnête.** Comparer systématiquement in-distribution **et**
cross-dataset. Si la fusion n'apporte pas de gain cross-dataset, l'écrire dans le
rapport. Ne pas sélectionner le modèle sur le test.

## Robustesse

Évaluée sous compression JPEG (75/50/30), flou gaussien (σ=1/2) et
redimensionnement (×0.5/×0.25). Les deepfakes réels sont souvent recompressés :
la dégradation est rapportée dans `reports/metrics.json` et le rapport.

## Biais et risques

- **Biais de données** : 140k est majoritairement StyleGAN et de qualité web ;
  déséquilibre démographique probable (âge, genre, carnation, lunettes).
- **Raccourcis** : le modèle peut apprendre des artefacts de compression ou de
  générateur plutôt que la sémantique du visage.
- **Faux positifs** : risque de stigmatiser des personnes réelles ; toujours
  permettre une revue humaine et un recours.
- **Faux négatifs** : risque de laisser passer un deepfake ; ne pas s'appuyer
  sur un score unique.
- **Grad-CAM** : indice d'attention, pas une explication causale.
- **Webcam** : qualité/lumière variables ; ne pas conclure sur un flux dégradé.

## Optimisation et déploiement

- ONNX FP32, FP16 (poids) et INT8 (quantification dynamique). **Ne pas déployer
  INT8 si l'AUC perd plus d'un point** ; comparer les tableaux générés.
- Backend CPU par défaut ; GPU optionnel. Grad-CAM en PyTorch chargé
  paresseusement.
- Seuil configurable (`DECISION_THRESHOLD`), exposé dans `/model/metadata`.

## Maintenance

- Réentraîner via `dvc repro` ; versionner données/poids avec DVC.
- Promouvoir via MLflow (`staging` → `production`), avec fallback local.
- Surveiller : dérive des données, taux de rejet, latence, taux d'erreur par
  sous-groupe (quand des labels de revue humaine existent).

## Contact / licence

Projet académique MLOps. Code sous MIT ; données et poids soumis à leurs
licences d'origine (voir README).
