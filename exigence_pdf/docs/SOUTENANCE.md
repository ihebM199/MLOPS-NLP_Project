# Aide-mémoire soutenance (interne, pas de slides)

## Démo 5 minutes

1. **Démarrage** (30 s) : `docker compose up --build -d` puis
   `curl localhost:8000/health`. Montrer les 3 services (`docker compose ps`) et
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

## Points à préparer (manuel)

- Révoquer le jeton Kaggle exposé, puis purger l'historique
  (`scripts/purge_secret_history.sh`) et force-push.
- Configurer le remote DVC (local/S3/GCS) et `dvc push`.
- Lancer l'entraînement GPU complet + `scripts/run_experiments.py`.
- Activer la protection de branche `main` (PR + checks requis).
- Compiler `rapport/main.tex` (TeX Live) et archiver `rapport_technique.pdf`.
