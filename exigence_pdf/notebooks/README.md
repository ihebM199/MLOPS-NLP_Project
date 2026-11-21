# Notebook nettoyé

`Deepfake_AdaptiveFusion_MLOps_officielle.ipynb` est la version **assainie** du
notebook Colab d'origine :

- la cellule contenant le jeton Kaggle a été **supprimée** ;
- la cellule `!kaggle competitions list` inutile a été supprimée ;
- toutes les sorties et compteurs d'exécution ont été effacés (109 Ko au lieu
  de 21 Mo) ;
- le code métier est importé depuis `src/` — pas de duplication.

Le notebook sert uniquement à l'exploration et à la démonstration. Pour
reproduire le pipeline, utiliser les modules `src/` ou `dvc repro`.

Régénérer / vérifier :

```bash
python scripts/sanitize_notebook.py --input notebook.ipynb --in-place
git log -p | grep -i "KGAT_"     # doit être vide
```
