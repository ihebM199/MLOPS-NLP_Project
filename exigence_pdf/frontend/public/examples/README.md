# Galerie d'exemples

Les 8 images attendues par `ExampleGallery.jsx` (`real_1.jpg` … `fake_4.jpg`) sont
extraites du **split test** du dataset 140k par :

```bash
python scripts/export_examples.py --dataset-root data/processed/real_vs_fake/real-vs-fake
```

Tant que le script n'a pas été lancé (dataset non téléchargé), l'interface
affiche des tuiles « exemple à générer » au lieu d'inventer des images : aucune
image factice n'est livrée avec le dépôt.
