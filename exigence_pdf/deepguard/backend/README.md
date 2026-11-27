# Backend — service modèle (Django)

Portage Django de l'API FastAPI d'origine (mêmes routes, mêmes contrats, mêmes
corps d'erreur). Il est intégré au projet `exigence_pdf` et réutilise le package
partagé `src/`, ainsi que `models/` et `reports/metrics.json`.

## Lancer

Depuis la racine du projet (`exigence_pdf/`) :

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r deepguard/backend/requirements.txt
python deepguard/backend/manage.py runserver
# -> http://127.0.0.1:8000
```

`manage.py` ajoute automatiquement la racine du projet au `PYTHONPATH`, donc
`import src.*` fonctionne sans variable supplémentaire.

## Endpoints

| Méthode | Route | Description |
|---|---|---|
| GET | `/health` | état du service et du modèle (503 si dégradé) |
| GET | `/model/metadata` | fiche modèle + métriques + seuil |
| POST | `/predict` | classification (multipart `file`) |
| POST | `/predict/visualize` | Grad-CAM (PNG ou JSON+base64) |
| GET | `/metrics` | exposition Prometheus |
| GET | `/monitoring/stats` | statistiques des prédictions |
| POST | `/monitoring/drift` | analyse de dérive (`?min_samples=50`) |
| POST | `/monitoring/log-prediction` | journaliser une prédiction |

Le service sert aussi le build React (`deepguard/frontend/dist`) en mode
mono-serveur. Voir [`../README.md`](../README.md) pour l'installation complète,
Docker et le monitoring.
