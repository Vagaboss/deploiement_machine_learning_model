# Documentation du Projet - Déploiement d'un Modèle de Machine Learning

## 1. Présentation Générale

Ce dépôt contient l’ensemble du code source et des ressources nécessaires pour entraîner, évaluer, déployer et maintenir un modèle de machine learning à l’aide de FastAPI, PostgreSQL, Gradio, et GitHub Actions (CI/CD).

---

## 2. Structure du Dépôt Git

Le modèle de machine learning, initialement développé dans un notebook, a été migré dans une architecture modulaire avec des fichiers `.py` pour assurer la maintenabilité :

- `train_model.py` : pour l'entraînement du modèle
- `evaluate_model.py` : pour évaluer sa performance
- `preprocessing.py` : pour les étapes de transformation
- `api/` : contient l'API FastAPI et sa logique de gestion de base de données
- `hf_spaces/` : contient l’interface Gradio pour Hugging Face Spaces

---

## 3. CI/CD avec GitHub Actions

### Intégration Continue (CI)

Une action GitHub déclenchée à chaque `push` ou `pull_request` sur les branches `main` et `dev` :

- Installe les dépendances (`requirements.txt`)
- Entraîne et évalue le modèle
- (Tests retirés dans la version finale pour éviter les conflits avec PostgreSQL)

### Déploiement Continu (CD)

Le déploiement vers Hugging Face Spaces est automatisé :
- Le workflow push automatiquement le contenu du dossier `hf_spaces/` vers l’espace distant.
- L’interface Gradio est ainsi mise à jour dès que le modèle est entraîné.

---

## 4. Interface Hugging Face (Gradio)

Une interface utilisateur intuitive a été développée avec Gradio pour permettre de tester le modèle :

- Saisir les caractéristiques du bâtiment
- Obtenir la prédiction de consommation énergétique
- Accessible à : [https://huggingface.co/spaces/yacineould/P3_ML_deployment](https://huggingface.co/spaces/yacineould/P3_ML_deployment)

---

## 5. API FastAPI

L’API expose plusieurs endpoints documentés via Swagger :

- `/predict` : prédiction à partir des données utilisateurs
- `/health` : vérifie si l’API fonctionne
- `/history` : historique des requêtes
- `/dataset` : visualisation d’un extrait du dataset

Documentation interactive disponible à : `http://127.0.0.1:8000/docs`

---

## 6. Base de Données PostgreSQL

Une base PostgreSQL héberge :

- Le dataset complet
- Les entrées (`inputs`) et sorties (`outputs`) des requêtes utilisateur

### Connexion via SQLAlchemy

La base est reliée à l’API FastAPI pour historiser les interactions.

---

## 7. Tests Unitaires et Fonctionnels

Trois types de tests ont été réalisés avec Pytest :

- **Preprocessing** : `pytest tests/test_preprocessing.py`
- **Modèle ML** : `pytest tests/test_train_model.py`
- **API** : `pytest tests/test_api.py`

Les tests couvrent les cas critiques et les erreurs. Des rapports de couverture ont été générés pendant la CI initiale.

---

## 8. Installation Locale

```bash
git clone https://github.com/Vagaboss/deploiement_machine_learning_model.git
cd deploiement_machine_learning_model
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Accès à l'API : `http://127.0.0.1:8000/docs`

---

## 9. Architecture Générale

```
📦 deploiement_machine_learning_model/
├── api/
│   ├── main.py
│   ├── db.py
│   └── models.py
├── hf_spaces/
│   └── P3_ML_deployment/
├── models/
│   └── best_rf_pipeline.joblib
├── train_model.py
├── evaluate_model.py
├── preprocessing.py
├── requirements.txt
└── tests/
```

---

## 10. Protocole de Maintenance

- **Mise à jour du modèle** : réentraînement via `train_model.py`
- **Déploiement auto** : push sur `main` déclenche le workflow
- **Secrets** : gérés via `HF_TOKEN` dans GitHub

---

## 11. Justification des Choix Techniques

- **FastAPI** : rapidité de développement, documentation automatique
- **PostgreSQL** : robustesse, requêtes complexes
- **Gradio** : interface rapide à déployer
- **GitHub Actions** : intégration fluide avec GitHub

---

## 12. Exemples d’Utilisation

### Gradio

Utiliser l’interface web pour faire une prédiction à partir d’un formulaire.

### API (via `curl`)

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "PrimaryPropertyType": "Office",
  "Neighborhood": "Belltown",
  "YearBuilt": 1990,
  "NumberofBuildings": 1,
  "ENERGYSTARScore": 75,
  "SiteEnergyUse(kBtu)": 1500000,
  "TotalGHGEmissions": 300
}'
```
