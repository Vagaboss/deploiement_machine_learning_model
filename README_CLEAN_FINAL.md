# Documentation du Projet - Déploiement d'un Modèle de Machine Learning

# 🏙️ Prédiction de la Consommation d'Énergie des Bâtiments à Seattle

Ce projet de Machine Learning vise à prédire la consommation d’énergie des bâtiments non résidentiels de Seattle, en se basant sur des données publiques de 2016.  
L’objectif est d’**industrialiser un modèle** en le structurant dans un projet Python versionné avec Git, prêt à être exécuté, évalué, maintenu et déployé via CI/CD.

### 1. Présentation Générale

Ce dépôt contient l’ensemble du code source et des ressources nécessaires pour entraîner, évaluer, déployer et maintenir un modèle de machine learning à l’aide de FastAPI, PostgreSQL, Gradio, et GitHub Actions (CI/CD).

---

## Installation Locale

```bash
git clone https://github.com/Vagaboss/deploiement_machine_learning_model.git
cd gitP3
pip install -r requirements.txt
```
## Historique des commit et tag
- git log
- git tag

---

## 📦 Données utilisées

Le fichier `2016_Building_Energy_Benchmarking.csv` doit être placé dans le dossier `data/`.
Le fichier `cleaned_data.csv` doit être placé dans le dossier `data/`. C'est le dataset nettoyé suite au preprocessing.

🔗 Ils contiennent des informations principales sur les bâtiments de Seattle (surface, type, année, usage, consommation...).

---

### 2. Structure du Dépôt Git 

Le modèle de machine learning, initialement développé dans un notebook, a été migré dans une architecture modulaire avec des fichiers `.py` pour assurer la maintenabilité 

📦 deploiement_machine_learning_model/
├── .github/
│   └── workflows/
│       └── test-model.yaml
├── DB schema/
├── api/
│   ├── main.py
│   ├── db.py
│   ├── models.py
│   └── schemas.py
├── data/
│   ├── cleaned_data.csv
│   └── 2016_Building_Energy_Benchmarking.csv
├── hf_spaces/
│   └── P3_ML_deployment/
├── notebooks/
│   └── p3deplo(1).ipynb
├── src/
│   └── preprocessing.py
├── tests/
│   ├── test_api.py
│   ├── test_preprocessing.py
│   └── test_train_model.py
├── .coverage
├── .gitignore
├── README.md
├── README_CLEAN_FINAL.md
├── create_db.py
├── evaluate_model.py
├── insert_data.py
├── insert_outputs.py
├── requirements.txt
└── train_model.py


---

## Lancer l'entrainement et évaluation du modèle (commandes GIT)
- python train_model.py
   - Nettoie, encode et standardise les données
   - Entraîne un `RandomForestRegressor` avec `GridSearchCV`
   - Enregistre le modèle dans `models/best_rf_pipeline.joblib`

- python evaluate_model.py
  - Recharge le modèle
  - Affiche les performances :
    - R² (coefficient de détermination)
    - MAE (erreur absolue moyenne)
    - RMSE (racine de l'erreur quadratique moyenne)

### 3. CI/CD avec GitHub Actions

## Intégration Continue (CI)

Une action GitHub déclenchée à chaque `push` ou `pull_request` sur les branches `main` et `dev` :

- Installe les dépendances (`requirements.txt`)
- Entraîne et évalue le modèle

📄 Fichier : `.github/workflows/test-model.yml`

## Déploiement Continu (CD)

Le déploiement vers Hugging Face Spaces est automatisé :
- Le workflow push automatiquement le contenu du dossier `hf_spaces/` vers l’espace distant.
- L’interface Gradio est ainsi mise à jour dès que le modèle est entraîné.

📄 Fichier : `.github/workflows/test-model.yml`
---

### 4. Interface Hugging Face (Gradio)

Une interface utilisateur intuitive a été développée avec Gradio pour permettre de tester le modèle :

- Saisir les caractéristiques du bâtiment
- Obtenir la prédiction de consommation énergétique (en kBtu)
- Accessible à : [https://huggingface.co/spaces/yacineould/P3_ML_deployment](https://huggingface.co/spaces/yacineould/P3_ML_deployment)

---

## 🌐 Exemple d’utilisation – Hugging Face Spaces
🎯 Objectif : faire une prédiction via l’interface Gradio en ligne

    - Accéder à l’espace déployé

    - Utiliser l’interface graphique

    - Renseigne les valeurs des champs affichés (type de bâtiment, nombre d'étages, etc.)

    - Clique sur “Prédire”

    Résultat affiché :

    - Une prédiction s'affiche dans un encadré, indiquant la consommation prédite.

### 5. API FastAPI

L’API expose plusieurs endpoints documentés via Swagger :

- `/predict` : prédiction à partir des données utilisateurs
- `/health` : vérifie si l’API fonctionne
- `/history` : historique des requêtes
- `/dataset` : visualisation d’un extrait du dataset

Documentation interactive disponible à : `http://127.0.0.1:8000/docs`

- Ouvrir la base de données PostgreSQL "build-seatle"
- Se mettre dans gitP3 et lancer la commande git : uvicorn api.main:app
- Ouvrir http://127.0.0.1:8000/docs 
- Vous pourrez tester l’API via Swagger.
---
## 🌐 Exemple d’utilisation – FASTAPI
- A rentrer dans l'endpoint "Predict"
- Cliquer sur "Try it out"
- Exemple : 
{
  "PropertyGFATotal": 1000000,
  "NumberofFloors": 4,
  "NumberofBuildings": 2,
  "YearBuilt": 2010,
  "HasGas": true,
  "HasElectricity": true,
  "HasSteam": true,
  "HasParking": true,
  "UsageCount": "string",
  "PropertyTypeGrouped": "string",
  "PrimaryPropertyType": "string",
  "Neighborhood": "string"
}
- Appuyer sur Execute
- La valeur prédite est visible juste en dessous


### 6. Base de Données PostgreSQL

Une base PostgreSQL ("build-seatle") héberge :

- Le dataset complet
- Les entrées (`inputs`) et sorties (`outputs`) des requêtes utilisateur effectuées dans l'API

## Connexion via SQLAlchemy

La base est reliée à l’API FastAPI pour historiser les interactions.

---

### 7. Tests Unitaires et Fonctionnels

Trois types de tests ont été réalisés avec Pytest :

- **Preprocessing** : `pytest tests/test_preprocessing.py`
- **Modèle ML** : `pytest tests/test_train_model.py`
- **API** : `pytest tests/test_api.py`

Les tests couvrent les cas critiques et les erreurs. Des rapports de couverture ont été générés.

## Comment les lancer ? 

- Se mettre dans gitP3
- effectuer les commandes git suivantes : 
  - pytest tests/test_preprocessing.py
  - pytest tests/test_train_model.py
  - pytest tests/test_api.py

---
### 8. Voir les rapports de couverture

 Il faut se rendre dans le sous dossier htmlcov
 
 - Cliquer sur les 2 fichiers commençant par "Couverture"


### 9. Protocole de Maintenance

- **Mise à jour du modèle** : réentraînement ou mise à jour via modification du `train_model.py`
- **Mise à jour de l'API** : mise à jour des fichier .py dans le sous dossier "api"
- **Déploiement auto** : push sur `main` ou `dev` déclenche le workflow CI/CD
- **Secrets** : gérés via `HF_TOKEN` dans GitHub

---

### 10. Justification des Choix Techniques

- **FastAPI** : rapidité de développement, documentation automatique
- **PostgreSQL** : robustesse, requêtes complexes
- **Gradio** : interface rapide à déployer
- **GitHub Actions** : intégration fluide avec GitHub

---

### 11. 📌 Documentation technique du modèle de Machine Learning

## Architecture du modèle

Le modèle est un pipeline scikit-learn entraîné sur un dataset de consommation énergétique des bâtiments. Il est composé de :

    Un préprocessing (encodage, gestion des valeurs manquantes, normalisation)

    Un estimateur final 

    Ce pipeline est sauvegardé avec joblib pour être utilisé dans l’API ou dans l’espace Hugging Face.


- train_model.py	: Entraîne et sauvegarde le modèle ML
- evaluate_model.py	: Évalue les performances sur un jeu de test (MAE, RMSE, R²)
- models/best_rf_pipeline.joblib	: Contient le pipeline entraîné, prêt à être utilisé par l’API

## Pour entraîner et évaluer le modèle :

python train_model.py
python evaluate_model.py

- Performances du modèle (exemple)
Métrique	Score
RMSE	12.8
MAE	9.5
R²	0.78

    Ces résultats sont indicatifs et peuvent varier selon la version du dataset ou les hyperparamètres.

### 12. 🔄 Maintenance et mise à jour du modèle et de l’API

## Mettre à jour le modèle

    Modifier ou remplacer le dataset ou la logique dans train_model.py

    Réentraîner :

python train_model.py
python evaluate_model.py

- Vérifier que le fichier best_rf_pipeline.joblib a bien été mis à jour

- Commiter et pousser :

    git add models/best_rf_pipeline.joblib
    git commit -m "🔄 Update modèle ML"
    git push

    Le déploiement vers Hugging Face sera automatique grâce à la CD configurée avec GitHub Actions

## Mettre à jour l’API FastAPI

    Modifier la logique dans api/main.py ou les dépendances associées (schemas.py, db.py, etc.)

    Tester localement :

uvicorn api.main:app --reload

- Si tout fonctionne :

    git add .
    git commit -m "✨ Mise à jour API"
    git push

    ⚠️ Attention : l’API n’est pas déployée automatiquement. Elle tourne localement ou doit être déployée manuellement (Render, Railway, etc.)

### 13. Protocole de Mise à jour 

🔁 Protocole de mise à jour régulière

Pour assurer la pérennité et la fiabilité du projet, voici un protocole de mise à jour structuré, à suivre tous les mois (ou à chaque évolution majeure) :

1. Mise à jour des données

    📅 Fréquence : mensuelle ou dès qu’un nouveau dataset est disponible.

    🔧 Action : insérer les nouvelles données dans la base PostgreSQL (via psql ou script Python).

    ⚠️ Impact : nécessite potentiellement un réentraînement du modèle.

2. Réentraînement du modèle

    📅 Fréquence : à chaque mise à jour significative des données ou dégradation des performances.

    ⚙️ Action :

    python train_model.py
    python evaluate_model.py

    ✅ Évaluation : valider que les nouvelles performances sont meilleures ou stables.

    💾 Sauvegarde : remplacer le modèle enregistré dans models/best_rf_pipeline.joblib.

3. Mise à jour de l’API

    🔁 Si la structure du modèle change (nouveaux inputs ou changement de format), mettre à jour :

        Le schéma Pydantic dans main.py

        Les endpoints concernés

        Les tables inputs et outputs de la base PostgreSQL

    Tester localement avec :

    uvicorn api.main:app --reload

4. Mise à jour de l’espace Hugging Face

    📦 Remplacer le fichier modèle dans hf_spaces/P3_ML_deployment/models/

    ⚙️ La CI/CD déclenche automatiquement un nouveau déploiement via GitHub Actions

    🎯 Vérifier le bon fonctionnement de l’interface Gradio

5. Vérification des tests

    🧪 Relancer tous les tests :

    pytest tests/test_preprocessing.py
    pytest tests/test_train_model.py
    pytest tests/test_api.py

    🔍 S’assurer qu’ils passent tous à 100% avec pytest-cov

6. Mise à jour de la documentation

    📘 Mettre à jour le README.md :

        Nouvelles instructions d’installation si besoin

        Changement dans les inputs, outputs, ou modèle

        Ajout de nouveaux exemples d’utilisation

7. Versioning

    📌 Utiliser des tags Git pour marquer les nouvelles versions stables :

git tag -a v1.1 -m "Nouveau modèle + API mise à jour"
git push origin v1.1

