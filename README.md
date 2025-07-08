# 🏙️ Prédiction de la Consommation d'Énergie des Bâtiments à Seattle

Ce projet de Machine Learning vise à prédire la consommation d’énergie des bâtiments non résidentiels de Seattle, en se basant sur des données publiques de 2016.  
L’objectif est d’**industrialiser un modèle** en le structurant dans un projet Python versionné avec Git, prêt à être exécuté, évalué, maintenu et déployé via CI/CD.

---

## ⚙️ Installation

### 1. Clonez le projet

```bash
git clone <url-du-repo>
cd gitP3
```

### 2. Installez les dépendances

```bash
pip install -r requirements.txt
```

---

## 📦 Données utilisées

Le fichier `2016_Building_Energy_Benchmarking.csv` doit être placé dans le dossier `data/`.

🔗 Il contient des informations principales sur les bâtiments de Seattle (surface, type, année, usage, consommation...).

---


## ▶️ Exécution

### 🔹 Entraînement du modèle

```bash
python train_model.py
```

- Nettoie, encode et standardise les données
- Entraîne un `RandomForestRegressor` avec `GridSearchCV`
- Enregistre le modèle dans `models/best_rf_pipeline.joblib`

### 🔹 Évaluation du modèle

```bash
python evaluate_model.py
```

- Recharge le modèle
- Affiche les performances :
  - R² (coefficient de détermination)
  - MAE (erreur absolue moyenne)
  - RMSE (racine de l'erreur quadratique moyenne)

---

## 🔁 Intégration Continue (CI)

Une **GitHub Action** est configurée pour tester automatiquement le projet à chaque `push` ou `pull request` sur les branches `main` ou `dev`.

Le pipeline :

- installe les dépendances (`requirements.txt`)
- lance l’entraînement du modèle (`train_model.py`)
- évalue le modèle (`evaluate_model.py`)

📄 Fichier : `.github/workflows/test-model.yml`

---

## 🚀 Déploiement (CD – Continuous Delivery)

Le projet est déployé sous forme d’une application web interactive avec **Gradio** sur [Hugging Face Spaces](https://huggingface.co/spaces/yacineould/P3_ML_deployment).

### 🌐 Fonctionnement

- L’utilisateur renseigne les caractéristiques d’un bâtiment
- Le modèle prédit sa consommation annuelle d’énergie (en kBtu)

### 🔄 Déploiement automatique

À chaque `git push` sur le dépôt du Space Hugging Face :

- le modèle est mis à jour
- l’app est relancée automatiquement

---
### Précision sur la CI/CD

    La CI est assurée par GitHub Actions sur gitP3

    La CD est déclenchée automatiquement lors du push vers le dépôt Hugging Face

    Donc la CI / CD sont indépendants l'un de l'autre

## 🌐 API avec FastAPI

Une API a été créée avec FastAPI pour exposer le modèle.

### 🔸 Lancer l’API localement

```bash
uvicorn api.main:app --reload

Une fois lancée, rendez-vous sur : http://127.0.0.1:8000/docs
Vous pourrez tester l’API via Swagger.

### 📍 Endpoints disponibles
Méthode	Endpoint	Description
 GET	/health	Vérifie si l'API fonctionne
 POST	/predict	Envoie des données et retourne une prédiction

## 💡 Objectifs pédagogiques

Ce projet permet de :

- Appliquer un pipeline de régression supervisée
- Nettoyer les données et créer de nouvelles variables
- Structurer un projet Python de façon modulaire
- Gérer le versioning avec Git et GitHub
- Mettre en place une CI avec GitHub Actions
- Déployer un modèle ML avec Gradio et Hugging Face

---

## 👤 Réalisé par

**Yacine Ould**  
Projet réalisé dans le cadre de la formation **Data Scientist – OpenClassrooms**

---
