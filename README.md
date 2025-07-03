# 🏙️ Prédiction de la Consommation d'Énergie des Bâtiments à Seattle

Ce projet de Machine Learning vise à prédire la consommation d’énergie des bâtiments non résidentiels de Seattle, en se basant sur des données publiques de 2016.  
L’objectif est d’industrialiser un modèle en le structurant dans un projet Python versionné avec Git, prêt à être utilisé, évalué, maintenu et déployé.

---

## ⚙️ Installation

### 1. Clonez le projet

```bash
git clone <url-du-repo>
cd gitP3
```

### 3. Installez les dépendances

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

- Effectue le nettoyage, la préparation, l'encodage et la standardisation des données
- Recherche les meilleurs hyperparamètres via GridSearchCV
- Enregistre le meilleur modèle dans `models/best_random_forest_model.joblib`

### 🔹 Évaluation du modèle

```bash
python evaluate_model.py
```

- Recharge le modèle sauvegardé
- Affiche les scores sur le jeu d'entraînement et de test (R², MAE, RMSE)

---

## 📌 Objectifs pédagogiques

Ce projet vise à démontrer :
- L’usage d’une structure modulaire avec des scripts Python
- La gestion de version avec Git et GitHub
- L’industrialisation d’un modèle de machine learning

---

## 👤 Réalisé par

**Yacine Ould**  
Projet réalisé dans le cadre de la formation Data Scientist – OpenClassrooms.