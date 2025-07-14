---
title: Estimation Consommation Énergétique
emoji: 🏢
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.35.0
app_file: app.py
pinned: false
---



# 🏢 Estimation de la Consommation Énergétique des Bâtiments

Bienvenue sur cette application Gradio qui permet d’estimer la consommation annuelle d’énergie (en kBtu) d’un bâtiment non résidentiel à Seattle.  
Le modèle utilisé est un **Random Forest Regressor** entraîné sur les données publiques de la ville de Seattle (2016).

---

## 🔍 Objectif

Cette application vise à fournir une estimation rapide de la consommation énergétique d’un bâtiment, à partir de ses caractéristiques structurelles et d’usage :

- Type de propriété (ex. Office, Hotel…)
- Quartier (Neighborhood)
- Surface totale
- Nombre d’étages / bâtiments
- Présence d’électricité, de gaz, de parking…
- Âge du bâtiment (calculé à partir de l’année de construction)

---

## ⚙️ Fonctionnement de l’application

L’application utilise un **modèle de Machine Learning préentraîné**, encapsulé dans un **pipeline scikit-learn** avec :

- encodage binaire des variables catégorielles (`category_encoders`)
- standardisation des variables numériques (`StandardScaler`)
- prédiction avec `RandomForestRegressor`

---

## ✅ Comment l’utiliser

1. Remplissez tous les champs du formulaire (type de bâtiment, surface, nombre d'étages, etc.)
2. Cliquez sur **"Soumettre"**
3. Obtenez immédiatement une estimation de la consommation énergétique annuelle du bâtiment (en kBtu)

---

## 🛠️ Technologies utilisées

- Python 3.10
- Gradio
- scikit-learn
- pandas
- joblib
- category_encoders

---

## 🔄 Mise à jour automatique (CI/CD)

Cette application est déployée automatiquement à chaque modification via Git (`git push`).  
Elle se reconstruit et se relance automatiquement grâce à la plateforme **Hugging Face Spaces**.

---

## 👤 Auteur

Développé par **Yacine Ould**, dans le cadre du projet de déploiement d’un modèle de Machine Learning.  
Dataset : [Seattle 2016 Building Energy Benchmarking](https://data.seattle.gov)

---