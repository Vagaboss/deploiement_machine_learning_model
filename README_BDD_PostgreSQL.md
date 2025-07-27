
# 📦 Base de données PostgreSQL – Documentation & Guide d’utilisation

## 🧭 Objectif

Cette base de données PostgreSQL permet de **centraliser les interactions avec le modèle de Machine Learning**. Elle est utilisée pour **stocker les données envoyées via l’API FastAPI** dans deux tables principales :

- `inputs` : stocke les données envoyées au modèle.
- `outputs` : stocke les prédictions générées par le modèle.

---

## 🛠️ 1. Architecture et Fonctionnement

### 📌 Connexion à la base
- **Nom de la base** : `build-seatle`  
- **Utilisateur dédié** : `OC_yacine`  
- **Mot de passe** : `ocyacine2025`  
- **Hôte** : `localhost`  
- **Port** : `5432`  
- **Schéma** : `public`  

### 📌 Tables créées
Les deux tables `inputs` et `outputs` sont créées via le script `create_db.py` à partir du modèle SQLAlchemy défini dans `api/models.py`.

- `inputs` : chaque ligne représente une requête utilisateur au modèle.
- `outputs` : chaque ligne contient la prédiction associée à une entrée donnée.

---

## 🧪 2. Tutoriel complet pour installation chez un collègue

### Étape 1 : Installer PostgreSQL et pgAdmin
Télécharger et installer [PostgreSQL](https://www.postgresql.org/download/) (inclut pgAdmin).

---

### Étape 2 : Créer la base et l’utilisateur via pgAdmin

#### 🧭 Lancer pgAdmin

1. Ouvrir **pgAdmin 4**.
2. Connecter un **serveur local** (si ce n’est pas encore fait) :
   - Nom : `OC_P5` (ou autre nom)
   - Utilisateur : `postgres`
   - Mot de passe : celui défini à l’installation de PostgreSQL.

---

#### 🧱 Créer la base de données et l’utilisateur

1. Aller dans l’onglet **Tools > Query Tool**.
2. Copier-coller et exécuter le SQL suivant :

```sql
-- 1. Créer la base
CREATE DATABASE "build-seatle";

-- 2. Créer l’utilisateur
CREATE USER OC_yacine WITH PASSWORD 'ocyacine2025';

-- 3. Accorder les droits
GRANT CONNECT ON DATABASE "build-seatle" TO OC_yacine;
GRANT USAGE ON SCHEMA public TO OC_yacine;
GRANT CREATE, SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO OC_yacine;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO OC_yacine;
```

✅ À ce stade, l’utilisateur `OC_yacine` est lié à la base `build-seatle` avec les droits nécessaires.

---

## ⚙️ 3. Création des tables via `create_db.py`

1. **Configurer les accès dans le fichier `.env` (déjà fourni)** :

```
DATABASE_URL=postgresql://OC_yacine:ocyacine2025@localhost:5432/build-seatle
```

2. **Lancer la commande suivante depuis la racine du projet :**

```bash
python create_db.py
```

📌 Cela va automatiquement créer les tables `inputs` et `outputs` dans `build-seatle`.

---

## 🔗 4. Intégration avec l’API FastAPI

L’API utilise `SQLAlchemy` pour se connecter à la base de données à chaque appel POST :

- Lorsqu’un utilisateur envoie des données au modèle via l’API, l’objet `inputs` est enregistré automatiquement dans la base.
- Le modèle effectue la prédiction, et la réponse (`output`) est également stockée dans la base.

📄 Fichiers impliqués :
- `api/db.py` : configure la connexion SQLAlchemy à partir du `.env`.
- `api/models.py` : contient les classes SQLAlchemy des tables.
- `api/main.py` : contient les endpoints qui enregistrent les entrées/sorties.

---

## 🔐 5. Sécurité et authentification

- L’accès à la base est restreint à l’utilisateur `OC_yacine`.
- Il est possible d’ajouter d’autres utilisateurs dans pgAdmin si besoin.
- Les identifiants sont stockés dans un fichier `.env` qui **ne doit pas être versionné** (`.gitignore`).

---

## 📦 Exemple de structure des données stockées

### Exemple de requête envoyée à l’API :
```json
{
  "surface_total": 11000,
  "nbr_batiments": 1,
  "nbr_floors": 3,
  ...
}
```

### Ce que la BDD enregistre :

- `inputs` : avec tous les champs de la requête.
- `outputs` : avec le `prediction_value` renvoyé par le modèle et un horodatage.

---

## ✅ Vérification finale

- Base PostgreSQL `build-seatle` : ✅
- Utilisateur `OC_yacine` avec droits : ✅
- `.env` configuré : ✅
- Tables créées via `create_db.py` : ✅
- API fonctionnelle avec base connectée : ✅
