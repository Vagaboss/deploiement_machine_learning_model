import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# Charger les variables d'environnement du fichier .env
load_dotenv()

# Lire l'URL de la base depuis le fichier .env
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

# Création de l'engine et de la session
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# Classe de base pour les modèles SQLAlchemy
Base = declarative_base()

