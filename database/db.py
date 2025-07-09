from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Connexion à ta base PostgreSQL locale
DATABASE_URL = "postgresql://postgres:motdepasse@localhost/energy_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# Classe de base pour définir les tables
Base = declarative_base()
