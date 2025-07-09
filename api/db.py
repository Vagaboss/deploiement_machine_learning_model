from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Connexion à ta base PostgreSQL locale
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:<narutolekake1>@localhost:5432/build-seatle"


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# Classe de base pour définir les tables
Base = declarative_base()
