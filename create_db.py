from database.db import engine
from database import models

# Création des tables
models.Base.metadata.create_all(bind=engine)
print("✅ Base de données initialisée.")
