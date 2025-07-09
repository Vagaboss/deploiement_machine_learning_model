import sys
import os

# Ajout du dossier 'api' au path pour permettre les imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))

from db import engine
from models import Base

Base.metadata.create_all(bind=engine)
print("✅ Base de données initialisée.")
