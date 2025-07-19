import os
from models.experiment_store import Base, engine

db_path = engine.url.database
print(f"Creating database at: {db_path}")
Base.metadata.create_all(engine)
print("Database created successfully!")
