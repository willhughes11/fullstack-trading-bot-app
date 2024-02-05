import os

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")

# PostgreSQL connection string
db_url = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@quantum-flow-postgres/{POSTGRES_DB}"

# Create an engine
engine = create_engine(db_url)

# Define a data model
Base = declarative_base()