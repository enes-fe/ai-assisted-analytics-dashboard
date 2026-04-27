from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

import os
# Move database outside backend/ to prevent uvicorn reloads
STORAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_storage"))
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

DB_PATH = os.path.join(STORAGE_DIR, "ai_dashboard.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
