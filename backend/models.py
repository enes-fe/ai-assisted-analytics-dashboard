from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from database import Base

class DatasetMeta(Base):
    __tablename__ = "dataset_meta"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    table_name = Column(String, unique=True, index=True)
    row_count = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
