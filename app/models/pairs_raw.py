import db
from sqlalchemy import JSON, Column, String


class PairsRaw(db.Base):
    __tablename__ = "pairs_raw"

    id = Column(String, primary_key=True)
    source = Column(JSON)
