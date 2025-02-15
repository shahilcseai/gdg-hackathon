import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Get database URL from environment variables
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class SatelliteData(Base):
    __tablename__ = "satellite_data"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    latitude = Column(Float)
    longitude = Column(Float)
    no2_value = Column(Float)
    resolution = Column(Float)  # in kilometers
    source = Column(String)
    measurements = relationship("GroundMeasurement", back_populates="satellite_data")

class GroundMeasurement(Base):
    __tablename__ = "ground_measurements"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    latitude = Column(Float)
    longitude = Column(Float)
    no2_value = Column(Float)
    station_name = Column(String)
    satellite_data_id = Column(Integer, ForeignKey("satellite_data.id"))
    satellite_data = relationship("SatelliteData", back_populates="measurements")

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
