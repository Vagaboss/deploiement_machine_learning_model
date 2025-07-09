from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from database.db import Base
from datetime import datetime

class Input(Base):
    __tablename__ = "inputs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    PropertyGFATotal = Column(Float)
    NumberofFloors = Column(Integer)
    NumberofBuildings = Column(Integer)
    YearBuilt = Column(Integer)
    HasGas = Column(Boolean)
    HasElectricity = Column(Boolean)
    HasSteam = Column(Boolean)
    HasParking = Column(Boolean)
    UsageCount = Column(String)
    PropertyTypeGrouped = Column(String)
    PrimaryPropertyType = Column(String)
    Neighborhood = Column(String)


     # Output lié
    prediction = relationship("Output", back_populates="input", uselist=False)

class Output(Base):
    __tablename__ = "outputs"

    id = Column(Integer, primary_key=True, index=True)
    input_id = Column(Integer, ForeignKey("inputs.id"))
    prediction = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

    input = relationship("Input", back_populates="prediction")
