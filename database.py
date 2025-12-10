from sqlalchemy import create_engine, Column, Integer, Float, String, Table, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# SQLite Database Connection
DATABASE_URL = "sqlite:///./database.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


application_components = Table(
    "application_components", Base.metadata,
    Column("application_id", Integer, ForeignKey("applications.id"), primary_key=True),
    Column("component_id",   Integer, ForeignKey("components.id"),   primary_key=True)
)

# Body Model
class Body(Base):
    __tablename__ = "bodies"

    id = Column(Integer, primary_key=True, index=True)
    material = Column(String, nullable=False)  # Metal, Wood, Plastic
    size = Column(String, nullable=False)  # Small, Medium, Large
    cost = Column(Float, nullable=False)
    mass_kg = Column(Float, nullable=False)  # Factor affecting payload

# Motor Model (Now includes torque)
class Motor(Base):
    __tablename__ = "motors"

    id = Column(Integer, primary_key=True, index=True)
    size = Column(String, nullable=False)
    cost = Column(Float, nullable=False)
    mass_kg = Column(Float, nullable=False)
    torque = Column(Float, nullable=False)  # Torque added
    power_watts = Column(Float, nullable=False) # Typical draw under load

# Battery Model
class Battery(Base):
    __tablename__ = "batteries"

    id = Column(Integer, primary_key=True, index=True)
    size = Column(String, nullable=False)
    cost = Column(Float, nullable=False)
    mass_kg = Column(Float, nullable=False)
    capacity_wh = Column(Float, nullable=False)   # Watt-hours stored

# Tires Model (Now includes wheel radius)
class Tires(Base):
    __tablename__ = "tires"

    id = Column(Integer, primary_key=True, index=True)
    size = Column(String, nullable=False)  # Small, Medium, Large
    cost = Column(Float, nullable=False)
    mass_kg = Column(Float, nullable=False)  # Weight factor affecting payload
    wheel_radius = Column(Float, nullable=False)  # Wheel radius

class DroneBody(Base):
    __tablename__ = "drone_bodies"
    id = Column(Integer, primary_key=True, index=True)
    material = Column(String, nullable=False)  # Metal, Wood, Plastic
    size = Column(String, nullable=False)  # Small, Medium, Large
    cost = Column(Float, nullable=False)
    mass_kg = Column(Float, nullable=False)  # Factor affecting payload

class DroneMotor(Base):
    __tablename__ = "drone_motors"
    id = Column(Integer, primary_key=True, index=True)
    size = Column(String, nullable=False)
    cost = Column(Float, nullable=False)
    mass_kg = Column(Float, nullable=False)
    torque = Column(Float, nullable=False)  # Torque added
    power_watts = Column(Float, nullable=False)

class DroneBattery(Base):
    __tablename__ = "drone_batteries"
    id = Column(Integer, primary_key=True, index=True)
    size = Column(String, nullable=False)
    cost = Column(Float, nullable=False)
    mass_kg = Column(Float, nullable=False)
    capacity_wh = Column(Float, nullable=False)   # Watt-hours stored

# Communication Model (Now includes range in km)
class Communication(Base):
    __tablename__ = "communications"
    id = Column(Integer, primary_key=True, index=True)
    type = Column(String, nullable=False)  # LoRa, WiFi
    cost = Column(Float, nullable=False)
    range_km = Column(Float, nullable=False)  # Range in kilometers
    power_watts = Column(Float, nullable=False, default=0.0)

# Computing Unit Model
class ComputingUnit(Base):
    __tablename__ = "computing_units"
    id = Column(Integer, primary_key=True, index=True)
    model = Column(String, nullable=False)
    cost = Column(Float, nullable=False)
    performance = Column(Float, nullable=False)
    power_watts = Column(Float, nullable=False)
    unit_type = Column(String, nullable=False)

class Application(Base):
    __tablename__ = "applications"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    processing_mode = Column(String, nullable=False)
    platform = Column(String, nullable=False)
    components      = relationship(
        "Component",
        secondary=application_components,
        back_populates="applications"
    )

class Component(Base):
    __tablename__ = "components"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    cost = Column(Float, nullable=False)
    power_watts = Column(Float, nullable=False, default=0.0)
    category = Column(String, nullable=False)
    applications = relationship(
        "Application",
        secondary=application_components,
        back_populates="components"
    )

class EdgeServer(Base):
    __tablename__ = "edge_servers"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    cost = Column(Float, nullable=False)
# Create Tables
Base.metadata.create_all(bind=engine)
