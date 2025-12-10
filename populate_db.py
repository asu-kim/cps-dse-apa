from database import (SessionLocal, Body, Motor, Battery, Communication, Tires, ComputingUnit, DroneBody, DroneMotor,
                      DroneBattery, Application, Component, application_components, EdgeServer)

# Create a new database session
db = SessionLocal()

# Clear existing data
db.query(Body).delete()
db.query(Motor).delete()
db.query(Battery).delete()
db.query(Communication).delete()
db.query(Tires).delete()

# Insert Body Components
bodies = [
    Body(material="Metal", size="Small", cost=300, mass_kg=20),
    Body(material="Metal", size="Medium", cost=800, mass_kg=50),
    Body(material="Metal", size="Large", cost=2000, mass_kg=100),
    Body(material="Wood", size="Small", cost=500, mass_kg=10),
    Body(material="Wood", size="Medium", cost=1000, mass_kg=25),
    Body(material="Wood", size="Large", cost=2000, mass_kg=75),
    Body(material="Plastic", size="Small", cost=100, mass_kg=5),
    Body(material="Plastic", size="Medium", cost=300, mass_kg=20),
    Body(material="Plastic", size="Large", cost=700, mass_kg=70)
]

# Insert Motors
motors = [
    Motor(size="Small",  cost=14.99, mass_kg=0.096, torque=18.91,  power_watts=46.8),
    Motor(size="Medium", cost=39.99, mass_kg=0.38, torque=57.27,  power_watts=240),
    Motor(size="Large",  cost=59.99, mass_kg=0.576, torque=64.52, power_watts=1440),
]

# Insert Batteries
batteries = [
    Battery(size="Small",  cost=319.99,  mass_kg=35, capacity_wh= 2400),
    Battery(size="Medium", cost=559.99, mass_kg=25.76, capacity_wh=3968),
    Battery(size="Large",  cost=729.99, mass_kg=47.17, capacity_wh=7200),
]

# Insert Tires (Now includes wheel radius values)
tires = [
    Tires(size="Small", cost=6.99, mass_kg=0.128, wheel_radius=0.055),
    Tires(size="Medium", cost=13.8, mass_kg=0.076, wheel_radius=0.0625),
    Tires(size="Large", cost=39.99, mass_kg=10.43, wheel_radius=0.19)
]

# Add Drone Bodies
drone_bodies = [
    DroneBody(material="Carbon Fiber", size="Small", cost=13.49, mass_kg=0.018),
    DroneBody(material="Carbon Fiber", size="Medium", cost=22.99, mass_kg=0.18),
    DroneBody(material="Metal",        size="Large", cost=348.34, mass_kg=3.5),
]

drone_motors = [
    DroneMotor(size="Small",  cost=27.08,  mass_kg=0.0137, torque=0.055,  power_watts= 280),
    DroneMotor(size="Medium", cost=56, mass_kg=0.0873, torque=0.283, power_watts=330),
    DroneMotor(size="Large",  cost=259.5, mass_kg=0.453, torque=5, power_watts=2437),
]

drone_batteries = [
    DroneBattery(size="Small",  cost=107,   mass_kg=0.728, capacity_wh=115.44),
    DroneBattery(size="Medium", cost=342.71,  mass_kg=1.44, capacity_wh=377.4),
    DroneBattery(size="Large",  cost=865.2,  mass_kg=4.13, capacity_wh=912),
]

# Modify Communication Methods (GPS Removed, New Added)
communications = [
    Communication(type="LoRa",  range_km=5, cost=32.19, power_watts=0.16),
    Communication(type="WiFi",  range_km=0.1,  cost=9.99, power_watts= 0.35),
    Communication(type="Zigbee",range_km=0.2,  cost=28.25, power_watts= 0.01),
    Communication(type="5G",    range_km=5, cost=44.99, power_watts=15),
]

computing_units = [
    # Model name           cost,  Geekbench 5 multi-core scores,  power (watts)
    ComputingUnit(model="Raspberry Pi 4 Model B 8GB",                  cost=75.00, performance=679, power_watts=15.0, unit_type="CPU"),
    ComputingUnit(model="GMKtec NucBoxG3 PLUS",                        cost=125.93, performance=2470, power_watts=6.0, unit_type="CPU"),
    ComputingUnit(model="Jetson Nano Dev Kit 2GB",                     cost=224.99, performance=864, power_watts=10.0, unit_type="GPU"),
    ComputingUnit(model="Jetson Xavier NX Dev Kit",                    cost=793, performance=2029, power_watts=15.0, unit_type="GPU"),
    ComputingUnit(model="Coral Dev Board",                             cost=129.99, performance=1277, power_watts=5.0, unit_type="TPU"),
]

applications = [
    Application(name="General crop monitoring",                    processing_mode="Off-board", platform="Both"),
    Application(name="Thermal imaging for canopy temperature",     processing_mode="Off-board", platform="Both"),
    Application(name="Image stitching",                            processing_mode="Off-board", platform="Both"),
    Application(name="Soil monitoring (close-up imaging)",         processing_mode="Off-board", platform="Rover"),
    Application(name="Yield estimation (fruit/veggie counting)",   processing_mode="Off-board", platform="Both"),
    Application(name="Quality control & grading of harvested produce", processing_mode="Off-board", platform="Rover"),
    Application(name="Autonomous fruit/vegetable picking",         processing_mode="Onboard", platform="Rover"),
    Application(name="Mechanical weeding & hoeing",                processing_mode="Onboard", platform="Rover"),
    Application(name="Soil pH sampling",                           processing_mode="Onboard", platform="Rover"),
    Application(name="Climate mapping (T/HR/CO₂/wind)",            processing_mode="Off-board", platform="Rover"),
    Application(name="Fence/irrigation/infrastructure inspection", processing_mode="Off-board", platform="Drone"),
    Application(name="Livestock monitoring & herding",             processing_mode="Onboard", platform="Both"),
    Application(name="Beehive inspection",                         processing_mode="Onboard", platform="Rover"),
    Application(name="Frost & pest early-warning systems",         processing_mode="Onboard", platform="Both"),
    Application(name="Fertilizing",                                processing_mode="Onboard", platform="Rover"),
]

db.add_all(applications)
db.commit()

components = [
    Component(name="High-resolution RGB camera",          cost=51.99, power_watts=1.4,  category="Sensor"),
    Component(name="GPS unit",                            cost=13.1,  power_watts=0.13, category="Sensor"),
    Component(name="IMU",                                 cost=34.95, power_watts=0.04, category="Sensor"),
    Component(name="Macro/microscope-lens camera",        cost=42.94, power_watts=0.7,  category="Sensor"),
    Component(name="Depth camera",                        cost=314,   power_watts=3.5,  category="Sensor"),
    Component(name="Robotic manipulator arm",             cost=274.7, power_watts=25,   category="Actuator"),
    Component(name="Gripper",                             cost=35.99, power_watts=5.0,  category="Actuator"),
    Component(name="LiDAR sensor",                        cost=129.95,power_watts=0.65, category="Sensor"),
    Component(name="Mechanical hoe/weed-slice attachment",cost=149.99,power_watts=0.0,  category="Peripheral"),
    Component(name="pH sensor",                           cost=30.99, power_watts=0.05, category="Sensor"),
    Component(name="Temp/humidity sensor",                cost=14.95, power_watts=0.003,category="Sensor"),
    Component(name="CO₂ sensor",                          cost=37.67, power_watts=0.75, category="Sensor"),
    Component(name="Anemometer",                          cost=44.95, power_watts=0.02, category="Sensor"),
    Component(name="Datalogger",                          cost=17.95, power_watts=0.1,  category="Peripheral"),
    Component(name="thermal camera",                      cost=74.95, power_watts=0.07, category="Sensor"),
    Component(name="Zoom lens",                           cost=62.5,  power_watts=0,    category="Actuator"),
    Component(name="Tag-reader (RFID/UWB)",               cost=5.69,  power_watts=0.1,  category="Sensor"),
    Component(name="Speaker/horn",                        cost=9.58,  power_watts=0.5,  category="Actuator"),
    Component(name="Chemical tank",                       cost=118.99,power_watts=0.0,  category="Structural"),
    Component(name="Nozzle array",                        cost=15.99, power_watts=0.0,  category="Structural"),
]

db.add_all(components)
db.commit()

edge_servers = [
    EdgeServer(name="Local Edge Server", cost=2000.0),
    ]

# Build a lookup of name→Component
comp_by_name = {c.name: c for c in db.query(Component).all()}
# Query back all application instances
apps = {a.name: a for a in db.query(Application).all()}

# mapping of application name → list of component names
mapping = {
    "General crop monitoring": [
        "High-resolution RGB camera", "GPS unit", "IMU"
    ],
    "Thermal imaging for canopy temperature": [
        "thermal camera"
    ],
    "Image stitching": [
        "High-resolution RGB camera", "GPS unit", "IMU"
    ],
    "Soil monitoring (close-up imaging)": [
        "Macro/microscope-lens camera"
    ],
    "Yield estimation (fruit/veggie counting)": [
        "Depth camera"
    ],
    "Quality control & grading of harvested produce": [
        "High-resolution RGB camera",
    ],
    "Autonomous fruit/vegetable picking": [
        "Robotic manipulator arm", "Gripper", "Depth camera",
        "LiDAR sensor"
    ],
    "Mechanical weeding & hoeing": [
        "Mechanical hoe/weed-slice attachment",
        "High-resolution RGB camera "
    ],
    "Soil pH sampling": [
        "pH sensor"
    ],
    "Climate mapping (T/HR/CO₂/wind)": [
        "Temp/humidity sensor", "CO₂ sensor", "Anemometer", "Datalogger", "GPS unit"
    ],
    "Fence/irrigation/infrastructure inspection": [
        "thermal camera", "Zoom lens", "GPS unit"
    ],
    "Livestock monitoring & herding": [
        "thermal camera", "Tag-reader (RFID/UWB)", "Speaker/horn"
    ],
    "Beehive inspection": [
        "Macro/microscope-lens camera"
    ],
    "Frost & pest early-warning systems": [
        "Temp/humidity sensor"
    ],
    "Fertilizing": [
        "Chemical tank", "Nozzle array", "GPS unit"
    ],
}

for app_name, comp_list in mapping.items():
    app = apps.get(app_name)
    for cname in comp_list:
        comp = comp_by_name.get(cname)
        if comp:
            app.components.append(comp)


# Add Components to the database
db.add_all(bodies)
db.add_all(motors)
db.add_all(batteries)
db.add_all(tires)
db.add_all(drone_bodies)
db.add_all(drone_motors)
db.add_all(drone_batteries)
db.add_all(communications)
db.add_all(computing_units)
db.add_all(edge_servers)
db.commit()
db.close()

print("Database populated successfully!")
