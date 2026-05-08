from controller import Supervisor
import math
import os
import json
import paho.mqtt.client as mqtt

"""
Webots Motor Twin - IoT Edge Node
Streams raw sensors via MQTT and reacts to AI diagnostics.
"""

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATASET_PATH = os.path.join(PROJECT_ROOT, '01_AI_and_Data', 'data', 'raw', 'CMAPSSData', 'test_FD001.txt')
SELECTED_UNIT = 34 
STEP_INTERVAL_SECONDS = 0.5

# MQTT Setup
BROKER_ADDRESS = "127.0.0.1"
PORT = 1883
RAW_TOPIC = "digital_twin/raw_sensors"
AI_TOPIC = "digital_twin/engine_telemetry"

# Rotor & Appearance
TARGET_ROTOR_RPM = 1000
TARGET_ROTOR_OMEGA = (TARGET_ROTOR_RPM / 60.0) * 2.0 * math.pi
ROTOR_MOTOR_NAME = "rotor_motor"
MOTOR_BODY_APPEARANCE_DEF = "MOTOR_BODY_APPEARANCE"
SENSOR_APPEARANCE_DEFS = ["SENSOR_APPEARANCE_1", "SENSOR_APPEARANCE_2", "SENSOR_APPEARANCE_3"]

# Colors
COLOR_PALETTES = {
    'body': {
        'healthy':  (0.40, 0.50, 0.55),
        'warning':  (0.50, 0.48, 0.40),
        'critical': (0.55, 0.35, 0.30),
    },
    'sensor': {
        'healthy':  (0.50, 0.50, 0.50),
        'warning':  (0.80, 0.65, 0.30),
        'critical': (0.80, 0.20, 0.20),
    }
}

def clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, value))

def interpolate_color(c1, c2, t):
    t = clamp(t)
    return tuple(c1[i] * (1.0 - t) + c2[i] * t for i in range(3))

def get_color_for_health(health_value, component_type='body'):
    palette = COLOR_PALETTES[component_type]
    health_value = clamp(health_value)

    # Adjusted thresholds for a more realistic degradation curve
    # >= 50% is Healthy, 20% to 50% is Warning, < 20% is Critical
    if health_value >= 0.5:
        return palette['healthy']
    elif health_value >= 0.2:
        t = (health_value - 0.2) / 0.3
        return interpolate_color(palette['critical'], palette['warning'], t)
    else:
        return palette['critical']

# ============================================================================
# DATASET PARSER
# ============================================================================
class CMAPSSDataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.units = {}
        self._load_dataset()

    def _load_dataset(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"[ERROR] Dataset not found at: {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.strip().split()
                if len(values) < 26: continue
                unit_id = int(values[0])
                self.units.setdefault(unit_id, []).append({
                    "engine_id": unit_id,
                    "cycle": int(values[1]),
                    "settings": [float(v) for v in values[2:5]],
                    "sensors": [float(v) for v in values[5:26]],
                })
    def get_unit_data(self, unit_id):
        return sorted(self.units.get(unit_id, []), key=lambda r: r["cycle"])

# ============================================================================
# HARDWARE CONTROLLERS
# ============================================================================
class PBRAppearanceManager:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.body_node = supervisor.getFromDef(MOTOR_BODY_APPEARANCE_DEF)
        self.sensor_nodes = [supervisor.getFromDef(d) for d in SENSOR_APPEARANCE_DEFS if supervisor.getFromDef(d)]

    def _set_color(self, node, color):
        if node and node.getField("baseColor"):
            node.getField("baseColor").setSFColor([color[0], color[1], color[2]])

    def update_colors(self, health_value):
        self._set_color(self.body_node, get_color_for_health(health_value, 'body'))
        for node in self.sensor_nodes:
            self._set_color(node, get_color_for_health(health_value, 'sensor'))

class RotorController:
    def __init__(self, supervisor):
        self.motor = supervisor.getDevice(ROTOR_MOTOR_NAME)
        if self.motor:
            self.motor.setPosition(float("inf"))
            self.motor.setVelocity(TARGET_ROTOR_OMEGA)

    def update(self):
        if self.motor: self.motor.setVelocity(TARGET_ROTOR_OMEGA)
    def stop(self):
        if self.motor: self.motor.setVelocity(0.0)

# ============================================================================
# MAIN EDGE NODE CONTROLLER
# ============================================================================
class IoTEdgeNode:
    def __init__(self):
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.dataset = CMAPSSDataset(DATASET_PATH)
        self.unit_data = self.dataset.get_unit_data(SELECTED_UNIT)
        self.current_index = 0
        self.elapsed_time = 0.0
        
        self.rotor = RotorController(self.supervisor)
        self.appearance = PBRAppearanceManager(self.supervisor)
        self.current_health = 1.0 # 1.0 means 100% healthy
        
        self.setup_mqtt()

    def setup_mqtt(self):
        # Setup MQTT Client (using Version 2 API as we fixed earlier)
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "Webots_IoT_Node")
        self.mqtt_client.on_message = self.on_mqtt_message
        try:
            self.mqtt_client.connect(BROKER_ADDRESS, PORT, 60)
            self.mqtt_client.subscribe(AI_TOPIC)
            self.mqtt_client.loop_start()
            print("[INFO] Webots Connected to MQTT Broker!")
        except Exception as e:
            print(f"[ERROR] MQTT Connection failed: {e}")

    def on_mqtt_message(self, client, userdata, msg):
        # Listen to AI predictions to change 3D model color
        try:
            payload = json.loads(msg.payload.decode())
            predicted_rul = payload.get("predicted_rul", 100)
            
            # Map RUL to Health (0.0-1.0) using 100 as a standard max for test engines
            # Example: RUL 75 -> Health 0.75 (Will stay in the 'healthy' color zone)
            self.current_health = clamp(predicted_rul / 100.0)
        except Exception as e:
            pass

    def run(self):
        print("\n[START] Webots IoT Node Streaming Data...\n")
        while self.supervisor.step(self.timestep) != -1:
            self.rotor.update()
            
            # Apply color updates based on AI feedback
            self.appearance.update_colors(self.current_health)

            # Stream data every interval
            dt = self.timestep / 1000.0
            self.elapsed_time += dt
            if self.elapsed_time >= STEP_INTERVAL_SECONDS:
                self.elapsed_time -= STEP_INTERVAL_SECONDS
                
                if self.current_index < len(self.unit_data):
                    cycle_data = self.unit_data[self.current_index]
                    
                    # Publish raw sensor data
                    self.mqtt_client.publish(RAW_TOPIC, json.dumps(cycle_data))
                    print(f"[PUBLISH] Sent Cycle {cycle_data['cycle']} to AI Engine | Current AI Health: {self.current_health:.2f}")
                    
                    self.current_index += 1
                else:
                    print("[INFO] End of engine lifecycle.")
                    break

    def cleanup(self):
        self.rotor.stop()
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        print("\n[SHUTDOWN] System offline.")

if __name__ == "__main__":
    node = IoTEdgeNode()
    try:
        node.run()
    finally:
        node.cleanup()