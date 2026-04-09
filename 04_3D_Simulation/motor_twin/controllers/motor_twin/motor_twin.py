from controller import Supervisor
import math
import os

"""
Webots Motor Twin Supervisor Controller - Dataset-Driven Health
Electric Motor Digital Twin with NASA C-MAPSS Data
"""

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_PATH = r"F:\motor_twin\train_FD001.txt"

SELECTED_UNIT = 1
WINDOW_SIZE = 50
STEP_INTERVAL_SECONDS = 0.5
LOOP_SEQUENCE = True

# ============================================================================
# ROTOR CONFIGURATION
# ============================================================================

TARGET_ROTOR_RPM = 1000
TARGET_ROTOR_OMEGA = (TARGET_ROTOR_RPM / 60.0) * 2.0 * math.pi
ROTOR_MOTOR_NAME = "rotor_motor"
ROTOR_POSITION_SENSOR_NAME = "rotor_position"

# ============================================================================
# PBRAppearance DEF NAMES
# ============================================================================

MOTOR_BODY_APPEARANCE_DEF = "MOTOR_BODY_APPEARANCE"
SENSOR_APPEARANCE_DEFS = [
    "SENSOR_APPEARANCE_1",
    "SENSOR_APPEARANCE_2",
    "SENSOR_APPEARANCE_3",
]

# ============================================================================
# COLOR PALETTES
# ============================================================================

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


def classify_health_state(health_value):
    health_value = clamp(health_value)
    if health_value >= 0.7:
        return 'healthy'
    elif health_value >= 0.4:
        return 'warning'
    return 'critical'


def interpolate_color(c1, c2, t):
    t = clamp(t)
    return tuple(c1[i] * (1.0 - t) + c2[i] * t for i in range(3))


def get_color_for_health(health_value, component_type='body'):
    palette = COLOR_PALETTES[component_type]
    health_value = clamp(health_value)

    if health_value >= 0.7:
        return palette['healthy']
    elif health_value >= 0.4:
        t = (health_value - 0.4) / 0.3
        return interpolate_color(palette['critical'], palette['warning'], t)
    else:
        return palette['critical']


class CMAPSSDataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.raw_data = []
        self.units = {}
        self._load_dataset()

    def _load_dataset(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"[ERROR] Dataset file not found: {self.dataset_path}\n"
                f"        Put train_FD001.txt in project root:\n"
                f"        {PROJECT_ROOT}"
            )

        print(f"[INFO] Loading dataset from: {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                values = line.split()
                if len(values) < 26:
                    print(f"[WARNING] Line {line_num} has {len(values)} columns, expected 26. Skipping.")
                    continue

                try:
                    unit_id = int(values[0])
                    cycle = int(values[1])
                    row = {
                        "unit_id": unit_id,
                        "cycle": cycle,
                        "op_setting_1": float(values[2]),
                        "op_setting_2": float(values[3]),
                        "op_setting_3": float(values[4]),
                        "sensor_values": [float(v) for v in values[5:26]],
                    }
                except Exception as e:
                    print(f"[WARNING] Parse error at line {line_num}: {e}. Skipping.")
                    continue

                self.raw_data.append(row)
                self.units.setdefault(unit_id, []).append(row)

        print(f"[INFO] Dataset loaded: {len(self.raw_data)} total rows")
        print(f"[INFO] Found {len(self.units)} unique units")

    def get_unit_data(self, unit_id):
        if unit_id not in self.units:
            available = sorted(self.units.keys())
            raise ValueError(
                f"[ERROR] Unit {unit_id} not found.\n"
                f"        Available units: {available}"
            )
        return sorted(self.units[unit_id], key=lambda r: r["cycle"])

    def extract_window(self, unit_id, window_size):
        unit_data = self.get_unit_data(unit_id)
        if window_size > len(unit_data):
            raise ValueError(
                f"[ERROR] WINDOW_SIZE={window_size} exceeds unit {unit_id} cycles ({len(unit_data)})."
            )
        return unit_data[-window_size:]


class PBRAppearanceManager:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.body_appearance_node = None
        self.sensor_appearance_nodes = {}
        self._cache_nodes()

    def _cache_nodes(self):
        self.body_appearance_node = self.supervisor.getFromDef(MOTOR_BODY_APPEARANCE_DEF)
        if self.body_appearance_node is None:
            print(f"[WARNING] Missing DEF: {MOTOR_BODY_APPEARANCE_DEF}")
        else:
            print(f"[INFO] Found motor body PBRAppearance: {MOTOR_BODY_APPEARANCE_DEF}")

        for def_name in SENSOR_APPEARANCE_DEFS:
            node = self.supervisor.getFromDef(def_name)
            if node is None:
                print(f"[WARNING] Missing DEF: {def_name}")
            else:
                self.sensor_appearance_nodes[def_name] = node
                print(f"[INFO] Found sensor PBRAppearance: {def_name}")

    def _set_base_color(self, pbr_node, color):
        if pbr_node is None:
            return False
        try:
            field = pbr_node.getField("baseColor")
            if field is None:
                print("[ERROR] Node has no baseColor field")
                return False
            field.setSFColor([color[0], color[1], color[2]])
            return True
        except Exception as e:
            print(f"[ERROR] Failed to update baseColor: {e}")
            return False

    def update_body_color(self, health_value):
        color = get_color_for_health(health_value, 'body')
        self._set_base_color(self.body_appearance_node, color)

    def update_sensor_colors(self, health_value):
        color = get_color_for_health(health_value, 'sensor')
        for node in self.sensor_appearance_nodes.values():
            self._set_base_color(node, color)

    def update_colors(self, health_value):
        self.update_body_color(health_value)
        self.update_sensor_colors(health_value)


class DatasetHealthManager:
    def __init__(self, dataset, unit_id, window_size, step_interval_seconds, loop_sequence):
        self.dataset = dataset
        self.unit_id = unit_id
        self.window_size = window_size
        self.step_interval_seconds = step_interval_seconds
        self.loop_sequence = loop_sequence

        self.window_data = self._extract_and_process_window()
        self.current_index = 0
        self.elapsed_time = 0.0
        self.is_finished = False

        print("[INFO] Dataset health manager initialized")
        print(f"       Unit: {unit_id}")
        print(f"       Window size: {window_size} cycles")
        print(f"       Step interval: {step_interval_seconds}s")
        print(f"       Loop on end: {loop_sequence}")
        print(f"       Ready to play back {len(self.window_data)} cycles")

    def _extract_and_process_window(self):
        raw_window = self.dataset.extract_window(self.unit_id, self.window_size)
        last_cycle = raw_window[-1]["cycle"]

        processed = []
        for row in raw_window:
            true_rul = last_cycle - row["cycle"]
            health = clamp(true_rul / float(self.window_size))
            processed.append({
                "cycle": row["cycle"],
                "true_rul": true_rul,
                "health": health,
            })
        return processed

    def get_health_value(self, dt):
        if self.is_finished and not self.loop_sequence:
            return self.window_data[-1]["health"]

        self.elapsed_time += dt
        if self.elapsed_time >= self.step_interval_seconds:
            self.elapsed_time -= self.step_interval_seconds
            self.current_index += 1

            if self.current_index >= len(self.window_data):
                if self.loop_sequence:
                    self.current_index = 0
                    print("[INFO] Reached end of window, looping back to start")
                else:
                    self.current_index = len(self.window_data) - 1
                    self.is_finished = True
                    print("[INFO] Reached end of window, stopping at final state")

        return self.window_data[self.current_index]["health"]

    def get_playback_info(self):
        row = self.window_data[self.current_index]
        return {
            "cycle": row["cycle"],
            "true_rul": row["true_rul"],
            "health": row["health"],
            "index": self.current_index,
            "total": len(self.window_data),
        }


class RotorController:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.motor = supervisor.getDevice(ROTOR_MOTOR_NAME)
        if self.motor is None:
            raise RuntimeError(f"Motor '{ROTOR_MOTOR_NAME}' not found")

        self.motor.setPosition(float("inf"))
        self.motor.setVelocity(TARGET_ROTOR_OMEGA)

        self.position_sensor = None
        try:
            self.position_sensor = supervisor.getDevice(ROTOR_POSITION_SENSOR_NAME)
            if self.position_sensor is not None:
                self.position_sensor.enable(int(supervisor.getBasicTimeStep()))
        except Exception:
            self.position_sensor = None

        print(f"[INFO] Rotor controller initialized")
        print(f"       Target speed: {TARGET_ROTOR_RPM} RPM ({TARGET_ROTOR_OMEGA:.2f} rad/s)")

    def update(self):
        self.motor.setVelocity(TARGET_ROTOR_OMEGA)

    def stop(self):
        self.motor.setVelocity(0.0)


class MotorTwinDatasetController:
    def __init__(self):
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())

        print("=" * 70)
        print("MOTOR TWIN DATASET-DRIVEN CONTROLLER INITIALIZED")
        print("=" * 70)

        self.dataset = CMAPSSDataset(DATASET_PATH)
        self.rotor = RotorController(self.supervisor)
        self.appearance = PBRAppearanceManager(self.supervisor)
        self.health_state = DatasetHealthManager(
            dataset=self.dataset,
            unit_id=SELECTED_UNIT,
            window_size=WINDOW_SIZE,
            step_interval_seconds=STEP_INTERVAL_SECONDS,
            loop_sequence=LOOP_SEQUENCE,
        )

        self.current_health_state = None
        print(f"[INFO] Timestep: {self.timestep}ms")
        print("=" * 70)

    def update_health_visualization(self, health_value):
        state = classify_health_state(health_value)
        if state != self.current_health_state:
            self.current_health_state = state
            info = self.health_state.get_playback_info()
            print(
                f"[HEALTH] State: {state} | "
                f"Cycle: {info['cycle']} | "
                f"RUL: {info['true_rul']} | "
                f"Health: {health_value:.2f}"
            )
        self.appearance.update_colors(health_value)

    def run(self):
        print("[START] Entering main control loop...\n")
        while self.supervisor.step(self.timestep) != -1:
            dt = self.timestep / 1000.0
            self.rotor.update()
            health_value = self.health_state.get_health_value(dt)
            self.update_health_visualization(health_value)

    def cleanup(self):
        try:
            self.rotor.stop()
        except Exception:
            pass
        print("\n[SHUTDOWN] Motor stopped. Exiting.")


if __name__ == "__main__":
    controller = None
    try:
        controller = MotorTwinDatasetController()
        controller.run()
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Ctrl+C pressed")
        if controller is not None:
            controller.cleanup()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        if controller is not None:
            controller.cleanup()
