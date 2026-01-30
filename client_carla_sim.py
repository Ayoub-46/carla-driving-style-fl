import sys
import glob
import os
import time
import random
import math
import threading
import pandas as pd
import numpy as np

# --- FLOWER & ML IMPORTS ---
import flwr as fl
from flwr.common import Code, FitIns, FitRes, Status, Parameters, EvaluateIns, EvaluateRes, GetParametersIns, GetParametersRes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

from utils.data_loader import load_data 

# --- CARLA SETUP (Paths) ---
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
sys.path.append('/home/ayoub/Documents/CARLA/PythonAPI/carla')

import carla
from agents.navigation.behavior_agent import BehaviorAgent

# --- CONFIGURATION ---
SERVER_ADDRESS = "127.0.0.1:8080"
BATCH_SIZE = 1000
BEHAVIOR = 'normal'
LABEL_MAP = {'cautious': 0, 'normal': 1, 'aggressive': 2}
SPEED_LIMIT = 50.0
DATA_PATH = "data/dataset.csv" 
NUM_NPC_VEHICLES = 50  # Number of traffic vehicles

# --- SHARED DATA STORE ---
class SharedDataManager:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_eval = None
        self.y_eval = None
        self._load_eval_set()
        
        self.new_data_event = threading.Event()  
        self.training_done_event = threading.Event() 
        self.stop_simulation_event = threading.Event() 

    def _load_eval_set(self):
        """Loads a STRATIFIED SUBSET of the dataset for evaluation."""
        if os.path.exists(DATA_PATH):
            print(f"[Data Manager] Loading diverse eval set from {DATA_PATH}...")
            try:
                X, y, _ = load_data(DATA_PATH)
                SUBSET_SIZE = 1000
                if len(X) > SUBSET_SIZE:
                    _, X_subset, _, y_subset = train_test_split(
                        X, y, 
                        test_size=SUBSET_SIZE, 
                        stratify=y, 
                        random_state=42
                    )
                    self.X_eval = X_subset
                    self.y_eval = y_subset
                    print(f"[Data Manager] Created diverse subset: {len(self.X_eval)} samples.")
                else:
                    self.X_eval = X
                    self.y_eval = y
                    print(f"[Data Manager] Loaded full dataset ({len(X)} samples).")
            except Exception as e:
                print(f"[Data Manager] Failed to load eval set: {e}")
                self.X_eval = None
        else:
            print(f"[Data Manager] Warning: {DATA_PATH} not found.")

    def set_data(self, data_buffer):
        print(f"\n[Data Manager] Processing training batch ({len(data_buffer)} samples)...")
        df = pd.DataFrame(data_buffer)
        self.X_train = df.drop(columns=['label'])
        self.y_train = df['label']
        self.training_done_event.clear()
        self.new_data_event.set()

    def wait_for_data(self):
        if not self.new_data_event.is_set():
            print("[FL Client] Waiting for simulation data...")
            self.new_data_event.wait()
        self.new_data_event.clear()

    def signal_training_complete(self):
        self.training_done_event.set()

shared_manager = SharedDataManager()

# --- CARLA SIMULATION ---
def run_carla_simulation(manager):
    client = None
    ego_vehicle = None
    imu_actor = None
    settings = None
    world = None
    npc_vehicles = [] # Keep track of NPCs to destroy them later

    try:
        client = carla.Client('localhost', 3000)
        client.set_timeout(20.0)
        world = client.load_world('Town03')
        spectator = world.get_spectator()
        
        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        
        # 1. Spawn Ego Vehicle
        ego_bp = blueprint_library.find('vehicle.tesla.model3')
        ego_bp.set_attribute('role_name', 'hero')
        start_pose = spawn_points[54]
        ego_vehicle = world.try_spawn_actor(ego_bp, start_pose)
        
        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_actor = world.try_spawn_actor(imu_bp, carla.Transform(carla.Location(x=0, z=1)), attach_to=ego_vehicle)
        
        agent = BehaviorAgent(ego_vehicle, behavior=BEHAVIOR)
        agent.set_destination(spawn_points[212].location)
        
        # --- NEW: Spawn Traffic (NPCs) ---
        print(f"[Sim Thread] Spawning {NUM_NPC_VEHICLES} NPC vehicles...")
        tm = client.get_trafficmanager(8000) # Default port is 8000
        tm.set_synchronous_mode(True)
        
        for _ in range(NUM_NPC_VEHICLES):
            try:
                bp = random.choice(blueprint_library.filter('vehicle'))
                sp = random.choice(spawn_points)
                npc = world.try_spawn_actor(bp, sp)
                if npc:
                    npc.set_autopilot(True, tm.get_port())
                    npc_vehicles.append(npc)
            except Exception:
                pass # Skip if spawn failed (collision etc.)
        print(f"[Sim Thread] {len(npc_vehicles)} NPCs spawned.")
        # ---------------------------------

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        data_buffer = []
        total_distance = 0.0
        overspeed_counter = 0
        last_location = ego_vehicle.get_location()

        print(f"[Sim Thread] Simulation Started. Mode: {BEHAVIOR}")
        while not manager.stop_simulation_event.is_set():
            world.tick()
            
            # Chase Camera
            if ego_vehicle:
                t = ego_vehicle.get_transform()
                yaw_rad = math.radians(t.rotation.yaw)
                cam_loc = carla.Location(
                    x=t.location.x - 10 * math.cos(yaw_rad),
                    y=t.location.y - 10 * math.sin(yaw_rad),
                    z=t.location.z + 5
                )
                spectator.set_transform(carla.Transform(cam_loc, carla.Rotation(pitch=-15, yaw=t.rotation.yaw)))

            if agent.done(): agent.set_destination(random.choice(spawn_points).location)
            
            control = agent.run_step()
            ego_vehicle.apply_control(control)
            
            vel = ego_vehicle.get_velocity()
            acc = ego_vehicle.get_acceleration()
            gyro = ego_vehicle.get_angular_velocity()
            speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            
            curr_loc = ego_vehicle.get_location()
            total_distance += curr_loc.distance(last_location)
            last_location = curr_loc
            if speed > SPEED_LIMIT: overspeed_counter += 1

            row = {
                "distance": total_distance, "speed": speed,
                "gyro_x": gyro.x, "gyro_y": gyro.y, "gyro_z": gyro.z,
                "accel_x": acc.x, "brake_x": control.brake, 
                "accel_y": acc.y, "brake_y": 0.0, "accel_z": acc.z, "brake_z": 0.0,
                "overspeed_count": overspeed_counter, "label": LABEL_MAP[BEHAVIOR]
            }
            data_buffer.append(row)

            if len(data_buffer) >= BATCH_SIZE:
                print(f"\n[Sim Thread] Buffer full ({len(data_buffer)}). Handing over...")
                manager.set_data(data_buffer)
                print("[Sim Thread] Pausing for training...")
                manager.training_done_event.wait()
                print("[Sim Thread] Resuming drive...")
                data_buffer = []
                
    except Exception as e: print(f"[Sim Error] {e}")
    finally:
        print("[Sim Thread] Cleaning up actors...")
        if settings: settings.synchronous_mode = False; world.apply_settings(settings)
        if ego_vehicle: ego_vehicle.destroy()
        if imu_actor: imu_actor.destroy()
        # Clean up NPCs
        for npc in npc_vehicles:
            if npc.is_alive: npc.destroy()

# --- FL CLIENT ---
class AsyncDrivingClient(fl.client.Client):
    def __init__(self, data_manager):
        self.manager = data_manager
        self.params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eta': 0.1,
            'max_depth': 6,
            'eval_metric': 'mlogloss',
        }
        self.bst = None

    def get_parameters(self, ins):
        return GetParametersRes(status=Status(code=Code.OK, message="OK"), parameters=Parameters(tensors=[], tensor_type="bytes"))

    def fit(self, ins: FitIns) -> FitRes:
        self.manager.wait_for_data()
        print("[FL Client] Training started on new simulation batch...")
        X_train = self.manager.X_train
        y_train = self.manager.y_train

        global_model = None
        if ins.parameters.tensors:
            try:
                global_model_bytes = bytearray(ins.parameters.tensors[0])
                global_model = xgb.Booster()
                global_model.load_model(global_model_bytes)
            except: pass

        dtrain = xgb.DMatrix(X_train, label=y_train)
        self.bst = xgb.train(self.params, dtrain, num_boost_round=1, xgb_model=global_model)
        local_model_bytes = bytes(self.bst.save_raw("json"))
        self.manager.signal_training_complete()
        return FitRes(status=Status(code=Code.OK, message="OK"), parameters=Parameters(tensors=[local_model_bytes], tensor_type="bytes"), num_examples=len(X_train), metrics={})

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        if self.manager.X_eval is None:
             return EvaluateRes(status=Status(code=Code.OK, message="No Data"), loss=0.0, num_examples=0, metrics={})
        
        model_bytes = bytearray(ins.parameters.tensors[0])
        bst = xgb.Booster()
        bst.load_model(model_bytes)
        
        dtest = xgb.DMatrix(self.manager.X_eval, label=self.manager.y_eval)
        preds = bst.predict(dtest)
        
        if len(preds.shape) > 1 and preds.shape[1] > 1:
            preds_labels = np.argmax(preds, axis=1)
        else:
            preds_labels = preds
        
        acc = accuracy_score(self.manager.y_eval, preds_labels)
        loss_val = 1.0 - acc

        print(f"[FL Client] Eval on Diverse Subset -> Accuracy: {acc:.4f} | Loss: {loss_val:.4f}")

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=float(loss_val),
            num_examples=len(self.manager.X_eval),
            metrics={"accuracy": float(acc), "loss": float(loss_val)}
        )

def main():
    sim_thread = threading.Thread(target=run_carla_simulation, args=(shared_manager,), daemon=True)
    sim_thread.start()
    
    print("[Main] Simulation started. Waiting for FIRST batch...")
    shared_manager.new_data_event.wait()
    print("[Main] Data ready! Connecting to FL Server...")

    client = AsyncDrivingClient(shared_manager)
    try:
        fl.client.start_client(server_address=SERVER_ADDRESS, client=client)
    except KeyboardInterrupt: pass
    except Exception as e: print(e)
    finally:
        shared_manager.stop_simulation_event.set()
        shared_manager.training_done_event.set()
        sim_thread.join(timeout=2)

if __name__ == '__main__':
    main()