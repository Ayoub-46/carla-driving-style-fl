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
BEHAVIOR = 'aggressive'
LABEL_MAP = {'cautious': 0, 'normal': 1, 'aggressive': 2}
SPEED_LIMIT = 50.0

# --- SHARED DATA STORE ---
class SharedDataManager:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.new_data_event = threading.Event()  # "Data is ready"
        self.training_done_event = threading.Event() # "Training finished"
        self.stop_simulation_event = threading.Event() # "Kill sim"

    def set_data(self, data_buffer):
        """Called by Sim Thread to pass data to FL"""
        df = pd.DataFrame(data_buffer)
        X = df.drop(columns=['label'])
        y = df['label']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)
        
        # Wake up FL Client
        self.new_data_event.set()
        self.training_done_event.clear()

    def wait_for_data(self):
        """Called by FL Client (Main Thread) to block until data arrives"""
        print("[FL Client] Waiting for simulation data...")
        self.new_data_event.wait()
        self.new_data_event.clear()

    def signal_training_complete(self):
        """Called by FL Client after training"""
        self.training_done_event.set()

shared_manager = SharedDataManager()

# --- CARLA SIMULATION (BACKGROUND THREAD) ---
def run_carla_simulation(manager):
    """
    This function runs the entire CARLA loop in a separate thread.
    """
    client = None
    ego_vehicle = None
    imu_actor = None
    settings = None
    world = None

    try:
        # 1. Setup
        client = carla.Client('localhost', 3000)
        client.set_timeout(20.0)
        world = client.load_world('Town03')
        
        blueprint_library = world.get_blueprint_library()
        ego_bp = blueprint_library.find('vehicle.tesla.model3')
        ego_bp.set_attribute('role_name', 'hero')
        spawn_points = world.get_map().get_spawn_points()
        
        # Spawn safe
        start_pose = spawn_points[54]
        ego_vehicle = world.try_spawn_actor(ego_bp, start_pose)
        
        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_actor = world.try_spawn_actor(imu_bp, carla.Transform(carla.Location(x=0, z=1)), attach_to=ego_vehicle)

        agent = BehaviorAgent(ego_vehicle, behavior=BEHAVIOR)
        agent.set_destination(spawn_points[212].location)

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
            
            if agent.done():
                agent.set_destination(random.choice(spawn_points).location)
            
            control = agent.run_step()
            ego_vehicle.apply_control(control)

            # Data Collection
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
                "accel_y": acc.y, "brake_y": 0.0,
                "accel_z": acc.z, "brake_z": 0.0,
                "overspeed_count": overspeed_counter,
                "label": LABEL_MAP[BEHAVIOR]
            }
            data_buffer.append(row)

            # --- HANDOVER POINT ---
            if len(data_buffer) >= BATCH_SIZE:
                print(f"\n[Sim Thread] Buffer full ({len(data_buffer)}). Handing over to FL...")
                
                # 1. Send data
                manager.set_data(data_buffer)
                
                # 2. Block until FL is done
                print("[Sim Thread] Pausing for training...")
                manager.training_done_event.wait()
                
                print("[Sim Thread] Resuming drive...")
                data_buffer = []

    except Exception as e:
        print(f"[Sim Thread] Error: {e}")
    finally:
        print("[Sim Thread] Cleaning up...")
        if settings:
            settings.synchronous_mode = False
            world.apply_settings(settings)
        if ego_vehicle: ego_vehicle.destroy()
        if imu_actor: imu_actor.destroy()


# --- FL CLIENT (MAIN THREAD) ---
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
        # BLOCK: Wait for Sim Thread
        self.manager.wait_for_data()
        
        print("[FL Client] Training started...")
        X_train = self.manager.X_train
        y_train = self.manager.y_train

        # Load Global Model
        global_model = None
        if ins.parameters.tensors:
            try:
                global_model_bytes = bytearray(ins.parameters.tensors[0])
                global_model = xgb.Booster()
                global_model.load_model(global_model_bytes)
            except: pass

        # Train
        dtrain = xgb.DMatrix(X_train, label=y_train)
        self.bst = xgb.train(self.params, dtrain, num_boost_round=1, xgb_model=global_model)
        
        # Save
        local_model_bytes = bytes(self.bst.save_raw("json"))

        # UNBLOCK: Tell Sim Thread to continue
        self.manager.signal_training_complete()

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensors=[local_model_bytes], tensor_type="bytes"),
            num_examples=len(X_train),
            metrics={}
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        if self.manager.X_test is None:
             return EvaluateRes(status=Status(code=Code.OK, message="No Data"), loss=0.0, num_examples=0, metrics={})
        
        # Load model for eval
        model_bytes = bytearray(ins.parameters.tensors[0])
        bst = xgb.Booster()
        bst.load_model(model_bytes)
        
        dtest = xgb.DMatrix(self.manager.X_test, label=self.manager.y_test)
        preds = bst.predict(dtest)
        preds_labels = np.argmax(preds, axis=1)
        
        acc = accuracy_score(self.manager.y_test, preds_labels)
        loss = 1.0 - acc

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=float(loss),
            num_examples=len(self.manager.X_test),
            metrics={"accuracy": float(acc)}
        )

# --- ENTRY POINT ---
def main():
    # 1. Start CARLA in Background Thread
    sim_thread = threading.Thread(target=run_carla_simulation, args=(shared_manager,), daemon=True)
    sim_thread.start()
    
    # Give sim a second to init
    time.sleep(2)

    # 2. Start Flower Client in Main Thread (This blocks!)
    print("[Main] Connecting to FL Server...")
    client = AsyncDrivingClient(shared_manager)
    
    try:
        # This will block the main thread until the server tells it to stop
        fl.client.start_client(server_address=SERVER_ADDRESS, client=client)
    except KeyboardInterrupt:
        print("[Main] Stopping...")
    except Exception as e:
        print(f"[Main] FL Error: {e}")
    finally:
        # Signal Sim Thread to stop
        shared_manager.stop_simulation_event.set()
        # Wake up Sim Thread if it's waiting for training
        shared_manager.training_done_event.set()
        sim_thread.join(timeout=2)
        print("[Main] Done.")

if __name__ == '__main__':
    main()