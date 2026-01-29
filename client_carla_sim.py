import sys
import glob
import os
import time
import random
import math
import argparse
import pandas as pd
import numpy as np

import flwr as fl
from sklearn.model_selection import train_test_split
from fl.client import DrivingXgbClient  

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
from agents.navigation.global_route_planner import GlobalRoutePlanner

SERVER_ADDRESS = "127.0.0.1:8080"
BATCH_SIZE = 1000       
BEHAVIOR = 'aggressive' # 'cautious', 'normal', 'aggressive'
LABEL_MAP = {'cautious': 0, 'normal': 1, 'aggressive': 2}
SPEED_LIMIT = 50.0      # km/h (for calculating overspeed_count)

def generate_route(world, start, end):
    """Generates a route using the GlobalRoutePlanner."""
    grp = GlobalRoutePlanner(world.get_map(), sampling_resolution=2.0)
    return grp.trace_route(start, end)

def main():
    # 1. SETUP CARLA WORLD
    client = carla.Client('localhost', 3000)
    client.set_timeout(20.0)
    
    world = client.load_world('Town03')
    map_inst = world.get_map()
    blueprint_library = world.get_blueprint_library()
    
    # 2. SPAWN HERO VEHICLE
    ego_bp = blueprint_library.find('vehicle.tesla.model3')
    ego_bp.set_attribute('role_name', 'hero')
    spawn_points = map_inst.get_spawn_points()
    start_spawn = spawn_points[54]
    ego_vehicle = world.try_spawn_actor(ego_bp, start_spawn)

    # 3. SETUP SENSORS (IMU)
    
    imu_bp = blueprint_library.find('sensor.other.imu')
    imu_actor = world.try_spawn_actor(imu_bp, carla.Transform(carla.Location(x=0, z=1)), attach_to=ego_vehicle)

    # 4. SETUP AGENT
    agent = BehaviorAgent(ego_vehicle, behavior=BEHAVIOR)
    destination = spawn_points[212].location
    agent.set_destination(destination)
    
    # 5. SETUP TRAFFIC (NPCs)
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    npc_vehicles = []
    
    for _ in range(30): 
        bp = random.choice(blueprint_library.filter('vehicle'))
        sp = random.choice(spawn_points)
        npc = world.try_spawn_actor(bp, sp)
        if npc:
            npc.set_autopilot(True, tm.get_port())
            npc_vehicles.append(npc)

    # 6. SETUP SIMULATION SETTINGS
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # 7. MAIN DATA LOOP
    data_buffer = []
    total_distance = 0.0
    overspeed_counter = 0
    last_location = ego_vehicle.get_location()

    print(f"Simulation started. Driving style: {BEHAVIOR.upper()}")
    print(f"Collecting {BATCH_SIZE} samples before FL training...")

    try:
        while True:
            world.tick()

            if agent.done():
                print("Destination reached. Resetting route...")
                agent.set_destination(random.choice(spawn_points).location)
            
            control = agent.run_step()
            ego_vehicle.apply_control(control)

            velocity = ego_vehicle.get_velocity()
            accel = ego_vehicle.get_acceleration()
            gyro = ego_vehicle.get_angular_velocity()
            
            speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # 2. Derived Metrics
            curr_loc = ego_vehicle.get_location()
            dist_step = curr_loc.distance(last_location)
            total_distance += dist_step
            last_location = curr_loc

            if speed > SPEED_LIMIT:
                overspeed_counter += 1
            

            row = {
                "distance": total_distance,
                "speed": speed,
                "gyro_x": gyro.x, "gyro_y": gyro.y, "gyro_z": gyro.z,
                "accel_x": accel.x, 
                "brake_x": control.brake, # Map brake control to X
                "accel_y": accel.y, 
                "brake_y": 0.0,           # Scalar brake usually applies to all, but we zero these to match schema
                "accel_z": accel.z, 
                "brake_z": 0.0,
                "overspeed_count": overspeed_counter,
                "label": LABEL_MAP[BEHAVIOR]
            }
            data_buffer.append(row)

            # FL TRIGGER 
            if len(data_buffer) >= BATCH_SIZE:
                print(f"\n[FL] Buffer full ({len(data_buffer)}). Pausing sim to train...")
                
                df = pd.DataFrame(data_buffer)
                avg_speed = df['speed'].mean()
                avg_dist = df['distance'].mean()
                print(f"[DEBUG] Batch Stats -> Avg Speed: {avg_speed:.2f} | Avg Distance: {avg_dist:.2f}")
                # --------------------------


                df = pd.DataFrame(data_buffer)
                X = df.drop(columns=['label'])
                y = df['label']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                
                fl_client = DrivingXgbClient(X_train, y_train, X_test, y_test)
                
                # 3. Connect to Server
                try:
                    # If this succeeds, it runs for 1 round then returns
                    fl.client.start_client(
                        server_address=SERVER_ADDRESS,
                        client=fl_client
                    )
                    print("[FL] Training round complete. Resuming simulation...")
                    
                    # 4. Reset Buffer ONLY if training was successful
                    data_buffer = []

                except Exception as e:
                    # If the server is down (Experiment Over), start_client will throw an exception
                    print(f"\n[FL] Connection failed: {e}")
                    print("[FL] The server appears to be down. Assuming experiment is complete.")
                    print("Stopping Simulation...")
                    break

                # overspeed_counter = 0 

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)
        if ego_vehicle: ego_vehicle.destroy()
        if imu_actor: imu_actor.destroy()
        for npc in npc_vehicles: 
            if npc.is_alive: npc.destroy()

if __name__ == '__main__':
    main()