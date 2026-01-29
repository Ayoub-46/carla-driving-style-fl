# utils/carla_monitor.py
import carla
import pandas as pd
import time
import math

class CarlaMonitor:
    def __init__(self, host='localhost', port=2000):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.vehicle = None
        self.imu_sensor = None
        self.data_buffer = []
        self.total_distance = 0.0
        self.last_location = None

    def attach_to_hero(self, role_name='hero'):
        """Finds the vehicle labeled 'hero' and attaches an IMU."""
        while self.vehicle is None:
            print("Waiting for hero vehicle...")
            actors = self.world.get_actors().filter('vehicle.*')
            for actor in actors:
                if actor.attributes.get('role_name') == role_name:
                    self.vehicle = actor
                    break
            time.sleep(1)
        
        print(f"Attached to {self.vehicle.type_id}")
        self.last_location = self.vehicle.get_location()
        self._attach_imu()

    def _attach_imu(self):
        """Spawns an IMU sensor on the hero vehicle."""
        bp = self.world.get_blueprint_library().find('sensor.other.imu')
        self.imu_sensor = self.world.spawn_actor(
            bp, carla.Transform(), attach_to=self.vehicle
        )
        # We need to listen to IMU data to get gyro/accel
        # For simplicity in this snippet, we'll poll vehicle telemetry directly
        # but in a real app, you'd use the sensor callback.

    def collect_step(self):
        """Captures one data frame."""
        if not self.vehicle: return None

        # 1. Physics / Telemetry
        vel = self.vehicle.get_velocity()
        acc = self.vehicle.get_acceleration()
        ang_vel = self.vehicle.get_angular_velocity()
        control = self.vehicle.get_control()
        
        # 2. Calculate Derived Metrics
        speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) # m/s to km/h
        
        curr_loc = self.vehicle.get_location()
        dist_step = curr_loc.distance(self.last_location)
        self.total_distance += dist_step
        self.last_location = curr_loc

        # 3. Format exactly like your CSV
        row = {
            "distance": self.total_distance,
            "speed": speed,
            "gyro_x": ang_vel.x, "gyro_y": ang_vel.y, "gyro_z": ang_vel.z,
            "accel_x": acc.x, "accel_y": acc.y, "accel_z": acc.z,
            "brake_x": 0.0, # CARLA brake is 0-1 scalar, mapping required
            "brake_y": 0.0,
            "brake_z": 0.0, 
            # Note: brake_x/y/z implies a vector brake force, usually specific 
            # to physical sensors. You might map control.brake to one of these.
            "overspeed_count": 0, # Logic needed: e.g. if speed > limit
            "label": 1 # Placeholder: You need logic to define current behavior
        }
        self.data_buffer.append(row)

    def get_batch(self):
        return pd.DataFrame(self.data_buffer)

    def clear_buffer(self):
        self.data_buffer = []