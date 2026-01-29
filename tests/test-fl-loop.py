import flwr as fl
from flwr.common import Context  # Import Context
from fl.client import DrivingXgbClient
from fl.aggregator import get_strategy
from utils.data_loader import load_data, get_client_partitions
import os

DATA_PATH = "data/dataset.csv"
NUM_CLIENTS = 2

# Global variable to hold partitions (loaded once)
partitions = []

def main():
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    print("Loading and partitioning data...")
    X, y, encoder = load_data(DATA_PATH)
    
    # Populate the global partitions list
    global partitions
    partitions = get_client_partitions(X, y, num_clients=NUM_CLIENTS)

    # 2. Define Client Factory (Updated Signature)
    def client_fn(context: Context) -> fl.client.Client:
        """
        Construct a Client instance.
        
        The 'context' argument contains metadata about the node.
        In simulation, 'partition-id' is usually passed in node_config.
        """
        # Extract partition ID. 
        # In standard simulation, node_id is often the partition ID, 
        # but relying on node_config['partition-id'] is safer if configured.
        
        # Fallback logic to get ID:
        try:
            # Try getting explicit partition-id from config
            partition_id = int(context.node_config["partition-id"])
        except (KeyError, ValueError):
            # Fallback: parse node_id (often a string "0", "1")
            partition_id = int(context.node_id)

        # Retrieve the data for this specific client ID
        X_train, y_train, X_test, y_test = partitions[partition_id]
        
        return DrivingXgbClient(X_train, y_train, X_test, y_test)

    # 3. Start Simulation
    # We must provide a backend config to map partition IDs to clients
    # This tells the simulation "Assign partition-id 0 to client 0", etc.
    def client_config_fn(partition_id: str):
         return {"partition-id": partition_id}

    print("Starting FL Simulation...")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=get_strategy(),
        # We generally don't need detailed client_resources for simple XGBoost 
        # unless you are using GPU. 
    )

if __name__ == "__main__":
    main()