# server.py
import flwr as fl
from fl.aggregator import get_strategy

if __name__ == "__main__":
    print("Starting FL Server...")
    
    # We use the same strategy as before
    strategy = get_strategy()

    # Start the server and wait for clients
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10), # Run for more rounds
        strategy=strategy,
    )