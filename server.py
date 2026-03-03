# server.py
import flwr as fl
from fl.aggregator import get_strategy

if __name__ == "__main__":
    print("Starting FL Server...")
    
    strategy = get_strategy()

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=8),
        strategy=strategy,
    )