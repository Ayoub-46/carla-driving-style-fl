import flwr as fl
from sklearn.metrics import accuracy_score
import xgboost as xgb
import numpy as np
from fl.model import DrivingStyleModel

from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    Parameters,
)

class DrivingXgbClient(fl.client.Client):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = DrivingStyleModel()

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        # The XGBoost strategy (FedXgbBagging) does not typically pull parameters 
        # via this method to initialize, but the signature is mandatory.
        return GetParametersRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensors=[], tensor_type="bytes"),
        )

    def fit(self, ins: FitIns) -> FitRes:
        # 1. Recover global model (if exists)
        global_model = None
        if ins.parameters.tensors:
            # We receive 'bytes', convert to 'bytearray' for XGBoost if needed, 
            # though load_model usually accepts bytes too.
            global_model_bytes = bytearray(ins.parameters.tensors[0])
            global_model = xgb.Booster()
            try:
                global_model.load_model(global_model_bytes)
            except Exception:
                global_model = None

        # 2. Local Training
        self.model.train(
            self.X_train, 
            self.y_train, 
            num_boost_round=1, 
            xgb_model=global_model
        )

        # 3. Serialize result
        local_model_bytearray = self.model.bst.save_raw("json")
        
        local_model_bytes = bytes(local_model_bytearray)

        # 4. Return FitRes
        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensors=[local_model_bytes], tensor_type="bytes"),
            num_examples=len(self.X_train),
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        if not ins.parameters.tensors:
            return EvaluateRes(status=Status(code=Code.OK, message="No model"), loss=0.0, num_examples=0, metrics={})

        # Load Global Model
        model_bytes = bytearray(ins.parameters.tensors[0])
        bst = xgb.Booster()
        bst.load_model(model_bytes)
        self.model.bst = bst

        # Predict
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        preds = self.model.bst.predict(dtest)
        
        # 1. Calculate Accuracy
        # If your model outputs probabilities (multi:softprob), use np.argmax(preds, axis=1)
        # If your model outputs labels (multi:softmax), use preds directly
        acc = accuracy_score(self.y_test, preds)

        # 2. Calculate Loss (Error Rate)
        # Flower server aggregates this value automatically!
        loss_value = 1.0 - acc

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=float(loss_value),  
            num_examples=len(self.X_test),
            metrics={"accuracy": float(acc)},
        )

import argparse
from utils.data_loader import load_data, get_client_partitions

if __name__ == "__main__":
    # Parse command line arguments to identify which "static" client this is
    parser = argparse.ArgumentParser(description='Flower Client')
    parser.add_argument('--partition-id', type=int, default=0, help='Partition ID for this client')
    parser.add_argument('--server-address', type=str, default="127.0.0.1:8080", help='Server address')
    args = parser.parse_args()

    # 1. Load Data (Same as before)
    DATA_PATH = "data/dataset.csv"
    X, y, _ = load_data(DATA_PATH)
    
    # 2. Get the specific partition for this client ID
    # We assume 2 static clients for this example
    partitions = get_client_partitions(X, y, num_clients=2)
    
    if args.partition_id >= len(partitions):
        print(f"Error: Partition ID {args.partition_id} out of range.")
        exit(1)

    X_train, y_train, X_test, y_test = partitions[args.partition_id]

    # 3. Initialize Client
    client = DrivingXgbClient(X_train, y_train, X_test, y_test)

    # 4. Start Client (Connects to the server)
    print(f"Starting Static Client {args.partition_id}...")
    fl.client.start_client(
        server_address=args.server_address,
        client=client
    )