import flwr as fl
from flwr.server.strategy import FedXgbBagging
from flwr.common import Metrics
from typing import List, Tuple

# Metric Aggregation Function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregation function for (num_examples, metrics) tuples.
    Calculates weighted average of 'accuracy'.
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and divide by total examples
    return {"accuracy": sum(accuracies) / sum(examples)}

def get_strategy():
    """
    Returns the XGBoost Bagging strategy with metrics aggregation.
    """
    strategy = FedXgbBagging(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        # Pass the aggregation function here:
        evaluate_metrics_aggregation_fn=weighted_average, 
    )
    return strategy