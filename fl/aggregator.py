import flwr as fl
from flwr.server.strategy import FedXgbBagging
from flwr.common import Metrics
from typing import List, Tuple

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregation function for (num_examples, metrics) tuples.
    Calculates weighted average of 'accuracy' AND 'loss'.
    """
    aggregated_metrics = {}

    # 1. Aggregate Accuracy
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    aggregated_metrics["accuracy"] = sum(accuracies) / sum(examples)

    # 2. Aggregate Loss (if present in metrics)
    # This solves the "Loss: 0" issue by making it an explicit metric
    if "loss" in metrics[0][1]:
        losses = [num_examples * m["loss"] for num_examples, m in metrics]
        aggregated_metrics["loss"] = sum(losses) / sum(examples)

    return aggregated_metrics

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