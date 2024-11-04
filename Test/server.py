# server.py

import flwr as fl
from typing import List, Tuple
from flwr.common import Metrics
from flwr.server.strategy import FedAvg

# Define a metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# Create a FedAvg strategy
strategy = FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=4,  # Minimum number of clients to be sampled for training
    min_evaluate_clients=2,  # Minimum number of clients to be sampled for evaluation
    min_available_clients=4,  # Minimum number of clients that need to connect to the server before training starts
    evaluate_metrics_aggregation_fn=weighted_average,  # Custom metric aggregation function
)

#strategy = FedAvg(
    #fraction_fit=1.0,  # Sample 100% of available clients for training
    #fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    #min_fit_clients=10,  # Minimum number of clients to be sampled for training
    #min_evaluate_clients=5,  # Minimum number of clients to be sampled for evaluation
    #min_available_clients=10,  # Minimum number of clients that need to connect to the server before training starts
    #evaluate_metrics_aggregation_fn=weighted_average,  # Custom metric aggregation function
#)

if __name__ == "__main__":
    # Start the Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",  # Listen on all interfaces (adjust as needed)
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )
