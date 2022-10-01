from typing import List, Tuple

import flwr.server.strategy
import flwr as fl
from flwr.common import Metrics



if __name__ == "__main__":

    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        print("accuracy : {}".format(sum(accuracies) / sum(examples)))
        return {"accuracy": sum(accuracies) / sum(examples)}

    # Define strategy
    strategy = flwr.server.strategy.FedAvg(min_fit_clients=3, min_available_clients=3, evaluate_metrics_aggregation_fn=weighted_average)
    # server = Server(client_manager=client_manager, strategy=strategy)

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=100),
        strategy=strategy,
    )