# Simple Flower Example

There are two files `server.py` and `client.py`.

## Configuration

In `server.py` you can play changing the following options.

```
strategy = FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=10,  # Minimum number of clients to be sampled for training
    min_evaluate_clients=5,  # Minimum number of clients to be sampled for evaluation
    min_available_clients=10,  # Minimum number of clients that need to connect to the server before training starts
    evaluate_metrics_aggregation_fn=weighted_average,  # Custom metric aggregation function
)
```

## Usage
Running the Server
Start the Flower server by running:

`python server.py`

This will:

- Start the server on 0.0.0.0:8080.
- Use the FedAvg strategy with specified parameters.
- Wait for clients to connect (minimum number specified in the strategy).

## Running the Clients

Each client represents a participant in the federated learning process.

`python client.py --partition <PARTITION_ID> [--server_address <SERVER_ADDRESS>]`

- `--partition`: (Required) The data partition ID (0-9).
- `--server_address`: The server address (default is 127.0.0.1:8080).


## Example:

`python client.py --partition 0`

To simulate multiple clients, run the above command in separate terminals with different `--partition IDs`.

