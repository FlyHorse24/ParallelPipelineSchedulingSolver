"""
main package
"""
from simulator.simulator import Simulator4Draw1F1B


def main():
    """main function"""
    config = {
        "pp_size": 6,
        "num_microbatches": 12,
        "forward_execution_time": [5, 5, 3, 6, 5, 3],  # forward time for each pp stage
        "backward_execution_time": [9, 7, 9, 11, 9, 12],  # backward time for each pp stage
        "sequential_order_constraint_strategy": "strict",
        "max_activation_counts": None,  # not used
    }

    simulator = Simulator4Draw1F1B(config)
    simulator.run()


if __name__ == "__main__":
    main()
