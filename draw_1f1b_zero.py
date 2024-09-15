"""
main package
"""
from simulator.simulator import Simulator4Draw1F1Bzero


def main():
    """main function"""
    # config = {
    #     "pp_size": 6,
    #     "num_microbatches": 12,
    #     "forward_execution_time": [5, 5, 3, 6, 5, 3],  # forward time for each pp stage
    #     "backward_execution_time": [9, 7, 9, 11, 9, 12],  # backward time for each pp stage
    #     "sequential_order_constraint_strategy": "strict",
    #     "max_activation_counts": None,  # not used
    # }
    config = {
        "pp_size": 4,
        "num_microbatches": 4,
        "forward_execution_time": [10 for _ in range(4)],
        "backward_execution_time": [24 for _ in range(4)],
        "weight_execution_time":[9 for _ in range(4)],
        # stratiges: "strict", "double_interleaving", "full_interleaving","zero"
        "sequential_order_constraint_strategy": "zero",
        "max_activation_counts": None,
        "P2Pcommunication_time": 1,
    }

    simulator = Simulator4Draw1F1Bzero(config)
    simulator.run()


if __name__ == "__main__":
    main()
