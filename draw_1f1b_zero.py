"""
main package
"""
from simulator.simulator import Simulator4Draw1F1Bzero


def main():
    """main function"""
    config = {
        "pp_size": 4,
        "num_microbatches": 4,
        "forward_execution_time": [4 for _ in range(4)],
        "backward_execution_time": [6 for _ in range(4)],
        "weight_execution_time":[4 for _ in range(4)],
        # stratiges: "strict", "double_interleaving", "full_interleaving","zero"
        "sequential_order_constraint_strategy": "zero",
        "max_activation_counts": None,
        "P2Pcommunication_time": 0,
    }

    simulator = Simulator4Draw1F1Bzero(config)
    simulator.run()


if __name__ == "__main__":
    main()
