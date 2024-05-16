"""
main package
"""
from simulator.simulator import Simulator


def main():
    """main function"""
    config = {
        "pp_size": 4,
        "num_microbatches": 8,
        "forward_execution_time": [2 for _ in range(4)],
        "backward_execution_time": [3 for _ in range(4)],
        # stratiges: "strict", "double_interleaving", "full_interleaving",
        "sequential_order_constraint_strategy": "strict",
        "max_activation_counts": [4 for _ in range(4)],
    }

    simulator = Simulator(config)
    simulator.run()


if __name__ == "__main__":
    main()
