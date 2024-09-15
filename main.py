"""
main package
"""
from simulator.simulator import Simulator


def main():
    """main function"""
    config = {
        "pp_size": 4,
        "num_microbatches": 4,
        "forward_execution_time": [10 for _ in range(4)],
        "backward_execution_time": [24 for _ in range(4)],
        "weight_execution_time":[9 for _ in range(4)],
        # stratiges: "double_interleaving_zero","zero"
        "sequential_order_constraint_strategy": "zero",
        "max_activation_counts": [4 for _ in range(4)],
        "P2Pcommunication_time": 1,
    }

    simulator = Simulator(config)
    simulator.run()

if __name__ == "__main__":
    main()
