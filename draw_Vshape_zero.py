"""
main package
"""
from simulator.simulator import Simulator4DrawVshapeZero


def main():
    """main function"""
    config = {
        "pp_size": 4,
        "num_microbatches": 5,
        "forward_execution_time": [4 for _ in range(4)],
        "backward_execution_time": [6 for _ in range(4)],
        "weight_execution_time":[4 for _ in range(4)],
        # stratiges: "strict", "double_interleaving", "full_interleaving","zero"
        "sequential_order_constraint_strategy": "zero",
        "max_activation_counts": [4 for _ in range(4)],#每个device的内存峰值，目前可以理解为最开始同时压入的microbatch的数量上限
        "P2Pcommunication_time": 0,
        #上述为模型均分的情况
        "layersPerStage":[1 for _ in range(4)],#每个Stage层数乘以F、B、W，以获取不同Stage的F、B、W的时间以及内存之比
        #这些参数是计算F、B、W之间的比值
        "recomputation":False,
        "attentionheads":32,
        "hidden":4096,
        "sequencelen":2048,
        
        #目前activationMem不开重计算：B、W模拟了是2：1 ， 开重计算为1：1
        "back_activationMem":[2 for _ in range(4)],
        "weight_activationMem":[1 for _ in range(4)]
        
        
    }

    simulator = Simulator4DrawVshapeZero(config)
    simulator.run()


if __name__ == "__main__":
    main()
