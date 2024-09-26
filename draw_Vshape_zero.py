"""
main package
"""
from simulator.simulator import Simulator4DrawVshapeZero


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
        "max_activation_counts": [4 for _ in range(4)],
        "P2Pcommunication_time": 0,
        #上述为模型均分的情况
        "layersPerStage":[],#这个时间与内存占比按照层数比即可，用于更新上面的time,memory
        #这些参数是计算F、B、W之间的比值
        "recomputation":False,
        "attentionheads":32,
        "hidden":128,
        "sequencelen":1024,
        #目前activationMem不开重计算：B、W模拟了是2：1 ， 开重计算为1：1
        "back_activationMem":2,
        "weight_activationMem":1
        
        
    }

    simulator = Simulator4DrawVshapeZero(config)
    simulator.run()


if __name__ == "__main__":
    main()
