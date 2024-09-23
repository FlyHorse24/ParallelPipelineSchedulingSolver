
import gurobipy as gp
from gurobipy import *

import time
"""
simulator package
"""
import itertools
import time
import z3

from .painter import SchedulingPainter
from .painter import SchedulingPainterVshape
from .utils import resort_microbatch_index, resort_microbatch_index_Vshape

class Simulator4DrawVshapeZero:
    def __init__(self, config: dict) -> None:
        self._pp_size = config["pp_size"]
        self._num_microbatches = config["num_microbatches"]
        self._max_activation_counts = config["max_activation_counts"]

        self._forward_length = config["forward_execution_time"]
        self._backward_length = config["backward_execution_time"]
        self._weight_length = config["weight_execution_time"]
        self._sequential_order_constraint_strategy = config["sequential_order_constraint_strategy"]
        self._p2pcomm_length = config["P2Pcommunication_time"]
        assert isinstance(self._forward_length, (list, tuple)), "forward_execution_time must be list or tuple"
        assert isinstance(self._backward_length, (list, tuple)), "backward_execution_time must be list or tuple"

        assert self._sequential_order_constraint_strategy in ("strict", "double_interleaving", "full_interleaving", "zero", "double_interleaving_zero"), "sequential order constraint strategy is not supported"

        self._model = gp.Model('Vshape')
        self._forward_offsets1 = [[self._model.addVar(vtype=GRB.INTEGER, name=f"f1_{mb}_{i}") for mb in range(self._num_microbatches)] for i in range(self._pp_size)]
        self._backward_offsets1 = [[self._model.addVar(vtype=GRB.INTEGER, name=f"b1_{mb}_{i}") for mb in range(self._num_microbatches)] for i in range(self._pp_size)]
        self._weight_offsets1 = [[self._model.addVar(vtype=GRB.INTEGER, name=f"w1_{mb}_{i}") for mb in range(self._num_microbatches)] for i in range(self._pp_size)]

        self._forward_offsets2 = [[self._model.addVar(vtype=GRB.INTEGER, name=f"f2_{mb}_{i}") for mb in range(self._num_microbatches)] for i in range(self._pp_size)]
        self._backward_offsets2 = [[self._model.addVar(vtype=GRBp.INTEGER, name=f"b2_{mb}_{i}") for mb in range(self._num_microbatches)] for i in range(self._pp_size)]
        self._weight_offsets2 = [[self._model.addVar(vtype=GRB.INTEGER, name=f"w2_{mb}_{i}") for mb in range(self._num_microbatches)] for i in range(self._pp_size)]

        self._forward_length1 = [_ / 2 for _ in self._forward_length]
        self._backward_length1 = [_ / 2 for _ in self._backward_length]
        self._weight_length1 = [_ / 2 for _ in self._weight_length]

        self._forward_length2 = [_ / 2 for _ in self._forward_length]
        self._backward_length2 = [_ / 2 for _ in self._backward_length]
        self._weight_length2 = [_ / 2 for _ in self._weight_length]

    def _sequential_order_constraint_Vshape_zero(self):
        for mb in range(self._num_microbatches):
            # forward
            for i in range(1, self._pp_size):
                self._model.addConstr(self._forward_offsets1[i][mb] >= self._forward_offsets1[i - 1][mb] + self._forward_length1[i - 1] + self._p2pcomm_length)

            for i in range(self._pp_size - 1, 0, -1):
                self._model.addConstr(self._forward_offsets2[i - 1][mb] >= self._forward_offsets2[i][mb] + self._forward_length2[i] + self._p2pcomm_length)
            self._model.addConstr(self._forward_offsets2[self._pp_size - 1][mb] >= self._forward_offsets1[self._pp_size - 1][mb] + self._forward_length1[self._pp_size - 1])

            # backward
            for i in range(1, self._pp_size):
                self._model.addConstr(self._backward_offsets2[i][mb] >= self._backward_offsets2[i - 1][mb] + self._backward_length2[i - 1] + self._p2pcomm_length)
            for i in range(self._pp_size - 1, 0, -1):
                self._model.addConstr(self._backward_offsets1[i - 1][mb] >= self._backward_offsets1[i][mb] + self._backward_length1[i])

            self._model.addConstr(self._backward_offsets1[self._pp_size - 1][mb] >= self._backward_offsets2[self._pp_size - 1][mb] + self._backward_length2[self._pp_size - 1])

            # connect for and back
            self._model.addConstr(self._backward_offsets2[0][mb] >= self._forward_offsets2[0][mb] + self._forward_length2[self._pp_size - 1])

            # weight
            for i in range(self._pp_size):
                self._model.addConstr(self._weight_offsets2[i][mb] >= self._weight_offsets1[i][mb] + self._weight_length1[i])

            for i in range(self._pp_size):
                self._model.addConstr(self._weight_offsets1[i][mb] >= self._backward_offsets1[i][mb] + self._backward_length1[i])

            for i in range(self._pp_size):
                self._model.addConstr(self._weight_offsets2[i][mb] >= self._backward_offsets2[i][mb] + self._backward_length2[i])

        # microbatch order
        for mb in range(1, self._num_microbatches):
            for i in range(self._pp_size):
                self._model.addConstr(self._forward_offsets1[i][mb] >= self._forward_offsets1[i][mb - 1] + self._forward_length1[i])
                self._model.addConstr(self._backward_offsets1[i][mb] >= self._backward_offsets1[i][mb - 1] + self._backward_length1[i])
                self._model.addConstr(self._weight_offsets1[i][mb] >= self._weight_offsets1[i][mb - 1] + self._weight_length1[i])

                self._model.addConstr(self._forward_offsets2[i][mb] >= self._forward_offsets2[i][mb - 1] + self._forward_length2[i])
                self._model.addConstr(self._backward_offsets2[i][mb] >= self._backward_offsets2[i][mb - 1] + self._backward_length2[i])
                self._model.addConstr(self._weight_offsets2[i][mb] >= self._weight_offsets2[i][mb - 1] + self._weight_length2[i])

    def _serial_computation_within_pipeline_constraint_Vshape_zero(self):
        for pp in range(self._pp_size):
            _pp_vars = self._forward_offsets1[pp] + self._backward_offsets1[pp] + self._weight_offsets1[pp] + \
                       self._forward_offsets2[pp] + self._backward_offsets2[pp] + self._weight_offsets2[pp]
            for i, _ in enumerate(_pp_vars):
                for j in range(i + 1, len(_pp_vars)):
                    judgeFBWi = i // self._num_microbatches
                    _i_length = 0
                    if judgeFBWi == 0:
                        _i_length = self._forward_length1[pp]
                    elif judgeFBWi == 1:
                        _i_length = self._backward_length1[pp]
                    elif judgeFBWi == 2:
                        _i_length = self._weight_length1[pp]
                    elif judgeFBWi == 3:
                        _i_length = self._forward_length2[pp]
                    elif judgeFBWi == 4:
                        _i_length = self._backward_length2[pp]
                    elif judgeFBWi == 5:
                        _i_length = self._weight_length2[pp]

                    judgeFBWj = j // self._num_microbatches
                    _j_length = 0
                    if judgeFBWj == 0:
                        _j_length = self._forward_length1[pp]
                    elif judgeFBWj == 1:
                        _j_length = self._backward_length1[pp]
                    elif judgeFBWj == 2:
                        _j_length = self._weight_length1[pp]
                    elif judgeFBWj == 3:
                        _j_length = self._forward_length2[pp]
                    elif judgeFBWj == 4:
                        _j_length = self._backward_length2[pp]
                    elif judgeFBWj == 5:
                        _j_length = self._weight_length2[pp]
                    self._model.addConstr(gp.or_(self._pp_vars[j] >= self._pp_vars[i] + _i_length,
                                                 self._pp_vars[j] + _j_length <= self._pp_vars[i]))

    def _build_constraints(self) -> None:
        for i in range(self._pp_size):
            for mb in range(self._num_microbatches):
                self._forward_offsets1[i][mb].start = 0
                self._backward_offsets1[i][mb].start = 0
                self._weight_offsets1[i][mb].start = 0

                self._forward_offsets2[i][mb].start = 0
                self._backward_offsets2[i][mb].start = 0
                self._weight_offsets2[i][mb].start = 0

        # constraint 1-0: forward and backward of each microbatch
        self._sequential_order_constraint_Vshape_zero()

        # constraint 2: no overlapping of forward and backward within each pipeline
        self._serial_computation_within_pipeline_constraint_Vshape_zero()

    def _build_optimize_objectives(self) -> None:
        max_stage = self._model.addVar(vtype=gp.INTEGER, name="max_start_offset")
        for pp in range(self._pp_size):
            max_var = self._weight_offsets2[pp][-1]
            min_var = self._forward_offsets1[pp][0]
            stage = max_var - min_var
            self._model.addConstr(max_stage >= stage)
        self._model.setObjective(max_stage, gp.MINIMIZE)

    def _draw(self, results: dict) -> None:
        painter_conf = {
            "pp_size": self._pp_size,
            "pp_height": 50,
            "pp_align": 10,
            "pixel_base": 15,
            "forward_length1": self._forward_length1,
            "backward_length1": self._backward_length1,
            "weight_length1": self._weight_length1,
            "forward_length2": self._forward_length2,
            "backward_length2": self._backward_length2,
            "weight_length2": self._weight_length2,
        }

        SchedulingPainterVshape(painter_conf).draw(results)

    def run(self) -> None:
        """run simulation"""
        # 1. builds the solver constraints.
        self._build_constraints()

        # 2. builds the solver optimize objectives.
        self._build_optimize_objectives()

        # 3. runs the solver.
        start_time = time.time()
        print("Gurobi Solver Solving...")
        self._model.optimize()
        end_time = time.time()
        if self._model.status == gp.OPTIMAL:
            print(f"Result: OPTIMAL, Cost: {end_time - start_time:.2f}")
            # transforms the result to a dictionary.
            results = {var.varName: var.x for var in self._model.getVars()}
            results.pop("max_start_offset")
            # 4. draws the result.
            self._draw(resort_microbatch_index_Vshape(self._num_microbatches, results))
        else:
            print(f"Result: INFEASIBLE, Cost: {end_time - start_time:.2f}")
