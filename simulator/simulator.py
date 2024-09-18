"""
simulator package
"""
import itertools
import time
import z3

from .painter import SchedulingPainter
from .utils import resort_microbatch_index


class Simulator:
    """Simulator"""

    def __init__(self, config: dict) -> None:
        self._pp_size = config["pp_size"]
        self._num_microbatches = config["num_microbatches"]
        self._max_activation_counts = config["max_activation_counts"]

        self._forward_length = config["forward_execution_time"]
        self._backward_length = config["backward_execution_time"]
        self._weight_length = config["weight_execution_time"]
        self._sequential_order_constraint_strategy = config[
            "sequential_order_constraint_strategy"
        ]
        self._p2pcomm_length = config["P2Pcommunication_time"]
        assert isinstance(
            self._forward_length, (list, tuple)
        ), "forward_execution_time must be list or tuple"
        assert isinstance(
            self._backward_length, (list, tuple)
        ), "backward_execution_time must be list or tuple"

        assert self._sequential_order_constraint_strategy in (
            "strict",
            "double_interleaving",
            "full_interleaving",
            "zero",
            "double_interleaving_zero"
        ), "sequential order constraint strategy is not supported"

        self._solver = z3.Optimize()
        self._forward_offsets = [[] for i in range(self._pp_size)]
        self._backward_offsets = [[] for i in range(self._pp_size)]
        self._weight_offsets = [[] for i in range(self._pp_size)]

    def _sequential_order_constraint_strict(self):
        for mb in range(self._num_microbatches):
            # forward stages sequential constraint
            for i in range(1, self._pp_size):
                self._solver.add(
                    self._forward_offsets[i][mb]
                    >= self._forward_offsets[i - 1][mb] + self._forward_length[i-1]
                )
            # backward stages sequential constraint
            for i in range(self._pp_size - 1, 0, -1):
                self._solver.add(
                    self._backward_offsets[i - 1][mb]
                    >= self._backward_offsets[i][mb] + self._backward_length[i]
                )
            # forward-backward connection sequential constraint
            self._solver.add(
                self._backward_offsets[self._pp_size - 1][mb]
                >= self._forward_offsets[self._pp_size - 1][mb] + self._forward_length[self._pp_size - 1]
            )
    
    def _sequential_order_constraint_zero(self):
        self._backward_length = [self._backward_length[i]-self._weight_length[i] for i in range(len(self._backward_length))]
        for mb in range(self._num_microbatches):
            # forward stages sequential constraint
            for i in range(1, self._pp_size):
                self._solver.add(
                    self._forward_offsets[i][mb]
                    >= self._forward_offsets[i - 1][mb] + self._forward_length[i-1] + self._p2pcomm_length
                )
            # backward stages sequential constraint
            for i in range(self._pp_size - 1, 0, -1):
                self._solver.add(
                    self._backward_offsets[i - 1][mb]
                    >= self._backward_offsets[i][mb] + self._backward_length[i] + self._p2pcomm_length
                )
            #weight stages sequential constraint
            for i in range(self._pp_size):
                self._solver.add(
                    self._weight_offsets[i][mb]
                    >= self._backward_offsets[i][mb] + self._backward_length[i]
                )
            
            # forward-backward connection sequential constraint
            self._solver.add(
                self._backward_offsets[self._pp_size - 1][mb]
                >= self._forward_offsets[self._pp_size - 1][mb] + self._forward_length[self._pp_size - 1]
            )

    def _sequential_order_constraint_double_interleaving(self):
        for mb in range(self._num_microbatches):
            # down pipe
            down_case = z3.And(
                *[
                    self._forward_offsets[i][mb]
                    >= self._forward_offsets[i - 1][mb] + self._forward_length[i-1]
                    for i in range(1, self._pp_size)
                ],
                *[
                    self._backward_offsets[i - 1][mb]
                    >= self._backward_offsets[i][mb] + self._backward_length[i]
                    for i in range(self._pp_size - 1, 0, -1)
                ],
                self._backward_offsets[self._pp_size - 1][mb]
                >= self._forward_offsets[self._pp_size - 1][mb] + self._forward_length[self._pp_size - 1],
            )
            # up pipe
            up_case = z3.And(
                *[
                    self._forward_offsets[i - 1][mb]
                    >= self._forward_offsets[i][mb] + self._forward_length[i]
                    for i in range(self._pp_size - 1, 0, -1)
                ],
                *[
                    self._backward_offsets[i][mb]
                    >= self._backward_offsets[i - 1][mb] + self._backward_length[i-1]
                    for i in range(1, self._pp_size)
                ],
                self._backward_offsets[0][mb]
                >= self._forward_offsets[0][mb] + self._forward_length[0],
            )

            self._solver.add(z3.Or(down_case, up_case))

    def _sequential_order_constraint_double_interleaving_zero(self):
        self._backward_length = [self._backward_length[i]-self._weight_length[i] for i in range(len(self._backward_length))]
        
        for mb in range(self._num_microbatches):
            # down pipe
            down_case = z3.And(
                *[
                    self._forward_offsets[i][mb]
                    >= self._forward_offsets[i - 1][mb] + self._forward_length[i-1] + self._p2pcomm_length
                    for i in range(1, self._pp_size)
                ],
                *[
                    self._backward_offsets[i - 1][mb]
                    >= self._backward_offsets[i][mb] + self._backward_length[i] + self._p2pcomm_length
                    for i in range(self._pp_size - 1, 0, -1)
                ],
                *[
                    self._weight_offsets[i][mb]
                    >= self._backward_offsets[i][mb] + self._backward_length[i]
                    for i in range(self._pp_size)
                ],
                self._backward_offsets[self._pp_size - 1][mb]
                >= self._forward_offsets[self._pp_size - 1][mb] + self._forward_length[self._pp_size - 1],
            )
            # up pipe
            up_case = z3.And(
                *[
                    self._forward_offsets[i - 1][mb]
                    >= self._forward_offsets[i][mb] + self._forward_length[i] + self._p2pcomm_length
                    for i in range(self._pp_size - 1, 0, -1)
                ],
                *[
                    self._backward_offsets[i][mb]
                    >= self._backward_offsets[i - 1][mb] + self._backward_length[i-1] + self._p2pcomm_length
                    for i in range(1, self._pp_size)
                ],
                *[
                    self._weight_offsets[i][mb]
                    >= self._backward_offsets[i][mb] + self._backward_length[i]
                    for i in range(self._pp_size)
                ],
                self._backward_offsets[0][mb]
                >= self._forward_offsets[0][mb] + self._forward_length[0],
            )

            self._solver.add(z3.Or(down_case, up_case))

    def _sequential_order_constraint_full_interleaving(self):
        for mb in range(self._num_microbatches):
            cases = []

            for perm in itertools.permutations(range(self._pp_size)):
                cases.append(
                    z3.And(
                        # forward sequential order
                        *[
                            self._forward_offsets[perm[i + 1]][mb]
                            >= self._forward_offsets[perm[i]][mb] + self._forward_length[perm[i]]
                            for i in range(len(perm) - 1)
                        ],
                        # corresponding backward order
                        *[
                            self._backward_offsets[perm[i - 1]][mb]
                            >= self._backward_offsets[perm[i]][mb]
                            + self._backward_length[perm[i]]
                            for i in range(len(perm) - 1, 0, -1)
                        ],
                        # forward-backward connection order
                        self._backward_offsets[perm[-1]][mb]
                        >= self._forward_offsets[perm[-1]][mb] + self._forward_length[perm[-1]],
                    )
                )

            # add all possibilities to z3 constraints
            self._solver.add(z3.Or(*cases))

    def _serial_computation_within_pipeline_constraint(self):
        for pp in range(self._pp_size):
            _pp_vars = self._forward_offsets[pp] + self._backward_offsets[pp]
            for i, _ in enumerate(_pp_vars):
                for j in range(i + 1, len(_pp_vars)):
                    _i_length = (
                        self._forward_length[pp]
                        if i // self._num_microbatches == 0
                        else self._backward_length[pp]
                    )
                    _j_length = (
                        self._forward_length[pp]
                        if j // self._num_microbatches == 0
                        else self._backward_length[pp]
                    )
                    self._solver.add(
                        z3.Or(
                            _pp_vars[j] >= _pp_vars[i] + _i_length,
                            _pp_vars[j] + _j_length <= _pp_vars[i],
                        )
                    )

    def _serial_computation_within_pipeline_constraint_zero(self):
        for pp in range(self._pp_size):
            _pp_vars = self._forward_offsets[pp] + self._backward_offsets[pp] + self._weight_offsets[pp]
            for i, _ in enumerate(_pp_vars):
                for j in range(i + 1, len(_pp_vars)):
                    judgeFBWi = i // self._num_microbatches
                    _i_length = 0
                    if judgeFBWi == 0:
                        _i_length = self._forward_length[pp]
                    elif judgeFBWi == 1:
                        _i_length = self._backward_length[pp]
                    elif judgeFBWi == 2:
                        _i_length = self._weight_length[pp]                
                    
                    judgeFBWj = j // self._num_microbatches
                    _j_length = 0
                    if judgeFBWj == 0:
                        _j_length = self._forward_length[pp]
                    elif judgeFBWj == 1:
                        _j_length = self._backward_length[pp]
                    elif judgeFBWj == 2:
                        _j_length = self._weight_length[pp]
                    self._solver.add(
                        z3.Or(
                            _pp_vars[j] >= _pp_vars[i] + _i_length,
                            _pp_vars[j] + _j_length <= _pp_vars[i],
                        )
                    )

    def _pipeline_activation_accumulation_constraint(self):
        for pp in range(self._pp_size):
            # calculate the maximum activation value for this pp
            for mb in range(self._num_microbatches):
                _backward_var = self._backward_offsets[pp][mb]
                _actvaition_count = 1

                for other_mb in range(self._num_microbatches):
                    if other_mb == mb:
                        continue
                    _actvaition_count += z3.If(
                        z3.And(
                            self._backward_offsets[pp][other_mb] > _backward_var,
                            self._forward_offsets[pp][other_mb] < _backward_var,
                        ),
                        1,
                        0,
                    )

                self._solver.add(_actvaition_count <= self._max_activation_counts[pp])

    def _build_constraints(self) -> None:
        for i in range(self._pp_size):
            for mb in range(self._num_microbatches):
                self._forward_offsets[i].append(z3.Int(f"f_{mb}_{i}"))
                self._solver.add(self._forward_offsets[i][-1] >= 0)
                self._backward_offsets[i].append(z3.Int(f"b_{mb}_{i}"))
                self._solver.add(self._backward_offsets[i][-1] >= 0)
                self._weight_offsets[i].append(z3.Int(f"w_{mb}_{i}"))
                self._solver.add(self._weight_offsets[i][-1] >= 0)

        if self._sequential_order_constraint_strategy == "zero":
            #
            #
            self._sequential_order_constraint_zero()
        elif self._sequential_order_constraint_strategy == "double_interleaving_zero":
            #
            #
            self._sequential_order_constraint_double_interleaving_zero()

        # constraint 2: no overlapping of forward and backward within each pipeline
        self._serial_computation_within_pipeline_constraint_zero()

        # constraint 3: the accumulation count of activations does not exceed max_activation_counts
        self._pipeline_activation_accumulation_constraint()

    def _build_optimize_objectives(self) -> None:
        # 1. minimize the execution time of each microbatch
        max_var = z3.Int("max_start_offset")

        for pp in range(self._pp_size):
            for var in self._weight_offsets[pp]:
                self._solver.add(max_var >= var)
        self._solver.minimize(max_var)

    def _draw(self, results: dict) -> None:
        painter_conf = {
            "pp_size": self._pp_size,
            "pp_height": 50,
            "pp_align": 10,
            "pixel_base": 5,
            "forward_length": self._forward_length,
            "backward_length": self._backward_length,
            "weight_length": self._weight_length
        }

        SchedulingPainter(painter_conf).draw(results)
    
    def run(self) -> None:
        """run simulation"""
        # 1. builds the solver constraints.
        self._build_constraints()

        # 2. builds the solver optimize objectives.
        self._build_optimize_objectives()

        # 3. runs the solver.
        start_time = time.time()
        print("Z3 Solver Solving...")
        check = self._solver.check()
        end_time = time.time()
        if  check == z3.sat:
            print(f"Result: SAT, Cost: {end_time - start_time:.2f}")
            # tranforms the result to a dictionary.
            model = self._solver.model()
            results = {str(key): model[key].as_long() for key in model}
            results.pop("max_start_offset")
            # 4. draws the result.
            self._draw(resort_microbatch_index(self._num_microbatches ,results))
        else:
            print(f"Result: UNSAT, Cost: {end_time - start_time:.2f}")

class Simulator4DrawVshapeZero(Simulator):
    def _sequential_order_constraint_Vshape_zero(self):
        self._backward_length = [self._backward_length[i]-self._weight_length[i] for i in range(len(self._backward_length))]
        #切割
        self._forward_length1, self._forward_length2 = [_/2 for _ in self._forward_length]
        self._backward_length1, self._backward_length2 = [_/2 for _ in self._backward_length]
        self._weight_length1, self._weight_length2 = [_/2 for _ in self._weight_length] 

        self._forward_offsets1, self._forward_offsets2 = self._forward_offsets
        self._backward_offsets1, self._backward_offsets2 = self._backward_offsets
        self._weight_offsets1, self._weight_offsets2 = self._weight_offsets
        
        for mb in range(self._num_microbatches):
            # down pipe
            forward_case = z3.And(
                *[
                    self._forward_offsets1[i][mb]
                    >= self._forward_offsets1[i - 1][mb] + self._forward_length1[i-1] + self._p2pcomm_length
                    for i in range(1, self._pp_size)
                ],
                *[
                    self._forward_offsets2[i - 1][mb]
                    >= self._forward_offsets2[i][mb] + self._forward_length2[i] + self._p2pcomm_length
                    for i in range(self._pp_size - 1, 0, -1)
                ],
                self._forward_offsets2[self._pp_size - 1][mb]
                >= self._forward_offsets1[self._pp_size - 1][mb] + self._forward_length1[self._pp_size - 1],
            )
            # up pipe
            backward_case = z3.And(
                *[
                    self._backward_offsets2[i][mb]
                    >= self._backward_offsets2[i - 1][mb] + self._backward_length2[i-1] + self._p2pcomm_length
                    for i in range(1, self._pp_size)
                ],
                *[
                    self._backward_offsets1[i - 1][mb]
                    >= self._backward_offsets1[i][mb] + self._backward_length1[i] + self._p2pcomm_length
                    for i in range(self._pp_size - 1, 0, -1)
                ], 
                *[
                    self._weight_offsets2[i][mb]
                    >= self._backward_offsets2[i][mb] + self._backward_length2[i]
                    for i in range(self._pp_size)
                ],
                *[
                    self._weight_offsets1[i][mb]
                    >= self._backward_offsets1[i][mb] + self._backward_length1[i]
                    for i in range(self._pp_size)
                ],
                self._backward_offsets1[self._pp_size - 1][mb]
                >= self._backward_offsets2[self._pp_size - 1][mb] + self._backward_length2[self._pp_size - 1],
            )

            #
            self._solver.add(
                self._backward_offsets2[0][mb]
                >= self._forward_offsets2[0][mb] + self._forward_length2[self._pp_size - 1]
            )  
            self._solver.add(z3.And(forward_case, backward_case))

    def _build_constraints(self) -> None:
        for i in range(self._pp_size):
            for mb in range(self._num_microbatches):
                self._forward_offsets[i].append(z3.Int(f"f_{mb}_{i}"))
                self._solver.add(self._forward_offsets[i][-1] >= 0)
                self._backward_offsets[i].append(z3.Int(f"b_{mb}_{i}"))
                self._solver.add(self._backward_offsets[i][-1] >= 0)
                self._weight_offsets[i].append(z3.Int(f"w_{mb}_{i}"))
                self._solver.add(self._weight_offsets[i][-1] >= 0)

        # constraint 1-0: forward and backward of each microbatch
        # are executed in sequential order
        
        self._sequential_order_constraint_zero()

        # constraint 2: no overlapping of forward and backward within each pipeline
        self._serial_computation_within_pipeline_constraint_zero()

    def _draw(self, results: dict) -> None:
        painter_conf = {
            "pp_size": self._pp_size,
            "pp_height": 50,
            "pp_align": 10,
            "pixel_base": 5,
            "forward_length": self._forward_length,
            "backward_length": self._backward_length,
            "weight_length": self._weight_length
        }

        SchedulingPainter(painter_conf).draw(results)
        
    

class Simulator4Draw1F1Bzero(Simulator):
    """Simulator for 1f1b drawing"""

    def _1f1b_scheduling_constraint(self) -> None:
        for i in range(0, self._pp_size):
            num_warmup_microsteps = self._pp_size - i - 1
            num_warmup_microsteps = min(num_warmup_microsteps, self._num_microbatches)
            num_1f1b_micropairs = self._num_microbatches - num_warmup_microsteps

            # warmup
            for j in range(1, num_warmup_microsteps):
                self._solver.add(self._forward_offsets[i][j] == self._forward_offsets[i][j-1] + self._forward_length[i])

            # 1f1b
            for j in range(1, num_1f1b_micropairs):
                _forward_mb, _backward_mb = j+num_warmup_microsteps, j
                self._solver.add(self._forward_offsets[i][_forward_mb] == self._backward_offsets[i][_backward_mb-1] + self._backward_length[i])


    def _build_constraints(self) -> None:
        for i in range(self._pp_size):
            for mb in range(self._num_microbatches):
                self._forward_offsets[i].append(z3.Int(f"f_{mb}_{i}"))
                self._solver.add(self._forward_offsets[i][-1] >= 0)
                self._backward_offsets[i].append(z3.Int(f"b_{mb}_{i}"))
                self._solver.add(self._backward_offsets[i][-1] >= 0)
                self._weight_offsets[i].append(z3.Int(f"w_{mb}_{i}"))
                self._solver.add(self._weight_offsets[i][-1] >= 0)

        # constraint 1-0: forward and backward of each microbatch
        # are executed in sequential order
        
        self._sequential_order_constraint_zero()

        self._1f1b_scheduling_constraint()

        # constraint 2: no overlapping of forward and backward within each pipeline
        self._serial_computation_within_pipeline_constraint_zero()

    def _draw(self, results: dict) -> None:
        painter_conf = {
            "pp_size": self._pp_size,
            "pp_height": 50,
            "pp_align": 10,
            "pixel_base": 5,
            "forward_length": self._forward_length,
            "backward_length": self._backward_length,
            "weight_length": self._weight_length
        }

        SchedulingPainter(painter_conf).draw(results)
