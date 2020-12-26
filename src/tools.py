import time
from typing import Dict, Callable, Tuple
import numpy as np
from qiskit.aqua.components.optimizers import Optimizer


def get_expected_values(counts: Dict[str, int]):
	qubit_cnt = len(next(iter(counts.keys())))
	dist = np.zeros((qubit_cnt, ))
	sum = 0

	for k, v in counts.items():
		sum += v

		for i, c in enumerate(k):
			if c == "1":
				dist[qubit_cnt - 1 - i] += v

	dist /= sum

	return dist


def train_circuit(num_vars: int, obj_func: Callable[[np.ndarray], float], optim: Optimizer) -> Tuple[float, float, np.ndarray]:
	params = np.random.rand(num_vars)
	time1 = time.time()
	ret = optim.optimize(num_vars=num_vars, objective_function=obj_func, initial_point=params)
	time2 = time.time()
	# print(ret[0])
	# print(ret[1])
	# print(ret[2])
	# print()

	return ret[1], time2 - time1, ret[0]
