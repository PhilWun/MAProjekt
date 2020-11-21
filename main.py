import time
from typing import Dict, Callable, Tuple, List

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer, execute
from qiskit.aqua.components.optimizers import Optimizer, ADAM, CG, COBYLA, L_BFGS_B, GSLS, NELDER_MEAD, NFT, P_BFGS, POWELL, SLSQP, SPSA, TNC
from qiskit.providers.aer import QasmSimulator


def create_simple_circuit(params):
	qr = QuantumRegister(1, name="q")
	cr = ClassicalRegister(1, name='c')
	qc = QuantumCircuit(qr, cr)
	qc.rx(params[0], qr[0])
	qc.measure(qr, cr)

	return qc


def get_expected_values(counts: Dict[str, int], shots_cnt: int):
	qubit_cnt = len(next(iter(counts.keys())))
	dist = np.zeros((qubit_cnt, ))

	for k, v in counts.items():
		for i, c in enumerate(k):
			if c == "1":
				dist[qubit_cnt - 1 - i] += v

	dist /= shots_cnt

	return dist


def objective_function(target: np.ndarray, backend, shots_cnt: int) -> Callable[[np.ndarray], float]:
	def callable_obj_func(params: np.ndarray):
		qc = create_simple_circuit(params)
		result = execute(qc, backend, shots=shots_cnt).result()
		output = get_expected_values(result.get_counts(qc), shots_cnt)

		cost: float = ((output - target) * (output - target)).sum()

		return cost

	return callable_obj_func


def test_all_optimizers():
	optimizers: List[Optimizer] = [
		ADAM(maxiter=100, tol=0, lr=0.1, eps=0.1),
		# CG(),
		# COBYLA(),
		# L_BFGS_B(),
		# GSLS(),
		# NELDER_MEAD(),
		# NFT(),
		# # P_BFGS(),
		# POWELL(),
		# SLSQP(),
		# SPSA(),
		# TNC()
	]

	target = np.array([0.8])
	backend = QasmSimulator()
	obj_func = objective_function(target, backend, 1000)

	for optim in optimizers:
		error_list: List[float] = []
		train_time_list: List[float] = []

		for _ in range(100):
			error, train_time = train_circuit(1, obj_func, optim)
			error_list.append(error)
			train_time_list.append(train_time)

		print(type(optim).__name__)
		print("median error:", np.median(error_list))
		print("median time:", np.median(train_time_list))
		print()


def train_circuit(num_vars: int, obj_func: Callable[[np.ndarray], float], optim: Optimizer) -> Tuple[float, float]:
	params = np.random.rand(num_vars)
	time1 = time.time()
	ret = optim.optimize(num_vars=num_vars, objective_function=obj_func, initial_point=params)
	time2 = time.time()
	print(ret[0])
	print(ret[1])
	print(ret[2])
	# print()

	return ret[1], time2 - time1


def main():
	backend = Aer.get_backend("qasm_simulator")
	NUM_SHOTS = 1000
	target = np.array([0.8])

	# Initialize the COBYLA optimizer
	optimizer = COBYLA()

	# train circuit
	obj_func = objective_function(target, backend, NUM_SHOTS)
	error, train_time = train_circuit(3, obj_func, COBYLA())

	print("Error:", error)
	print("Time:", train_time)


	# Obtain the output distribution using the final parameters
	# qc = create_simple_circuit(ret[0])
	# counts = execute(qc, backend, shots=NUM_SHOTS).result().get_counts(qc)
	# output = get_expected_values(counts, NUM_SHOTS)

	# print("Target", target)
	# print("Output:", output)
	# print("Output Error:", ret[1])
	# print("Number of objectiv function calls:", ret[2])
	# print("Parameters Found:", ret[0])


if __name__ == "__main__":
	# main()
	test_all_optimizers()
