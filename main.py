from typing import List, Callable

import numpy as np
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.aqua.components.optimizers import Optimizer, ADAM, CG, COBYLA, L_BFGS_B, GSLS, NELDER_MEAD, NFT, P_BFGS, POWELL, SLSQP, SPSA, TNC
from qiskit.providers import BaseBackend
from qiskit.providers.aer import QasmSimulator, AerJob

import QNN1
import QNN2
import QNN3
from tools import train_circuit, get_expected_values


def test_all_optimizers(obj_func: Callable[[np.ndarray], float], param_num: int):
	optimizers: List[Optimizer] = [
		ADAM(maxiter=100, tol=0, lr=0.1, eps=0.1),
		CG(maxiter=100, gtol=0, eps=0.1),
		COBYLA(maxiter=100),  # gradient-free optimization
		L_BFGS_B(maxiter=100, epsilon=0.1),
		GSLS(maxiter=100, min_gradient_norm=0),
		NELDER_MEAD(maxfev=100, xatol=0),  # gradient-free optimization
		NFT(maxiter=100),  # gradient-free optimization
		POWELL(maxiter=100, xtol=0),  # gradient-free optimization
		SLSQP(maxiter=100, ftol=0, eps=0.1),
		SPSA(maxiter=100),  # recommended for noisy measurements, approximates gradient with two measurements
		TNC(maxiter=100, eps=0.1, ftol=0, xtol=0, gtol=0)
	]

	for optim in optimizers:
		error_list: List[float] = []
		train_time_list: List[float] = []

		for _ in range(10):
			error, train_time, param_values = train_circuit(param_num, obj_func, optim)
			error_list.append(error)
			train_time_list.append(train_time)

		print(type(optim).__name__)
		print("error (mean):", np.mean(error_list))
		print("error (std):", np.std(error_list))
		print("mean time:", np.median(train_time_list))
		print()


def _bind_params(circ: QuantumCircuit, values: np.ndarray):
	value_dict = {}

	for param, value in zip(circ.parameters, values):
		value_dict[param] = value

	return circ.bind_parameters(value_dict)


def measure_output(values: np.ndarray, circ: QuantumCircuit, backend: BaseBackend) -> np.ndarray:
	qobj = assemble(_bind_params(circ, values))
	job: AerJob = backend.run(qobj)
	result = job.result()
	counts = result.get_counts()
	output = get_expected_values(counts)

	return output


def obj_func(values: np.ndarray, circ: QuantumCircuit, backend: BaseBackend, target: np.ndarray, error_mask: np.ndarray) -> float:
	output = measure_output(values, circ, backend)
	mse = ((output - target) * (output - target) * error_mask).sum() / error_mask.sum()

	return mse


def test_qnn():
	qnn: QuantumCircuit = QNN3.create_qiskit_circuit("", 3)
	backend = QasmSimulator()
	transpiled_qnn: QuantumCircuit = transpile(qnn, backend=backend)
	test_all_optimizers(
		Optimizer.wrap_function(obj_func, (transpiled_qnn, backend, np.array([0.8, 0.8, 0.8]), np.array([1, 1, 1]))),
		len(transpiled_qnn.parameters))

	# _, _, param_values = train_circuit(
	# 	len(transpiled_qnn.parameters),
	# 	Optimizer.wrap_function(obj_func, (transpiled_qnn, backend, np.array([0.8, 0.8, 0.8]), np.array([1, 1, 1]))),
	# 	COBYLA(maxiter=100))

	# print(measure_output(param_values, transpiled_qnn, backend))


if __name__ == "__main__":
	# main()
	# test_all_optimizers()
	test_qnn()
