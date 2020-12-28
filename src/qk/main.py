from typing import List, Callable, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.aqua.components.optimizers import Optimizer, ADAM, CG, COBYLA, L_BFGS_B, GSLS, NELDER_MEAD, NFT, POWELL, SLSQP, SPSA, TNC
from qiskit.providers import BaseBackend
from qiskit.providers.aer import QasmSimulator, AerJob

import QNN1
from src.qk.tools import train_circuit, get_expected_values


def execute_optimizer(
		transpiled_circuit: QuantumCircuit, backend, target: np.ndarray, param_num: int, optim: Optimizer, num_runs: int,
		target_error: Optional[float] = None):
	print(type(optim).__name__)
	iter_num_list: List[float] = []
	last_error_list: List[float] = []
	time_list: List[float] = []

	for _ in range(num_runs):
		intermediate_errors = []

		def error_callback(err: float):
			intermediate_errors.append(err)

		obj_func = Optimizer.wrap_function(mean_squared_error, (transpiled_circuit, backend, target, error_callback))
		error, train_time, param_values = train_circuit(param_num, obj_func, optim)
		last_error_list.append(error)
		time_list.append(train_time)
		print("objective function calls:", len(intermediate_errors))

		if target_error is not None:
			converged = False

			for i, err in enumerate(intermediate_errors):
				if err < 0.01:
					iter_num_list.append(i + 1)
					converged = True
					break

			if not converged:
				print("not converged")

	print()

	if target_error is not None:
		print("number of executed circuits till the target error was reached")
		print("mean:", np.mean(iter_num_list))
		print("standard deviation:", np.std(iter_num_list))
		print()

	print("error (mean):", np.mean(last_error_list))
	print("error (std):", np.std(last_error_list))


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


def mean_squared_error(
		values: np.ndarray, circ: QuantumCircuit, backend: BaseBackend, target: np.ndarray,
		callback: Optional[Callable[[float], None]] = None) -> float:
	output = measure_output(values, circ, backend)
	mse = ((output - target) * (output - target)).mean()

	if callback is not None:
		callback(mse)

	return mse


optimizers: List[Optimizer] = [
	ADAM(maxiter=100, tol=0, lr=0.1, eps=0.1),
	CG(maxiter=100, gtol=0, eps=0.1),
	COBYLA(maxiter=100),  # gradient-free optimization
	L_BFGS_B(maxiter=100, epsilon=0.1),
	GSLS(maxiter=100, min_gradient_norm=-1, min_step_size=-1),
	NELDER_MEAD(maxfev=100, xatol=0),  # gradient-free optimization
	NFT(maxiter=100),  # gradient-free optimization
	POWELL(maxiter=100, xtol=0),  # gradient-free optimization
	SLSQP(maxiter=100, ftol=0, eps=0.1),
	SPSA(maxiter=100),  # recommended for noisy measurements, approximates gradient with two measurements
	TNC(maxiter=100, eps=0.1)
]


def test_all_optimizers():
	qnn: QuantumCircuit = QNN1.create_qiskit_circuit("", 3)
	backend = QasmSimulator()
	transpiled_qnn: QuantumCircuit = transpile(qnn, backend=backend)

	for optim in optimizers:
		execute_optimizer(transpiled_qnn, backend, np.array([0.8, 0.8, 0.8]), len(transpiled_qnn.parameters), optim, 10, target_error=0.01)


def test_training():
	qnn: QuantumCircuit = QNN1.create_qiskit_circuit("", 3)
	backend = QasmSimulator()
	transpiled_qnn: QuantumCircuit = transpile(qnn, backend=backend)

	_, _, param_values = train_circuit(
		len(transpiled_qnn.parameters),
		Optimizer.wrap_function(mean_squared_error, (transpiled_qnn, backend, np.array([0.8, 0.8, 0.8]))),
		COBYLA(maxiter=100))

	print(measure_output(param_values, transpiled_qnn, backend))


if __name__ == "__main__":
	test_all_optimizers()
	# test_training()
