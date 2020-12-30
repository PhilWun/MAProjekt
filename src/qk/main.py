import argparse
from typing import List, Callable, Optional

import mlflow
import numpy as np
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.aqua.components.optimizers import Optimizer, ADAM, CG, COBYLA, L_BFGS_B, GSLS, NELDER_MEAD, NFT, POWELL, SLSQP, SPSA, TNC
from qiskit.providers import BaseBackend
from qiskit.providers.aer import QasmSimulator, AerJob

import QNN1
import QNN2
import QNN3
from tools import train_circuit, get_expected_values


# TODO: add input to circuit
def execute_optimizer(transpiled_circuit: QuantumCircuit, backend, target: np.ndarray, param_num: int, optim: Optimizer):
	eval_num = [0]  # hacky way to store an integer in an object to use it inside the callback function

	def error_callback(err: float):
		print("Function evaluation:", eval_num[0], "MSE: ", err)
		mlflow.log_metric("Training MSE", err, eval_num[0])
		eval_num[0] = eval_num[0] + 1

	obj_func = Optimizer.wrap_function(mean_squared_error, (transpiled_circuit, backend, target, error_callback))
	error, train_time, param_values = train_circuit(param_num, obj_func, optim)


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
		execute_optimizer(transpiled_qnn, backend, np.array([0.8, 0.8, 0.8]), len(transpiled_qnn.parameters), optim)


def test_training():
	qnn: QuantumCircuit = QNN1.create_qiskit_circuit("", 3)
	backend = QasmSimulator()
	transpiled_qnn: QuantumCircuit = transpile(qnn, backend=backend)

	_, _, param_values = train_circuit(
		len(transpiled_qnn.parameters),
		Optimizer.wrap_function(mean_squared_error, (transpiled_qnn, backend, np.array([0.8, 0.8, 0.8]))),
		COBYLA(maxiter=100))

	print(measure_output(param_values, transpiled_qnn, backend))


def cli():
	parser = argparse.ArgumentParser()
	parser.add_argument("--qnn", type=str)
	parser.add_argument("--qnum", type=int)
	parser.add_argument("--optimizer", type=str)
	parser.add_argument("--maxiter", type=int)
	parser.add_argument("--tol", type=float)
	parser.add_argument("--lr", type=float)
	parser.add_argument("--beta1", type=float)
	parser.add_argument("--beta2", type=float)
	parser.add_argument("--noise_factor", type=float)
	parser.add_argument("--eps", type=float)  # alias: epsilon
	parser.add_argument("--gtol", type=float)  # alias: min_gradient_norm
	parser.add_argument("--rhobeg", type=float)
	parser.add_argument("--maxfun", type=int)  # alias: max_eval, maxfev
	parser.add_argument("--factr", type=float)
	parser.add_argument("--sampling_radius", type=float)
	parser.add_argument("--sampling_size_factor", type=int)
	parser.add_argument("--initial_step_size", type=float)
	parser.add_argument("--min_step_size", type=float)
	parser.add_argument("--step_size_multiplier", type=float)
	parser.add_argument("--armijo_parameter", type=float)
	parser.add_argument("--max_failed_rejection_sampling", type=int)
	parser.add_argument("--xtol", type=float)  # alias: xatol
	parser.add_argument("--adaptive", type=int)  # 0: false, 1: true
	parser.add_argument("--reset_interval", type=int)
	parser.add_argument("--ftol", type=float)
	parser.add_argument("--c0", type=float)
	parser.add_argument("--c1", type=float)
	parser.add_argument("--c2", type=float)
	parser.add_argument("--c3", type=float)
	parser.add_argument("--c4", type=float)
	parser.add_argument("--skip_calibration", type=int)  # 0: false, 1: true
	parser.add_argument("--accuracy", type=float)

	args = parser.parse_args()

	qnn_name = args.qnn
	qnum = args.qnum
	optimizer_name = args.optimizer
	maxiter = args.maxiter
	tol = args.tol
	lr = args.lr
	beta1 = args.beta1
	beta2 = args.beta2
	noise_factor = args.noise_factor
	eps = args.eps
	gtol = args.gtol
	rhobeg = args.rhobeg
	maxfun = args.maxfun
	factr = args.factr
	sampling_radius = args.sampling_radius
	sampling_size_factor = args.sampling_size_factor
	initial_step_size = args.initial_step_size
	min_step_size = args.min_step_size
	step_size_multiplier = args.step_size_multiplier
	armijo_parameter = args.armijo_parameter
	max_failed_rejection_sampling = args.max_failed_rejection_sampling
	xtol = args.xtol
	adaptive = args.adaptive
	reset_interval = args.reset_interval
	ftol = args.ftol
	c0 = args.c0
	c1 = args.c1
	c2 = args.c2
	c3 = args.c3
	c4 = args.c4
	skip_calibration = args.skip_calibration
	accuracy = args.accuracy

	optimizer = None

	if optimizer_name == "ADAM":
		optimizer = ADAM(maxiter, tol, lr, beta1, beta2, noise_factor, eps, False)
	elif optimizer_name == "AMSGRAD":
		optimizer = ADAM(maxiter, tol, lr, beta1, beta2, noise_factor, eps, True)
	elif optimizer_name == "CG":
		optimizer = CG(maxiter, False, gtol, tol, eps)
	elif optimizer_name == "COBYLA":
		optimizer = COBYLA(maxiter, False, rhobeg, tol)
	elif optimizer_name == "L_BFGS_B":
		optimizer = L_BFGS_B(maxfun, maxiter, factr, -1, eps)
	elif optimizer_name == "GSLS":
		optimizer = GSLS(maxiter, maxfun, False, sampling_radius, sampling_size_factor, initial_step_size, min_step_size, step_size_multiplier, armijo_parameter, gtol, max_failed_rejection_sampling)
	elif optimizer_name == "NELDER_MEAD":
		optimizer = NELDER_MEAD(maxiter, maxfun, False, xtol, tol, bool(adaptive))
	elif optimizer_name == "NFT":
		optimizer = NFT(maxiter, maxfun, False, reset_interval)
	elif optimizer_name == "POWELL":
		optimizer = POWELL(maxiter, maxfun, False, xtol, tol)
	elif optimizer_name == "SLSQP":
		optimizer = SLSQP(maxiter, False, ftol, tol, eps)
	elif optimizer_name == "SPSA":
		optimizer = SPSA(maxiter, 1, 1, c0, c1, c2, c3, c4, bool(skip_calibration))
	elif optimizer_name == "TNC":
		optimizer = TNC(maxiter, False, accuracy, ftol, xtol, gtol, tol, eps)
	else:
		raise ValueError()

	qnn: QuantumCircuit

	if qnn_name == "QNN1":
		qnn = QNN1.create_qiskit_circuit("", qnum)
	elif qnn_name == "QNN2":
		qnn = QNN2.create_qiskit_circuit("", qnum)
	elif qnn_name == "QNN3":
		qnn = QNN3.create_qiskit_circuit("", qnum)
	else:
		raise ValueError()

	backend = QasmSimulator()
	transpiled_qnn: QuantumCircuit = transpile(qnn, backend=backend)

	execute_optimizer(transpiled_qnn, backend, np.array([0.8] * qnum), len(transpiled_qnn.parameters), optimizer)


if __name__ == "__main__":
	# test_all_optimizers()
	# test_training()
	cli()
