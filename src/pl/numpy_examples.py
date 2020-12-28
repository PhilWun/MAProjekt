import argparse
from typing import Callable, Tuple

import mlflow
import numpy as onp
import pennylane as qml
import pennylane.numpy as np
from pennylane import QNode

import QNN1
import QNN2
import QNN3


def normalize_output(output):
	return output / -2.0 + 0.5


def cost_func_constructor(qnode: QNode, input_values: np.tensor, target: np.tensor):
	def cost(weights: np.tensor):
		result = np.mean((normalize_output(qnode(input_values, weights)) - target) ** 2)

		return result

	return cost


def training_loop(qnode: QNode, input_values: np.tensor, target: np.tensor, params: np.tensor, optimizer, steps: int):
	for i in range(steps):
		cost_list = []

		for j in range(input_values.shape[0]):
			cost_func = cost_func_constructor(qnode, input_values[j], target[j])
			params, cost = optimizer.step_and_cost(cost_func, params)
			cost_list.append(cost)

		mean_cost = onp.mean(cost_list)
		print("Step:", i, "Error:", mean_cost)
		mlflow.log_metric("Training MSE", mean_cost, step=i)

	return params


def test_qnn(q_num: int, qnn_name: str, optimizer, steps: int):
	dev = qml.device('default.qubit', wires=q_num, shots=1000, analytic=False)

	# choose the QNN depending on the argument
	if qnn_name == "QNN1":
		circ_func, param_num = QNN1.constructor(q_num)
	elif qnn_name == "QNN2":
		circ_func, param_num = QNN2.constructor(q_num)
	elif qnn_name == "QNN3":
		circ_func, param_num = QNN3.constructor(q_num)
	else:
		raise ValueError()

	qnode = qml.QNode(circ_func, dev)

	# create the parameter and data arrays
	params = np.array(np.random.rand(param_num))
	input_values = np.array([[0] * q_num], requires_grad=False)
	target = np.array([[0.8] * q_num], requires_grad=False)

	training_loop(qnode, input_values, target, params, optimizer, steps)


def print_circuit(constructor_func: Callable[[int], Tuple[Callable, int]]):
	q_num = 3
	dev = qml.device('default.qubit', wires=q_num, shots=1000, analytic=False)
	circ_func, param_num = constructor_func(q_num)
	qnode = qml.QNode(circ_func, dev)
	qnode(np.zeros(q_num), np.zeros(param_num))
	print(qnode.draw())


if __name__ == "__main__":
	# optimizer = qml.AdamOptimizer(0.1)
	# test_qnn(3, "QNN1", optimizer, 100)
	# print_circuit(QNN2.constructor)
	parser = argparse.ArgumentParser()
	parser.add_argument("--qnn", type=str)
	parser.add_argument("--qnum", type=int)
	parser.add_argument("--optimizer", type=str)
	parser.add_argument("--stepsize", type=float)
	parser.add_argument("--eps", type=float)
	parser.add_argument("--beta1", type=float)
	parser.add_argument("--beta2", type=float)
	parser.add_argument("--momentum", type=float)
	parser.add_argument("--decay", type=float)
	parser.add_argument("--steps", type=int)

	args = parser.parse_args()
	optimizer = None

	if args.optimizer == "Adagrad":
		optimizer = qml.AdagradOptimizer(args.stepsize, args.eps)
	elif args.optimizer == "Adam":
		optimizer = qml.AdamOptimizer(args.stepsize, args.beta1, args.beta2, args.eps)
	elif args.optimizer == "GradientDescent":
		optimizer = qml.GradientDescentOptimizer(args.stepsize)
	elif args.optimizer == "Momentum":
		optimizer = qml.MomentumOptimizer(args.stepsize, args.momentum)
	elif args.optimizer == "NesterovMomentum":
		optimizer = qml.NesterovMomentumOptimizer(args.stepsize, args.momentum)
	elif args.optimizer == "RMSProp":
		optimizer = qml.RMSPropOptimizer(args.stepsize, args.decay, args.eps)
	elif args.optimizer == "Rotosolve":
		optimizer = qml.RotosolveOptimizer()

	test_qnn(args.qnum, args.qnn, optimizer, args.steps)
