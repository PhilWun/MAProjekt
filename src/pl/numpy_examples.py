from typing import Callable, Tuple

import mlflow
import numpy as onp
import pennylane as qml
import pennylane.numpy as np
from pennylane import QNode

from pl import QNN1, QNN2, QNN3


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
	mlflow.log_param("Number of qubits", q_num)
	mlflow.log_param("Number of steps", steps)
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

	mlflow.log_param("QNN", qnn_name)
	qnode = qml.QNode(circ_func, dev)

	# create the parameter and data arrays
	params = np.array(np.random.rand(param_num))
	input_values = np.array([[0] * q_num], requires_grad=False)
	target = np.array([[0.8] * q_num], requires_grad=False)

	if type(optimizer) == qml.AdagradOptimizer:
		mlflow.log_param("Optimizer", "AdagradOptimizer")
		mlflow.log_param("stepsize", optimizer._stepsize)
		mlflow.log_param("eps", optimizer.eps)
	if type(optimizer) == qml.AdamOptimizer:
		mlflow.log_param("Optimizer", "AdamOptimizer")
		mlflow.log_param("stepsize", optimizer._stepsize)
		mlflow.log_param("beta1", optimizer.beta1)
		mlflow.log_param("beta2", optimizer.beta2)
		mlflow.log_param("eps", optimizer.eps)
	if type(optimizer) == qml.GradientDescentOptimizer:
		mlflow.log_param("Optimizer", "GradientDescentOptimizer")
		mlflow.log_param("stepsize", optimizer._stepsize)
	if type(optimizer) == qml.MomentumOptimizer:
		mlflow.log_param("Optimizer", "MomentumOptimizer")
		mlflow.log_param("stepsize", optimizer._stepsize)
		mlflow.log_param("momentum", optimizer.momentum)
	if type(optimizer) == qml.NesterovMomentumOptimizer:
		mlflow.log_param("Optimizer", "NesterovMomentumOptimizer")
		mlflow.log_param("stepsize", optimizer._stepsize)
		mlflow.log_param("momentum", optimizer.momentum)
	if type(optimizer) == qml.RMSPropOptimizer:
		mlflow.log_param("Optimizer", "RMSPropOptimizer")
		mlflow.log_param("stepsize", optimizer._stepsize)
		mlflow.log_param("decay", optimizer.decay)
		mlflow.log_param("eps", optimizer.eps)
	if type(optimizer) == qml.RotosolveOptimizer:
		mlflow.log_param("Optimizer", "RotosolveOptimizer")

	training_loop(qnode, input_values, target, params, optimizer, steps)


def print_circuit(constructor_func: Callable[[int], Tuple[Callable, int]]):
	q_num = 3
	dev = qml.device('default.qubit', wires=q_num, shots=1000, analytic=False)
	circ_func, param_num = constructor_func(q_num)
	qnode = qml.QNode(circ_func, dev)
	qnode(np.zeros(q_num), np.zeros(param_num))
	print(qnode.draw())


if __name__ == "__main__":
	optimizer = qml.AdamOptimizer(0.1)
	test_qnn(3, "QNN1", optimizer, 100)
	# print_circuit(QNN2.constructor)
