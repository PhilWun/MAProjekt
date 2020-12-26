from typing import Callable, Tuple

import pennylane as qml
import pennylane.numpy as np
from pennylane import QNode

from pl import QNN1


def normalize_output(output):
	return output / -2.0 + 0.5


def cost_func_constructor(qnode: QNode, input_values: np.tensor, target: np.tensor):
	def cost(weights: np.tensor):
		result = np.mean((normalize_output(qnode(input_values, weights)) - target) ** 2)
		print(result)

		return result

	return cost


def training_loop(qnode: QNode, input_values: np.tensor, target: np.tensor, params: np.tensor, optimizer, steps: int):
	for i in range(steps):
		for j in range(input_values.shape[0]):
			cost = cost_func_constructor(qnode, input_values[j], target[j])
			params = optimizer.step(cost, params)

	return params


def test_qnn1():
	dev = qml.device('default.qubit', wires=3, shots=1000, analytic=False)
	circ_func, param_num = QNN1.constructor(3)
	qnode = qml.QNode(circ_func, dev)

	params = np.array(np.random.rand(param_num))
	input_values = np.array([[0, 0, 0]], requires_grad=False)
	target = np.array([[0.8, 0.8, 0.8]], requires_grad=False)

	opt = qml.GradientDescentOptimizer(stepsize=0.1)
	steps = 100

	params = training_loop(qnode, input_values, target, params, opt, steps)

	print(params)
	print(normalize_output(qnode(input_values[0], params)))
	print(qnode.draw(show_variable_names=True))
	# print(qnode(input_values[1], params))


def print_circuit(constructor_func: Callable[[int], Tuple[Callable, int]]):
	q_num = 3
	dev = qml.device('default.qubit', wires=q_num, shots=1000, analytic=False)
	circ_func, param_num = constructor_func(q_num)
	qnode = qml.QNode(circ_func, dev)
	qnode(np.zeros(q_num), np.zeros(param_num))
	print(qnode.draw())


if __name__ == "__main__":
	test_qnn1()
	# print_circuit(QNN2.constructor)
