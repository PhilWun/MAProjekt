from typing import List

import pennylane as qml
import pennylane.numpy as np
from pennylane import QNode


def qnn3_constructor(q_num: int) :
	def circuit(input_values: np.tensor, weights: np.tensor) -> List[np.tensor]:
		# angle encoding of the input
		for i in range(q_num):
			qml.RX(input_values[i], wires=i)

		# RY layer
		for i in range(q_num):
			qml.RY(weights[i], wires=i)

		# CNOT layer
		for i in range(1, q_num):
			for j in range(0, i):
				qml.CNOT(wires=[j, i])

		# RY layer
		for i in range(q_num):
			qml.RY(weights[i + q_num], wires=i)

		return [qml.expval(qml.PauliZ(wires=i)) for i in range(q_num)]

	return circuit


def cost_func_constructor(qnode: QNode, input_values: np.tensor, target: np.tensor):
	def cost(weights: np.tensor):
		result = np.mean((qnode(input_values, weights) - target) ** 2)
		print(result)

		return result

	return cost


def training_loop(qnode: QNode, input_values: np.tensor, target: np.tensor, params: np.tensor, optimizer, steps: int):
	for i in range(steps):
		for j in range(input_values.shape[0]):
			cost = cost_func_constructor(qnode, input_values[j], target[j])
			params = optimizer.step(cost, params)

	return params


def test_qnn3():
	dev = qml.device('default.qubit', wires=3, shots=1000, analytic=False)
	qnode = qml.QNode(qnn3_constructor(3), dev)

	params = np.array(np.random.rand(6))
	input_values = np.array([[0, np.pi, 0]], requires_grad=False)
	target = np.array([[-1, 1, -1]], requires_grad=False)

	opt = qml.GradientDescentOptimizer(stepsize=0.1)
	steps = 100

	params = training_loop(qnode, input_values, target, params, opt, steps)

	print(params)
	print(qnode(input_values[0], params))
	# print(qnode(input_values[1], params))


if __name__ == "__main__":
	test_qnn3()
