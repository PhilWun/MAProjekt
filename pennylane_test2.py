from typing import List

import pennylane as qml
import pennylane.numpy as np

q_num = 3


def qnn3(input_values: np.tensor, weights: np.tensor) -> List[np.tensor]:
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


opt = qml.GradientDescentOptimizer(stepsize=0.1)
steps = 100
# params = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
params = np.array(np.random.rand(6))
input_values = np.array([[0, np.pi, 0], [np.pi, 0, np.pi]], requires_grad=False)
target = np.array([[-1, 1, -1], [1, -1, 1]], requires_grad=False)
dev = qml.device('default.qubit', wires=3, shots=1000, analytic=False)
qnode = qml.QNode(qnn3, dev)


def generate_cost_func(input_values: np.tensor, target: np.tensor):
	def cost(weights: np.tensor):
		result = np.mean((qnode(input_values, weights) - target) ** 2)
		print(result)

		return result

	return cost


for i in range(steps):
	for j in range(input_values.shape[0]):
		cost = generate_cost_func(input_values[j], target[j])
		params = opt.step(cost, params)

print(params)
print(qnode(input_values[0], params))
print(qnode(input_values[1], params))
