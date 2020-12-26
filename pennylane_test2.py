from typing import List, Callable, Tuple

import pennylane as qml
import pennylane.numpy as np
from pennylane import QNode


def qnn1_constructor(q_num: int) -> Tuple[Callable, int]:
	"""
	Implements circuit A from J. Romero, J. P. Olson, and A. Aspuru-Guzik, “Quantum autoencoders for efficient compression
	of quantum data,” Quantum Sci. Technol., vol. 2, no. 4, p. 045001, Dec. 2017, doi: 10.1088/2058-9565/aa8072.

	:param q_num:
	:return: function that constructs the circuit
	"""
	def circuit(input_values: np.tensor, weights: np.tensor) -> List[np.tensor]:
		# angle encoding of the input
		for i in range(q_num):
			qml.RX(input_values[i], wires=i)

		idx = 0  # current index for the weights

		for i in range(q_num - 1):
			for j in range(q_num - 1 - i):
				add_two_qubit_gate(j, j + i + 1, weights[15 * idx:15 * idx + 15])
				idx += 1

		return [qml.expval(qml.PauliZ(wires=i)) for i in range(q_num)]

	def add_two_qubit_gate(q1, q2, weights: np.tensor):
		"""
		Adds a general two-qubit gate.
		Implements a general two-qubit gate as seen in F. Vatan and C. Williams, “Optimal Quantum Circuits for General
		Two-Qubit Gates,” Phys. Rev. A, vol. 69, no. 3, p. 032315, Mar. 2004, doi: 10.1103/PhysRevA.69.032315.

		:param q1: first input qubit for the gate
		:param q2: second input qubit for the gate
		"""

		qml.Rot(weights[0], weights[1], weights[2], wires=q1)
		qml.Rot(weights[3], weights[4], weights[5], wires=q2)

		qml.CNOT(wires=[q2, q1])

		qml.RZ(weights[6], wires=q1)
		qml.RY(weights[7], wires=q2)

		qml.CNOT(wires=[q1, q2])
		qml.RY(weights[8], wires=q2)
		qml.CNOT(wires=[q2, q1])

		qml.Rot(weights[9], weights[10], weights[11], wires=q1)
		qml.Rot(weights[12], weights[13], weights[14], wires=q2)

	param_num = 0

	for i in range(q_num - 1):
		for j in range(q_num - 1 - i):
			param_num += 15

	return circuit, param_num


def qnn2_constructor(q_num: int) -> Tuple[Callable, int]:
	"""
	Implements circuit B from J. Romero, J. P. Olson, and A. Aspuru-Guzik, “Quantum autoencoders for efficient compression
	of quantum data,” Quantum Sci. Technol., vol. 2, no. 4, p. 045001, Dec. 2017, doi: 10.1088/2058-9565/aa8072.

	:param q_num: number of qubits
	:return:
	"""

	def circuit(input_values: np.tensor, weights: np.tensor) -> List[np.tensor]:
		# angle encoding of the input
		for i in range(q_num):
			qml.RX(input_values[i], wires=i)

		idx = 0  # current index for the weights

		# layer of single qubit rotations
		for i in range(q_num):
			qml.Rot(weights[idx], weights[idx + 1], weights[idx + 2], wires=i)
			idx += 3

		# layer of controlled single qubit rotations
		for i in range(q_num):
			for j in range(q_num):
				if i != j:
					qml.CRot(weights[idx], weights[idx + 1], weights[idx + 2], wires=[i, j])
					idx += 3

		# layer of single qubit rotations
		for i in range(q_num):
			qml.Rot(weights[idx], weights[idx + 1], weights[idx + 2], wires=i)
			idx += 3

		return [qml.expval(qml.PauliZ(wires=i)) for i in range(q_num)]

	return circuit, 2 * q_num + q_num * (q_num - 1)


def qnn3_constructor(q_num: int) -> Tuple[Callable, int]:
	"""
	Implements the circuit from A. Abbas, D. Sutter, C. Zoufal, A. Lucchi, A. Figalli, and S. Woerner, “The power of
	quantum neural networks,” arXiv:2011.00027 [quant-ph], Oct. 2020, Accessed: Nov. 08, 2020. [Online]. Available:
	http://arxiv.org/abs/2011.00027.

	:param q_num: number of qubits
	:return: function that constructs the circuit
	"""
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

	return circuit, 2 * q_num


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
	circ_func, param_num = qnn1_constructor(3)
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
	# test_qnn1()
	print_circuit(qnn1_constructor)
