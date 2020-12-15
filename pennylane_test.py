from math import pi
from typing import List

import pennylane as qml
import torch
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

import QNN1
import QNN2
import QNN3


def generate_circuit_func(params: List[Parameter], qc: QuantumCircuit):
	def func(weights: torch.Tensor):
		value_dict = {}

		for param, value in zip(params, weights):
			value_dict[param] = value

		qml.from_qiskit(qc)(value_dict)

		return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

	return func


def cost(weights: torch.Tensor, target: torch.Tensor, qnode: qml.QNode):
	output = (qnode(weights) - 1) / -2.0  # normalizes the expected values

	return torch.mean((output - target) ** 2)


def train_loop(
		iterations: int, weights: torch.Tensor, target: torch.Tensor, qnode: qml.QNode):
	opt = torch.optim.Adam([weights], lr=0.1)

	def closure():
		opt.zero_grad()
		loss = cost(weights, target, qnode)
		loss.backward()
		print(loss.item())

		return loss.item()

	for i in range(iterations):
		opt.step(closure)


def test_training():
	dev = qml.device("default.qubit", wires=3, shots=1000, analytic=False)
	qc = QNN1.create_qiskit_circuit("", 3)
	params: List[Parameter] = list(qc.parameters)

	weights = torch.tensor(np.random.rand(len(params)) * 2 * pi, requires_grad=True)
	target = torch.tensor([0.8, 0.8, 0.8], requires_grad=False)

	qnode = qml.QNode(generate_circuit_func(params, qc), dev, interface="torch")

	train_loop(100, weights, target, qnode)


if __name__ == "__main__":
	test_training()
