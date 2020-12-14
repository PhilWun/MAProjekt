from math import pi
from typing import List

import pennylane as qml
import torch
import numpy as np
from qiskit.circuit import Parameter

import QNN1
import QNN2
import QNN3

dev = qml.device("default.qubit", wires=3, shots=1000, analytic=False)
qc = QNN2.create_qiskit_circuit("", 3)
params: List[Parameter] = list(qc.parameters)


@qml.qnode(dev, interface="torch")
def circuit(weights: torch.Tensor):
	value_dict = {}

	for param, value in zip(params, weights):
		value_dict[param] = value

	qml.from_qiskit(qc)(value_dict)

	return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))


def cost(weights: torch.Tensor, target: torch.Tensor):
	output = (circuit(weights) - 1) / -2.0  # normalizes the expected values

	return torch.mean((output - target) ** 2)


weights = torch.tensor(np.random.rand(len(params)) * 2 * pi, requires_grad=True)
target = torch.tensor([0.8, 0.8, 0.8], requires_grad=False)

opt = torch.optim.Adam([weights], lr=0.1)
steps = 200


def closure():
	opt.zero_grad()
	loss = cost(weights, target)
	loss.backward()
	print(loss.item())

	return loss


for i in range(steps):
	opt.step(closure)
