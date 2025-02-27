import sys
from typing import Callable

import pennylane as qml

sys.path.append(".")  # add root of the project to the PYTHONPATH so that the other modules can be imported

from src.pl import QNN1, QNN2, QNN3


def create_qlayer(constructor_func: Callable, q_num: int) -> qml.qnn.TorchLayer:
	"""
	Input of the created quantum layer should be in the range [0, pi]. The output will be in the range [-1, 1] if the
	measurement is done with pauli-z.

	:param constructor_func: Function that constructs the circuit.
	:param q_num: Number of qubits.
	:return: Pennylane TorchLayer.
	"""
	dev = qml.device('default.qubit', wires=q_num, shots=1000, analytic=False)
	circ_func, param_num = constructor_func(q_num)
	qnode = qml.QNode(circ_func, dev, interface="torch")

	weight_shapes = {"weights": param_num}
	qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

	return qlayer


qnn_constructors = {
	"QNN1": QNN1.constructor,
	"QNN2": QNN2.constructor,
	"QNN3": QNN3.constructor,
}
