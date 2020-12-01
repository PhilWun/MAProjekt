from typing import Callable

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
import numpy as np

from tools import get_expected_values


def create_simple_circuit(params):
	qr = QuantumRegister(1, name="q")
	cr = ClassicalRegister(1, name='c')
	qc = QuantumCircuit(qr, cr)
	qc.rx(params[0], qr[0])
	qc.measure(qr, cr)

	return qc


def objective_function(target: np.ndarray, backend, shots_cnt: int) -> Callable[[np.ndarray], float]:
	def callable_obj_func(params: np.ndarray):
		qc = create_simple_circuit(params)
		result = execute(qc, backend, shots=shots_cnt).result()
		output = get_expected_values(result.get_counts(qc))

		cost: float = ((output - target) * (output - target)).sum()

		return cost

	return callable_obj_func
