from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import UGate


def create_unit_cell(param_prefix: str, num_qubits: int):
	"""
	Implements the circuit from A. Abbas, D. Sutter, C. Zoufal, A. Lucchi, A. Figalli, and S. Woerner, “The power of
	quantum neural networks,” arXiv:2011.00027 [quant-ph], Oct. 2020, Accessed: Nov. 08, 2020. [Online]. Available:
	http://arxiv.org/abs/2011.00027.

	:param param_prefix:
	:param num_qubits:
	:return:
	"""
	register = QuantumRegister(num_qubits)
	unit_cell = QuantumCircuit(register)
	layer_idx = 0

	# first layer of single qubit y-rotations
	for i in range(num_qubits):
		unit_cell.ry(
			Parameter(param_prefix + str(layer_idx) + "_" + str(i)),
			register[i])

	layer_idx += 1
	unit_cell.barrier()

	# one layer of controlled ratations per qubit
	for i in range(1, num_qubits):
		for j in range(0, i):
			unit_cell.cnot(j, i)

	unit_cell.barrier()

	# last layer of single qubit y-rotations
	for i in range(num_qubits):
		unit_cell.ry(
			Parameter(param_prefix + str(layer_idx) + "_" + str(i)),
			register[i])

	unit_cell.barrier()

	# TODO: measure in z-basis

	print(unit_cell.draw(output="text"))


def main():
	create_unit_cell("", 4)


if __name__ == "__main__":
	main()
