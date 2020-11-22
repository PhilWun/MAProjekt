from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import UGate


def create_unit_cell(param_prefix: str, num_qubits: int):
	"""
	Implements circuit B from J. Romero, J. P. Olson, and A. Aspuru-Guzik, “Quantum autoencoders for efficient compression
	of quantum data,” Quantum Sci. Technol., vol. 2, no. 4, p. 045001, Dec. 2017, doi: 10.1088/2058-9565/aa8072.

	:param param_prefix:
	:param num_qubits:
	:return:
	"""
	register = QuantumRegister(num_qubits)
	unit_cell = QuantumCircuit(register)
	layer_idx = 0

	# first layer of single qubit rotations
	for i in range(num_qubits):
		unit_cell.u(
			Parameter(param_prefix + str(layer_idx) + "_" + str(i) + "a"),
			Parameter(param_prefix + str(layer_idx) + "_" + str(i) + "b"),
			Parameter(param_prefix + str(layer_idx) + "_" + str(i) + "c"),
			register[i])

	unit_cell.barrier()

	layer_idx += 1

	# one layer of controlled ratations per qubit
	for i in range(num_qubits):
		for j in range(num_qubits):
			if i != j:
				controlled_rotation_gate = UGate(
					Parameter(param_prefix + str(layer_idx + i) + "_" + str(i) + "_" + str(j) + "a"),
					Parameter(param_prefix + str(layer_idx + i) + "_" + str(i) + "_" + str(j) + "b"),
					Parameter(param_prefix + str(layer_idx + i) + "_" + str(i) + "_" + str(j) + "c")
				).control()

				unit_cell.append(controlled_rotation_gate, [register[i], register[j]])

	unit_cell.barrier()
	layer_idx += num_qubits

	# last layer of single qubit rotations
	for i in range(num_qubits):
		unit_cell.u(
			Parameter(param_prefix + str(layer_idx) + "_" + str(i) + "a"),
			Parameter(param_prefix + str(layer_idx) + "_" + str(i) + "b"),
			Parameter(param_prefix + str(layer_idx) + "_" + str(i) + "c"),
			register[i])

	unit_cell.barrier()

	print(unit_cell.draw(output="text"))


def main():
	create_unit_cell("", 3)


if __name__ == "__main__":
	main()
