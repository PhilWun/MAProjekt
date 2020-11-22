from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Parameter, Instruction


def create_unit_cell(param_prefix: str, num_qubits: int):
	"""
	Implements circuit A from J. Romero, J. P. Olson, and A. Aspuru-Guzik, “Quantum autoencoders for efficient compression
	of quantum data,” Quantum Sci. Technol., vol. 2, no. 4, p. 045001, Dec. 2017, doi: 10.1088/2058-9565/aa8072.

	:param param_prefix:
	:param num_qubits:
	:return:
	"""
	register = QuantumRegister(num_qubits)
	unit_cell = QuantumCircuit(register)
	idx = 1

	for i in range(num_qubits - 1):
		for j in range(num_qubits - 1 - i):
			inst = create_two_qubit_gate(param_prefix + "U" + str(idx) + "_")
			unit_cell.append(inst, [register[j], register[j + i + 1]])
			idx += 1

	print(unit_cell.draw(output="text"))


def create_two_qubit_gate(param_prefix: str) -> Instruction:
	"""
	Implements a general two-qubit gate as seen in F. Vatan and C. Williams, “Optimal Quantum Circuits for General
	Two-Qubit Gates,” Phys. Rev. A, vol. 69, no. 3, p. 032315, Mar. 2004, doi: 10.1103/PhysRevA.69.032315.

	:param param_prefix:
	:return:
	"""
	register = QuantumRegister(2)
	circ = QuantumCircuit(register, name="U")

	circ.u(
		Parameter(param_prefix + "a1_a"),
		Parameter(param_prefix + "a1_b"),
		Parameter(param_prefix + "a1_c"),
		register[0])

	circ.u(
		Parameter(param_prefix + "a2_a"),
		Parameter(param_prefix + "a2_b"),
		Parameter(param_prefix + "a2_c"),
		register[1]
	)

	circ.cnot(register[1], register[0])

	circ.rz(Parameter(param_prefix + "t1"), register[0])
	circ.ry(Parameter(param_prefix + "t2"), register[1])

	circ.cnot(register[0], register[1])

	circ.ry(Parameter(param_prefix + "t3"), register[1])

	circ.cnot(register[1], register[0])

	circ.u(
		Parameter(param_prefix + "a3_a"),
		Parameter(param_prefix + "a3_b"),
		Parameter(param_prefix + "a3_c"),
		register[0]
	)

	circ.u(
		Parameter(param_prefix + "a4_a"),
		Parameter(param_prefix + "a4_b"),
		Parameter(param_prefix + "a4_c"),
		register[1]
	)

	return circ.to_instruction()


def main():
	create_unit_cell("", 3)


if __name__ == "__main__":
	main()
