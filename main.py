from typing import Dict, Callable

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer, execute
from qiskit.aqua.components.optimizers import COBYLA


def get_var_form(params):
    qr = QuantumRegister(1, name="q")
    cr = ClassicalRegister(1, name='c')
    qc = QuantumCircuit(qr, cr)
    qc.u(params[0], params[1], params[2], qr[0])
    qc.measure(qr, cr)

    return qc


def get_expected_values(counts: Dict[str, int], shots_cnt: int):
    qubit_cnt = len(next(iter(counts.keys())))
    dist = np.zeros((qubit_cnt, ))

    for k, v in counts.items():
        for i, c in enumerate(k):
            if c == "1":
                dist[qubit_cnt - 1 - i] += v

    dist /= shots_cnt

    return dist


def objective_function(target: np.ndarray, backend, shots_cnt: int) -> Callable[[np.ndarray], float]:
    def callable_obj_func(params: np.ndarray):
        qc = get_var_form(params)
        result = execute(qc, backend, shots=shots_cnt).result()
        output = get_expected_values(result.get_counts(qc), shots_cnt)

        cost = ((output - target) * (output - target)).sum()

        return cost

    return callable_obj_func


def main():
    backend = Aer.get_backend("qasm_simulator")
    NUM_SHOTS = 10000
    target = np.array([0.8])

    # Initialize the COBYLA optimizer
    optimizer = COBYLA(maxiter=500, tol=0.0001)

    # Create the initial parameters (noting that our single qubit variational form has 3 parameters)
    params = np.random.rand(3)
    ret = optimizer.optimize(num_vars=3, objective_function=objective_function(target, backend, NUM_SHOTS), initial_point=params)

    # Obtain the output distribution using the final parameters
    qc = get_var_form(ret[0])
    counts = execute(qc, backend, shots=NUM_SHOTS).result().get_counts(qc)
    output = get_expected_values(counts, NUM_SHOTS)

    print("Target", target)
    print("Output:", output)
    print("Output Error:", ret[1])
    print("Number of objectiv function calls:", ret[2])
    print("Parameters Found:", ret[0])


if __name__ == "__main__":
    main()
