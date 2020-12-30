from math import pi

import pennylane as qml
import torch
import QNN1


def cost(circuit: qml.QNode, input_data: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.mean((circuit(input_data, weights) - target) ** 2)


def main():
    q_num = 3
    dev = qml.device('default.qubit', wires=q_num, shots=1000, analytic=False)
    circ_func, param_num = QNN1.constructor(q_num)
    qnode = qml.QNode(circ_func, dev, interface="torch")

    input_data = torch.tensor([0.0] * q_num, requires_grad=False)
    target = torch.tensor([0.8] * q_num, requires_grad=False)
    weights = torch.tensor(torch.rand(param_num).clone().detach() * 2 * pi, requires_grad=True)

    opt = torch.optim.Adam([weights], lr=0.1)
    steps = 100

    for i in range(steps):
        opt.zero_grad()
        loss = cost(qnode, input_data, target, weights)
        loss.backward()
        opt.step()
        print(loss.item())


if __name__ == "__main__":
    main()
