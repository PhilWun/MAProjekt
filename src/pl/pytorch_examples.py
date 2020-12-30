from math import pi

import pennylane as qml
import torch
import QNN1


def main():
    q_num = 3
    dev = qml.device('default.qubit', wires=q_num, shots=1000, analytic=False)
    circ_func, param_num = QNN1.constructor(q_num)
    qnode = qml.QNode(circ_func, dev, interface="torch")
    weight_shapes = {"weights": param_num}
    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
    model = torch.nn.Sequential(
        qlayer
    )

    input_data = torch.tensor([0.0] * q_num, requires_grad=False)
    target = torch.tensor([0.8] * q_num, requires_grad=False)

    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_func = torch.nn.MSELoss()
    steps = 100

    for i in range(steps):
        opt.zero_grad()
        loss = loss_func(model(input_data), target)
        loss.backward()
        opt.step()
        print(loss.item())


if __name__ == "__main__":
    main()
