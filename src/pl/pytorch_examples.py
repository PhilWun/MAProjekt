from math import pi
from typing import Callable, Tuple

import pennylane as qml
import torch
import QNN1


def create_qlayer(constructor_func: Callable, q_num: int) -> qml.qnn.TorchLayer:
    dev = qml.device('default.qubit', wires=q_num, shots=1000, analytic=False)
    circ_func, param_num = constructor_func(q_num)
    qnode = qml.QNode(circ_func, dev, interface="torch")

    weight_shapes = {"weights": param_num}
    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    return qlayer


def training_loop(
        model: torch.nn.Module, inputs: torch.Tensor, target: torch.Tensor, opt: torch.optim.Optimizer, steps: int):
    loss_func = torch.nn.MSELoss()

    for i in range(steps):
        opt.zero_grad()
        loss = loss_func(model(inputs), target)
        loss.backward()
        opt.step()
        print(loss.item())


def example_train_qnn1():
    q_num = 3
    qlayer = create_qlayer(QNN1.constructor, q_num)

    model = torch.nn.Sequential(
        qlayer
    )

    inputs = torch.tensor([0.0] * q_num, requires_grad=False)
    target = torch.tensor([0.8] * q_num, requires_grad=False)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)

    training_loop(model, inputs, target, opt, 100)


if __name__ == "__main__":
    example_train_qnn1()
