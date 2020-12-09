import pennylane as qml
import torch


dev = qml.device("default.qubit", wires=2, shots=1000, analytic=False)


@qml.qnode(dev, interface="torch")
def circuit(phi, theta):
	qml.RX(phi[0], wires=0)
	qml.RZ(phi[1], wires=1)
	qml.CNOT(wires=[0, 1])
	qml.RX(theta, wires=0)

	return qml.expval(qml.PauliZ(0))


def cost(phi, theta):
	return torch.abs(circuit(phi, theta) - 0.5)**2


phi = torch.tensor([0.011, 0.012], requires_grad=True)
theta = torch.tensor(0.05, requires_grad=True)

opt = torch.optim.Adam([phi, theta], lr=0.1)
steps = 200


def closure():
	opt.zero_grad()
	loss = cost(phi, theta)
	loss.backward()
	print(loss)

	return loss


for i in range(steps):
	opt.step(closure)
