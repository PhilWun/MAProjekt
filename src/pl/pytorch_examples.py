import argparse
import os
import pickle
from typing import Callable

import mlflow
import pandas as pd
import pennylane as qml
import torch
from mlflow.pyfunc import PythonModel, PythonModelContext

import QNN1
import QNN2
import QNN3


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
		print("Step:", i, "MSE:", loss.item())
		mlflow.log_metric("Training MSE", loss.item(), i)


class MLFModel(PythonModel):
	"""
	MLflow model to store a trained model in a folder.
	"""
	def __init__(self, q_num: int):
		self.q_num = q_num

	def predict(self, context: PythonModelContext, model_input: pd.DataFrame):
		qlayer = create_qlayer(QNN1.constructor, self.q_num)

		model = torch.nn.Sequential(
			qlayer
		)

		sd = pickle.load(open(context.artifacts["circuit_parameters"], "rb"))
		model.load_state_dict(sd)

		return model(torch.tensor(model_input.to_numpy()))


def log_model(pyt_model: torch.nn.Module, mlf_model: MLFModel):
	sd = pyt_model.state_dict()
	pickle.dump(sd, open("state_dict.pickle", mode="wb"))
	mlflow.log_artifact("state_dict.pickle")

	mlflow.pyfunc.log_model(
		"hybrid_model",
		python_model=mlf_model,
		conda_env="conda.yaml",
		code_path=[
			"src/pl/QNN1.py",
			"src/pl/QNN2.py",
			"src/pl/QNN3.py",
			"src/pl/TwoQubitGate.py",
		],
		artifacts={"circuit_parameters": "runs:/" + mlflow.active_run().info.run_id + "/state_dict.pickle"})


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

	sd = model.state_dict()
	pickle.dump(sd, open("state_dict.pickle", mode="wb"))
	mlflow.log_artifact("state_dict.pickle")
	hybrid_model = MLFModel(q_num)
	mlflow.pyfunc.save_model(
		"hybrid_model",
		python_model=hybrid_model,
		conda_env="conda.yaml",
		code_path=[
			"src/pl/QNN1.py",
			"src/pl/QNN2.py",
			"src/pl/QNN3.py",
			"src/pl/TwoQubitGate.py",
		],
		artifacts={"circuit_parameters": "runs:/" + mlflow.active_run().info.run_id + "/state_dict.pickle"})


def example_load_model():
	q_num = 3
	model = mlflow.pyfunc.load_model("hybrid_model")
	inputs = pd.DataFrame([[0.0] * q_num])
	print(model.predict(inputs))


def cli():
	parser = argparse.ArgumentParser()
	parser.add_argument("--qnn", type=str)
	parser.add_argument("--qnum", type=int)
	parser.add_argument("--steps", type=int)
	parser.add_argument("--optimizer", type=str)
	parser.add_argument("--lr", type=float)
	parser.add_argument("--rho", type=float)
	parser.add_argument("--eps", type=float)
	parser.add_argument("--weight_decay", type=float)
	parser.add_argument("--lr_decay", type=float)
	parser.add_argument("--initial_accumulator_value", type=float)
	parser.add_argument("--beta1", type=float)
	parser.add_argument("--beta2", type=float)
	parser.add_argument("--amsgrad", type=int)  # 0: false, 1: true
	parser.add_argument("--lambd", type=float)
	parser.add_argument("--alpha", type=float)
	parser.add_argument("--t0", type=float)
	parser.add_argument("--max_iter", type=int)
	parser.add_argument("--max_eval", type=float)
	parser.add_argument("--tolerance_grad", type=float)
	parser.add_argument("--tolerance_change", type=float)
	parser.add_argument("--history_size", type=int)
	parser.add_argument("--momentum", type=float)
	parser.add_argument("--centered", type=int)
	parser.add_argument("--eta1", type=float)
	parser.add_argument("--eta2", type=float)
	parser.add_argument("--step_size1", type=float)
	parser.add_argument("--step_size2", type=float)
	parser.add_argument("--dampening", type=float)
	parser.add_argument("--nesterov", type=int)

	args = parser.parse_args()

	qnn_name = args.qnn
	qnum = args.qnum
	steps = args.steps
	optimizer_name = args.optimizer
	lr = args.lr
	rho = args.rho
	eps = args.eps
	weight_decay = args.weight_decay
	lr_decay = args.lr_decay
	initial_accumulator_value = args.initial_accumulator_value
	beta1 = args.beta1
	beta2 = args.beta2
	amsgrad = bool(args.amsgrad)
	lambd = args.lambd
	alpha = args.alpha
	t0 = args.t0
	max_iter = args.max_iter
	max_eval = args.max_eval
	tolerance_grad = args.tolerance_grad
	tolerance_change = args.tolerance_change
	history_size = args.history_size
	momentum = args.momentum
	centered = args.centered
	eta1 = args.eta1
	eta2 = args.eta2
	step_size1 = args.step_size1
	step_size2 = args.step_size2
	dampening = args.dampening
	nesterov = bool(args.nesterov)

	if qnn_name == "QNN1":
		constructor_func = QNN1.constructor
	elif qnn_name == "QNN2":
		constructor_func = QNN2.constructor
	elif qnn_name == "QNN3":
		constructor_func = QNN3.constructor
	else:
		raise ValueError()

	qlayer = create_qlayer(constructor_func, qnum)

	model = torch.nn.Sequential(
		qlayer
	)

	inputs = torch.tensor([0.0] * qnum, requires_grad=False)
	target = torch.tensor([0.8] * qnum, requires_grad=False)

	optimizer = None

	if optimizer_name == "Adadelta":
		optimizer = torch.optim.Adadelta(model.parameters(), lr, rho, eps, weight_decay)
	elif optimizer_name == "Adagrad":
		optimizer = torch.optim.Adagrad(model.parameters(), lr, lr_decay, weight_decay, initial_accumulator_value, eps)
	elif optimizer_name == "Adam":
		optimizer = torch.optim.Adam(model.parameters(), lr, (beta1, beta2), eps, weight_decay, amsgrad)
	elif optimizer_name == "AdamW":
		optimizer = torch.optim.AdamW(model.parameters(), lr, (beta1, beta2), eps, weight_decay, amsgrad)
	elif optimizer_name == "Adamax":
		optimizer = torch.optim.Adamax(model.parameters(), lr, (beta1, beta2), eps, weight_decay)
	elif optimizer_name == "ASGD":
		optimizer = torch.optim.ASGD(model.parameters(), lr, lambd, alpha, t0, weight_decay)
	elif optimizer_name == "LBFGS":
		optimizer = torch.optim.LBFGS(model.parameters(), lr, max_iter, max_eval, tolerance_grad, tolerance_change, history_size)
	elif optimizer_name == "RMSprop":
		optimizer = torch.optim.RMSprop(model.parameters(), lr, alpha, eps, weight_decay, momentum, centered)
	elif optimizer_name == "Rprop":
		optimizer = torch.optim.Rprop(model.parameters(), lr, (eta1, eta2), (step_size1, step_size2))
	elif optimizer_name == "SGD":
		optimizer = torch.optim.SGD(model.parameters(), lr, momentum, dampening, weight_decay, nesterov)

	training_loop(model, inputs, target, optimizer, steps)
	mlf_model = MLFModel(qnum)
	log_model(model, mlf_model)


if __name__ == "__main__":
	# example_train_qnn1()
	# example_load_model()
	cli()
