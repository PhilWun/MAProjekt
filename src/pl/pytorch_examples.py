import argparse
import os
import pickle
import sys
from itertools import chain
from math import pi
from os import listdir
from random import getrandbits
from typing import Callable, Optional, Type, List, Iterator

import mlflow
import numpy as np
import pandas as pd
import pennylane as qml
import torch
from mlflow.pyfunc import PythonModel, PythonModelContext
from sklearn.preprocessing import StandardScaler, RobustScaler
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(".")  # add root of the project to the PYTHONPATH so that the other modules can be imported

import QNN1
import QNN2
import QNN3
import datasets.creditcardfraud.load as creditcardfraud
import datasets.fashion_mnist.load as fashion_mnist
import datasets.heart_disease_uci.load as heart_disease_uci


def create_qlayer(constructor_func: Callable, q_num: int) -> qml.qnn.TorchLayer:
	"""
	Input of the created quantum layer should be in the range [0, pi]. The output will be in the range [-1, 1] if the
	measurement is done with pauli-z.

	:param constructor_func: Function that constructs the circuit.
	:param q_num: Number of qubits.
	:return: Pennylane TorchLayer.
	"""
	dev = qml.device('default.qubit', wires=q_num, shots=1000, analytic=False)
	circ_func, param_num = constructor_func(q_num)
	qnode = qml.QNode(circ_func, dev, interface="torch")

	weight_shapes = {"weights": param_num}
	qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

	return qlayer


def training_loop(
		model: torch.nn.Module, train_input: torch.Tensor, train_target: torch.Tensor, test_input: Optional[torch.Tensor],
		test_target: Optional[torch.Tensor], optis: List, steps: int, batch_size: int):
	loss_func = torch.nn.MSELoss()
	train_dataset = TensorDataset(train_input, train_target)
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

	test_data_available = test_input is not None and test_target is not None

	if test_data_available:
		test_dataset = TensorDataset(test_input, test_target)
		test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

	batch_cnt_overall = 0

	for i in range(steps):
		# training on the training dataset
		error_sum = 0
		batch_cnt = 0

		for batch_input, batch_target in train_dataloader:
			for opt in optis:
				opt.zero_grad()

			loss = loss_func(model(batch_input), batch_target)
			loss.backward()

			for opt in optis:
				opt.step()

			error_sum += loss.item()
			mlflow.log_metric("Training Batch MSE", loss.item(), batch_cnt_overall)
			batch_cnt += 1
			batch_cnt_overall += 1

		error_mean = error_sum / batch_cnt
		print("Step:", i, "Training MSE:", error_mean)
		mlflow.log_metric("Training MSE", error_mean, i)
		log_model_parameters(model, i)

		if test_data_available:
			# calculating the error on the test dataset
			error_sum = 0
			batch_cnt = 0

			for batch_input, batch_target in test_dataloader:
				with torch.no_grad():
					loss = loss_func(model(batch_input), batch_target)
					error_sum += loss.item()
					batch_cnt += 1

			error_mean = error_sum / batch_cnt
			print("Step:", i, "Test MSE:", error_mean)
			mlflow.log_metric("Test MSE", error_mean, i)


qnn_constructors = {
	"QNN1": QNN1.constructor,
	"QNN2": QNN2.constructor,
	"QNN3": QNN3.constructor,
}


class QuantumModel(torch.nn.Module):
	def __init__(self, q_num: int, qnn_name: str, autoencoder: bool, embedding_size: int):
		super(QuantumModel, self).__init__()

		self.q_num = q_num
		self.autoencoder = autoencoder
		self.embedding_size = embedding_size

		self.q_layer1 = create_qlayer(qnn_constructors[qnn_name], q_num)

		if autoencoder:
			self.q_layer2 = create_qlayer(qnn_constructors[qnn_name], q_num)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Calculates the output of the model.

		:param x: The input. Values should be in the range [0, pi].
		:return: Output of the model. Values are in the range [-1, 1] if autoencoder is true, otherwise [0, pi].
		"""
		embedding = self.q_layer1(x)
		# scaling the values to be in the range [0, pi]
		embedding = (embedding / 2.0 + 0.5) * pi

		if self.autoencoder:
			embedding[:, 0:self.q_num - self.embedding_size] = 0
			reconstruction = self.q_layer2(embedding)

			return reconstruction
		else:
			return embedding


class HybridAutoencoder(torch.nn.Module):
	def __init__(self, input_size: int, q_num: int, embedding_size: int, qnn_name: str):
		super(HybridAutoencoder, self).__init__()

		self.q_num = q_num
		self.embedding_size = embedding_size

		self.fc1 = torch.nn.Linear(input_size, q_num)
		self.q_layer1 = create_qlayer(qnn_constructors[qnn_name], q_num)
		self.q_layer2 = create_qlayer(qnn_constructors[qnn_name], q_num)
		self.fc2 = torch.nn.Linear(q_num, input_size)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Calculates the output of the model.

		:param x: Input. The range of the values can be anything, but should be appropriately scaled.
		:return: Output of the model. The range of the values can be anything.
		"""
		# encoder
		x = torch.sigmoid(self.fc1(x))
		x = x * pi  # scale with pi because input to the quantum layer should be in the range [0, pi]
		embedding = self.q_layer1(x)  # output in the range [-1, 1]

		# scaling the values to be in the range [0, pi]
		embedding = (embedding / 2.0 + 0.5) * pi
		# bottleneck
		embedding[:, 0:self.q_num - self.embedding_size] = 0

		# decoder
		x = self.q_layer2(embedding)
		reconstruction = self.fc2(x)

		return reconstruction

	def get_classical_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
		return chain(self.fc1.parameters(), self.fc2.parameters())

	def get_quantum_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
		return chain(self.q_layer1.parameters(), self.q_layer2.parameters())


class ClassicalAutoencoder(torch.nn.Module):
	def __init__(self, input_size: int, intermediate_size: int, embedding_size: int):
		super(ClassicalAutoencoder, self).__init__()

		self.fc1 = torch.nn.Linear(input_size, intermediate_size)
		self.fc2 = torch.nn.Linear(intermediate_size, embedding_size)
		self.fc3 = torch.nn.Linear(embedding_size, intermediate_size)
		self.fc4 = torch.nn.Linear(intermediate_size, input_size)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = torch.sigmoid(self.fc1(x))
		x = torch.sigmoid(self.fc2(x))
		x = torch.sigmoid(self.fc3(x))
		x = torch.sigmoid(self.fc4(x))

		return x


class MLFModel(PythonModel):
	"""
	MLflow model to store a trained model in a folder.
	"""
	def __init__(self, model_class: Type, *args):
		self.model_class = model_class
		self.args = args

	def predict(self, context: PythonModelContext, model_input: pd.DataFrame):
		model = self.model_class(*self.args)

		files = listdir(context.artifacts["parameters"])
		sd = pickle.load(open(sorted(files)[-1], "rb"))
		model.load_state_dict(sd)

		return model(torch.tensor(model_input.to_numpy()))


def log_model(mlf_model: MLFModel):
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
		artifacts={"parameter_folder": "runs:/" + mlflow.active_run().info.run_id + "/parameters"})


def log_model_parameters(model: torch.nn.Module, step: int):
	sd = model.state_dict()
	folder_path = "/tmp/" + mlflow.active_run().info.run_id

	if not os.path.exists(folder_path):
		os.mkdir(folder_path)

	file_name = folder_path + "/{:06}.pickle".format(step)
	pickle.dump(sd, open(file_name, mode="wb"))
	mlflow.log_artifact(file_name, "parameters")


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
	model = mlflow.pyfunc.load_model("file:///home/philipp/MEGA/Studium/PlanQK/MA/Projekt/mlruns/0/433e0d15b3ff4ea783ea2ab60cb2782b/artifacts/hybrid_model")
	inputs = pd.DataFrame([[0.0] * q_num])
	print(model.predict(inputs))


def parse_arg_str(args_str: str, arg_types: List[type], arg_names: List[str]) -> List:
	split_args = args_str.split(",")
	parsed = []

	for value_str, t, name in zip(split_args, arg_types, arg_names):
		if t == bool:
			value = value_str.lower() == "true"
		else:
			value = t(value_str)

		parsed.append(value)
		mlflow.set_tag(name, value)

	return parsed


def parse_optimizer_and_args(
		optimizer_name: str, optimizer_args_str: str, params_q: Iterator[torch.nn.parameter.Parameter],
		params_c: Iterator[torch.nn.parameter.Parameter], is_hybrid: bool) -> List:
	class_and_args = {
		"Adadelta": (
			torch.optim.Adadelta,
			[float, float, float, float],
			["lr", "rho", "eps", "weight_decay"]
		),
		"Adagrad": (
			torch.optim.Adagrad,
			[float, float, float, float, float],
			["lr", "lr_decay", "weight_decay", "initial_accumulator_value", "eps"]
		),
		"Adam": (
			torch.optim.Adam,
			[float, float, float, float, float, bool],
			["lr", "beta1", "beta2", "eps", "weight_decay", "amsgrad"]
		),
		"AdamW": (
			torch.optim.AdamW,
			[float, float, float, float, float, bool],
			["lr", "beta1", "beta2", "eps", "weight_decay", "amsgrad"]
		),
		"Adamax": (
			torch.optim.Adamax,
			[float, float, float, float, float],
			["lr", "beta1", "beta2", "eps", "weight_decay"]
		),
		"ASGD": (
			torch.optim.ASGD,
			[float, float, float, float, float],
			["lr", "lambd", "alpha", "t0", "weigth_decay"]
		),
		"LBFGS": (
			torch.optim.LBFGS,
			[float, int, int, float, float, int],
			["lr", "max_iter", "max_eval", "tolerance_grad", "tolerance_change", "history_size"]
		),
		"RMSprop": (
			torch.optim.RMSprop,
			[float, float, float, float, float, bool],
			["lr", "alpha", "eps", "weight_decay", "momentum", "centered"]
		),
		"Rprop": (
			torch.optim.Rprop,
			[float, float, float, float, float],
			["lr", "eta1", "eta2", "step_size1", "step_size2"]
		),
		"SGD": (
			torch.optim.SGD,
			[float, float, float, float, bool],
			["lr", "momentum", "dampening", "weight_decay", "nesterov"]
		)
	}

	lr_c = 0

	if is_hybrid:
		split = optimizer_args_str.split(",", 1)
		lr_c = float(split[0])
		optimizer_args_str = split[1]

	ca = class_and_args[optimizer_name]
	opt_args = parse_arg_str(optimizer_args_str, ca[1], ca[2])

	if optimizer_name == "Adam":
		opt_args = [opt_args[0]] + [(opt_args[1], opt_args[2])] + opt_args[3:]
	elif optimizer_name == "AdamW":
		opt_args = [opt_args[0]] + [(opt_args[1], opt_args[2])] + opt_args[3:]
	elif optimizer_name == "Adamax":
		opt_args = [opt_args[0]] + [(opt_args[1], opt_args[2])] + opt_args[3:]
	elif optimizer_name == "Rprop":
		opt_args = [opt_args[0], (opt_args[1], opt_args[2]), (opt_args[3], opt_args[4])]

	optimizer_q = ca[0](params_q, *opt_args)

	if is_hybrid:
		optimizer_c = ca[0](params_c, lr_c, *opt_args[1:])

		return [optimizer_q, optimizer_c]
	else:
		return [optimizer_q]


def cli():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", type=str)
	parser.add_argument("--dataset_fraction", type=float)
	parser.add_argument("--scaler", type=str)
	parser.add_argument("--model", type=str)
	parser.add_argument("--model_args", type=str)
	parser.add_argument("--steps", type=int)
	parser.add_argument("--batch_size", type=int)
	parser.add_argument("--optimizer", type=str)
	parser.add_argument("--optimizer_args", type=str)
	parser.add_argument("--seed", type=str)

	args = parser.parse_args()

	dataset_name: str = args.dataset
	dataset_fraction: float = args.dataset_fraction
	scaler_name: str = args.scaler
	model_name: str = args.model
	model_args_str: str = args.model_args
	steps: int = args.steps
	batch_size: int = args.batch_size
	optimizer_name: str = args.optimizer
	optimizer_args_str: str = args.optimizer_args
	seed_str: str = args.seed

	train_input = None
	train_target = None
	test_input = None
	test_target = None

	if seed_str == "":
		seed = getrandbits(32)
		mlflow.set_tag("seed", seed)
	else:
		seed = int(seed_str)

	torch.random.manual_seed(seed)

	if dataset_name == "trivial":
		train_input = np.array([[0.8] * 3], dtype=np.float32)
	elif dataset_name == "fashion_mnist":
		train_input, _, test_input, _ = fashion_mnist.load_dataset()
	elif dataset_name == "heart_disease_uci":
		train_input, _, test_input, _ = heart_disease_uci.load_dataset(rnd_seed=seed)
	elif dataset_name == "creditcardfraud":
		train_input, _, test_input, _ = creditcardfraud.load_dataset(rnd_seed=seed)
	else:
		raise ValueError(dataset_name)

	if dataset_fraction != 1.0:
		rng = np.random.default_rng(seed)

		# shuffle the rows
		rng.shuffle(train_input, axis=0)
		rng.shuffle(test_input, axis=0)
		# keep only a fraction of the whole dataset
		train_input = train_input[0:int(train_input.shape[0] * dataset_fraction)]
		test_input = test_input[0:int(test_input.shape[0] * dataset_fraction)]

	if scaler_name == "none":
		pass
	elif scaler_name == "standard":
		scaler = StandardScaler()
		scaler.fit(train_input)
		train_input = scaler.transform(train_input)
		test_input = scaler.transform(test_input)
	elif scaler_name == "robust":
		scaler = RobustScaler()
		scaler.fit(train_input)
		train_input = scaler.transform(train_input)
		test_input = scaler.transform(test_input)

	# convert to tensor
	train_input = torch.tensor(train_input, requires_grad=False)

	if test_input is not None:
		test_input = torch.tensor(test_input, requires_grad=False)

	# set the target equal to the input, because the model is used as an autoencoder
	train_target = train_input
	test_target = test_input

	if model_name == "hybrid":
		model_args = parse_arg_str(
			model_args_str,
			[int, int, str],
			["q_num", "embedding_size", "qnn_name"]
		)
		model_args = (train_input.shape[1], *model_args)  # infer the input size from the dataset
		model = HybridAutoencoder(*model_args)
		params_primary = model.get_quantum_parameters()
		params_secondary = model.get_classical_parameters()
	elif model_name == "quantum":
		model_args = parse_arg_str(
			model_args_str,
			[str, bool, int],
			["qnn_name", "autoencoder", "embedding_size"]
		)
		model_args = (train_input.shape[1], *model_args)  # infer the number of qubits from the dataset
		model = QuantumModel(*model_args)
		params_primary = model.parameters()
		params_secondary = []
	elif model_name == "classical":
		model_args = parse_arg_str(
			model_args_str,
			[int, int],
			["intermediate_size", "embedding_size"]
		)
		model_args = (train_input.shape[1], *model_args)  # infer the number of qubits from the dataset
		model = ClassicalAutoencoder(*model_args)
		params_primary = model.parameters()
		params_secondary = []
	else:
		raise ValueError()

	is_hybrid = model_name == "hybrid"
	optimizer = parse_optimizer_and_args(optimizer_name, optimizer_args_str, params_primary, params_secondary, is_hybrid)

	training_loop(model, train_input, train_target, test_input, test_target, optimizer, steps, batch_size)
	mlf_model = MLFModel(model.__class__, *model_args)
	log_model(mlf_model)


if __name__ == "__main__":
	# example_train_qnn1()
	# example_load_model()
	cli()
