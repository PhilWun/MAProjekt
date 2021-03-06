import argparse
import sys
from random import getrandbits
from typing import List, Iterator

import mlflow
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

sys.path.append(".")  # add root of the project to the PYTHONPATH so that the other modules can be imported

from src.parsing import parse_arg_str, parse_optimizer_and_args
from src.pl.pytorch.log import log_model, MLFModel
from src.pl.pytorch.models.classical import ClassicalAutoencoder
from src.pl.pytorch.models.hybrid import HybridAutoencoder
from src.pl.pytorch.models.quantum import QuantumModel
from src.pl.pytorch.training import training_loop

import datasets.creditcardfraud.load as creditcardfraud
import datasets.fashion_mnist.load as fashion_mnist
import datasets.heart_disease_uci.load as heart_disease_uci
import datasets.breast_cancer_wisconsin.load as breast_cancer_wisconsin
import datasets.iris.load as iris


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

	if seed_str == "":
		seed = getrandbits(32)
		mlflow.set_tag("seed", seed)
	else:
		seed = int(seed_str)

	torch.random.manual_seed(seed)

	test_input = None
	train_label = None
	test_label = None

	if dataset_name == "trivial":
		train_input = np.array([[0.8] * 3], dtype=np.float32)
	elif dataset_name == "fashion_mnist":
		train_input, train_label, test_input, test_label = fashion_mnist.load_dataset()
	elif dataset_name == "heart_disease_uci":
		train_input, train_label, test_input, test_label = heart_disease_uci.load_dataset(rnd_seed=seed)
	elif dataset_name == "creditcardfraud":
		train_input, train_label, test_input, test_label = creditcardfraud.load_dataset(rnd_seed=seed)
	elif dataset_name == "breast_cancer_wisconsin":
		train_input, train_label, test_input, test_label = breast_cancer_wisconsin.load_dataset(rnd_seed=seed)
	elif dataset_name == "iris":
		train_input, train_label, test_input, test_label = iris.load_dataset(rnd_seed=seed)
	else:
		raise ValueError(dataset_name)

	if dataset_fraction != 1.0:
		# choose a random subset of the training / test data
		train_input, _, train_label, _ = train_test_split((train_input, train_label), train_size=dataset_fraction, random_state=seed)
		test_input, _, test_label, _ = train_test_split((test_input, test_label), train_size=dataset_fraction, random_state=seed)

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

	if train_label is not None:
		train_label = torch.tensor(train_label, requires_grad=False)

	if test_label is not None:
		test_label = torch.tensor(test_label, requires_grad=False)

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

	training_loop(model, train_input, train_target, train_label, test_input, test_target, test_label, optimizer, steps, batch_size)
	mlf_model = MLFModel(model.__class__, *model_args)
	log_model(mlf_model)


if __name__ == "__main__":
	cli()
