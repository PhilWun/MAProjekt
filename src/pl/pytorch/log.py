import os
import pickle
from os import listdir
from typing import Type

import mlflow
import torch
from mlflow.pyfunc import PythonModel, PythonModelContext
import pandas as pd
import numpy as np


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

		return model.embed(torch.tensor(model_input.to_numpy()))


def log_model(mlf_model: MLFModel):
	mlflow.pyfunc.log_model(
		"model",
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


def log_embeddings(embeddings: torch.Tensor, labels: torch.Tensor, step: int):
	folder_path = "/tmp/" + mlflow.active_run().info.run_id

	if not os.path.exists(folder_path):
		os.mkdir(folder_path)

	embeddings_file_name = folder_path + "/{:06}_embeddings.csv".format(step)
	labels_file_name = folder_path + "/{:06}_labels.csv".format(step)

	np.savetxt(embeddings_file_name, embeddings.detach().numpy(), delimiter=",")
	np.savetxt(labels_file_name, labels.detach().numpy(), delimiter=",")

	mlflow.log_artifact(embeddings_file_name, "embeddings")
	mlflow.log_artifact(labels_file_name, "labels")
