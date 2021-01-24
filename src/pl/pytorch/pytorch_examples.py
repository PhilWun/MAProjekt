import pickle
import sys

import mlflow
import pandas as pd
import torch

sys.path.append(".")  # add root of the project to the PYTHONPATH so that the other modules can be imported

from src.pl.pytorch.log import MLFModel
from src.pl.pytorch.models.common_functions import create_qlayer
from src.pl.pytorch.training import training_loop
import src.pl.QNN1 as QNN1


def example_train_qnn1():
	q_num = 3
	qlayer = create_qlayer(QNN1.constructor, q_num)

	model = torch.nn.Sequential(
		qlayer
	)

	inputs = torch.tensor([0.0] * q_num, requires_grad=False)
	target = torch.tensor([0.8] * q_num, requires_grad=False)
	opt = torch.optim.Adam(model.parameters(), lr=0.1)

	training_loop(model, inputs, target, None, None, [opt], 100, 1)

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


if __name__ == "__main__":
	# example_train_qnn1()
	# example_load_model()
	pass
