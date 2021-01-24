import sys
from typing import Optional, List

import mlflow
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(".")  # add root of the project to the PYTHONPATH so that the other modules can be imported

from src.pl.pytorch.log import log_model_parameters


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
