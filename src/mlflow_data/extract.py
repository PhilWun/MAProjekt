import csv
import json
import os
from typing import List

from mlflow.entities import Metric
from mlflow.tracking import MlflowClient
import numpy as np


def extract_metric(run_id: str, metric_name: str) -> List[float]:
	client = MlflowClient()
	metric_history: List[Metric] = client.get_metric_history(run_id, metric_name)
	metric_list = []

	for m in metric_history:
		v: float = m.value
		metric_list.append(v)

	return metric_list


def extract_metrics_from_csv(run_id_file_name: str, output_file_name: str, metric_name: str):
	reader = csv.DictReader(open(run_id_file_name, mode="rt"))
	run_ids = []

	for row in reader:
		run_ids.append(row["Run ID"])

	metrics = []

	for run_id in run_ids:
		metrics.append(extract_metric(run_id, metric_name))

	json.dump(metrics, open(output_file_name, mode="wt"))


def extract_all_metrics():
	input_folder_path = "src/mlflow_data/clustering/model_comp/iris/mlflow_csv"
	input_files = os.listdir(input_folder_path)

	for input_file in input_files:
		input_file_path = os.path.join(input_folder_path, input_file)
		extract_metrics_from_csv(
			input_file_path,
			input_file_path.replace("runs", "metrics").replace("mlflow_csv", "metrics").replace(".csv", ".json"),
			"adjusted_mutual_info_score")


def load_metrics(file_path: str) -> List[List[float]]:
	return json.load(open(file_path, mode="rt"))


def first_index_value_under(values: List[float], threshold: float) -> int:
	"""
	:param values: list of values
	:param threshold: threshold
	:return: Index of the first element, that is smaller than threshold. -1 if no value is smaller.
	"""
	return np.argmax(np.array([float("inf")] + values) < threshold) - 1


def print_convergence_indexes():
	folder_path = "src/mlflow_data/optimizer_test/metrics"
	files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
	convergence_threshold = 0.01

	for f in files:
		print(f)
		metrics = load_metrics(f)

		for m in metrics:
			print(first_index_value_under(m, convergence_threshold))

		print()


# print(extract_metric("2330677808fc4a4c875041f691547be2", "Training MSE"))
# extract_metrics_from_csv("src/mlflow_data/adam_runs.csv", "src/mlflow_data/optimizer_test/output.csv", "Training MSE")

if __name__ == "__main__":
	extract_all_metrics()
	# print_convergence_indexes()
