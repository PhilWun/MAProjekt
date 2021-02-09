import argparse
import os
import sys

import mlflow
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, DBSCAN, FeatureAgglomeration, OPTICS, \
	SpectralClustering
import numpy as np

sys.path.append(".")

from src.mlflow_tools import get_artifact_uri
from src.parsing import parse_arg_str

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--run_id", type=str)
	parser.add_argument("--cluster_algo", type=str)
	parser.add_argument("--cluster_args", type=str)
	parser.add_argument("--step", type=int)

	args = parser.parse_args()

	run_id: str = args.run_id
	cluster_algo_name: str = args.cluster_algo
	cluster_args: str = args.cluster_args
	step: int = args.step

	# TODO: add other clustering algorithms
	if cluster_algo_name == "AffinityPropagation":
		args = parse_arg_str(
			cluster_args,
			[float, int, int, str],
			["damping", "max_iter", "convergence_iter"])
		cluster_algo = AffinityPropagation(*args)
	elif cluster_algo_name == "AgglomerativeClustering":
		args = parse_arg_str(
			cluster_args,
			[int, str],
			["n_clusters", "linkage"]
		)
		cluster_algo = AgglomerativeClustering(n_clusters=args[0], linkage=args[1])
	elif cluster_algo_name == "DBSCAN":
		args = parse_arg_str(
			cluster_args,
			[float, int],
			["eps", "min_samples"]
		)
		cluster_algo = DBSCAN(*args, metric="precomputed")
	elif cluster_algo_name == "FeatureAgglomeration":
		args = parse_arg_str(
			cluster_args,
			[int, str],
			["n_clusters", "linkage"]
		)
		cluster_algo = FeatureAgglomeration(n_clusters=args[0], linkage=args[1])
	elif cluster_algo_name == "OPTICS":
		args = parse_arg_str(
			cluster_args,
			[int],
			["min_samples"]
		)
		cluster_algo = OPTICS(min_samples=args[0])
	elif cluster_algo_name == "SpectralClustering":
		args = parse_arg_str(
			cluster_args,
			[int, int, int],
			["n_clusters", "n_components", "n_init"]
		)
		cluster_algo = SpectralClustering(n_clusters=args[0], n_components=args[1], n_init=args[2])
	else:
		raise ValueError(cluster_algo_name)

	artifact_path = get_artifact_uri(run_id)
	embeddings_folder_path = os.path.join(artifact_path, "embeddings")
	labels_folder_path = os.path.join(artifact_path, "labels")

	if step == -1:
		embeddings_path = os.path.join(embeddings_folder_path, sorted(os.listdir(embeddings_folder_path))[-1])
		labels_path = os.path.join(labels_folder_path, sorted(os.listdir(labels_folder_path))[-1])
	else:
		embeddings_path = os.path.join(embeddings_folder_path, "{:06}_embeddings.csv".format(step))
		labels_path = os.path.join(labels_folder_path, "{:06}_labels.csv".format(step))

	print(embeddings_path)
	print(labels_path)
	embeddings = np.loadtxt(embeddings_path, delimiter=",")
	labels = np.loadtxt(labels_path, delimiter=",")

	cluster_labels = cluster_algo.fit_predict(embeddings)
	clusters = {}

	for cluster_label in np.unique(cluster_labels):
		clusters[int(cluster_label)] = list(np.compress(cluster_labels == cluster_label, labels))

	mlflow.log_dict(clusters, "clusters.json")
