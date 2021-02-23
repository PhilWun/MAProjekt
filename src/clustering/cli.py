import argparse
import os
import sys

import mlflow
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, DBSCAN, FeatureAgglomeration, OPTICS, \
	SpectralClustering, Birch, KMeans, MiniBatchKMeans, MeanShift, SpectralBiclustering, SpectralCoclustering
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

	if cluster_algo_name == "AffinityPropagation":
		args = parse_arg_str(
			cluster_args,
			[float],
			["damping"])
		cluster_algo = AffinityPropagation(*args)
	elif cluster_algo_name == "AgglomerativeClustering":
		args = parse_arg_str(
			cluster_args,
			[int, str],
			["n_clusters", "linkage"]
		)
		cluster_algo = AgglomerativeClustering(n_clusters=args[0], linkage=args[1])
	elif cluster_algo_name == "Birch":
		args = parse_arg_str(
			cluster_args,
			[float, int, int],
			["threshold", "branching_factor", "n_clusters"]
		)
		cluster_algo = Birch(threshold=args[0], branching_factor=args[1], n_clusters=args[2])
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
	elif cluster_algo_name == "KMeans":
		args = parse_arg_str(
			cluster_args,
			[int],
			["n_clusters"]
		)
		cluster_algo = KMeans(n_clusters=args[0])
	elif cluster_algo_name == "MiniBatchKMeans":
		args = parse_arg_str(
			cluster_args,
			[int],
			["n_clusters"]
		)
		cluster_algo = MiniBatchKMeans(n_clusters=args[0])
	elif cluster_algo_name == "MeanShift":
		args = parse_arg_str(
			cluster_args,
			[float],
			["bandwidth"]
		)
		cluster_algo = MeanShift(bandwidth=args[0])
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
			[int],
			["n_clusters"]
		)
		cluster_algo = SpectralClustering(n_clusters=args[0], n_components=args[1], n_init=args[2])
	elif cluster_algo_name == "SpectralBiclustering":
		args = parse_arg_str(
			cluster_args,
			[int],
			["n_clusters"]
		)
		cluster_algo = SpectralBiclustering(n_clusters=args[0])
	elif cluster_algo_name == "SpectralCoclustering":
		args = parse_arg_str(
			cluster_args,
			[int, str],
			["n_clusters"]
		)
		cluster_algo = SpectralCoclustering(n_clusters=args[0])
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
	real_labels = np.loadtxt(labels_path, delimiter=",")

	cluster_labels = cluster_algo.fit_predict(embeddings)
	clusters = {}

	for cluster_label in np.unique(cluster_labels):
		labels_in_cluster = np.compress(cluster_labels == cluster_label, real_labels).astype(dtype=np.int)
		label_counts = np.bincount(labels_in_cluster)
		label_dict = {}

		for i, count in enumerate(label_counts):
			label_dict["Label " + str(i)] = int(count)

		clusters["Cluster " + str(cluster_label)] = label_dict

	mlflow.log_dict(clusters, "clusters.json")
