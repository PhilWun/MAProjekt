import os
from typing import Optional

import mlflow
import numpy as np
import plotly.graph_objects as go

from mlflow_tools import get_artifact_uri


def plot_embeddings(embeddings_path: str, labels_path: str) -> go.Figure:
	embeddings = np.loadtxt(embeddings_path, delimiter=",")
	labels = np.loadtxt(labels_path, delimiter=",")

	x0 = np.compress(labels == 0, embeddings[:, 0])
	y0 = np.compress(labels == 0, embeddings[:, 1])
	x1 = np.compress(labels == 1, embeddings[:, 0])
	y1 = np.compress(labels == 1, embeddings[:, 1])
	x2 = np.compress(labels == 2, embeddings[:, 0])
	y2 = np.compress(labels == 2, embeddings[:, 1])

	fig = go.Figure()
	fig.add_trace(go.Scatter(
		x=x0,
		y=y0,
		mode="markers", marker_color="red", name="Klasse 0"))
	fig.add_trace(go.Scatter(
		x=x1,
		y=y1,
		mode="markers", marker_color="blue", name="Klasse 1"))
	fig.add_trace(go.Scatter(
		x=x2,
		y=y2,
		mode="markers", marker_color="orange"))

	return fig


def plot_embeddings_from_run(run_id: str, step: int) -> Optional[go.Figure]:
	artifact_path = get_artifact_uri(run_id)
	embedding_path = artifact_path + "/embeddings/{:06}_embeddings.csv".format(step)
	labels_path = artifact_path + "/labels/{:06}_labels.csv".format(step)

	if (not os.path.exists(embedding_path)) or (not os.path.exists(labels_path)):
		return None

	return plot_embeddings(embedding_path, labels_path)


def plot_all_steps_to_file(run_id: str, folder: str):
	os.makedirs(folder, exist_ok=True)
	step = 0

	while True:
		fig = plot_embeddings_from_run(run_id, step)

		if fig is None:
			break
		else:
			fig.write_image(os.path.join(folder, "{:03}.pdf".format(step)), scale=2)
			step += 1


if __name__ == "__main__":
	# for i in range(1000):
	# 	plot_embeddings_from_run("b360ae685c6847e4aa4b0e54d2453320", i).write_image("src/plot/images/{:03}.png".format(i), scale=2)
	plot_all_steps_to_file("d5114f246f034516929ee4405e09d44a", "src/plot/d5114f246f034516929ee4405e09d44a")
