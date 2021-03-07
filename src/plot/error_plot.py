import json
import os

import plotly.graph_objects as go
import numpy as np


def add_line_with_std(fig: go.Figure, x: np.ndarray, mean: np.ndarray, std: np.ndarray, name: str, dash="solid"):
	std_upper = mean + std
	std_lower = mean - std

	x_rev = x[::-1]
	std_lower = std_lower[::-1]

	# fig.add_trace(go.Scatter(
	# 	x=np.concatenate([x, x_rev]),
	# 	y=np.concatenate([std_upper, std_lower]),
	# 	fill="toself",
	# 	fillcolor="rgba({r}, {g}, {b}, 0.2)".format(r=r, g=g, b=b),
	# 	line_color="rgba(255, 255, 255, 0)",
	# 	name=name,
	# 	showlegend=False
	# ))

	fig.add_trace(go.Scatter(
		x=x,
		y=mean,
		line=dict(dash=dash),
		name=name
	))


def main():
	folder_path = "../mlflow_data/optimizer_test/metrics"
	files = sorted(os.listdir(folder_path))
	files = [files[0], files[10], files[3], files[7]]

	fig = go.Figure()

	for file in files:
		name = file.replace("_metrics.json", "").upper()

		metrics = json.load(open(folder_path + "/" + file, mode="rt"))
		min_len = 1000

		for m in metrics:
			if min_len > len(m):
				min_len = len(m)

		for i in range(len(metrics)):
			metrics[i] = metrics[i][0:min_len]

		metrics_arr = np.array(metrics).transpose()
		mean = np.mean(metrics_arr, axis=1)[0:1000]
		std = np.std(metrics_arr, axis=1)[0:1000]
		x = np.arange(0, mean.shape[0])
		dash = "solid"

		if name in ["NFT", "COBYLA"]:
			dash = "dot"

		add_line_with_std(fig, x, mean, std, name, dash)

	fig.update_traces(mode="lines")
	fig.update_layout(xaxis_title="Funktionsauswertungen", yaxis_title="mittlerer quadratischer Fehler")
	fig.show()
	fig.write_image("error_plot.pdf")


main()
