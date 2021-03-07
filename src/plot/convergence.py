import numpy as np
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

optimizer_names = np.array([
	"ADAM",
	"AMSGRAD",
	"CG",
	"COBYLA",
	"L_BFGS_B",
	"GSLS",
	"NELDER_MEAD",
	"NFT",
	"POWELL",
	"SLSQP",
	"SPSA",
	"TNC"
])

failed_convergence = np.array([
	0,
	10,
	20,
	0,
	20,
	80,
	100,
	0,
	10,
	0,
	0,
	100
])

avg_func_evals = np.array([
	440.9,
	551.66666,
	311.875,
	37.9,
	255.25,
	184,
	0,
	61.3,
	513.55555,
	278.7,
	81.9,
	0
])

gradient_based = [0, 1, 2, 4, 5, 9, 10, 11]
gradient_free = [3, 6, 7, 8]

data_gradient_based = pd.DataFrame(data={
	"optimizer": optimizer_names[gradient_based],
	"failed_convergence": failed_convergence[gradient_based],
	"avg_func_evals": avg_func_evals[gradient_based]
})

data_gradient_free = pd.DataFrame(data={
	"optimizer": optimizer_names[gradient_free],
	"failed_convergence": failed_convergence[gradient_free],
	"avg_func_evals": avg_func_evals[gradient_free]
})

# fig = px.bar(data, x="optimizer", y="avg_func_evals", color="failed_convergence", color_continuous_scale="bluered", facet_col="optim_type")
fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_width=[0.65, 0.35], horizontal_spacing=0.03)

fig.add_trace(
	go.Bar(
		x=data_gradient_based["optimizer"],
		y=data_gradient_based["avg_func_evals"],
		marker=dict(color=data_gradient_based["failed_convergence"], coloraxis="coloraxis")),
	1, 1
)

fig.add_trace(
	go.Bar(
		x=data_gradient_free["optimizer"],
		y=data_gradient_free["avg_func_evals"],
		marker=dict(color=data_gradient_free["failed_convergence"], coloraxis="coloraxis")),
	1, 2
)

fig.update_yaxes(title_text="Funktionsauswertungen", row=1, col=1)
fig.update_xaxes(title_text="gradientenbasiert", row=1, col=1)
fig.update_xaxes(title_text="gradientenfrei", row=1, col=2)
fig.update_layout(coloraxis=dict(colorscale="bluered", colorbar_title_text="nicht konvergiert in Prozent", colorbar_title_side="right"), showlegend=False)
fig.show()
fig.write_image("convergence.pdf")
