import plotly.graph_objects as go
import numpy as np

optimizer_names = np.array([
	"ADAM",
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

qnn1_error_mean = np.array([
	0.0004343859354654954,
	0.04714000066121421,
	0.00017936706542968767,
	0.02161944071451824,
	0.17851107279459638,
	0.15722877502441412,
	0.018146546681722,
	5.762736002604175e-05,
	0.00012531280517578153,
	0.06911483128865561,
	0.16062077840169273
])

qnn1_error_std = np.array([
	0.00044023932358428104,
	0.0553221613443943,
	0.0002691437093958139,
	0.037034221208012356,
	0.10016101686384433,
	0.04323771913647513,
	0.012194458595336693,
	5.820048140285114e-05,
	7.739919644372208e-05,
	0.05926069911273663,
	0.05672759380419152
])

qnn1_time = np.array([
	44.58244276046753,
	7.106216549873352,
	1.08284592628479,
	9.064624309539795,
	11.406931400299072,
	1.162330150604248,
	2.003806948661804,
	21.551619172096252,
	50.274120450019836,
	2.586902379989624,
	17.28065037727356
])

qnn2_error_mean = np.array([
	0.00020294825236002616,
	0.03941155751546224,
	0.0002006340026855469,
	0.03989215215047202,
	0.35980304082234704,
	0.3268645795186361,
	0.019694614410400386,
	0.0003503926595052083,
	0.00018444697062174485,
	0.10249197006225588,
	0.38398916244506837
])

qnn2_error_std = np.array([
	0.00014254836629974427,
	0.08233059991411082,
	0.00018289764728015686,
	0.11610695133747644,
	0.08843680700263734,
	0.0713308312677562,
	0.014629714461906024,
	0.0003876761116802383,
	0.000190701632783842,
	0.09137762043510603,
	0.13730675627047542
])

qnn2_time = np.array([
	55.95191264152527,
	13.681773900985718,
	2.4715410470962524,
	12.511336088180542,
	29.306902766227722,
	2.3067010641098022,
	3.123115301132202,
	28.77156960964203,
	75.12416195869446,
	7.731313467025757,
	26.27416467666626
])

qnn3_error_mean = np.array([
	0.00034091949462890713,
	0.030092983245849615,
	0.0005755996704101567,
	0.015884119669596353,
	0.30696142832438156,
	0.3704960378011068,
	0.033363583882649726,
	8.241653442382776e-05,
	0.004076932271321617,
	0.12331768035888672,
	0.19021179835001628
])

qnn3_error_std = np.array([
	0.00030341183932524094,
	0.0579801024686729,
	0.0005693157343306548,
	0.03359526018897168,
	0.12844475451220433,
	0.0787397558924727,
	0.01832604307799434,
	8.73090468082859e-05,
	0.008404034747920074,
	0.16971084435839862,
	0.18631584264568918
])

qnn3_time = np.array([
	3.2893320322036743,
	0.763243556022644,
	0.3120748996734619,
	0.45738160610198975,
	0.786461591720581,
	0.4978039264678955,
	0.9782583713531494,
	4.983641266822815,
	2.8636887073516846,
	1.1721922159194946,
	0.9781012535095215
])

tmp = [
	(qnn1_error_mean, qnn1_error_std, qnn1_time, "QNN1"),
	(qnn2_error_mean, qnn2_error_std, qnn2_time, "QNN2"),
	(qnn3_error_mean, qnn3_error_std, qnn3_time, "QNN3")
]

gradient_based = [0, 1, 3, 4, 8, 9, 10]
gradient_free = [2, 5, 6, 7]

for error_mean, error_std, time, name in tmp:
	# plot error
	fig = go.Figure()
	fig.add_trace(go.Scatter(
		x=optimizer_names[gradient_based],
		y=error_mean[gradient_based],
		mode="markers",
		name="gradientenbasiert",
		error_y={
			"type": "data",
			"array": error_std,
			"visible": True
		}
	))

	fig.add_trace(go.Scatter(
		x=optimizer_names[gradient_free],
		y=error_mean[gradient_free],
		mode="markers",
		name="gradientenfrei",
		error_y={
			"type": "data",
			"array": error_std,
			"visible": True
		}
	))

	fig.update_yaxes(range=[0, 0.6])
	fig.update_layout(template="plotly_white", title="Fehler nach 100 Iterationen (" + name + ")", xaxis_title="Optimierer", yaxis_title="Fehler (MSE)")
	fig.show()

	# plot time
	fig = go.Figure()
	fig.add_trace(go.Bar(
		x=optimizer_names[gradient_based],
		y=time[gradient_based],
		name="gradientenbasiert"
	))

	fig.add_trace(go.Bar(
		x=optimizer_names[gradient_free],
		y=time[gradient_free],
		name="gradientenfrei"
	))

	fig.update_yaxes(range=[0, 80])
	fig.update_layout(template="plotly_white", title="Zeit für 100 Iterationen (" + name + ")", xaxis_title="Optimierer", yaxis_title="Zeit (s)")
	fig.show()
