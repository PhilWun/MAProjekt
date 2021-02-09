from typing import List, Iterator

import mlflow
import torch


def parse_arg_str(args_str: str, arg_types: List[type], arg_names: List[str]) -> List:
	split_args = args_str.split(",")
	parsed = []

	for value_str, t, name in zip(split_args, arg_types, arg_names):
		if t == bool:
			value = value_str.lower() == "true"
		else:
			value = t(value_str)

		parsed.append(value)
		mlflow.set_tag(name, value)

	return parsed


def parse_optimizer_and_args(
		optimizer_name: str, optimizer_args_str: str, params_q: Iterator[torch.nn.parameter.Parameter],
		params_c: Iterator[torch.nn.parameter.Parameter], is_hybrid: bool) -> List:
	class_and_args = {
		"Adadelta": (
			torch.optim.Adadelta,
			[float, float, float, float],
			["lr", "rho", "eps", "weight_decay"]
		),
		"Adagrad": (
			torch.optim.Adagrad,
			[float, float, float, float, float],
			["lr", "lr_decay", "weight_decay", "initial_accumulator_value", "eps"]
		),
		"Adam": (
			torch.optim.Adam,
			[float, float, float, float, float, bool],
			["lr", "beta1", "beta2", "eps", "weight_decay", "amsgrad"]
		),
		"AdamW": (
			torch.optim.AdamW,
			[float, float, float, float, float, bool],
			["lr", "beta1", "beta2", "eps", "weight_decay", "amsgrad"]
		),
		"Adamax": (
			torch.optim.Adamax,
			[float, float, float, float, float],
			["lr", "beta1", "beta2", "eps", "weight_decay"]
		),
		"ASGD": (
			torch.optim.ASGD,
			[float, float, float, float, float],
			["lr", "lambd", "alpha", "t0", "weigth_decay"]
		),
		"LBFGS": (
			torch.optim.LBFGS,
			[float, int, int, float, float, int],
			["lr", "max_iter", "max_eval", "tolerance_grad", "tolerance_change", "history_size"]
		),
		"RMSprop": (
			torch.optim.RMSprop,
			[float, float, float, float, float, bool],
			["lr", "alpha", "eps", "weight_decay", "momentum", "centered"]
		),
		"Rprop": (
			torch.optim.Rprop,
			[float, float, float, float, float],
			["lr", "eta1", "eta2", "step_size1", "step_size2"]
		),
		"SGD": (
			torch.optim.SGD,
			[float, float, float, float, bool],
			["lr", "momentum", "dampening", "weight_decay", "nesterov"]
		)
	}

	lr_c = 0

	if is_hybrid:
		split = optimizer_args_str.split(",", 1)
		lr_c = float(split[0])
		optimizer_args_str = split[1]

	ca = class_and_args[optimizer_name]
	opt_args = parse_arg_str(optimizer_args_str, ca[1], ca[2])

	if optimizer_name == "Adam":
		opt_args = [opt_args[0]] + [(opt_args[1], opt_args[2])] + opt_args[3:]
	elif optimizer_name == "AdamW":
		opt_args = [opt_args[0]] + [(opt_args[1], opt_args[2])] + opt_args[3:]
	elif optimizer_name == "Adamax":
		opt_args = [opt_args[0]] + [(opt_args[1], opt_args[2])] + opt_args[3:]
	elif optimizer_name == "Rprop":
		opt_args = [opt_args[0], (opt_args[1], opt_args[2]), (opt_args[3], opt_args[4])]

	optimizer_q = ca[0](params_q, *opt_args)

	if is_hybrid:
		optimizer_c = ca[0](params_c, lr_c, *opt_args[1:])

		return [optimizer_q, optimizer_c]
	else:
		return [optimizer_q]
