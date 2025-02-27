import sys
from itertools import chain
from math import pi
from typing import Iterator

import torch

sys.path.append(".")  # add root of the project to the PYTHONPATH so that the other modules can be imported

from src.pl.pytorch.models.common_functions import create_qlayer, qnn_constructors


class HybridAutoencoder(torch.nn.Module):
	def __init__(self, input_size: int, q_num: int, embedding_size: int, qnn_name: str):
		super(HybridAutoencoder, self).__init__()

		self.q_num = q_num
		self.embedding_size = embedding_size

		self.fc1 = torch.nn.Linear(input_size, q_num)
		self.q_layer1 = create_qlayer(qnn_constructors[qnn_name], q_num)
		self.q_layer2 = create_qlayer(qnn_constructors[qnn_name], q_num)
		self.fc2 = torch.nn.Linear(q_num, input_size)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Calculates the output of the model.

		:param x: Input. The range of the values can be anything, but should be appropriately scaled.
		:return: Output of the model. The range of the values can be anything.
		"""
		embedding = self.embed(x)
		reconstruction = self.reconstruct(embedding)

		return reconstruction

	def embed(self, x: torch.Tensor) -> torch.Tensor:
		# encoder
		x = torch.sigmoid(self.fc1(x))
		x = x * pi  # scale with pi because input to the quantum layer should be in the range [0, pi]
		embedding = self.q_layer1(x)  # output in the range [-1, 1]

		# scaling the values to be in the range [0, pi]
		embedding = (embedding / 2.0 + 0.5) * pi
		# bottleneck
		embedding = embedding[:, 0:self.embedding_size]

		return embedding

	def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
		# padding with zeros to match the input size of the quantum layer
		x = torch.cat((x, torch.zeros((x.shape[0], self.q_num - self.embedding_size))), dim=1)
		# decoder
		x = self.q_layer2(x)
		reconstruction = self.fc2(x)

		return reconstruction

	def get_classical_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
		return chain(self.fc1.parameters(), self.fc2.parameters())

	def get_quantum_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
		return chain(self.q_layer1.parameters(), self.q_layer2.parameters())
