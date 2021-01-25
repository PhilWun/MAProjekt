import sys
from math import pi

import torch

sys.path.append(".")  # add root of the project to the PYTHONPATH so that the other modules can be imported

from src.pl.pytorch.models.common_functions import create_qlayer, qnn_constructors


class QuantumModel(torch.nn.Module):
	def __init__(self, q_num: int, qnn_name: str, autoencoder: bool, embedding_size: int):
		super(QuantumModel, self).__init__()

		self.q_num = q_num
		self.autoencoder = autoencoder
		self.embedding_size = embedding_size

		self.q_layer1 = create_qlayer(qnn_constructors[qnn_name], q_num)

		if autoencoder:
			self.q_layer2 = create_qlayer(qnn_constructors[qnn_name], q_num)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Calculates the output of the model.

		:param x: The input. Values should be in the range [0, pi].
		:return: Output of the model. Values are in the range [-1, 1] if autoencoder is true, otherwise [0, pi].
		"""
		embedding = self.embed(x)

		if self.autoencoder:
			reconstruction = self.reconstruct(embedding)

			return reconstruction
		else:
			return embedding

	def embed(self, x: torch.Tensor) -> torch.Tensor:
		embedding = self.q_layer1(x)
		# scaling the values to be in the range [0, pi]
		embedding = (embedding / 2.0 + 0.5) * pi

		if self.autoencoder:
			embedding = embedding[:, 0:self.embedding_size]

		return embedding

	def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
		x = torch.cat((x, torch.zeros((x.shape[0], self.q_num - self.embedding_size))))
		reconstruction = self.q_layer2(x)

		return reconstruction
