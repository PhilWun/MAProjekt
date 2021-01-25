import torch


class ClassicalAutoencoder(torch.nn.Module):
	def __init__(self, input_size: int, intermediate_size: int, embedding_size: int):
		super(ClassicalAutoencoder, self).__init__()

		self.fc1 = torch.nn.Linear(input_size, intermediate_size)
		self.fc2 = torch.nn.Linear(intermediate_size, embedding_size)
		self.fc3 = torch.nn.Linear(embedding_size, intermediate_size)
		self.fc4 = torch.nn.Linear(intermediate_size, input_size)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.embed(x)
		x = self.reconstruct(x)

		return x

	def embed(self, x: torch.Tensor) -> torch.Tensor:
		x = torch.sigmoid(self.fc1(x))
		x = torch.sigmoid(self.fc2(x))

		return x

	def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
		x = torch.sigmoid(self.fc3(x))
		x = torch.sigmoid(self.fc4(x))

		return x
