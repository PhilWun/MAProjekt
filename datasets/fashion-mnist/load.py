import gzip
import numpy as np


def load_dataset():
	"""
	Loads the fashion-mnist dataset. https://github.com/zalandoresearch/fashion-mnist

	:return: training images, training labels, test images, test labels
	"""
	train_images_path = "train-images-idx3-ubyte.gz"
	train_labels_path = "train-labels-idx1-ubyte.gz"
	test_images_path = "t10k-images-idx3-ubyte.gz"
	test_labels_path = "t10k-labels-idx1-ubyte.gz"

	train_images, train_labels = _load_images_labels(train_images_path, train_labels_path)
	test_images, test_labels = _load_images_labels(test_images_path, test_labels_path)

	return train_images, train_labels, test_images, test_labels


def _load_images_labels(images_path: str, labels_path: str):
	images_raw = gzip.open(images_path, 'rb')
	images = np.frombuffer(images_raw.read(), dtype=np.uint8, offset=16).reshape(-1, 784)
	labels_raw = gzip.open(labels_path, 'rb')
	labels = np.frombuffer(labels_raw.read(), dtype=np.uint8, offset=8)

	return images, labels
