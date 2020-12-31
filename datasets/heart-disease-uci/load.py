from enum import Enum

import numpy as np
import pandas
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split


class ColumnType(Enum):
	NUMERIC = 0
	CATEGORICAL = 1
	LABEL = 2


def load_dataset(test_set_fraction: float = 0.3, rnd_seed: float = 0):
	"""
	Loads the Heart Disease UCI dataset. https://www.kaggle.com/ronitf/heart-disease-uci

	:return:
	"""
	column_types = [
		ColumnType.NUMERIC,
		ColumnType.CATEGORICAL,
		ColumnType.CATEGORICAL,
		ColumnType.NUMERIC,
		ColumnType.NUMERIC,
		ColumnType.CATEGORICAL,
		ColumnType.CATEGORICAL,
		ColumnType.NUMERIC,
		ColumnType.CATEGORICAL,
		ColumnType.NUMERIC,
		ColumnType.CATEGORICAL,
		ColumnType.CATEGORICAL,
		ColumnType.CATEGORICAL,
		ColumnType.LABEL
	]

	dataframe = pandas.read_csv("heart.csv")
	np_data = dataframe.to_numpy()
	preprocessed_columns = []
	labels = None

	for i, ct in enumerate(column_types):
		column_data = np_data[:, i].reshape((-1, 1))

		if ct == ColumnType.NUMERIC:
			scaler = preprocessing.MinMaxScaler()
			preprocessed_columns.append(scaler.fit_transform(column_data))
		elif ct == ColumnType.CATEGORICAL:
			enc = preprocessing.OneHotEncoder(sparse=False)
			preprocessed_columns.append(enc.fit_transform(column_data))
		elif ct == ColumnType.LABEL:
			enc = preprocessing.OrdinalEncoder()
			labels = enc.fit_transform(column_data).reshape((-1, ))
		else:
			raise ValueError()

	preprocessed_array = np.concatenate(preprocessed_columns, axis=1)
	input_train, input_test, target_train, target_test = train_test_split(
		preprocessed_array, labels, test_size=test_set_fraction, random_state=rnd_seed)

	return input_train, target_train, input_test, target_test


if __name__ == "__main__":
	load_dataset()
