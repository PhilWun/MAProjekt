from enum import Enum
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn import preprocessing


class ColumnType(Enum):
	NUMERIC = 0
	CATEGORICAL = 1
	LABEL = 2
	DISCARD = 3


def load_csv(file: str, column_types: List[ColumnType]) -> Tuple[np.ndarray, np.ndarray]:
	dataframe = pd.read_csv(file)
	np_data = dataframe.to_numpy()
	preprocessed_columns = []
	labels = None

	for i, ct in enumerate(column_types):
		column_data = np_data[:, i].reshape((-1, 1))

		if ct == ColumnType.NUMERIC:
			preprocessed_columns.append(column_data)
		elif ct == ColumnType.CATEGORICAL:
			enc = preprocessing.OneHotEncoder(sparse=False)
			preprocessed_columns.append(enc.fit_transform(column_data))
		elif ct == ColumnType.LABEL:
			enc = preprocessing.OrdinalEncoder()
			labels = enc.fit_transform(column_data).reshape((-1,))
		elif ct == ColumnType.DISCARD:
			pass
		else:
			raise ValueError()

	preprocessed_array = np.concatenate(preprocessed_columns, axis=1).astype(np.float32)

	return preprocessed_array, labels
