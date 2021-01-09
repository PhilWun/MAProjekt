from sklearn.model_selection import train_test_split

from datasets.tools import ColumnType, load_csv


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

	data, labels = load_csv("heart.csv", column_types)
	input_train, input_test, target_train, target_test = train_test_split(
		data, labels, test_size=test_set_fraction, random_state=rnd_seed)

	return input_train, target_train, input_test, target_test


if __name__ == "__main__":
	load_dataset()
