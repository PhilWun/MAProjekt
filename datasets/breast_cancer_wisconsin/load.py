from sklearn.model_selection import train_test_split

from datasets.tools import ColumnType, load_csv


def load_dataset(test_set_fraction: float = 0.3, rnd_seed: int = 0):
	"""
	Loads the Breast Cancer Wisconsin (Diagnostic) Data Set. https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

	:return:
	"""
	column_types = [
		ColumnType.DISCARD,
		ColumnType.LABEL
	] + 30 * [ColumnType.NUMERIC]

	data, labels = load_csv("/home/philipp/MEGA/Studium/PlanQK/MA/Projekt/datasets/breast_cancer_wisconsin/data.csv", column_types)
	input_train, input_test, target_train, target_test = train_test_split(
		data, labels, test_size=test_set_fraction, random_state=rnd_seed, shuffle=True)

	return input_train, target_train, input_test, target_test


if __name__ == "__main__":
	load_dataset()
