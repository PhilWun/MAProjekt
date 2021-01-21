import os
import yaml


def fix_artifact_paths(mlruns_path: str):
	abspath = os.path.abspath(mlruns_path)
	new_mlruns_path = "file://" + abspath
	experiment_folders = os.listdir(abspath)

	for exp_folder in experiment_folders:
		if not exp_folder.startswith("."):
			meta_file = open(os.path.join(abspath, exp_folder, "meta.yaml"), mode="r")
			content = yaml.load(meta_file, Loader=yaml.FullLoader)
			meta_file.close()
			old_mlruns_path = os.path.split(content["artifact_location"])[0]

			content["artifact_location"] = content["artifact_location"].replace(old_mlruns_path, new_mlruns_path)
			meta_file = open(os.path.join(abspath, exp_folder, "meta.yaml"), mode="w")
			yaml.dump(content, meta_file)
			meta_file.close()

			run_folders = os.listdir(os.path.join(abspath, exp_folder))

			for run_folder in run_folders:
				if run_folder != "meta.yaml":
					meta_file = open(os.path.join(abspath, exp_folder, run_folder, "meta.yaml"), mode="r")
					content = yaml.load(meta_file, Loader=yaml.FullLoader)
					meta_file.close()

					content["artifact_uri"] = content["artifact_uri"].replace(old_mlruns_path, new_mlruns_path)
					meta_file = open(os.path.join(abspath, exp_folder, run_folder, "meta.yaml"), mode="w")
					yaml.dump(content, meta_file)
					meta_file.close()


if __name__ == "__main__":
	fix_artifact_paths("/home/philipp/Documents/tmp/mlruns")
