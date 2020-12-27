from random import random

import mlflow


def example_resume_run():
	run_id = "174cc448545b47039bd0cff676d9bc02"
	mlflow.start_run(run_id=run_id, run_name="TEST3")
	# mlflow.log_param("param1", randint(0, 100))
	mlflow.log_param("run_id", mlflow.active_run().info.run_id)
	mlflow.log_metric("foo", random())
	mlflow.log_metric("foo", random() + 1)
	mlflow.log_metric("foo", random() + 2)


if __name__ == "__main__":
	example_resume_run()
