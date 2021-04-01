# Project for the master thesis "Quantum-assisted clustering with hybrid neural networks"

This project contains the code that was used to run the experiments for the master thesis "Quantum-assisted clustering with hybrid neural networks".

## Prerequisites

### Software
* Python 3.8.5
* MLflow 1.13.1
* conda 4.9.2

Everything else gets installed automatically by MLflow into a conda environment.

### Datasets
The datasets folder contains functions to load five different datasets.

* Breast Cancer Wisconsin (Diagnostic) Data Set (https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
* Credit Card Fraud Detection (https://www.kaggle.com/mlg-ulb/creditcardfraud)
* Fashion-MNIST (https://github.com/zalandoresearch/fashion-mnist)
* Heart Disease UCI (https://www.kaggle.com/ronitf/heart-disease-uci)
* Iris Species (https://www.kaggle.com/uciml/iris)

You have to download the datasets, that you want to use and copy them into the respective folders.

## Project structure

```
MAProject
|--datasets
|--src
|  |--clustering
|  |--pl
|  |--plot
|  |--qk
|
|--MLproject
|--conda.yaml
```

* `datasets` folder: functions to load five different datasets.
* `clustering` folder: code that executes the cluster algorithms.
* `pl` folder: code for the Pennylane + PyTorch implementation of the different autoencoders.
* `plot` folder: code that was used to create the plots for the master thesis.
* `qk` folder: Qiskit implementation of the quantum neural networks.
* `MLproject` file: contains the available entry points for MLflow and the arguments that can be specified.
* `conda.yaml` file: contains the dependencies that MLflow will automatically install into a new environment.

## Running experiments

You can use the `mlflow run -e <entry_point> -P <arg1>=<value1> -P <arg2>=<value2> ... <path>` command to run the experiments.

`<entry_point>` specifies which part of the program will be executed,
`-P <arg>=<value>` sets the argument `<arg>` to `<value>`,
`<path>` specifies the path to the root of the project.

All available entry points and arguments can be found in the `MLproject` file.
The possible values for the arguments are specified in the files of the respective entry points.


### Example

`mlflow run -e pennylane_pytorch -P dataset=heart_disease_uci -P scaler=standard -P model=hybrid -P model_args=3,2,QNN1 -P optimizer_args=0.001,0.1,0.9,0.999,1e-08,0,false .`

Explanation:
* `-e pennylane_pytorch` use the Pennylane + PyTorch implementation
* `-P dataset=heart_disease_uci` use the Heart Disease UCI dataset
* `-P scaler=standard` scale the data with the standard normalizer a.k.a. zero-mean normalizer
* `-P model=hybrid` use the hybrid autoencoder as model
* `-P model_args=3,2,QNN1` set the input size of the model to 3, the embedding size to 2 and use the first variant of the QNN
* `-P optimizer_args=0.001,0.1,0.9,0.999,1e-08,0,false` set the learning rate to 0.001 for the classical part and 0.1 for the quantum part of the hybrid autoencoder, set the parameter for ADAM to beta=(0.9, 0.999), eps=1e-08, amsgrad=false

