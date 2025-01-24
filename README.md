# Network-based machine learning models applied to binary classification tasks using high-dimensional metabolomics data

This repository contains scripts related to the manuscript "Comparison of network-based machine learning approaches to binary classification via metabolomic data" submitted to the journal "Metabolites". Specifically, this repository contains scripts which implement five network-based machine learning models (Bayesian neural networks, feedforward neural networks, convolutional neural networks, Kolmogorov-Arnold networks, and spiking neural networks) to binary classification tasks based on high-dimensional metabolomics data. In our study, 17 publicly available datasets were considered: 6 from "MetaboLights" and 11 from "Metabolomics Workbench". More detail on the datasets chosen can be found in the manuscript. To avoid plagiarizing, we intentionally do not share the datasets in this repository and instead provide a toy dataset for reproducibility.

## Table of Contents
- [1. Abstract of manuscript](#abstract)
- [2. Dependencies](#dependencies)
- [3. Data preprocessing](#preprocessing)
  - [3.1 Description of input data](#input-data-description)
  - [3.2 Preprocessing transformations](#preprocessing-transformations)
- [4. Network-based machine learning approaches](#network-based-ml-approaches)
  - [4.1 Optimal hyperparameter selection](#optimal-hyperparameter-selection)
  - [4.2 Evaluation of network-based machine learning approaches](#evaluation)
- [5. Reproducible example](#example)

<a name="abstract"></a>
## 1. Abstract of manuscript
**Paste abstract when ready**

<a name="dependencies"></a>
## 2. Dependencies
The dependencies for our workflow and their versions can be found in the environment.yml file. To avoid potential dependency conflicts, we recommend creating a conda environment. If you don't already have conda installed, see [https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). Once conda is installed and this repository cloned, a conda environment can be created by navigating to this repository's directory (which contains the environment.yml file) and using the command:
```
conda env create -f environment.yml
```
This conda environment can then be activated with:
```
conda activate NNs_for_binary_classification_env
```
To deactivate a conda environment, run:
```
conda deactivate
```

<a name="preprocessing"></a>
## 3. Data preprocessing

<a name="input-data-description"></a>
### 3.1 Description of input data
Our workflow can be applied to any tabular data with rows corresponding to observations and columns corresponding to features. In our study specifically, rows correspond to samples analyzed using mass spectrometry techniques, columns correspond to metabolite, and cells correspond to the abundance of the given metabolite in the given sample. A toy dataset can be found ___.

<a name="preprocessing-transformations"></a>
### 3.2 Preprocessing transformations
A preprocessing approach similar to that used in "A comparative evaluation of the generalised predictive ability of eight machine learning algorithms across ten clinical metabolomics data sets for binary classification" by Mendez et al (https://pmc.ncbi.nlm.nih.gov/articles/PMC6856029/) is used. Specifically, each dataset is split into training data (2/3) and test data (1/3), and the base e logarithm transformation is applied to non-missing metabolite abundances. The metabolite abundance of each sample in the training data is then scaled to have mean 0 and variance 1, and the metabolite abundance of each sample in the test data is scaled to have an approximate standard normal distribution by subtracting the mean abundance and dividing by the abundance standard deviation of the sample in the training data. Missing metabolite abundance values are imputed using K-nearest neighbors with K=3.

<a name="network-based-ml-approaches"></a>
## 4. Network-based machine learning approaches
Five network-based machine learning models are implemented: 
1. Bayesian neural network
2. Convolutional neural network
3. Feedforward neural network
4. Kolmogorov-Arnold network
5. Spiking neural network.

<a name="optimal-hyperparameter-selection"></a>
## 4.1 Optimal hyperparameter selection
For each combination of dataset and model, 5-fold stratified cross validation with 10 different partitions is performed. Optimal hyperparameters are selected as those with the largest mean AUC across the 5x10=50 test folds within the training data. The Adam optimizer was used for each of the five network-based machine learning models.

<a name="evaluation"></a>
## 4.2 Evaluation of network-based machine learning approaches
For each combination of dataset and mode, the model is evaluated on the test data with respect to AUC, F1-score, and accuracy using its optimal hyperparameters. Obviously, a user can modify this approach to consider other evaluation metrics of interest such as precision, positive predictive value, etc.

<a name="example"></a>
## 5. Reproducible example
Once you've created a conda conda environment with the necessary dependencies, a toy example of our workflow can be reproduced with the scripts in the "toy_example/scripts" directory. The R script used to simulate the toy dataset (create_toy_data.R) is included, although this R script does not need to be run because the toy data is also provided (toy_example/data/raw_toy_data.csv). To determine optimal hyperparameters, run the five python scripts in the toy_example/scripts/hyperparameter_grid_search directory. Note that one can modify the grid search of hyperparameters. Very small grid searches of hyperparameters are currently performed for the sake of making this toy example run fast. In real-world applications on larger datasets with larger hyperparameter grid searches, one may want to parallelize these computations across multiple nodes. Once the hyperparameter grid searches are complete, run the R script toy_example/scripts/hyperparameter_grid_search/get_optimal_hyperparameters.R to create CSV files of the optimal hyperparameters. Lastly, run the five python scripts in the toy_example/scripts/test_data_evaluation directory to evaluate each of the network-based machine learning models on the test data with optimal hyperparameters. The prediction results will be written to the directory toy_example/data/test_data_evaluation, where one can evaluate performance via accuracy, AUC, F1-score, etc. An implementation of the workflow is shown below. To reduce possibilities of dependency conflicts and creation time of the conda environment, the R dependencies are not installed by default in the conda environment. We recommend using RStudio to run the R script. Additionally, note that the path to the toy_example directory will have to modified in these scripts.
```
python toy_example/scripts/hyperparameter_grid_search/BNN.py
python toy_example/scripts/hyperparameter_grid_search/CNN.py
python toy_example/scripts/hyperparameter_grid_search/FNN.py
python toy_example/scripts/hyperparameter_grid_search/KAN.py
python toy_example/scripts/hyperparameter_grid_search/SNN.py

Rscript toy_example/scripts/hyperparameter_grid_search/get_optimal_hyperparameters.R

python toy_example/scripts/test_data_evaluation/BNN.py
python toy_example/scripts/test_data_evaluation/CNN.py
python toy_example/scripts/test_data_evaluation/FNN.py
python toy_example/scripts/test_data_evaluation/KAN.py
python toy_example/scripts/test_data_evaluation/SNN.py
```




