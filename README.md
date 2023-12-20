# Prototype-based learning for real estate data

## Summary

This is the code necessary to reproduce the experiments in the article "Predicting real estate proces with prototype-based models". 

This repository only contains the scripts necessary to reproduce the results; the data and model can be found in a [separate repository](https://github.com/prof-jose/prototype_based_learning). It is assume you have already cloned this. 

## Setup

* Clone this repository and navigate to the corresponding folder. Make sure you add symbolic links to the `data` and `protolearn` folders of the model repository. 

* It is recommended that you set up a virtual environment. A `requirements.txt` file is provided with the dependencies. 

* The file `requirements_frozen.txt` contains exactly all the modules and exact versions of the virtual environment used to prepare the results for the paper. 

## Running the prototype model experiments

The provided Makefile provides the targets to reproduce the experiments as run for preparing the paper. 

* `make validation`: Runs the experiments for the prototype-based model. On a CPU only server it will take several hours. 

* `make validation_kmeans` and `make validation_random:`: Runs the k-means baseline (should take a few minutes). 

To use a different number of prototypes, use e.g. `make validation N_PROTOTYPES=100`. Inspect the different parameters in the Makefile.

## Running the ML baselines

Call `python baselines.py`. 

Modify the configuration variables inside the script to try different models and parameter combinations. 


## Collecting the results

The notebook in `analysis/analysis_mlflow.ipynb` contains an example of collecting the MLflow data produced and compiling tables and plots.