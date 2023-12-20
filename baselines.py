"""
This script evaluates the performance of the baseline models.

Usage:
    python baselines.py

Instructions:
- Set the CASE constant to one of 'tree', 'forest', 'neural_net'.
- Review the config_* dictionaries to set the hyperparameters.
"""

# Import loader
import itertools
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf

from experiments.loaders import Loader

# Set the baseline model to evaluate
CASE = 'tree'  # 'tree', 'forest', 'neural_net'

# Specify the hyperparameters to evaluate for each model

# Decision tree parameters
config_tree = {
    'max_leaf_nodes': [10, 25, 50, 125, 175, 150, 200, 250]
}

# Random forest parameters
config_forest = {
    'max_leaf_nodes': [5, 10, 22, 25, 50],
    'n_estimators': [2, 5, 10, 20, 50],
}

# Neural network parameters
config_neural_net = {
    'n_units': [[90], [30, 30], [40, 20], [22, 22, 22]],
    'epochs': [40],
    'batch_size': [64, 128, 256],
    'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4]
}
#   'n_units': [[40], [10, 10], [8, 8, 8]],

# Specify the data to use
CONFIG = './data/paris_per_m2_all.json'
#CONFIG = './data/king_county_per_sqft_all.json'


# Below are the functions to train each model

# Each function has the same signature:
#   X_train, y_train, X_val, y_val, X_test, y_test, **params
# where params is a dictionary of hyperparameters
# The function returns the validation error, test error, and number of
# parameters


def train_tree(X_train, y_train, X_val, y_val, X_test, y_test, normalizer, **params):
    """
    Train a decision tree regressor.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data.
    y_train : pd.Series
        Training labels.
    X_val : pd.DataFrame
        Validation data.
    y_val : pd.Series
        Validation labels.
    X_test : pd.DataFrame
        Test data.
    y_test : pd.Series
        Test labels.
    params : dict
        Hyperparameters.
    """
    model = DecisionTreeRegressor(max_leaf_nodes=params['max_leaf_nodes'])
    model.fit(X_train, y_train)

    yhat_val = model.predict(X_val)
    yhat_test = model.predict(X_test)

    err_val = np.sqrt(np.square((yhat_val-y_val)*normalizer).mean())
    err_test = np.sqrt(np.square((yhat_test-y_test)*normalizer).mean())
    nparams = 2*model.tree_.node_count + 3*model.tree_.n_leaves

    return err_val, err_test, nparams


def train_forest(X_train, y_train, X_val, y_val, X_test, y_test, normalizer, **params):
    """
    Train a random forest regressor.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data.
    y_train : pd.Series
        Training labels.
    X_val : pd.DataFrame
        Validation data.
    y_val : pd.Series
        Validation labels.
    X_test : pd.DataFrame
        Test data.
    y_test : pd.Series
        Test labels.
    params : dict
        Hyperparameters.

    """
    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_leaf_nodes=params['max_leaf_nodes']
        )
    model.fit(X_train, y_train)

    yhat_val = model.predict(X_val)
    yhat_test = model.predict(X_test)
    err_val = np.sqrt(np.square((yhat_val-y_val)*normalizer).mean())
    err_test = np.sqrt(np.square((yhat_test-y_test)*normalizer).mean())

    # Compute the number of parameters
    params = 0
    for tree in model.estimators_:
        params += 2*tree.tree_.node_count + 3*tree.tree_.n_leaves
    return err_val, err_test, params


def train_neural_net(X_train, y_train, X_val, y_val, X_test, y_test, normalizer, **params):
    """
    Train a neural network regressor.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data.
    y_train : pd.Series
        Training labels.
    X_val : pd.DataFrame
        Validation data.
    y_val : pd.Series
        Validation labels.
    X_test : pd.DataFrame
        Test data.
    y_test : pd.Series
        Test labels.
    params : dict
        Hyperparameters.

    """
    input = tf.keras.Input(shape=(X_train.shape[1],))
    for i, n_units in enumerate(params['n_units']):
        if i == 0:
            x = tf.keras.layers.Dense(n_units, activation='relu')(input)
        else:
            x = tf.keras.layers.Dense(n_units, activation='relu')(x)
    output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=params['learning_rate']
        ),
        loss='mse'
        )
    model.fit(
        X_train, y_train, epochs=params['epochs'],
        batch_size=params['batch_size'], verbose=False
        )

    yhat_val = model.predict(X_val, verbose=0).reshape(-1)
    yhat_test = model.predict(X_test, verbose=0).reshape(-1)

    err_val = np.sqrt(np.square((yhat_val-y_val)*normalizer).mean())
    err_test = np.sqrt(np.square((yhat_test-y_test)*normalizer).mean())

    nparams = model.count_params()
    return err_val, err_test, nparams


def main():
    """
    Main experiment training and evaluation loop.
    """

    # Disable GPU
    tf.config.set_visible_devices([], 'GPU')

    # Load and prepare the data
    loader = Loader(CONFIG)
    X_train, X_test, y_train, y_test = loader.get_splits()

    VALIDATION_SPLIT = 0.2
    n_val = int(VALIDATION_SPLIT * len(X_train))
    X_train = X_train.iloc[:-n_val, :]
    X_val = X_train.iloc[-n_val:, :]
    y_train = y_train.iloc[:-n_val]
    y_val = y_train.iloc[-n_val:]

    # Select the model
    if CASE == 'tree':
        train_fn = train_tree
        config = config_tree
    elif CASE == 'forest':
        train_fn = train_forest
        config = config_forest
    elif CASE == 'neural_net':
        train_fn = train_neural_net
        config = config_neural_net
    else:
        raise ValueError('Unknown case: {}'.format(CASE))

    # Run the experiment
    all_results = []
    for run in range(10):
        for params in itertools.product(*config.values()):
            kw_args = dict(zip(config.keys(), params))
            normalizer = loader._config['target']['normalizer']
            err_val, err_test, nparams = train_fn(
                X_train, y_train, X_val, y_val, X_test, y_test, normalizer, **kw_args
                )
            print(
                'Run: {}, {}, Val: {}, Test: {}, Params: {}'
                .format(run, kw_args, err_val, err_test, nparams)
                )

            params_formatted = []
            for p in params:
                if isinstance(p, list):
                    params_formatted.append(','.join(map(str, p)))
                else:
                    params_formatted.append(p)

            all_results.append(
                [run] + params_formatted + [err_val, err_test, nparams]
                )

    # Convert results to dataframe
    print(all_results)
    all_results = np.array(all_results)
    param_names = list(config.keys())
    all_results = pd.DataFrame(
        all_results,
        columns=['run'] + param_names + ['val', 'test', 'nparams'],
        )
    # Convert numeric columns to numeric
    number_cols = ['run', 'val', 'test', 'nparams']
    all_results[number_cols] = all_results[number_cols].apply(pd.to_numeric)
    print(all_results)
    print(all_results.dtypes)
    avg_results = all_results\
        .groupby(param_names)\
        .agg({
            'nparams': 'mean',
            'val': ['mean', 'std'],
            'test': ['mean', 'std']}
            ).reset_index()
    print(avg_results.sort_values(by=('val', 'mean'), ascending=True))
    avg_results.sort_values(
        by=('val', 'mean'),
        ascending=True
        ).to_csv('results_paris_{}.csv'.format(CASE), index=False)


if __name__ == '__main__':
    main()