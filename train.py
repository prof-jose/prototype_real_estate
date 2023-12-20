"""
Execute a run of the RLVQ model training.

Usage:
    python train.py
        --config <config>
        --n_prototypes <n_prototypes>
        --verbose
        --epochs <epochs>
        --learning-rate <learning-rate>
        --scale <scale>
        --regularization <regularization>
        --batch-size <batch-size>
        --logdir <logdir>
        --attempt-gpu
        --seed <seed>


Arguments:
    config: Path to the configuration file.
    n_prototypes: Number of prototypes to use.
    verbose: Whether to print the training progress.
    epochs: Number of epochs to train for.
    learning-rate: Learning rate to use.
    scale: Scale to use.
    regularization: Regularization constant to use.
    batch-size: Batch size to use.
    logdir: Directory to save the logs.
    attempt-gpu: Whether to attempt to use GPU.
    seed: Random seed to use.

Example:
    python train.py --config data/king_county.json --n_prototypes 10
"""

import argparse
import datetime
import matplotlib.pyplot as plt
import mlflow
import os
import tensorflow as tf

from protolearn.loaders import Loader
from protolearn.model_wrapper import PrototypeModel
from protolearn.general import get_logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    # Argument: location of the configuration file
    parser.add_argument('--config', type=str)
    # Argument: number of prototypes to use
    parser.add_argument('--n_prototypes', type=int)
    # Argument: verbose
    parser.add_argument('--verbose', action='store_true', default=False)
    # Argument: number of epochs
    parser.add_argument('--epochs', type=int, default=100)
    # Argument: learning rate
    parser.add_argument('--learning-rate', type=float, default=0.0005)
    # Argument: scale
    parser.add_argument('--scale', type=float, default=1e-2)
    # Argument: regularization constant
    parser.add_argument('--regularization', type=float, default=0.)
    # Argument: batch size
    parser.add_argument('--batch-size', type=int, default=256)
    # Argument: log directory
    parser.add_argument('--logdir', type=str, default=None)
    # Argument: attempt to use GPU
    parser.add_argument('--attempt-gpu', action='store_true', default=False)
    # Argument: random seed
    parser.add_argument('--seed', type=int, default=0)
    # Argument: init method
    parser.add_argument('--init-method', type=str, default="kmeans")

    args = parser.parse_args()

    # Just one check
    if args.init_method != "kmeans" and args.epochs > 0:
        raise ValueError(
            "Initialization method must be kmeans if epochs is > 0."
            )

    return args


def plot_results(args, model, X_train):
    """Plot the results of the training."""
    if args.logdir is not None:

        # Create a directory for the logs that includes
        # the current timestamp
        logdir = os.path.join(
            args.logdir,
            datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        )
        os.makedirs(logdir)

        plt.scatter(
            X_train.iloc[:, 0],
            X_train.iloc[:, 1],
            c='gray',
            alpha=0.1
        )
        initial_means = model.get_initial_prototypes()
        learned_means = model._model.layers[1].weights[0].numpy()
        plt.scatter(initial_means[:, 0], initial_means[:, 1], c='r')
        plt.scatter(learned_means[:, 0], learned_means[:, 1], c='b')
        plt.savefig(os.path.join(logdir, 'prototypes.png'))
        plt.close()


def main():
    """Main function."""

    logger = get_logger()
    args = parse_arguments()

    if not (args.attempt_gpu):
        tf.config.set_visible_devices([], 'GPU')

    # Set random seeds for reproducibility
    tf.keras.utils.set_random_seed(args.seed)

    # Log arguments, one in each line
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')

    loader = Loader(args.config, seed=args.seed)
    X_tr, X_test, y_tr, y_test = loader.get_splits()
    VALIDATION_SPLIT = 0.2
    n_val = int(VALIDATION_SPLIT * len(X_tr))
    X_train = X_tr.iloc[:-n_val, :]
    X_val = X_tr.iloc[-n_val:, :]
    y_train = y_tr.iloc[:-n_val]
    y_val = y_tr.iloc[-n_val:]

    logger.info(f'X_train size: {X_train.shape}')
    logger.info(f'X_test size: {X_test.shape}')
    logger.info(f'X_val size: {X_val.shape}')
    logger.info(f'y_val size: {y_val.shape}')
    logger.info(f'y_train size: {y_train.shape}')
    logger.info(f'y_test size: {y_test.shape}')

    logger.info(f'X_train sample:\n{X_train.head()}')
    logger.info(f'y_train sample:\n{y_train.head()}')

    epochs_per_iter = 0 if args.epochs == 0 else 1
    if args.epochs > 0:
        method = "repr"
    else:
        method = args.init_method

    model = PrototypeModel(
        n_prototypes=args.n_prototypes,
        scale=args.scale,
        reg_constant=args.regularization,
        learning_rate=args.learning_rate,
        epochs=epochs_per_iter,
        batch_size=args.batch_size,
        verbose=args.verbose,
        validation_data=(X_val.values, y_val.values),
        restart=False,
        init_method=args.init_method
    )

    # Train the model
    # Since each epoch is 1 experiment, we loop over epochs 1 by 1
    # But make sure to run for 0 epochs if args.epochs is 0

    for actual_epoch in range(args.epochs or 1):

        # Each epoch is logged as a run
        mlflow.start_run()
        mlflow.log_param('epoch', actual_epoch)
        mlflow.log_param('learning_rate', args.learning_rate)
        mlflow.log_param('scale', args.scale)
        mlflow.log_param('regularization', args.regularization)
        mlflow.log_param('batch_size', args.batch_size)
        mlflow.log_param('total_epochs', args.epochs)
        mlflow.log_param('n_prototypes', args.n_prototypes)
        mlflow.log_param('seed', args.seed)
        mlflow.log_param('method', method)

        model.fit(X_train.values, y_train.values)
        training_log = model._training_log
        if args.epochs == 0:
            val_evaluation = model._model.evaluate(
                X_val,
                y_val,
                verbose=args.verbose
                )
            loss_val, mse_val, mean_radii_val = val_evaluation
        evaluation = model._model.evaluate(
            X_test,
            y_test,
            verbose=args.verbose
            )
        loss_test, mse_test, mean_radii_test = evaluation
        # Print: epoch, training loss, val mse, test mse, training mean of
        # radii, val mean of radii, test mean of radii
        if args.epochs > 0:
            mlflow.log_metric('train_loss', training_log.history["loss"][-1])
            mlflow.log_metric('val_mse', training_log.history["val_mse"][-1])
            mlflow.log_metric('test_mse', mse_test)
            mlflow.log_metric(
                'train_mean_of_radii',
                training_log.history["mean_of_radii"][-1]
                )
            mlflow.log_metric(
                'val_mean_of_radii',
                training_log.history["val_mean_of_radii"][-1]
                )
            mlflow.log_metric('test_mean_of_radii', mean_radii_test)
        else:
            mlflow.log_metric('val_mse', mse_val)
            mlflow.log_metric('test_mse', mse_test)
            mlflow.log_metric('val_mean_of_radii', mean_radii_val)
            mlflow.log_metric('test_mean_of_radii', mean_radii_test)

        run = mlflow.active_run()
        data, info = mlflow.get_run(run.info.run_id)
        logger.info(data[1].params)
        logger.info(data[1].metrics)
        mlflow.end_run()

    logger.info(f'model: {model._model.summary()}')
    plot_results(args, model, X_train)


if __name__ == '__main__':
    main()
