import argparse
import logging
import os
import sys

from datetime import datetime
import tensorflow as tf

from input import get_input
from model import SmileGan


def get_args():
    """Define the task arguments with the default values.
    Returns:
        experiment parameters
    """

    args_parser = argparse.ArgumentParser()

    # Data files arguments
    args_parser.add_argument(
        '--train-files',
        help='GCS or local paths to training data.',
        nargs='+',
        required=True)

    args_parser.add_argument(
        '--batch-size',
        help='Batch size for each training and evaluation step.',
        type=int,
        default=1)

    args_parser.add_argument(
        '--num-epochs',
        help="""
        Maximum number of training data epochs on which to train.
        If both --train-size and --num-epochs are specified,
        --train-steps will default to:
            (train-size/train-batch-size) * num-epochs.
        """,
        default=50,
        type=int,
    )

    # Estimator arguments
    args_parser.add_argument(
        '--learning-rate',
        help='Learning rate value for the optimizers.',
        default=0.1,
        type=float)

    args_parser.add_argument(
        '--cycle-lambda',
        help='Cycle Loss Lambda.',
        default=10,
        type=int)

    # Saved model arguments
    args_parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models.',
        required=True)

    return args_parser.parse_args()


def _setup_logging():
    """Sets up logging."""
    root_logger = logging.getLogger()
    root_logger_previous_handlers = list(root_logger.handlers)
    for h in root_logger_previous_handlers:
        root_logger.removeHandler(h)
    root_logger.setLevel(logging.INFO)
    root_logger.propagate = False

    # Set tf logging to avoid duplicate logging. If the handlers are not removed
    # then we will have duplicate logging
    tf_logger = logging.getLogger('SmileGan')
    while tf_logger.handlers:
        tf_logger.removeHandler(tf_logger.handlers[0])

    # Redirect INFO logs to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    root_logger.addHandler(stdout_handler)

    # Suppress C++ level warnings.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    args = get_args()
    _setup_logging()

    logging.info('Epoch count: {}.'.format(args.num_epochs))
    logging.info('Batch size: {}.'.format(args.batch_size))

    # Create the Estimator

    # Run the train and evaluate experiment
    time_start = datetime.utcnow()
    logging.info('Experiment started...')
    logging.info('.......................................')

    train_neutral, train_smile = get_input(args.train_files)
    gan = SmileGan(args)
    gan.train(train_neutral, train_smile)

    time_end = datetime.utcnow()
    logging.info('.......................................')
    logging.info('Experiment finished.')
    time_elapsed = time_end - time_start
    logging.info('Experiment elapsed time: {} seconds'.format(
        time_elapsed.total_seconds()))


if __name__ == '__main__':
    main()