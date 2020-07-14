import argparse
from datetime import datetime

from .fid_calculator import FIDCalculator
from .input import get_input
from .model import SmileGan


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
        type=str)

    args_parser.add_argument(
        '--test-files',
        help='GCS or local paths to testing data.',
        type=str)

    args_parser.add_argument(
        '--num-epochs',
        help="""
        Maximum number of training data epochs on which to train.
        """,
        default=50,
        type=int,
    )

    # Estimator arguments
    args_parser.add_argument(
        '--g-lr',
        help='Learning rate value for the optimizers.',
        default=0.0002,
        type=float)

    args_parser.add_argument(
        '--d-lr',
        help='Learning rate value for the optimizers.',
        default=0.0002,
        type=float)

    args_parser.add_argument(
        '--g-b1',
        help='Learning rate value for the optimizers.',
        default=0.05,
        type=float)

    args_parser.add_argument(
        '--d-b1',
        help='Learning rate value for the optimizers.',
        default=0.05,
        type=float)

    args_parser.add_argument(
        '--cycle-lambda',
        help='Cycle Loss Lambda.',
        default=10,
        type=int)

    args_parser.add_argument(
        '--generator-model',
        help="unet/resnet",
        default="unet",
        type=str
    )

    args_parser.add_argument(
        '--sample-train',
        help="sample train image to use in tensorboard",
        type=str
    )

    args_parser.add_argument(
        '--sample-test',
        help="sample test image to use in tensorboard",
        type=str
    )

    args_parser.add_argument(
        '--calculate-fid',
        help="Should calculate FID",
        default=False,
        type=str
    )
    # Saved model arguments
    args_parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models.',
        required=True)

    return args_parser.parse_args()


def main():
    args = get_args()
    print('Epoch count: {}.'.format(args.num_epochs))

    # Run the train and evaluate experiment
    time_start = datetime.utcnow()
    print('Experiment started...')

    train, test = get_input(args.train_files, args.test_files)

    fid_calculator = FIDCalculator()
    fid_calculator.calc_stats(test[1])

    gan = SmileGan(args, fid_calculator)
    gan.train(train[0], train[1], test[0], calculate_fid=args.calculate_fid)

    time_end = datetime.utcnow()
    print('Experiment finished.')
    time_elapsed = time_end - time_start
    print('Experiment elapsed time: {} seconds'.format(time_elapsed.total_seconds()))
