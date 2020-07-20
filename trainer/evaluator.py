from .fid_calculator import FIDCalculator
from .input import get_test_input, process_test, denormalize
from .model import SmileGan
import argparse
import tensorflow as tf


class Evaluator:
    def __init__(self, job_dir, generator_model="unet", stats_data_gen=None):
        args = argparse.Namespace()
        args.job_dir = job_dir
        args.generator_model = generator_model
        args.cycle_lambda = 0
        args.num_epochs = 0
        args.g_lr = 0
        args.g_b1 = 0
        args.d_lr = 0
        args.d_b1 = 0
        args.sample_train = ""
        args.sample_test = ""

        self.model = SmileGan(args)

        if stats_data_gen:
            self.fid_calculator = FIDCalculator()
            self.fid_calculator.calc_stats(stats_data_gen)

    def get_smile(self, image_path):
        img_test = tf.expand_dims(process_test(image_path), 0)
        return denormalize(self.model.generator_g(img_test)[0])

    def calculate_fid_single(self, image_generated):
        img_test = tf.expand_dims(image_generated, 0)
        return self.fid_calculator.calculate(img_test)
