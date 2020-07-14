import numpy
import tensorflow as tf
from numpy import cov, iscomplexobj, trace, asarray
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import preprocess_input

from trainer.input import AUTOTUNE


def scale_images(images):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = tf.image.resize(image, [299, 299],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # store
        images_list.append(new_image)
    return asarray(images_list)


@tf.function
def preprocess_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [299, 299],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image


def preprocess_evaluator(image_gen):
    return image_gen.map(preprocess_image, num_parallel_calls=AUTOTUNE).batch(5).map(preprocess_input,
                                                                                     num_parallel_calls=AUTOTUNE)


class FIDCalculator:
    def __init__(self):
        self.model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    def calc_stats(self, train_gen):
        self.act_real = self.model.predict(preprocess_evaluator(train_gen))
        self.mu_real, self.sigma_real = self.act_real.mean(axis=0), cov(self.act_real, rowvar=False)

    def calculate(self, images):
        images = tf.cast(images, tf.float32)
        images = scale_images(images)
        images = preprocess_input(images)

        # Preprocess
        act_fake = self.model.predict(images)
        mu_fake, sigma_fake = act_fake.mean(axis=0), cov(act_fake, rowvar=False)

        # calculate sum squared difference between means
        ssdiff = numpy.sum((self.mu_real - mu_fake) ** 2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(self.sigma_real.dot(sigma_fake))
        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + trace(self.sigma_real + sigma_fake - 2.0 * covmean)
        return fid
