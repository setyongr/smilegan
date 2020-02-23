import numpy
from numpy import cov, iscomplexobj, trace, asarray
from numpy.random.mtrand import randint
from scipy.linalg import sqrtm
import tensorflow as tf
from skimage.transform import rescale, resize, downscale_local_mean
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


class Evaluator:
    def __init__(self):
        self.model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    def calculate_fid(self, model, images1, images2):
        # calculate activations
        act1 = model.predict(images1)
        act2 = model.predict(images2)
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)

        # calculate sum squared difference between means
        ssdiff = numpy.sum((mu1 - mu2) ** 2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def calc_stats(self, train_gen):
        self.actReal = self.model.predict_generator(preprocess_evaluator(train_gen))
        self.muReal, self.sigmaReal = self.actReal.mean(axis=0), cov(self.actReal, rowvar=False)

    def evaluate(self, images):
        images = tf.cast(images, tf.float32)
        images = scale_images(images)
        images = preprocess_input(images)

        # Preprocess
        actFake = self.model.predict(images)
        muFake, sigmaFake = actFake.mean(axis=0), cov(actFake, rowvar=False)

        # calculate sum squared difference between means
        ssdiff = numpy.sum((self.muReal - muFake) ** 2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(self.sigmaReal.dot(sigmaFake))
        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + trace(self.sigmaReal + sigmaFake - 2.0 * covmean)
        return fid

    # def evaluateX(self):
    #     model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
    #     images1 = randint(0, 255, 20 * 32 * 32 * 3)
    #     images1 = images1.reshape((20, 32, 32, 3))
    #     images2 = images1[0:10, :]
    #     print('Prepared', images1.shape, images2.shape)
    #     # convert integer to floating point values
    #     images1 = images1.astype('float32')
    #     images2 = images2.astype('float32')
    #     # resize images
    #     images1 = self.scale_images(images1, (299, 299, 3))
    #     images2 = self.scale_images(images2, (299, 299, 3))
    #     print('Scaled', images1.shape, images2.shape)
    #     # pre-process images
    #     images1 = preprocess_input(images1)
    #     images2 = preprocess_input(images2)
    #     # fid between images1 and images1
    #     fid = self.calculate_fid(model, images1, images1)
    #     print('FID (same): %.3f' % fid)
    #     # fid between images1 and images2
    #     fid = self.calculate_fid(model, images1, images2)
    #     print('FID (different): %.3f' % fid)
