import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


def process_img(file_path):
    img = tf.io.read_file(file_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def random_crop(image):
    cropped_image = tf.image.random_crop(
        image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image


# normalizing the images to [-1, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def random_jitter(image):
    # resizing to 286 x 286 x 3
    image = tf.image.resize(image, [286, 286],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # randomly cropping to 256 x 256 x 3
    image = random_crop(image)

    # random mirroring
    image = tf.image.random_flip_left_right(image)

    return image


def preprocess_image_train(image):
    image = random_jitter(image)
    image = normalize(image)
    return image

def get_input(data_dir, train_size):
    neutral_ds = tf.data.Dataset.list_files(data_dir + '*a.jpg').map(
        process_img, num_parallel_calls=AUTOTUNE)
    smile_ds = tf.data.Dataset.list_files(data_dir + '*b.jpg').map(
        process_img, num_parallel_calls=AUTOTUNE)

    return (neutral_ds.take(train_size), smile_ds.take(train_size)), (neutral_ds.skip(train_size), smile_ds.skip(train_size))

def preprocess_input(gen):
    return gen.map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE
    ).batch(1)

def denormalize(image):
    return (image + 1) * 127.5

def process_test(data):
    image = process_img(data)
    image = tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT])
    image = normalize(image)
    return image
