import time

import tensorflow as tf
from network import unet_generator, discriminator

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)

    generated_loss = loss_obj(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss  # total range 0 - 2

    return total_disc_loss * 0.5  # make range 0 - 1


def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)


def calc_cycle_loss(cycle_labmda, real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return cycle_labmda * loss1


def identity_loss(cycle_labmda, real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return cycle_labmda * 0.5 * loss


class SmileGan:
    def __init__(self, args):
        OUTPUT_CHANNELS = 3
        self.CHECKPOINT_PATH = args.job_dir
        self.cycle_lambda = args.cycle_lambda
        self.epochs = args.num_epochs

        self.g_lr = args.g_lr
        self.g_b1 = args.g_b1

        self.d_lr = args.d_lr
        self.d_b1 = args.d_b1

        print("Initializing Model")

        self.generator_g = unet_generator(OUTPUT_CHANNELS)
        self.generator_f = unet_generator(OUTPUT_CHANNELS)

        self.discriminator_x = discriminator()
        self.discriminator_y = discriminator()

        self.generator_g_optimizer = tf.keras.optimizers.Adam(self.g_lr, beta_1=self.g_b1)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(self.g_lr, beta_1=self.g_b1)

        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(self.d_lr, beta_1=self.d_b1)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(self.d_lr, beta_1=self.d_b1)

        ckpt = tf.train.Checkpoint(generator_g=self.generator_g,
                                   generator_f=self.generator_f,
                                   discriminator_x=self.discriminator_x,
                                   discriminator_y=self.discriminator_y,
                                   generator_g_optimizer=self.generator_g_optimizer,
                                   generator_f_optimizer=self.generator_f_optimizer,
                                   discriminator_x_optimizer=self.discriminator_x_optimizer,
                                   discriminator_y_optimizer=self.discriminator_y_optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(ckpt, self.CHECKPOINT_PATH, max_to_keep=10)

        print("Loading Checkpoint")
        # if a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

        print("Model Initialized")

    @tf.function
    def train_step(self, real_x, real_y):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = generator_loss(disc_fake_y)
            gen_f_loss = generator_loss(disc_fake_x)

            total_cycle_loss = calc_cycle_loss(self.cycle_lambda, real_x, cycled_x) + calc_cycle_loss(self.cycle_lambda,
                                                                                                      real_y,
                                                                                                      cycled_y)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss
            total_gen_f_loss = gen_f_loss + total_cycle_loss

            disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss,
                                              self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss,
                                              self.generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss,
                                                  self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss,
                                                  self.discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                                       self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                                       self.generator_f.trainable_variables))

        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                           self.discriminator_x.trainable_variables))

        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                           self.discriminator_y.trainable_variables))

    def train(self, train_neutral, train_smile):
        for epoch in range(self.epochs):
            start = time.time()

            n = 0
            for image_x, image_y in tf.data.Dataset.zip((train_neutral, train_smile)):
                self.train_step(image_x, image_y)
                if n % 10 == 0:
                    print(n, end=' ')
                n += 1

            if (epoch + 1) % 10 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                    ckpt_save_path))

            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                               time.time() - start))
