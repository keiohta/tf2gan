import logging

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy

from tf2gan.networks.simple_net import Generator, Discriminator


class GAN:
    def __init__(self, generator=None, discriminator=None, batch_size=64, z_dim=100, device="/gpu:0", logdir="results"):
        self._G = generator or Generator()
        self._D = discriminator or Discriminator()
        self._batch_size = batch_size
        self._z_dim = z_dim
        self._optimizer_G = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self._optimizer_D = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self._device = device

        self._train_acc = tf.keras.metrics.BinaryAccuracy()

        # Logger
        self._writer = tf.summary.create_file_writer(logdir)
        self._writer.set_as_default()

    def train(self, real_imgs, n_epochs=100, summary_freq=1, save_model_freq=10, test_freq=1):
        n_imgs = real_imgs.shape[0]
        train_dataset = tf.data.Dataset.from_tensor_slices(real_imgs)
        train_dataset = train_dataset.shuffle(n_imgs).repeat(n_epochs).batch(self._batch_size)

        n_iter_per_epoch = int(n_imgs / self._batch_size)
        n_trained_epochs = 0
        tf.summary.experimental.set_step(n_trained_epochs)

        for n_iter, _real_imgs in enumerate(train_dataset):
            noise = tf.random.uniform([self._batch_size, self._z_dim], -1., 1.)
            loss_dict = self._train_body(_real_imgs, noise)

            if (n_iter + 1) % n_iter_per_epoch == 0:
                n_trained_epochs += 1
                tf.summary.experimental.set_step((n_iter + 1) // n_iter_per_epoch)

                if n_trained_epochs % summary_freq == 0:
                    tf.summary.scalar(name="Generator/loss", data=loss_dict["loss_G"])
                    tf.summary.scalar(name="Discriminator/loss", data=loss_dict["loss_D"])
                    tf.summary.scalar(name="Discriminator/train_acc", data=self._train_acc.result())
                    self._reset_states()

                if n_trained_epochs % save_model_freq == 0:
                    pass

                if n_trained_epochs % test_freq == 0:
                    self._test(n_trained_epochs)

    @tf.function
    def _train_body(self, real_imgs, noise):
        with tf.device(self._device):
            with tf.GradientTape(persistent=True) as tape:
                fake_imgs = self._G(noise)
                real_logits = self._D(real_imgs)
                fake_logits = self._D(fake_imgs)

                loss_G = tf.reduce_mean(
                    binary_crossentropy(y_true=tf.ones_like(fake_logits), y_pred=fake_logits, from_logits=True))
                loss_D = tf.reduce_mean(
                    binary_crossentropy(y_true=tf.ones_like(real_logits), y_pred=real_logits, from_logits=True) +
                    binary_crossentropy(y_true=tf.zeros_like(fake_logits), y_pred=fake_logits, from_logits=True))

            vars_G = self._G.trainable_variables
            vars_D = self._D.trainable_variables
            self._optimizer_G.apply_gradients(zip(tape.gradient(loss_G, vars_G), vars_G))
            self._optimizer_D.apply_gradients(zip(tape.gradient(loss_D, vars_D), vars_D))

            self._train_acc.update_state(tf.ones_like(real_logits), tf.nn.sigmoid(real_logits))
            self._train_acc.update_state(tf.zeros_like(fake_logits), tf.nn.sigmoid(fake_logits))

        return {"loss_G": loss_G, "loss_D": loss_D}

    def _test(self, n_trained_epochs):
        noise = tf.random.uniform([16, self._z_dim], -1., 1.)
        generated_imgs = self._G(noise)
        plt.close()
        plt.figure(figsize=(4, 4))
        for i, img in enumerate(generated_imgs):
            plt.subplot(4, 4, i + 1)
            plt.imshow(img[:, :, 0], cmap='gray')
        plt.tick_params(axis='both', labelsize=0, length=0)
        plt.tight_layout()
        filename = "gen_imgs_epoch_{}".format(n_trained_epochs)
        logging.info("Saving {}...".format(filename))
        plt.savefig(filename)

    def _reset_states(self):
        self._train_acc.reset_states()
