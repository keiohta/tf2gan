import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2gan.networks.spectral_norm import SpectralNormalization as SN


class Base(tf.keras.Model):
    def __init__(self, name):
        super().__init__(name=name)

    def call(self, inputs):
        features = inputs
        for layer in self._layers:
            features = layer(features)
        return features


class Generator(Base):
    def __init__(self, img_size, name="Generator"):
        super().__init__(name=name)

        self._layers = [
            tf.keras.layers.Dense(units=8 * 8 * 32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(8, 8, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=(1, 1), padding="SAME")]

        dummy_noise = tf.constant(np.zeros(shape=(1, 100), dtype=np.float32))
        with tf.device("/cpu:0"):
            test_out = self(dummy_noise)
        assert test_out.numpy().shape[1:] == img_size


class Discriminator(Base):
    def __init__(self, img_size, enable_sn=False, name="Discriminator"):
        super().__init__(name=name)

        self._layers = [
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            Dense(256, activation='relu'),
            Dense(1)]

        if enable_sn:
            for idx, layer in enumerate(self._layers):
                self._layers[idx] = SN(layer)

        dummy_img = tf.constant(np.zeros(shape=(1,) + img_size, dtype=np.float32))
        with tf.device("/cpu:0"):
            self(dummy_img)
