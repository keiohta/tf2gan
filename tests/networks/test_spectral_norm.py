import unittest

import numpy as np
from tensorflow.keras.layers import Conv2D, Dense

from tf2gan.networks.spectral_norm import SpectralNormalization


class TestSNDense(unittest.TestCase):
    def test_conv(self):
        layer = SpectralNormalization(
            Conv2D(filters=4, kernel_size=3))
        layer(np.zeros(shape=(1, 10, 10, 1), dtype=np.float32))

    def test_fc(self):
        layer = SpectralNormalization(
            Dense(units=10))
        layer(np.zeros(shape=(1, 10), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
