import unittest

import numpy as np

from tf2gan.networks.simple_net import Generator, Discriminator


class TestSNDense(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.img_size = (32, 32, 1)
        cls.dummy_img = np.zeros(shape=(1,) + cls.img_size, dtype=np.float32)
        cls.dummy_noize = np.zeros(shape=(1, 100), dtype=np.float32)

    def test_init(self):
        gen_net = Generator(self.img_size)
        gen_net(self.dummy_noize)

        disc_net = Discriminator(self.img_size)
        disc_net(self.dummy_img)


if __name__ == "__main__":
    unittest.main()
