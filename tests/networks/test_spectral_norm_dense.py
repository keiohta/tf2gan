import unittest

import numpy as np

from tf2gan.networks.spectral_norm_dense import SNDense


class TestSNDense(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.unit_size = 10
        cls.batch_size = 10

    def test_init(self):
        layer = SNDense(units=self.unit_size)
        layer(np.zeros(shape=(self.batch_size, self.unit_size), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
