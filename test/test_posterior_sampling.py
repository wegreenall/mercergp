import torch
import torch.distributions as D
import matplotlib.pyplot as plt

import unittest
from mercergp.posterior_sampling import MercerSpectralDistribution


class TestMercerSpectralDistribution(unittest.TestCase):
    def setUp(self):
        self.frequency = 1000
        self.probs = torch.ones(10000)
        self.spectral_distribution = MercerSpectralDistribution(
            self.frequency, self.probs
        )
        self.feature_count = 100
        return

    def test_sample_shape(self):
        sample = self.spectral_distribution.sample((self.feature_count,))
        self.assertEqual(sample.shape, torch.Size([self.feature_count, 1]))


@unittest.skip("")
class test_kernel_fft_decomposed(unittest.TestCase):
    def setUp(self):
        # self.kernel = MercerKernel()
        return

    def test_sample_shape(self):
        return


@unittest.skip("")
class test_histogram_spectral_distribution(unittest.TestCase):
    def setUp(self):
        return

    def test_sample_shape(self):
        return


if __name__ == "__main__":
    """
    The program begins here
    """
    unittest.main()
