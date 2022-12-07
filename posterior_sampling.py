import torch
import torch.distributions as D

import math
from mercergp.kernels import MercerKernel

"""
This file contains classes and functions useful for the purpose of appropriate posterior
sampling for non-stationary (specifically, Mercer) kernels. The idea is based on work
in Wilson(2020) using Matheron's rule to generate Gaussian processes using a
prior component and a posterior component.
"""


class NonStationarySpectralDistribution:
    """
    Represents the spectral distribution, where sampling has been
    converted to the shape necessary for the Random Fourier Feature
    basis format.
    """

    def __init__(
        self,
        spectral_distribution: D.Distribution,
    ):
        self.spectral_distribution = spectral_distribution
        # self.dim_1_dist = dim_1_marginal
        # self.dim_2_dist = dim_2_marginal
        return

    def sample(self, sample_size: torch.Size) -> torch.Tensor:
        """
        Returns a sample from the 2d-spectral distribution corresponding to
        a kernel via Yaglom's theorem.

        The spectral distribution as passed in should generate a sample of size
        [N, 2], since it is ostensibly a 2-d spectral distribution.
        """
        if sample_size[0] % 2 != 0:
            raise ValueError(
                "Please pass an even number for the sample size for the NonStationaryDistribution. It is likely that you have selected an odd number for the order for the Random Fourier Features prior component on a non-stationary Fourier Features model"
            )
        first_half_sample_size = sample_size.clone()
        second_half_sample_size = sample_size.clone()
        first_half_sample_size[0] = math.floor(first_half_sample_size[0] / 2)
        second_half_sample_size[0] = math.ceiling(
            second_half_sample_size[0] / 2
        )
        # sample_part_1 = self.dim_1_dist.sample(sample_size)
        # sample_part_2 = self.dim_2_dist.sample(sample_size)
        sample = self.spectral_distribution.sample(sample_size)
        sample = torch.cat((sample[:, 0], sample[:, 1]))
        return sample


def histogram_spectral_distribution(kernel: MercerKernel) -> D.Distribution:
    """
    Given a kernel, returns a spectral distribution approximation
    via histogram sampling and a 2-d FFT.
    """
    raise NotImplementedError


def mixtures_spectral_distribution(kernel: MercerKernel) -> D.Distribution:
    """
    Given a kernel, returns a spectral distribution approximation
    via mixture Gaussian sampling and a 2-d FFT.
    """
    raise NotImplementedError
