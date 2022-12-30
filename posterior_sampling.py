import torch
import torch.distributions as D

import math
from mercergp.kernels import MercerKernel
from mercergp.eigenvalue_gen import SmoothExponentialFasshauer
from ortho.builders import get_orthonormal_basis_from_sample
from ortho.basis_functions import smooth_exponential_eigenvalues
import matplotlib.pyplot as plt

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
        second_half_sample_size[0] = math.ceil(second_half_sample_size[0] / 2)
        # sample_part_1 = self.dim_1_dist.sample(sample_size)
        # sample_part_2 = self.dim_2_dist.sample(sample_size)
        sample = self.spectral_distribution.sample(sample_size)
        sample = torch.cat((sample[:, 0], sample[:, 1]))
        return sample


def kernel_fft(
    begin: torch.Tensor,
    end: torch.Tensor,
    frequency: float,
    kernel: MercerKernel,
) -> torch.Tensor:
    """
    Given a beginning range, end range, and a frequency, calculates the 2-d FFT
    for a Mercer kernel for the purpose of utilising Yaglom's theorem.
    """
    x_range = torch.linspace(begin, end, int(frequency))
    y_range = torch.linspace(begin, end, int(frequency))
    grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing="ij")
    values = kernel(x_range, y_range)
    fft = torch.fft.fft2(values)
    fft_shifted = torch.fft.fftshift(fft)
    return fft_shifted


def kernel_fft_decomposed(
    begin: torch.Tensor,
    end: torch.Tensor,
    frequency: float,
    kernel: MercerKernel,
) -> torch.Tensor:
    """
    Given a beginning range, end range, and a frequency, calculates the 2-d FFT
    for a Mercer kernel for the purpose of utilising Yaglom's theorem.

    This case utilises the fact that:
        F(ω_1, ω_2) = \int \int k(x,y)e^(-j2π(ω_1 x + ω_2 y))dxdy
                    = \int \int \sum_i λ_i φ_i(x) φ_i(y) e^(-j2π(ω_1 x)e^(-j2π(ω_2 y))dxdy
                    = \sum_i λ_i \int φ_i(x) e^(-2jπ(ω_1 x))dx \int φ_i(y) e^(-2j π(ω_2 y))dy

    """
    x_range = torch.linspace(begin, end, int(frequency))
    y_range = torch.linspace(begin, end, int(frequency))

    basis = kernel.basis
    phis = basis(x_range)
    phis_2 = basis(y_range)

    # get the per-side FFTs of the function
    fft_data = torch.fft.fftshift(
        torch.fft.fft(torch.fft.fftshift(phis), norm="ortho")
    )
    fft_data_2 = torch.fft.fftshift(
        torch.fft.fft(torch.fft.fftshift(phis_2), norm="ortho")
    )

    eigens = kernel.get_eigenvalues()

    # Outer product for the 2-d FFT
    full_fft = torch.einsum("il, kl -> ikl", fft_data, fft_data_2)

    complex_eigens = torch.complex(eigens, torch.zeros(eigens.shape))

    spectral_density = torch.einsum("l,ijl->ij", complex_eigens, full_fft).real
    return spectral_density


def histogram_spectral_distribution(
    kernel: MercerKernel,
    begin: torch.Tensor,
    end: torch.Tensor,
    frequency: float,
) -> D.Distribution:
    """
    Given a kernel, returns a spectral distribution approximation
    via histogram sampling and a 2-d FFT.
    """
    spectral_density = kernel_fft_decomposed(begin, end, frequency, kernel)

    raise NotImplementedError


def mixtures_spectral_distribution(kernel: MercerKernel) -> D.Distribution:
    """
    Given a kernel, returns a spectral distribution approximation
    via mixture Gaussian sampling and a 2-d FFT.
    """
    raise NotImplementedError


if __name__ == "__main__":
    """
    Test examples
    """

    def weight_function(x: torch.Tensor):
        """A standard weight function for test cases."""
        return torch.exp(-(x ** 2) / 2)

    order = 15
    sample_size = 1000
    sample_shape = torch.Size([sample_size])
    mixture_dist = False
    if mixture_dist:
        mixture = D.Normal(torch.Tensor([-2.0, 2.0]), torch.Tensor([2.0, 2.0]))
        categorical = D.Categorical(torch.Tensor([0.2, 0.8]))
        input_sample = D.MixtureSameFamily(categorical, mixture).sample(
            sample_shape
        )
    else:
        dist = D.Normal(0.0, 1.0)
        input_sample = dist.sample(sample_shape)

    basis = get_orthonormal_basis_from_sample(
        input_sample, weight_function, order
    )
    params = {
        "ard_parameter": torch.Tensor([[1.0]]),
        "variance_parameter": torch.Tensor([1.0]),
        "precision_parameter": torch.Tensor([1.0]),
        "noise_parameter": torch.Tensor([0.1]),
    }

    eigenvalues = SmoothExponentialFasshauer(order)(params)
    kernel = MercerKernel(order, basis, eigenvalues, params)
    begin = -3
    end = 3
    frequency = 1000
    fft_data = kernel_fft(begin, end, frequency, kernel)
    fft_data_2 = kernel_fft_decomposed(begin, end, frequency, kernel)
    # print(fft_data)
    # plt.imshow(fft_data.real)
    # plt.show()

    print(fft_data_2)
    plt.imshow(fft_data_2.real)
    plt.show()
    # plt.imshow(fft_data.real)
    # plt.show()
