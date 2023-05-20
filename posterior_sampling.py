import torch
import torch.distributions as D

import math
from mercergp.kernels import MercerKernel
from mercergp.eigenvalue_gen import SmoothExponentialFasshauer
from ortho.builders import OrthoBuilder
from ortho.basis_functions import (
    Basis,
    # smooth_exponential_eigenvalues_fasshauer,
    smooth_exponential_basis_fasshauer,
)

# from mercergp.builders import build_mercer_gp_fourier_posterior

# from ortho.basis_functions import smooth_exponential_eigenvalues
import matplotlib.pyplot as plt
from termcolor import colored

"""
This file contains classes and functions useful for the purpose of appropriate posterior
sampling for non-stationary (specifically, Mercer) kernels. The idea is based on work
in Wilson(2020) using Matheron's rule to generate Gaussian processes using a
prior component and a posterior component.
"""


class MercerSpectralDistribution(D.Categorical):
    """
    Represents the spectral distribution, where sampling has been
    converted to the shape necessary for the Random Fourier Feature
    basis format.
    """

    def __init__(self, frequency: int, probs: torch.Tensor):
        self.frequency = frequency
        self.half_frequency = math.floor(frequency / 2)
        self.pre_probs = probs
        self.flattened_probs = probs.flatten()
        super().__init__(probs=self.flattened_probs)

    def sample(self, feature_count: torch.Size):
        """
        To sample from the 2-d spectral density,
        we can treat the FFT result as a large categorical distribution.
        This has had its (2-d) probability "matrix" flattened so the corresponding
        probability refers to a single location in the 2-d matrix, so we get
        the right correlation (like a 2-d probability table). Then we can use
        modulo and floor division operators to extract the 2-d sample from the
        stride and step of the sample with respect to the frequency.

        Leaving the sample at integer values leads to periodic samples.
        As a result it is recommended to use either the histogram sampling
        or the Gaussian sampling.
        """
        categorical_sample = super().sample(
            torch.Size([math.floor(feature_count[0] / 2), 1])
        )
        omega_1 = torch.div(
            categorical_sample, self.frequency, rounding_mode="trunc"
        )
        omega_2 = categorical_sample % self.frequency
        # omega_2 = torch.abs(categorical_sample % self.frequency)
        omega_1 -= self.half_frequency
        omega_2 -= self.half_frequency
        sample = torch.cat((omega_1, omega_2)).double()
        return sample


class HistogramSpectralDistribution(MercerSpectralDistribution):
    def sample(self, feature_count: torch.Size):
        """
        Histogram sampling from the spectral distribution involves constructing
        a standard discrete integer frequency distribution (as represented by
        the MercerSpectralDistribution class), and adding Unif[-0.5, 0.5]
        random variables to the sample along the ω_1 and ω_2 axes.
        """
        sample = super().sample(feature_count)
        sample = sample + D.Uniform(-0.5, 0.5).sample(feature_count).unsqueeze(
            1
        )
        return sample


class GaussianMixtureSpectralDistribution(MercerSpectralDistribution):
    def sample(self, feature_count: torch.Size):
        """
        Gaussian mixture sampling from the spectral distribution involves constructing
        a standard discrete integer frequency distribution (as represented by
        the MercerSpectralDistribution class), and adding Gaussian distributed
        random variables to the sample along the ω_1 and ω_2 axes.
        The result should lead to a smoothing and perhaps a decrease in smaller frequencies...
        """
        sample = super().sample(feature_count)
        sample = sample + D.Normal(0.0, 0.5).sample(feature_count).unsqueeze(1)
        return sample


def kernel_fft(
    kernel: MercerKernel,
    begin: torch.Tensor,
    end: torch.Tensor,
    frequency: int,
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
    kernel: MercerKernel,
    begin: torch.Tensor,
    end: torch.Tensor,
    frequency: float,
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
    eigens = kernel.get_eigenvalues()
    phis = basis(x_range)
    phis_2 = basis(y_range)

    # get the per-side FFTs of the function
    fft_data = torch.fft.fftshift(
        torch.fft.fft(
            torch.fft.fftshift(phis),
            # n=math.floor(frequency / 2),
            # n=2 * frequency,
            norm="ortho",
            dim=0,
        )
    )
    fft_data_2 = torch.fft.fftshift(
        torch.fft.fft(
            torch.fft.fftshift(phis_2),
            # n=math.floor(frequency / 2),
            # n=2 * frequency,
            norm="ortho",
            dim=0,
        )
    )

    # fft_data = torch.fft.fft(torch.fft.fftshift(phis), norm="ortho", dim=0)

    # fft_data_2 = torch.fft.fft(torch.fft.fftshift(phis_2), norm="ortho", dim=0)
    # Outer product for the 2-d FFT
    full_fft = torch.einsum(
        "l, il, jl -> ij", eigens, fft_data.real, fft_data_2.real
    )

    return torch.abs(full_fft)
    # return full_fft


def kernel_fft_decomposed_real(
    kernel: MercerKernel,
    begin: torch.Tensor,
    end: torch.Tensor,
    frequency: float,
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
    eigens = kernel.get_eigenvalues()
    phis = basis(x_range)
    phis_2 = basis(y_range)

    # get the per-side FFTs of the function
    fft_data = torch.fft.fftshift(
        torch.fft.rfft(torch.fft.fftshift(phis), norm="ortho", dim=0)
    )
    fft_data_2 = torch.fft.fftshift(
        torch.fft.rfft(torch.fft.fftshift(phis_2), norm="ortho", dim=0)
    )

    # fft_data = torch.fft.fft(torch.fft.fftshift(phis), norm="ortho", dim=0)

    # fft_data_2 = torch.fft.fft(torch.fft.fftshift(phis_2), norm="ortho", dim=0)
    # Outer product for the 2-d FFT
    full_fft = torch.einsum("l, il, jl -> ij", eigens, fft_data, fft_data_2)

    return torch.abs(full_fft)


def integer_spectral_distribution(
    kernel: MercerKernel,
    begin: torch.Tensor,
    end: torch.Tensor,
    frequency: int,
) -> D.Distribution:
    """
    Given a kernel, returns a spectral distribution approximation
    via histogram sampling and a 2-d FFT.
    """
    # generate a matrix of spectral density evaluations
    spectral_density = kernel_fft_decomposed(kernel, begin, end, frequency)
    # plt.imshow(spectral_density)
    # plt.show()
    distribution = MercerSpectralDistribution(frequency, spectral_density)
    return distribution


def histogram_spectral_distribution(
    kernel: MercerKernel,
    begin: torch.Tensor,
    end: torch.Tensor,
    frequency: int,
) -> D.Distribution:
    """
    Given a kernel, returns a spectral distribution approximation
    via histogram sampling and a 2-d FFT.
    """
    # generate a matrix of spectral density evaluations
    spectral_density = kernel_fft_decomposed(kernel, begin, end, frequency)
    distribution = HistogramSpectralDistribution(frequency, spectral_density)
    return distribution


def gaussian_spectral_distribution(
    kernel: MercerKernel,
    begin: torch.Tensor,
    end: torch.Tensor,
    frequency: int,
) -> D.Distribution:
    """
    Given a kernel, returns a spectral distribution approximation
    via histogram sampling and a 2-d FFT.
    """
    # generate a matrix of spectral density evaluations
    spectral_density = kernel_fft_decomposed(kernel, begin, end, frequency)
    distribution = GaussianMixtureSpectralDistribution(
        frequency, spectral_density
    )
    return distribution


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
    test_posterior_sampling_correlation = True

    def weight_function(x: torch.Tensor):
        """A standard weight function for test cases."""
        return torch.exp(-(x**2) / 2)

    order = 20
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

    basis = (
        OrthoBuilder(order)
        .set_sample(input_sample)
        .set_weight_function(weight_function)
        .get_orthonormal_basis()
    )

    params = {
        "ard_parameter": torch.Tensor([[10.0]]),
        "variance_parameter": torch.Tensor([1.0]),
        "precision_parameter": torch.Tensor([1.0]),
        "noise_parameter": torch.Tensor([0.1]),
    }

    eigenvalues = SmoothExponentialFasshauer(order)(params)
    basis = Basis(smooth_exponential_basis_fasshauer, 1, order, params)
    kernel = MercerKernel(order, basis, eigenvalues, params)
    begin = -4
    end = 4
    frequency = 2000
    fft_data = kernel_fft(kernel, begin, end, frequency)
    fft_data_2 = kernel_fft_decomposed(kernel, begin, end, frequency)
    integer = True
    if integer:
        spectral_dist = integer_spectral_distribution(
            kernel, begin, end, frequency
        )
    else:
        spectral_dist = histogram_spectral_distribution(
            kernel, begin, end, frequency
        )

    sample = spectral_dist.sample(torch.Size([8000]))
    plt.hist(sample.numpy().flatten(), bins=87)
    # plt.hist(sample[:, 1].numpy().flatten(), bins=87)
    plt.show()
